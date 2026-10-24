"""Stage 1 for the Poly sealed structured store (issue #8; workstation only; Poly's tree is read, never written).

    python -I scripts/canon/poly_dedup.py --root D:/MNLMData --out D:/MNLMData --version p0micro-v1 --census p0-v1 \
        --store C:/code/poly/var/poly/native-reset-v1/state/production-source-store [--workers 12] [--step hash|layers|all]

Contract `poly-canon-v1` (frozen): the canonical store is Poly's sealed store *by reference*; each generation's canonical
artifact (manifest.json role=canonical) is re-digested with sha256 and must equal the manifest's; its `rows` are read in
file order; per row two BLAKE3-128 digests of the sort-keyed compact JSON: `full` (every field) and `rec` (the row minus
the receipt clocks RECEIPT_CLOCKS). Exact layer = collisions of `full` (keep-first by source, captured_at, generation,
row); recurrence layer = collisions of `rec`, counted and never removed (spec A5: a re-observed association is evidence);
`rows_in_repeats`/`recurrence_rows` count repeat occurrences beyond the first per key (N minus distinct keys), not the
rows sitting in groups of size >= 2. A generation's `sha256` field is the artifact's sha256 for one artifact and, for a
partitioned generation, the BLAKE3 over the artifact sha256 list (`artifacts_key`); the field name is kept for the seal.
Outputs: <out>/canon/<v>/poly-sealed-structured/<src>/<gen>.{full,rec}.bin + manifest.jsonl (skip-if-sealed by sha256,
PC-07); <out>/dedup/<v>/poly.* arrays and summaries; one ledger entry. Cost O(rows) hashing + one lexsort over rows.
Files above STREAM_BYTES are parsed with ijson (streaming); smaller ones with json.loads (C, faster).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import blake3
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, append_jsonl, digest_file, ledger_append, now_iso, read_jsonl, write_json  # noqa: E402

CONTRACT_ID = "poly-canon-v1"
SOURCE_ID = "poly-sealed-structured"
RECEIPT_CLOCKS = ("observed_at_unix_ms", "ingested_at_unix_ms")
STREAM_BYTES = 256 << 20


def canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def artifacts_key(canon: list[dict]) -> str:
    """One key for a generation's canonical artifact set: BLAKE3 over the newline-joined sha256 list in manifest order."""
    return blake3.blake3("\n".join(a["sha256"] for a in canon).encode()).hexdigest()


def canonical_artifacts(store: Path, src: str, gen: str) -> list[dict]:
    m = json.loads((store / "sources" / src / "generations" / gen / "manifest.json").read_bytes())
    return [a for a in m.get("artifacts", []) if a.get("role") == "canonical"]


def row_digests(row: dict) -> tuple[bytes, bytes]:
    full = blake3.blake3(canon_json(row)).digest(length=16)
    rec = blake3.blake3(canon_json({k: v for k, v in row.items() if k not in RECEIPT_CLOCKS})).digest(length=16)
    return full, rec


ENCODINGS = ("json_rows_object", "json_array", "kalshi_markets_gzip_json_v1", "csv_with_headers", "sec_nport_holding_tsv_v1")


def iter_rows(path: Path, size: int, encoding: str):
    """One record per declared canonical encoding (measured on 2026-10-15 across all 22,036 artifacts: the five below).
    An undeclared encoding refuses; nothing is guessed from the bytes."""
    if encoding == "json_rows_object":
        if size <= STREAM_BYTES:
            d = json.loads(path.read_bytes())
            if not isinstance(d, dict) or "rows" not in d:
                raise ValueError("MNLM_POLY_SCHEMA_MISMATCH: json_rows_object artifact has no 'rows'")
            yield from d["rows"]
        else:
            import ijson

            with path.open("rb") as f:
                yield from ijson.items(f, "rows.item", use_float=True)
    elif encoding == "json_array":
        d = json.loads(path.read_bytes())
        if not isinstance(d, list):
            raise ValueError("MNLM_POLY_SCHEMA_MISMATCH: json_array artifact is not a list")
        yield from d
    elif encoding == "kalshi_markets_gzip_json_v1":
        import gzip

        d = json.loads(gzip.decompress(path.read_bytes()))
        if not isinstance(d, dict) or not isinstance(d.get("markets"), list):
            raise ValueError("MNLM_POLY_SCHEMA_MISMATCH: kalshi gzip artifact has no 'markets' list")
        yield from d["markets"]
    elif encoding in ("csv_with_headers", "sec_nport_holding_tsv_v1"):
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter="," if encoding == "csv_with_headers" else "\t")
    else:
        raise ValueError(f"MNLM_POLY_ENCODING_UNKNOWN: {encoding!r} is not one of {ENCODINGS}")


def process_generation(args) -> dict:
    entry, store, out_dir = args
    src, gen = entry["path"].split("/")
    t0 = time.time()
    try:
        gdir = store / "sources" / src / "generations" / gen
        m = json.loads((gdir / "manifest.json").read_bytes())
        canon = [a for a in m.get("artifacts", []) if a.get("role") == "canonical"]  # partitioned generations hold several, in manifest order
        if not canon:
            raise ValueError("MNLM_POLY_CANON_ARTIFACTS: no canonical artifact")
        fulls, recs, n, size, parts, streamed = bytearray(), bytearray(), 0, 0, [], False
        for a in canon:
            p = gdir / a["relative_path"]
            psize = p.stat().st_size
            if psize != int(a["byte_count"]):
                raise ValueError(f"MNLM_POLY_CANON_SIZE: {a['relative_path']} {psize} != manifest {a['byte_count']}")
            h = hashlib.sha256()
            with p.open("rb") as f:
                while b := f.read(1 << 22):
                    h.update(b)
            if h.hexdigest() != a["sha256"]:
                raise ValueError(f"MNLM_POLY_CANON_DIGEST_MISMATCH: {a['relative_path']} {h.hexdigest()[:12]} != {a['sha256'][:12]}")
            n0 = n
            for row in iter_rows(p, psize, a.get("canonical_record_encoding")):
                fu, re_ = row_digests(row)
                fulls += fu
                recs += re_
                n += 1
            if a.get("canonical_record_count") is not None and n - n0 != int(a["canonical_record_count"]):
                raise ValueError(f"MNLM_POLY_ROWCOUNT_MISMATCH: {a['relative_path']} read {n - n0}, artifact {a['canonical_record_count']}")
            parts.append({"path": a["relative_path"], "sha256": a["sha256"], "bytes": psize, "rows": n - n0, "encoding": a.get("canonical_record_encoding")})
            size += psize
            streamed |= psize > STREAM_BYTES
        if n != int(m.get("canonical_record_count") or 0):
            raise ValueError(f"MNLM_POLY_ROWCOUNT_MISMATCH: read {n}, manifest {m.get('canonical_record_count')}")
        d = out_dir / src
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{gen}.full.bin").write_bytes(bytes(fulls))
        (d / f"{gen}.rec.bin").write_bytes(bytes(recs))
        return {"source": src, "generation": gen, "contract": CONTRACT_ID, "artifacts": parts, "sha256": artifacts_key(canon), "bytes": size,
                "rows": n, "captured_at_unix_ms": m.get("captured_at_unix_ms"), "completeness": m.get("completeness"), "streamed": streamed,
                "secs": round(time.time() - t0, 3), "host": HOST, "ts": now_iso()}
    except Exception as e:  # noqa: BLE001 - recorded as a typed row (PC-32)
        return {"source": src, "generation": gen, "contract": CONTRACT_ID, "host": HOST, "ts": now_iso(),
                "error": {"code": str(e).split(":")[0] if str(e).startswith("MNLM_") else "MNLM_POLY_GENERATION_FAILED", "message": str(e)[:500], "trace": traceback.format_exc()[-1500:]}}


def step_hash(paths: Paths, census: str, store: Path, workers: int) -> dict:
    entries = read_jsonl(paths.root / "census" / census / SOURCE_ID / "manifest.jsonl")
    out_dir = paths.canon(SOURCE_ID)
    have = {}
    for r in read_jsonl(out_dir / "manifest.jsonl"):
        if not r.get("error"):
            have[(r["source"], r["generation"])] = r
    todo = []
    for e in entries:
        src, gen = e["path"].split("/")
        r = have.get((src, gen))
        if r and (out_dir / src / f"{gen}.full.bin").exists():
            canon = canonical_artifacts(store, src, gen)
            key = artifacts_key(canon)
            legacy = "artifacts" not in r and len(canon) == 1 and r["sha256"] == canon[0]["sha256"]  # rows sealed before partition support
            if r["sha256"] == key or legacy:
                continue
        todo.append(e)
    todo.sort(key=lambda e: -e["bytes"])  # biggest first so the tail of the pool is short
    print(f"poly: {len(entries)} generations, {len(entries) - len(todo)} sealed, {len(todo)} to hash, workers={workers}", flush=True)
    stats = {"generations": len(entries), "skipped": len(entries) - len(todo), "hashed": 0, "errors": 0, "rows": 0, "bytes": 0}
    t0 = time.time()
    if todo:
        with mp.Pool(min(workers, len(todo))) as pool:
            for i, r in enumerate(pool.imap_unordered(process_generation, [(e, store, out_dir) for e in todo])):
                append_jsonl(out_dir / "manifest.jsonl", r)
                if r.get("error"):
                    stats["errors"] += 1
                    print(f"  REFUSED {r['source']}/{r['generation'][:20]}: {r['error']['code']} {r['error']['message'][:120]}", flush=True)
                else:
                    stats["hashed"] += 1
                    stats["rows"] += r["rows"]
                    stats["bytes"] += r["bytes"]
                if i % 500 == 0:
                    print(f"  {i}/{len(todo)} done, {stats['rows']} rows, {round(time.time() - t0)} s", flush=True)
    stats["secs"] = round(time.time() - t0, 1)
    print(json.dumps(stats), flush=True)
    return stats


def step_layers(paths: Paths) -> dict:
    out_dir = paths.canon(SOURCE_ID)
    rows = {}
    for r in read_jsonl(out_dir / "manifest.jsonl"):
        if not r.get("error"):
            rows[(r["source"], r["generation"])] = r
    gens = sorted(rows.values(), key=lambda r: (r["source"], r["captured_at_unix_ms"] or 0, r["generation"]))
    fulls, recs, gen_i, counts = [], [], [], []
    for k, r in enumerate(gens):
        f = np.fromfile(out_dir / r["source"] / f"{r['generation']}.full.bin", dtype=np.uint8).reshape(-1, 16)
        c = np.fromfile(out_dir / r["source"] / f"{r['generation']}.rec.bin", dtype=np.uint8).reshape(-1, 16)
        if len(f) != r["rows"] or len(c) != r["rows"]:
            raise SystemExit(f"MNLM_POLY_SIDECAR_COUNT {r['source']}/{r['generation']}: {len(f)} {len(c)} != {r['rows']}")
        fulls.append(f)
        recs.append(c)
        gen_i.append(np.full(len(f), k, dtype=np.uint32))
        counts.append(len(f))
    full, rec, gi = np.concatenate(fulls), np.concatenate(recs), np.concatenate(gen_i)
    del fulls, recs
    n = len(full)
    src_names = sorted({r["source"] for r in gens})
    src_of_gen = np.array([src_names.index(r["source"]) for r in gens], dtype=np.uint8)
    d = paths.dedup
    d.mkdir(parents=True, exist_ok=True)

    def collisions(keys: np.ndarray):
        k = keys.view(np.uint64).reshape(-1, 2)
        order = np.lexsort((k[:, 1], k[:, 0]))
        s = k[order]
        same = np.all(s[1:] == s[:-1], axis=1)
        starts = np.flatnonzero(np.concatenate(([True], ~same)))
        sizes = np.diff(np.concatenate((starts, [n])))
        grp = np.repeat(np.arange(len(starts)), sizes)
        head = order[starts][grp]
        link = order != head
        rem, kept = order[link], head[link]
        o2 = np.argsort(rem, kind="stable")
        return rem[o2], kept[o2], sizes

    t0 = time.time()
    ex_r, ex_k, ex_sizes = collisions(full)
    np.save(d / "poly.exact.removed.npy", ex_r)
    np.save(d / "poly.exact.kept_of.npy", ex_k)
    starts_of_gen = np.concatenate(([0], np.cumsum(counts)))
    with (d / "poly.exact.jsonl.zst").open("wb") as fh, zstandard.ZstdCompressor(level=6).stream_writer(fh) as zw:
        for r_, k_ in zip(ex_r.tolist(), ex_k.tolist()):
            gr, gk = gens[int(gi[r_])], gens[int(gi[k_])]
            zw.write((json.dumps({"removed": r_, "ref": f"poly|{gr['source']}|{gr['generation']}|{r_ - int(starts_of_gen[gi[r_]])}", "dup_of": k_,
                                  "dup_of_ref": f"poly|{gk['source']}|{gk['generation']}|{k_ - int(starts_of_gen[gi[k_]])}"}, sort_keys=True) + "\n").encode())
    rc_r, rc_k, rc_sizes = collisions(rec)
    per_src = {}
    for i, s in enumerate(src_names):
        m = src_of_gen[gi] == i
        per_src[s] = {"rows": int(m.sum()), "exact_removed": int(np.sum(src_of_gen[gi[ex_r]] == i)), "recurrence_rows": int(np.sum(src_of_gen[gi[rc_r]] == i)),
                      "generations": int(np.sum(src_of_gen == i))}
    alive = np.ones(n, dtype=bool)
    alive[ex_r] = False
    np.save(d / "poly.unique.npy", np.flatnonzero(alive))
    pop = {"version": paths.version, "source": SOURCE_ID, "contract": CONTRACT_ID, "receipt_clocks": list(RECEIPT_CLOCKS), "sources": src_names,
           "generations": [{"source": r["source"], "generation": r["generation"], "rows": r["rows"], "sha256": r["sha256"], "captured_at_unix_ms": r["captured_at_unix_ms"]} for r in gens],
           "rows": int(n), "host": HOST, "ts": now_iso()}
    write_json(d / "poly.population.json", pop)
    summary = {"version": paths.version, "rows": int(n), "generations": len(gens), "sources": len(src_names),
               "exact": {"removed": int(len(ex_r)), "groups": int(len(ex_sizes)), "largest_group": int(ex_sizes.max())},
               "recurrence": {"distinct_keys": int(len(rc_sizes)), "rows_in_repeats": int(len(rc_r)), "keys_seen_ge2": int(np.sum(rc_sizes >= 2)),
                              "max_multiplicity": int(rc_sizes.max()), "removed": 0, "note": "counted, never removed (spec A5)"},
               "unique": int(alive.sum()), "per_source": per_src, "secs": round(time.time() - t0, 1), "host": HOST, "ts": now_iso()}
    digests = {f: digest_file(d / f)[0] for f in ("poly.exact.removed.npy", "poly.exact.kept_of.npy", "poly.unique.npy", "poly.population.json")}
    digests["manifest.jsonl"] = digest_file(out_dir / "manifest.jsonl")[0]
    e = ledger_append(paths, f"poly_dedup@{HOST}", "dedup.poly", paths.version, {"summary": summary, "digests": digests})
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(d / "poly.summary.json", summary)
    print(json.dumps(summary, indent=1)[:1500], flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--census", default="p0-v1")
    ap.add_argument("--store", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--step", default="all", choices=["hash", "layers", "all"])
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    rc = 0
    if args.step in ("hash", "all"):
        st = step_hash(paths, args.census, Path(args.store), args.workers)
        rc = 3 if st["errors"] else 0
    if args.step in ("layers", "all"):
        step_layers(paths)
    return rc


if __name__ == "__main__":
    sys.exit(main())
