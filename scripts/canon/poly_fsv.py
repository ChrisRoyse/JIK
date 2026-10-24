"""Stage 1 FSV for the Poly sealed store layer (issue #8; workstation). Independent readback of poly_dedup.py's outputs.

    python -I scripts/canon/poly_fsv.py --root D:/MNLMData --out D:/MNLMData --version V --store S --issue 8 [--sample 12]

B1 Poly artifact bytes -> row digests: for sampled generations re-digest every canonical artifact (sha256) against
   Poly's manifest and ours; re-parse the rows with an inline reader per declared encoding (not poly_dedup.iter_rows) and
   recompute both BLAKE3-128 digests; the bytes must equal the stored .full.bin / .rec.bin exactly; every census
   generation must have a sealed row.
B2 exact map -> recount: every removed row re-hashes equal to its kept row; kept rows are never removed; unique == rows
   minus removed; the recurrence counts are recomputed with an independent np.unique; the ledger chain verifies and the
   latest dedup.poly digests equal the files on disk.
Negative controls: one changed character in a field changes both digests; a changed receipt clock changes `full` only
   (the recurrence key ignores exactly the declared clocks); a byte-flipped copy of a .full.bin changes its digest; a
   byte-flipped copy of the ledger refuses MNLM_LEDGER_CHAIN_BROKEN.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, digest_file, read_jsonl, run_stamp, write_json  # noqa: E402
from ledger import LedgerError, read_entries, verify_chain  # noqa: E402

CLOCKS = ("observed_at_unix_ms", "ingested_at_unix_ms")
CHECKS: list[dict] = []


def check(name: str, passed: bool, **detail):
    detail.pop("ok", None)
    CHECKS.append({"check": name, "ok": bool(passed), **detail})
    print(("  ok   " if passed else "  FAIL ") + name + (f"  {json.dumps(detail, default=str)[:240]}" if not passed else ""), flush=True)


def pick(n: int, k: int, seed: str) -> list[int]:
    return sorted({int.from_bytes(blake3.blake3(f"{seed}:{i}".encode()).digest(8), "big") % n for i in range(k)}) if n else []


def dig(row: dict) -> tuple[bytes, bytes]:
    c = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")  # noqa: E731
    return blake3.blake3(c(row)).digest(length=16), blake3.blake3(c({k: v for k, v in row.items() if k not in CLOCKS})).digest(length=16)


def ours_artifacts(r: dict, canon: list[dict]) -> list[dict]:
    """Our manifest row's artifact list; rows sealed before partition support carry one artifact as canonical_path/sha256."""
    if "artifacts" in r:
        return [{**a, "encoding": a.get("encoding") or c.get("canonical_record_encoding")} for a, c in zip(r["artifacts"], canon)]
    return [{"path": r["canonical_path"], "sha256": r["sha256"], "encoding": canon[0].get("canonical_record_encoding") if canon else None}]


def rows_of(path: Path, encoding: str) -> list[dict]:
    """Inline reader per declared encoding (separate from poly_dedup.iter_rows)."""
    if encoding == "json_rows_object":
        return json.loads(path.read_bytes())["rows"]
    if encoding == "json_array":
        return json.loads(path.read_bytes())
    if encoding == "kalshi_markets_gzip_json_v1":
        return json.loads(gzip.decompress(path.read_bytes()))["markets"]
    if encoding in ("csv_with_headers", "sec_nport_holding_tsv_v1"):
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="," if encoding == "csv_with_headers" else "\t"))
    raise ValueError(f"unknown encoding {encoding}")


def b1(paths: Paths, store: Path, census: str, sample: int, rows_by_gen: dict) -> None:
    entries = read_jsonl(paths.root / "census" / census / "poly-sealed-structured" / "manifest.jsonl")
    missing = [e["path"] for e in entries if tuple(e["path"].split("/")) not in rows_by_gen]
    check("B1.every_census_generation_sealed", not missing, census=len(entries), sealed=len(rows_by_gen), missing=missing[:5])
    gens = sorted(rows_by_gen)
    small = [g for g in gens if rows_by_gen[g]["bytes"] < 300 << 20]
    for j in pick(len(small), sample, "b1poly"):
        src, gen = small[j]
        r = rows_by_gen[(src, gen)]
        gdir = store / "sources" / src / "generations" / gen
        m = json.loads((gdir / "manifest.json").read_bytes())
        canon = [a for a in m["artifacts"] if a.get("role") == "canonical"]
        ok_sha, full, rec, n = True, bytearray(), bytearray(), 0
        ours_list = ours_artifacts(r, canon)
        for a, ours in zip(canon, ours_list):
            p = gdir / a["relative_path"]
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            ok_sha &= sha == a["sha256"] == ours["sha256"] and a["relative_path"] == ours["path"]
            for row in rows_of(p, a.get("canonical_record_encoding")):
                f, c = dig(row)
                full += f
                rec += c
                n += 1
        check(f"B1.artifact_digests.{src}.{gen[:16]}", ok_sha and len(canon) == len(ours_list), artifacts=len(canon))
        sf = (paths.canon("poly-sealed-structured") / src / f"{gen}.full.bin").read_bytes()
        sr = (paths.canon("poly-sealed-structured") / src / f"{gen}.rec.bin").read_bytes()
        check(f"B1.row_digests_replay.{src}.{gen[:16]}", bytes(full) == sf and bytes(rec) == sr and n == r["rows"] == int(m["canonical_record_count"]), rows=n, sealed=r["rows"])


def b2(paths: Paths, rows_by_gen: dict) -> dict:
    d = paths.dedup
    pop = json.loads((d / "poly.population.json").read_text(encoding="utf-8"))
    fulls, recs = [], []
    for g in pop["generations"]:
        fulls.append(np.fromfile(paths.canon("poly-sealed-structured") / g["source"] / f"{g['generation']}.full.bin", dtype=np.uint8).reshape(-1, 16))
        recs.append(np.fromfile(paths.canon("poly-sealed-structured") / g["source"] / f"{g['generation']}.rec.bin", dtype=np.uint8).reshape(-1, 16))
    full, rec = np.concatenate(fulls), np.concatenate(recs)
    n = len(full)
    check("B2.population_recount", n == pop["rows"] == sum(g["rows"] for g in pop["generations"]) and len(pop["generations"]) == len(rows_by_gen), rows=n)
    ex_r, ex_k = np.load(d / "poly.exact.removed.npy"), np.load(d / "poly.exact.kept_of.npy")
    removed = np.zeros(n, dtype=bool)
    removed[ex_r] = True
    f64 = full.view(np.uint64).reshape(-1, 2)
    check("B2.exact_pairs_rehash_equal", bool(np.all(f64[ex_r] == f64[ex_k])) and not np.any(removed[ex_k]) and bool(np.all(np.diff(ex_r) > 0)) and bool(np.all(ex_k < ex_r)), pairs=int(len(ex_r)))
    uniq = np.load(d / "poly.unique.npy")
    summ = json.loads((d / "poly.summary.json").read_text(encoding="utf-8"))
    check("B2.unique_recount", len(uniq) == n - len(ex_r) == summ["unique"] and not np.any(removed[uniq]), unique=int(len(uniq)))
    fv = f64.view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
    _, cnt = np.unique(fv, return_counts=True)
    check("B2.exact_groups_independent_unique", int(cnt.sum() - len(cnt)) == len(ex_r) and int(cnt.max()) == summ["exact"]["largest_group"], removed_recount=int(cnt.sum() - len(cnt)))
    rv = rec.view(np.uint64).reshape(-1, 2).view([("a", np.uint64), ("b", np.uint64)]).reshape(-1)
    _, rcnt = np.unique(rv, return_counts=True)
    check("B2.recurrence_counts_independent_unique", int(len(rcnt)) == summ["recurrence"]["distinct_keys"] and int(rcnt.sum() - len(rcnt)) == summ["recurrence"]["rows_in_repeats"]
          and int(rcnt.max()) == summ["recurrence"]["max_multiplicity"] and summ["recurrence"]["removed"] == 0, distinct=int(len(rcnt)))
    chain = verify_chain(paths.ledger)
    latest = None
    for e in read_entries(paths.ledger):
        if e["kind"] == "dedup.poly" and e["subject"] == paths.version:
            latest = e
    dig_ok = latest is not None
    if latest:
        for f, h in latest["payload_digests"]["digests"].items():
            p = d / f if (d / f).exists() else paths.canon("poly-sealed-structured") / f
            dig_ok &= p.exists() and digest_file(p)[0] == h
    check("B2.ledger_chain_and_digests", chain.get("ok", False) and dig_ok, head=chain.get("head", "")[:16], seq=latest["seq"] if latest else None)
    return {"rows": int(n), "exact_removed": int(len(ex_r)), "unique": int(len(uniq)), "recurrence_distinct": int(len(rcnt)), "ledger_head": chain.get("head")}


def negatives(paths: Paths, store: Path, rows_by_gen: dict) -> None:
    src, gen = sorted(g for g in rows_by_gen if rows_by_gen[g]["bytes"] < 1 << 20)[0]
    r = rows_by_gen[(src, gen)]
    m = json.loads((store / "sources" / src / "generations" / gen / "manifest.json").read_bytes())
    a = ours_artifacts(r, [x for x in m["artifacts"] if x.get("role") == "canonical"])[0]
    row = rows_of(store / "sources" / src / "generations" / gen / a["path"], a["encoding"])[0]
    f0, c0 = dig(row)
    k = next(k for k in row if k not in CLOCKS and isinstance(row[k], str) and row[k])
    mut = dict(row)
    mut[k] = row[k][:-1] + ("x" if row[k][-1] != "x" else "y")
    f1, c1 = dig(mut)
    check("NEG.field_change_changes_both_digests", f1 != f0 and c1 != c0, field=k)
    if any(c in row for c in CLOCKS):
        clk = dict(row)
        for c in CLOCKS:
            if c in clk:
                clk[c] = (clk[c] or 0) + 1
        f2, c2 = dig(clk)
        check("NEG.receipt_clock_change_changes_full_only", f2 != f0 and c2 == c0)
    else:
        check("NEG.receipt_clock_change_changes_full_only", False, note=f"sampled row of {src} carries no receipt clock; pick another")
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "x.full.bin"
        shutil.copy(paths.canon("poly-sealed-structured") / src / f"{gen}.full.bin", t)
        b = bytearray(t.read_bytes())
        b[0] ^= 0x01
        t.write_bytes(bytes(b))
        check("NEG.tampered_sidecar_digest_differs", digest_file(t)[0] != digest_file(paths.canon("poly-sealed-structured") / src / f"{gen}.full.bin")[0])
        lt = Path(td) / "ledger.jsonl"
        lines = paths.ledger.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[-1])
        e["payload_digests"] = {"tampered": "x"}
        lines[-1] = json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        lt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            verify_chain(lt)
            check("NEG.tampered_ledger_refuses", False)
        except LedgerError as err:
            check("NEG.tampered_ledger_refuses", err.code == "MNLM_LEDGER_CHAIN_BROKEN", code=err.code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--census", default="p0-v1")
    ap.add_argument("--issue", default="8")
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    rows_by_gen = {}
    for r in read_jsonl(paths.canon("poly-sealed-structured") / "manifest.jsonl"):
        key = (r["source"], r["generation"])
        if r.get("error"):
            rows_by_gen.pop(key, None)
        else:
            rows_by_gen[key] = r
    print(f"FSV Stage 1 (Poly) {args.version} on {HOST}: {len(rows_by_gen)} sealed generations")
    b1(paths, Path(args.store), args.census, args.sample, rows_by_gen)
    counts = b2(paths, rows_by_gen)
    negatives(paths, Path(args.store), rows_by_gen)
    run = f"stage1-poly-{args.version}-{run_stamp()}"
    ev = paths.evidence(args.issue, run)
    passed = sum(c["ok"] for c in CHECKS)
    write_json(ev / "poly-fsv.json", {"version": args.version, "host": HOST, "run": run, "checks": CHECKS, "passed": passed, "total": len(CHECKS), "counts": counts})
    for f in ("poly.summary.json", "poly.population.json"):
        shutil.copy(paths.dedup / f, ev / f)
    print(f"{passed}/{len(CHECKS)} checks passed; report {ev / 'poly-fsv.json'}")
    return 0 if passed == len(CHECKS) else 3


if __name__ == "__main__":
    sys.exit(main())
