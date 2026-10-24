"""Stage 1 FSV (issue #8, spec §17): read the persisted canonical store, dedup maps and ledger back through separate code
paths and audit three boundaries with negative controls. Report: <root>/evidence/fsv/<issue>/<run>/stage1-fsv.json.

    python -I scripts/canon/fsv_stage1.py --root R --out O --version V --issue 8 [--sample-shards 2] [--plant]

B1 raw bytes -> canonical bytes: re-digest sampled raw shards against the manifest (receipt) digests; re-digest the canon
   shard file and every stored doc hash; re-extract sampled rows from raw with pyarrow/zstd/zip/gzip directly and an
   inline normalisation (not contracts.py) and compare bytes.
B2 duplicate map -> recount of unique identities: population recount from manifests; every exact pair re-hashes equal;
   every dup_of is kept; layers are disjoint; unique == docs - exact - near; sampled near pairs re-estimated from the
   signature files; sampled spans compared byte-for-byte between the cut doc and the kept doc; ledger chain verified and
   its digests equal the files on disk.
B3 planted corpus (--plant): exact_b, near_b and the 300-byte passage are caught with the right dup_of; the 150-byte
   passage, the 30%-overlap pair and the distinct doc survive.
Negative controls: a byte-flipped copy of a canon shard must not reproduce its sealed digest; a byte-flipped copy of the
   ledger must refuse MNLM_LEDGER_CHAIN_BROKEN.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, digest_file, iter_canon_docs, load_sidecars, manifest_rows, read_jsonl, run_stamp, sealed_shards, write_json  # noqa: E402
from ledger import LedgerError, read_entries, verify_chain  # noqa: E402

CHECKS: list[dict] = []


def check(name: str, passed: bool, **detail):
    detail.pop("ok", None)
    CHECKS.append({"check": name, "ok": bool(passed), **detail})
    print(("  ok   " if passed else "  FAIL ") + name + (f"  {json.dumps(detail, default=str)[:200]}" if not passed else ""), flush=True)


RAW_DIR_ALIAS = {"sefaria-export-nc": "sefaria-export"}  # restated from scripture.py: the NC partition shares its raw tree
SCRIPTURE_SOURCES = {"ebible-english", "berean-bsb", "open-english-bible", "unfoldingword-uhb", "unfoldingword-ugnt", "unfoldingword-ult", "unfoldingword-ust",
                     "tanach-us-uxlc", "sblgnt", "byztxt-rp2018", "nestle1904", "ubsicap-versification", "crosswire-sword-pd", "openbible-crossrefs",
                     "sefaria-export", "sefaria-export-nc", "stepbible-data"}


def norm(s: str | None) -> str | None:
    """Independent restatement of canon-v1 (kept separate from contracts.py on purpose)."""
    if s is None:
        return None
    t = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    ls = [x.rstrip() for x in t.split("\n")]
    while ls and ls[0] == "":
        ls = ls[1:]
    while ls and ls[-1] == "":
        ls = ls[:-1]
    return "\n".join(ls) if ls else None


def pick(n: int, k: int, seed: str) -> list[int]:
    if n == 0:
        return []
    out = sorted({int.from_bytes(blake3.blake3(f"{seed}:{i}".encode()).digest(8), "big") % n for i in range(k)})
    return out


def raw_row_text(source: str, row: dict, raw_dir: Path, target: int, cuy_root: Path, f: dict | None = None, text: str | None = None) -> str | None:
    """Re-extracts the raw text of row ordinal `target` through libraries directly (B1 independent path)."""
    files = row["raw_files"]
    if source in ("fineweb-edu-sample-10BT", "wikipedia-en-20231101", "opencodeinstruct"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(raw_dir / files[0])
        base = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if target < base + n:
                if source == "opencodeinstruct":
                    t = pf.read_row_group(rg, columns=["input", "output"]).slice(target - base, 1).to_pylist()[0]
                    return norm("\n\n".join(str(t[c]) for c in ("input", "output") if t[c] is not None))
                return norm(pf.read_row_group(rg, columns=["text"]).slice(target - base, 1).to_pylist()[0]["text"])
            base += n
        return None
    if source.startswith("nemotron-cc"):
        import zstandard

        with (raw_dir / files[0]).open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
            for i, line in enumerate(io.TextIOWrapper(zr, encoding="utf-8")):
                if i == target:
                    return norm(json.loads(line).get("text")) if line.strip() else None
        return None
    if source == "cuyahoga-eighth-district":
        with (cuy_root / "ingest" / files[0]).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == target:
                    return norm(json.loads(line)["text"])
        return None
    if source == "plant-v1":
        with (raw_dir / files[0]).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == target:
                    return norm(json.loads(line)["text"])
        return None
    if source == "gdelt-events-90d":
        i = 0
        for name in files:
            with zipfile.ZipFile(raw_dir / name) as z:
                for line in z.read(z.namelist()[0]).decode("utf-8", "surrogateescape").split("\n"):
                    line = line.rstrip("\r")
                    if not line:
                        continue
                    if i == target:
                        return line
                    i += 1
        return None
    if source == "wikidata5m":
        if row["shard"] == "wikidata5m_text":
            with gzip.open(raw_dir / files[0], "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == target:
                        return norm(line.rstrip("\n").partition("\t")[2])
            return None
        return "SKIP"
    if source == "identity-v1":  # issue #10: the raw row re-read without the contract module
        from fsv_identity import reparse as id_reparse

        return norm(id_reparse(raw_dir / files[0], row["shard"], target))
    if f is not None and source in SCRIPTURE_SOURCES:  # issue #9: own parsers locate the record by its explicit coordinate
        from fsv_reparse import reparse

        cands = reparse(source, raw_dir / files[0] if not Path(files[0]).is_absolute() else Path(files[0]), f, text or "")
        if cands == "SKIP":
            return "SKIP"
        normed = [norm(c) for c in cands]
        return text if text in normed else (normed[0] if normed else None)
    return "SKIP"


def raw_path(src: str, name: str, raw_dir: Path, cuy_root: Path) -> Path:
    return (cuy_root / "ingest" / name) if src == "cuyahoga-eighth-district" else raw_dir / name


def b1_shard(job: dict) -> list[tuple]:
    """One canon shard: shard digest, every doc hash, and 3 independently re-extracted rows. Returns check tuples."""
    src, sh, row, raw_dir, cuy_root = job["src"], job["sh"], job["row"], Path(job["raw_dir"]), Path(job["cuy_root"])
    canon_dir = Path(job["canon_dir"])
    out = []
    fb3, fsize = digest_file(canon_dir / f"{sh['shard']}.jsonl.zst")
    out.append((f"B1.shard_digest.{src}.{sh['shard']}", fb3 == row["out_blake3"] and fsize == row["out_bytes"], {"got": fb3[:12], "sealed": row["out_blake3"][:12]}))
    b3, rws, ns = load_sidecars(canon_dir, sh["shard"])
    n, bad, nonconf, raw_kind, texts = 0, 0, 0, 0, {}
    targets = set(pick(sh["docs"], 3, f"b1rows:{src}:{sh['shard']}"))
    for i, rec in enumerate(iter_canon_docs(canon_dir, sh["shard"])):
        tb = rec["text"].encode("utf-8")
        h = blake3.blake3(tb).digest(length=16)
        if h.hex() != rec["b3"] or b3[i].tobytes() != h or int(ns[i]) != len(tb) or int(rws[i]) != rec["row"]:
            bad += 1
        if norm(rec["text"]) != rec["text"]:  # canon-v1 conformance of the stored text (readback finding, issue #9)
            if rec["kind"] == "prose":
                nonconf += 1
            else:  # row and alias records are the raw line by contract (the exact-row and alias layers key on raw bytes);
                raw_kind += 1  # 8 of 4.8 M wikidata5m alias lines carry decomposed combining sequences and are reported, not failed
        if i in targets:
            texts[i] = (rec["row"], rec["text"], rec.get("f"))
        n += 1
    out.append((f"B1.doc_hashes.{src}.{sh['shard']}", bad == 0 and n == sh["docs"] == row["docs"], {"docs": n, "bad": bad, "sealed": sh["docs"]}))
    out.append((f"B1.canon_text_conformance.{src}.{sh['shard']}", nonconf == 0, {"docs": n, "nonconforming": nonconf, "raw_kind_not_nfc": raw_kind}))
    agree = skipped = 0
    bad_rows = []
    for i, (rowno, text, f) in texts.items():
        try:
            got = raw_row_text(src, row, raw_dir, rowno, cuy_root, f, text)
        except Exception as exc:  # noqa: BLE001 a re-parse failure is a disagreement with its cause, never a skip
            got = f"MNLM_FSV_REPARSE_FAILED {type(exc).__name__}: {exc}"[:200]
        if got == "SKIP":
            skipped += 1
        elif got == text:
            agree += 1
        else:
            bad_rows.append({"row": rowno, "ref": (f or {}).get("ref"), "got": (got or "")[:160], "want": text[:160]})
    out.append((f"B1.reextract.{src}.{sh['shard']}", agree + skipped == len(texts), {"agree": agree, "skipped": skipped, "sampled": len(texts), "rows": bad_rows[:3]}))
    return out


def b1(paths: Paths, meta: dict, sample_shards: int, cuy_root: Path, census_root: Path, all_shards: bool, workers: int) -> None:
    """Sampled mode: `sample_shards` population shards per source. Census mode (--all-shards): every sealed shard of every
    source under the canon version, alias and synthetic-twin shards included, and every raw file re-digested once."""
    import multiprocessing as mp

    jobs, raw_files = [], {}
    if all_shards:
        src_dirs = sorted(d for d in (paths.out / "canon" / paths.canon_version).iterdir() if (d / "manifest.jsonl").exists())
        per_src = {d.name: [{"source": d.name, "shard": r["shard"], "docs": r["docs"], "canon_blake3": r["out_blake3"], "kind": r["kind"]} for r in sealed_shards(d)] for d in src_dirs}
    else:
        per_src = {}
        for sh in meta["shards"]:
            per_src.setdefault(sh["source"], []).append(sh)
    raw_loc: dict[tuple, Path] = {}
    for src, shards in per_src.items():
        rows = manifest_rows(paths.canon(src))
        raw_dir = census_root / "raw" / RAW_DIR_ALIAS.get(src, src)
        stored = {r["path"]: r["stored"] for r in read_jsonl(raw_dir / "receipts.jsonl") if r.get("stored")} if (raw_dir / "receipts.jsonl").exists() else {}
        idx = range(len(shards)) if all_shards else pick(len(shards), sample_shards, f"b1:{src}")
        for j in idx:
            sh = shards[j]
            row = rows[sh["shard"]]
            if src != "plant-v1":
                for name, sha in zip(row["raw_files"], row["raw_sha256"]):
                    raw_files[(src, name)] = sha
                    st = stored.get(name)  # an over-long basename is stored digest-suffixed; the receipt names the real file
                    raw_loc[(src, name)] = (Path(st) if Path(st).is_absolute() else raw_dir / st) if st else raw_path(src, name, raw_dir, cuy_root)
            jobs.append({"src": src, "sh": sh, "row": row, "raw_dir": str(raw_dir), "cuy_root": str(cuy_root), "canon_dir": str(paths.canon(src))})
    print(f"B1: {len(jobs)} shards, {len(raw_files)} raw files, workers={workers}", flush=True)
    keys = list(raw_files)
    with mp.Pool(min(workers, max(1, len(keys)))) as pool:
        digests = dict(zip(keys, pool.map(sha256_stream, [raw_loc[k] for k in keys])))
    bad = [k for k in keys if digests[k] != raw_files[k]]
    check("B1.raw_digests_match_receipts", not bad, files=len(keys), mismatched=[f"{s}/{n}" for s, n in bad[:5]])
    with mp.Pool(min(workers, len(jobs))) as pool:
        for out in pool.imap_unordered(b1_shard, jobs):
            for name, ok, detail in out:
                check(name, ok, **detail)


def sha256_stream(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


EXPECTED_REFUSALS = ("_LICENCE_REFUSED", "_EMPTY_VERSION")  # a licence gate or an empty edition is a lawful refusal; any other code is a defect


def b0_sources_sealed(paths: Paths) -> None:
    """Every source directory under the canon version holds at least one sealed shard, and every refusal carries an
    expected code. A source whose every shard was refused must not vanish from the population silently (run 11 of
    scripture-v1 lost openbible-crossrefs to a header-parse refusal with every other check green); a source with 99 of
    100 shards refused by a parser error must not pass on the strength of the one."""
    base = paths.out / "canon" / paths.canon_version
    for d in sorted(p for p in base.iterdir() if (p / "manifest.jsonl").exists()):
        rows = manifest_rows(d)
        codes: dict[str, int] = {}
        for r in rows.values():
            if r.get("error"):
                codes[r["error"].get("code", "?")] = codes.get(r["error"].get("code", "?"), 0) + 1
        sealed = len(sealed_shards(d))
        unexpected = {c: n for c, n in codes.items() if not c.endswith(EXPECTED_REFUSALS)}
        check(f"B0.source_sealed.{d.name}", sealed > 0 and not unexpected, sealed=sealed, refused=codes, unexpected=unexpected)


def b0_receipts(paths: Paths, census: str) -> None:
    """Every census manifest entry of every downloadable source has an accepted receipt (digest or content-length)."""
    cdir = paths.root / "census" / census
    rows = json.loads((cdir / "census.json").read_text(encoding="utf-8"))["sources"]
    for mpath in sorted(cdir.glob("*/manifest.jsonl")):
        src = mpath.parent.name
        access = rows.get(src, {}).get("access", {})
        if access.get("type") == "by-host":
            access = access.get(HOST) or {}
        if access.get("type") not in ("hf-tree", "cc-pathlist", "gdelt-masterfilelist", "http-head", "git-tree", "gcs-prefix", "local-authored"):
            continue  # aiwonder-local or manifest-copy sources are not downloaded here (local-authored carries in-place sha256 receipts)
        rpath = paths.root / "raw" / src / "receipts.jsonl"
        rec = {r["path"]: r for r in read_jsonl(rpath)}
        ents = [json.loads(l) for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries = [e["path"] for e in ents]
        acc = []
        for e in ents:
            r = rec.get(e["path"], {})
            ok = r.get("verified") is True or (r.get("verified") is None and r.get("attestation") == "content-length" and r.get("size_match") is True)
            # a receipt for an older publication or build (other digest of any kind, other size) is not a receipt
            stale = (e.get("digest") is not None and (r.get("expected") != e["digest"] or r.get("expected_kind") != e.get("digest_kind"))) \
                or (isinstance(e.get("bytes"), int) and e["bytes"] >= 0 and r.get("bytes") != e["bytes"])
            if ok and not stale:
                acc.append(e["path"])
        check(f"B0.receipts_complete.{src}", len(acc) == len(entries), accepted=len(acc), manifest=len(entries))


def _texts_of(job: tuple) -> dict[str, bytes]:
    canon_dir, s, sh, rows = job
    rows, last, out = set(rows), rows[-1], {}
    for rec in iter_canon_docs(Path(canon_dir), sh):
        if rec["row"] > last:
            break
        if rec["row"] in rows:
            out[f"{s}|{sh}|{rec['row']}"] = rec["text"].encode("utf-8")
    return out


def b2(paths: Paths, meta: dict, pop: dict, threshold: int, big: bool = False) -> dict:
    d = paths.dedup
    n = len(pop["n"])
    expected = sum(sh["docs"] for sh in meta["shards"])
    check("B2.population_recount", n == expected == meta["docs"], population=n, manifests=expected)
    ex_r, ex_k = np.load(d / "exact.removed.npy"), np.load(d / "exact.kept_of.npy")
    b3 = pop["b3"].view(np.uint64).reshape(-1, 2)
    removed_all = np.zeros(n, dtype=bool)
    removed_all[ex_r] = True
    check("B2.exact_pairs_rehash_equal", bool(np.all(b3[ex_r] == b3[ex_k])) and not np.any(removed_all[ex_k]) and bool(np.all(np.diff(ex_r) > 0)) and bool(np.all(ex_k < ex_r)),
          pairs=int(len(ex_r)))
    nr_r, nr_k, est = np.load(d / "near.removed.npy"), np.load(d / "near.kept_of.npy"), np.load(d / "near.est_jaccard.npy")
    near_in_exact = bool(np.any(removed_all[nr_r]))
    removed_all[nr_r] = True
    check("B2.near_pairs_disjoint_kept", (not near_in_exact) and not np.any(removed_all[nr_k]) and bool(np.all(np.diff(nr_r) > 0)) and bool(np.all(nr_k < nr_r)) and bool(np.all(pop["prose"][nr_r])),
          pairs=int(len(nr_r)))
    agree, tested = 0, 0
    sigs: dict[int, np.ndarray] = {}
    for j in pick(len(nr_r), 2000 if big else 200, "b2near"):
        pair = []
        for idx in (int(nr_r[j]), int(nr_k[j])):
            k = int(pop["shard"][idx])
            if k not in sigs:
                sigs[k] = np.load(paths.sig(meta["shards"][k]["source"]) / f"{meta['shards'][k]['shard']}.sig.npy", mmap_mode="r")
            first = int(np.searchsorted(pop["shard"], k, side="left"))
            pair.append(sigs[k][idx - first])
        tested += 1
        agree += int(abs(float(np.mean(pair[0] == pair[1])) - float(est[j])) < 1e-6)
    check("B2.near_estimates_replay", agree == tested, tested=tested, agree=agree)
    if (d / "near.links.npy").exists():
        from contracts import near_removable

        links, lrem = np.load(d / "near.links.npy"), np.load(d / "near.links_removed.npy")
        removable = np.array([near_removable(s) for s in meta["sources"]], dtype=bool)
        same_removed = set(links[lrem, 0].tolist()) == set(nr_r.tolist())
        check("B2.near_links_consistent", same_removed and bool(np.all(removable[pop["src"][links[lrem, 0]]])) and not np.any(removable[pop["src"][links[~lrem, 0]]])
              and not np.any(removed_all[links[~lrem, 0]]), links=int(len(links)), exempt_kept=int(np.sum(~lrem)))
    uniq = np.load(d / "unique.npy")
    us = json.loads((d / "unique.summary.json").read_text(encoding="utf-8"))
    check("B2.unique_recount", len(uniq) == n - len(ex_r) - len(nr_r) == us["unique"] and bool(np.all(np.diff(uniq) > 0)) and not np.any(removed_all[uniq]),
          unique=int(len(uniq)), docs=n, exact=int(len(ex_r)), near=int(len(nr_r)))
    # An exact-removed doc's kept copy may itself be near-removed later (readback of p0-v1 found 9,745); the chain must end
    # in the unique set after one hop, and a near kept copy is never removed, so two hops never occur.
    near_to = np.full(n, -1, dtype=np.int64)
    near_to[nr_r] = nr_k
    hop = near_to[ex_k]
    final = np.where(hop >= 0, hop, ex_k)
    check("B2.exact_chain_ends_in_unique", not np.any(removed_all[final]), exact_kept_of_near_removed=int(np.sum(hop >= 0)))
    spans = np.load(d / "spans.npy") if (d / "spans.npy").exists() else np.empty((0, 5), dtype=np.int64)
    alive = ~removed_all
    ok_struct = bool(np.all(alive[spans[:, 0]]) and np.all(alive[spans[:, 3]]) and np.all(pop["prose"][spans[:, 0]]) and np.all(spans[:, 2] - spans[:, 1] >= threshold)) if len(spans) else True
    check("B2.spans_structure", ok_struct, spans=int(len(spans)))
    ss = json.loads((d / "spans.summary.json").read_text(encoding="utf-8")) if (d / "spans.summary.json").exists() else {}
    check("B2.spans_count_matches_summary", len(spans) == ss.get("spans_cut", 0), spans=int(len(spans)), summary=ss.get("spans_cut"))
    agree, tested = 0, 0
    if len(spans):
        import zstandard

        want = pick(len(spans), 1000 if big else 100, "b2spans")
        wanted, recs = set(want), {}
        with (d / "spans.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
            for i, line in enumerate(io.TextIOWrapper(zr, encoding="utf-8")):
                if i in wanted:  # only the sampled records are parsed (8.6 M records, 10.9 GB resident, before)
                    recs[i] = json.loads(line)
        need: dict[tuple[int, str], set] = {}
        for j in want:
            r = recs[j]
            for ref in (r["ref"], r["dup_of_ref"]):
                s, sh, rw = ref.split("|")
                need.setdefault((s, sh), set()).add(int(rw))
        texts: dict[str, bytes] = {}
        import multiprocessing as mp

        with mp.Pool(min(16, len(need))) as pool:  # the sampled spans touch 117 shards on p0-v1 (84 GB canonical): read in parallel, not serially
            for part in pool.imap_unordered(_texts_of, [(str(paths.canon(s)), s, sh, sorted(rows)) for (s, sh), rows in need.items()]):
                texts.update(part)
        by_kind: dict[str, list[int]] = {}
        for j in want:
            r = recs[j]
            a = texts[r["ref"]][r["start"]:r["end"]]
            b = texts[r["dup_of_ref"]][r["dup_start"]:r["dup_start"] + (r["end"] - r["start"])]
            tested += 1
            ok = int(a == b and len(a) >= threshold and a.decode("utf-8", "strict") is not None
                     and texts[r["ref"]][:r["start"]].decode("utf-8", "strict") is not None)
            agree += ok
            by_kind.setdefault(r["resolved_by"], [0, 0])
            by_kind[r["resolved_by"]][0] += ok
            by_kind[r["resolved_by"]][1] += 1
    check("B2.spans_bytes_equal_kept_copy", agree == tested, tested=tested, agree=agree, by_resolved=by_kind if len(spans) else {})
    chain = verify_chain(paths.ledger)
    latest: dict[str, dict] = {}  # the ledger is append-only: a re-run supersedes, and the latest seal per kind must match disk
    for e in read_entries(paths.ledger):
        if e["subject"] == paths.version and e["kind"] in ("dedup.layers", "dedup.spans"):
            latest[e["kind"]] = e
    entries = list(latest.values())
    dig_ok, dig_n = True, 0
    for e in entries:
        for f, h in e["payload_digests"]["digests"].items():
            p = d / f if (d / f).exists() else paths.sa / f
            dig_n += 1
            dig_ok &= p.exists() and digest_file(p)[0] == h
    check("B2.ledger_chain_and_digests", chain.get("ok", False) and dig_ok and len(entries) >= 2, entries=len(entries), digests=dig_n, head=chain.get("head", "")[:16])
    return {"exact": int(len(ex_r)), "near": int(len(nr_r)), "unique": int(len(uniq)), "spans": int(len(spans)), "ledger_head": chain.get("head")}


def b3_plant(paths: Paths, meta: dict, pop: dict) -> None:
    labels = {}
    for rec in iter_canon_docs(paths.canon("plant-v1"), "plant"):
        labels[rec["row"]] = rec["f"]["label"]
    idx = {v: k for k, v in labels.items()}  # row == population index for the single plant shard
    d = paths.dedup
    ex = dict(zip(np.load(d / "exact.removed.npy").tolist(), np.load(d / "exact.kept_of.npy").tolist()))
    nr = dict(zip(np.load(d / "near.removed.npy").tolist(), np.load(d / "near.kept_of.npy").tolist()))
    sp = np.load(d / "spans.npy")
    uniq = set(np.load(d / "unique.npy").tolist())
    check("B3.exact_b_removed_dup_of_exact_a", ex == {idx["exact_b"]: idx["exact_a"]}, exact=ex)
    check("B3.near_b_removed_dup_of_near_a", nr == {idx["near_b"]: idx["near_a"]}, near=nr)
    rows = [tuple(x) for x in sp.tolist()]
    check("B3.span_b_cut_dup_of_span_a", len(rows) == 1 and rows[0][0] == idx["span_b"] and rows[0][3] == idx["span_a"] and rows[0][2] - rows[0][1] >= 300, spans=rows)
    check("B3.neg.short_passage_not_cut", all(r[0] not in (idx["short_a"], idx["short_b"]) for r in rows))
    check("B3.neg.near_miss_survives", idx["miss_a"] in uniq and idx["miss_b"] in uniq and all(r[0] != idx["miss_b"] for r in rows))
    check("B3.neg.distinct_survives", idx["distinct"] in uniq and all(r[0] != idx["distinct"] for r in rows))
    check("B3.empty_counted_not_doc", "empty" not in idx and manifest_rows(paths.canon("plant-v1"))["plant"]["empty"] == 1)


def negatives(paths: Paths, meta: dict) -> None:
    sh = meta["shards"][0]
    src_file = paths.canon(sh["source"]) / f"{sh['shard']}.jsonl.zst"
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "shard.jsonl.zst"
        shutil.copy(src_file, t)
        b = bytearray(t.read_bytes())
        b[len(b) // 2] ^= 0x01
        t.write_bytes(bytes(b))
        check("NEG.tampered_shard_digest_differs", digest_file(t)[0] != sh["canon_blake3"])
        lt = Path(td) / "ledger.jsonl"
        lines = paths.ledger.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[-1])
        e["payload_digests"] = {"tampered": "x"}
        lines[-1] = json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        lt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            verify_chain(lt)
            check("NEG.tampered_ledger_refuses", False, note="verify_chain accepted a tampered entry")
        except LedgerError as err:
            check("NEG.tampered_ledger_refuses", err.code == "MNLM_LEDGER_CHAIN_BROKEN", code=err.code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--issue", default="8")
    ap.add_argument("--sample-shards", type=int, default=2)
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--all-shards", action="store_true", help="census mode: every shard of every source, every raw file")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--census", default="p0-v1")
    ap.add_argument("--cuyahoga-root", default="/zfs/archive/calyx/freelaw-cuyahoga")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    from artifact_seals import Audit
    Audit(paths).require("spans")
    meta = json.loads((paths.dedup / "population.json").read_text(encoding="utf-8"))
    z = np.load(paths.dedup / "population.npz")
    pop = {k: z[k] for k in z.files}
    print(f"FSV Stage 1 {args.version} on {HOST}: {meta['docs']} docs over {len(meta['shards'])} shards")
    b0_receipts(paths, args.census)
    b0_sources_sealed(paths)
    b1(paths, meta, args.sample_shards, Path(args.cuyahoga_root), paths.root, args.all_shards, args.workers)
    counts = b2(paths, meta, pop, args.threshold, big=args.all_shards)
    if args.plant:
        b3_plant(paths, meta, pop)
    if "identity-v1" in meta["sources"]:  # issue #10: every anchor must resolve to the sealed verifier
        from fsv_identity import checks as identity_checks

        identity_checks(paths, check)
    negatives(paths, meta)
    run = f"stage1-{args.version}-{run_stamp()}"
    ev = paths.evidence(args.issue, run)
    passed = sum(c["ok"] for c in CHECKS)
    report = {"version": args.version, "host": HOST, "run": run, "mode": "census" if args.all_shards else "sampled", "checks": CHECKS, "passed": passed, "total": len(CHECKS), "counts": counts,
              "population": {k: meta[k] for k in ("docs", "prose_docs", "canonical_bytes", "sources")}}
    write_json(ev / "stage1-fsv.json", report)
    for f in ("population.json", "exact.summary.json", "near.summary.json", "unique.summary.json", "spans.summary.json"):
        if (paths.dedup / f).exists():
            shutil.copy(paths.dedup / f, ev / f)
    print(f"{passed}/{len(CHECKS)} checks passed; report {ev / 'stage1-fsv.json'}")
    return 0 if passed == len(CHECKS) else 3


if __name__ == "__main__":
    sys.exit(main())
