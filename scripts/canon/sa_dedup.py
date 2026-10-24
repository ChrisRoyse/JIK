"""Stage 1 op 2 (issue #8): exact-substring deduplication by suffix array (Lee et al. 2022) over the unique prose population.

    python -I scripts/canon/sa_dedup.py --root R --out O --version V --tool ~/mnlm/tools/deduplicate-text-datasets \
        [--step corpus|table|similar|map|all] [--threshold 200] [--part-bytes 1500000000] [--concurrent 4] [--threads 32]

corpus:  <out>/sa/<v>/corpus.bin = canonical text bytes of every unique prose doc in population order, each followed by
         one 0x00 separator; offsets.npy (int64, doc starts, plus the end) and docs.npy (population indices). O(bytes).
table:   `make-part` over parts of --part-bytes (+100000 byte overlap, the tool's HACK), --concurrent at once (each
         needs about 10x its part in RAM), then `merge` and concatenation into corpus.bin.table.bin. Pointer width is
         ceil(log2(bytes)/8); the table must be an exact multiple of the corpus size.
similar: `self-similar --length-threshold T` then `collect` -> ranges.txt ("a b" byte ranges duplicated at least twice).
map:     read exact duplicate groups before collect merges overlapping windows. Verify each window, exclude crossings
         of item boundaries, keep the earliest valid occurrence, then merge windows only with identical source/target
         displacement. Record UTF-8-aligned spans with byte-equal counterparts; never infer provenance from union ranges.
         Spans are recorded, never applied here. Reuse requires an accepted, parent-bound completion seal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import blake3
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, digest_file, iter_canon_docs, ledger_append, now_iso, save_npy, write_json  # noqa: E402
from dedup import load_population, ref_of  # noqa: E402
from artifact_seals import Audit, accept, publish, refuse  # noqa: E402
from suffix_protocol import normalize_collect, read_ranges  # noqa: E402
from substring_ranges import resolve_groups, group_files, RESOLVER  # noqa: E402

HACK = 100000
SEP = b"\x00"


def sh(cmd: list[str], log: Path, cwd: Path | None = None) -> subprocess.Popen:
    f = log.open("ab")
    f.write((" ".join(cmd) + "\n").encode())
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)


# ---------------------------------------------------------------------------------------------------------------- corpus
def build_corpus(paths: Paths) -> dict:
    audit = Audit(paths)
    audit.require("unique")
    pop, meta = load_population(paths, audit)
    uniq = np.load(paths.dedup / "unique.npy")
    docs = uniq[pop["prose"][uniq]]
    sa = paths.sa
    sa.mkdir(parents=True, exist_ok=True)
    if (sa / "corpus.json").exists():
        audit.require("corpus")
        prior = json.loads((sa / "corpus.json").read_text(encoding="utf-8"))
        if prior["docs"] == len(docs) and np.array_equal(np.load(sa / "docs.npy"), docs) and (sa / "corpus.bin").stat().st_size == prior["bytes"] \
                and len(np.load(sa / "offsets.npy", mmap_mode="r")) == len(docs) + 1:
            print(f"corpus: {prior['docs']} docs already written for this unique population; skipping (PC-07)", flush=True)
            return prior
        refuse("MNLM_SA_CORPUS_MISMATCH", "cached corpus geometry")
    parents = audit.begin("corpus")
    offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    want = {}  # shard index -> set of rows wanted, in order
    for d in docs.tolist():
        want.setdefault(int(pop["shard"][d]), []).append(d)
    t0, pos, j = time.time(), 0, 0
    with (sa / "corpus.bin.tmp").open("wb") as out:
        for k, sh_meta in enumerate(meta["shards"]):
            if k not in want:
                continue
            rows_wanted = pop["row"][want[k]]
            if not np.all(np.diff(rows_wanted) > 0):
                raise SystemExit(f"MNLM_SA_ORDER shard {k}: wanted rows not ascending")
            it = iter(rows_wanted.tolist())
            nxt = next(it, None)
            if digest_file(paths.canon(sh_meta["source"]) / (sh_meta["shard"] + ".jsonl.zst"))[0] != sh_meta["canon_blake3"]:
                refuse("MNLM_SA_CANON_MISMATCH", sh_meta["shard"])
            for rec in iter_canon_docs(paths.canon(sh_meta["source"]), sh_meta["shard"]):
                if nxt is None:
                    break
                if rec["row"] != nxt:
                    continue
                tb = rec["text"].encode("utf-8")
                if blake3.blake3(tb).digest(length=16) != bytes(pop["b3"][docs[j]]):
                    refuse("MNLM_SA_CANON_MISMATCH", f"{sh_meta['shard']} row {nxt}")
                if len(tb) != int(pop["n"][docs[j]]):
                    raise SystemExit(f"MNLM_SA_BYTES {sh_meta['source']}/{sh_meta['shard']} row {nxt}: {len(tb)} != population {int(pop['n'][docs[j]])}")
                out.write(tb)
                out.write(SEP)
                pos += len(tb) + 1
                j += 1
                offsets[j] = pos
                nxt = next(it, None)
            if nxt is not None:
                raise SystemExit(f"MNLM_SA_MISSING_ROW {sh_meta['source']}/{sh_meta['shard']} row {nxt} not found in canon shard")
        out.flush()
        os.fsync(out.fileno())
    if j != len(docs):
        raise SystemExit(f"MNLM_SA_DOC_COUNT wrote {j} of {len(docs)}")
    os.replace(sa / "corpus.bin.tmp", sa / "corpus.bin")
    save_npy(sa / "offsets.npy", offsets)
    save_npy(sa / "docs.npy", docs)
    b3, nbytes = digest_file(sa / "corpus.bin")
    info = {"docs": int(len(docs)), "bytes": int(nbytes), "corpus_blake3": b3, "secs": round(time.time() - t0, 1), "host": HOST, "ts": now_iso()}
    write_json(sa / "corpus.json", info)
    audit.finish("corpus", parents)
    return info


# ----------------------------------------------------------------------------------------------------------------- table
def build_table(paths: Paths, tool: Path, part_bytes: int, concurrent: int, threads: int) -> dict:
    audit = Audit(paths)
    audit.require("corpus")
    sa, data = paths.sa, paths.sa / "corpus.bin"
    size = data.stat().st_size
    if min(part_bytes, concurrent, threads) < 1 or size == 0:
        refuse("MNLM_SA_GEOMETRY", "positive worker/part settings and a nonempty prose corpus required")
    exe = tool / "target" / "release" / "dedup_dataset"
    log = sa / "table.log"
    parts = []
    s = 0
    while s < size:
        e = min(s + part_bytes + HACK, size)
        parts.append((s, e))
        if e == size:
            break
        s += part_bytes
    fact = max(1, int(math.ceil(math.log2(size) / 8)))
    t0 = time.time()
    final = sa / "corpus.bin.table.bin"
    corpus_b3 = corpus_identity(paths, audit)
    if (sa / "table.json").exists():
        audit.require("table")
        print(f"table: corpus.bin.table.bin present at {size * fact} B for corpus {corpus_b3[:12]}; skipping (PC-07)", flush=True)
        return json.loads((sa / "table.json").read_text(encoding="utf-8"))
    parents = audit.begin("table")
    part_parents = {"corpus": parents["corpus"], "tool_blake3": digest_file(exe)[0]}
    pending = list(parts)
    for attempt in range(3):
        todo = []
        for s, e in pending:
            pf = sa / f"corpus.bin.part.{s}-{e}"
            tf = sa / f"corpus.bin.part.{s}-{e}.table.bin"
            pfact = int(math.ceil(math.log2(e - s) / 8))  # a part's pointers address the part, not the corpus (the tool's own rule)
            part_name = f"part-{s}-{e}"
            if (sa / (part_name + ".seal.json")).exists():
                accept(sa, part_name, (pf.name, tf.name), part_parents)
                continue
            pf.unlink(missing_ok=True)
            tf.unlink(missing_ok=True)
            todo.append((s, e))
        if not todo:
            break
        print(f"make-part: {len(todo)} parts to build (attempt {attempt + 1}), {concurrent} concurrent, fact={fact}", flush=True)
        running: list[subprocess.Popen] = []
        launched = []
        for s, e in todo:
            while len(running) >= concurrent:
                running = [p for p in running if p.poll() is None]
                time.sleep(2)
            running.append(sh([str(exe), "make-part", "--data-file", str(data), "--start-byte", str(s), "--end-byte", str(e)], log))
            launched.append((s, e, running[-1]))
        for p in running:
            p.wait()
        for s, e, p in launched:
            pf, tf = sa / f"corpus.bin.part.{s}-{e}", sa / f"corpus.bin.part.{s}-{e}.table.bin"
            pfact = max(1, int(math.ceil(math.log2(e - s) / 8)))
            if p.wait() != 0 or not pf.exists() or not tf.exists() or pf.stat().st_size != e - s or tf.stat().st_size != (e - s) * pfact:
                refuse("MNLM_SA_MAKE_PART_FAILED", f"part {s}-{e}; see table.log")
            publish(sa, f"part-{s}-{e}", (pf.name, tf.name), part_parents)
        pending = todo
    else:
        raise SystemExit("MNLM_SA_MAKE_PART_FAILED: parts still missing or mis-sized after 3 attempts; see table.log")
    (sa / "tmp").mkdir(exist_ok=True)
    for p in (sa / "tmp").glob("out.table.bin*"):
        p.unlink()
    cmd = [str(exe), "merge", "--output-file", str(sa / "tmp" / "out.table.bin"), "--num-threads", str(threads)]
    for s, e in parts:
        cmd += ["--suffix-path", str(sa / f"corpus.bin.part.{s}-{e}")]
    p = sh(cmd, log)
    if p.wait() != 0:
        raise SystemExit("MNLM_SA_MERGE_FAILED: see table.log (ulimit -Sn 100000 is required)")
    pieces = sorted((sa / "tmp").glob("out.table.bin.*"), key=lambda q: int(q.suffix[1:]))
    with (sa / "corpus.bin.table.bin.tmp").open("wb") as out:
        for q in pieces:
            with q.open("rb") as f:
                while b := f.read(1 << 24):
                    out.write(b)
    os.replace(sa / "corpus.bin.table.bin.tmp", data.with_name("corpus.bin.table.bin"))
    tsize = (sa / "corpus.bin.table.bin").stat().st_size
    if tsize % size != 0 or tsize // size != fact:
        raise SystemExit(f"MNLM_SA_TABLE_SIZE table {tsize} B is not {fact} x corpus {size} B")
    for q in pieces:
        q.unlink()
    for s, e in parts:  # parts are regenerable working files; retire them now (PC-28)
        (sa / f"corpus.bin.part.{s}-{e}").unlink(missing_ok=True)
        (sa / f"corpus.bin.part.{s}-{e}.table.bin").unlink(missing_ok=True)
        (sa / f"part-{s}-{e}.seal.json").unlink(missing_ok=True)
    info = {"parts": len(parts), "part_bytes": part_bytes, "pointer_bytes": fact, "table_bytes": int(tsize), "corpus_blake3": corpus_b3, "secs": round(time.time() - t0, 1), "ts": now_iso()}
    info["tool_blake3"] = digest_file(exe)[0]
    write_json(sa / "table.json", info)
    audit.finish("table", parents)
    return info


def corpus_identity(paths: Paths, audit=None) -> str:
    """The sealed corpus digest every later artifact (table, ranges, spans) is stamped with and checked against."""
    (audit or Audit(paths)).require("corpus")
    return json.loads((paths.sa / "corpus.json").read_text(encoding="utf-8"))["corpus_blake3"]


def require_corpus(paths: Paths, name: str, audit=None) -> str:
    """Refuses when <name>.json describes a different corpus than corpus.json (a stale table or ranges file)."""
    audit = audit or Audit(paths)
    audit.require(name.removesuffix(".json"))
    want = corpus_identity(paths, audit)
    p = paths.sa / name
    got = json.loads(p.read_text(encoding="utf-8")).get("corpus_blake3") if p.exists() else None
    if got != want:
        raise SystemExit(f"MNLM_SA_CORPUS_MISMATCH {name}: built for corpus {str(got)[:12]}, corpus.json is {want[:12]}")
    return want


# --------------------------------------------------------------------------------------------------------------- similar
def self_similar(paths: Paths, tool: Path, threshold: int, threads: int) -> dict:
    audit = Audit(paths)
    corpus_b3 = require_corpus(paths, "table.json", audit)
    if threshold < 1 or threads < 1:
        refuse("MNLM_SA_GEOMETRY", "positive threshold and threads required")
    sa, data = paths.sa, paths.sa / "corpus.bin"
    exe = tool / "target" / "release" / "dedup_dataset"
    if digest_file(exe)[0] != json.loads((sa / "table.json").read_text())["tool_blake3"]:
        refuse("MNLM_SA_TOOL_MISMATCH", "table and similar executables differ")
    parents = audit.begin("similar")
    cache = sa / "cache"
    cache.mkdir(exist_ok=True)
    for p in cache.iterdir():
        p.unlink()
    log = sa / "similar.log"
    t0 = time.time()
    p = sh([str(exe), "self-similar", "--data-file", str(data), "--length-threshold", str(threshold), "--cache-dir", str(cache), "--num-threads", str(threads)], log)
    if p.wait() != 0:
        raise SystemExit("MNLM_SA_SELF_SIMILAR_FAILED: see similar.log")
    with (sa / "collect.txt").open("wb") as out:
        p = subprocess.Popen([str(exe), "collect", "--data-file", str(data), "--cache-dir", str(cache), "--length-threshold", str(threshold)], stdout=out, stderr=log.open("ab"))
        if p.wait() != 0:
            raise SystemExit("MNLM_SA_COLLECT_FAILED: see similar.log")
    protocol = normalize_collect(sa)
    read_ranges(sa / "ranges.txt")  # unknown output cannot acquire a completion seal
    info = {"threshold": threshold, "secs": round(time.time() - t0, 1), "ranges_bytes": (sa / "ranges.txt").stat().st_size, "corpus_blake3": corpus_b3, "ts": now_iso()}
    info.update(protocol)
    info["group_files"] = group_files(data)
    write_json(sa / "similar.json", info)
    audit.finish("similar", parents)
    return info


# ------------------------------------------------------------------------------------------------------------------- map
def map_spans(paths: Paths, threshold: int) -> dict:
    sa = paths.sa
    audit = Audit(paths)
    corpus_b3 = require_corpus(paths, "similar.json", audit)
    pop, meta = load_population(paths, audit)
    if json.loads((sa / "similar.json").read_text())["threshold"] != threshold:
        refuse("MNLM_SA_THRESHOLD_MISMATCH", "map threshold differs from collected ranges")
    offsets, docs = np.load(sa / "offsets.npy"), np.load(sa / "docs.npy")
    if len(offsets) != len(docs) + 1:
        raise SystemExit(f"MNLM_SA_OFFSETS {sa}: offsets.npy has {len(offsets)} entries for {len(docs)} docs")
    r = read_ranges(sa / "ranges.txt")
    if r.ndim != 2 or r.shape[1] != 2 or np.any(r[:, 0] < 0) or np.any(r[:, 1] <= r[:, 0]) or np.any(r[:, 1] > offsets[-1]):
        refuse("MNLM_SA_RANGES_INVALID", "range outside the corpus or empty/reversed")
    t0 = time.time()
    info = json.loads((sa / "similar.json").read_text())
    if not info.get("group_files") or info["group_files"] != group_files(sa / "corpus.bin"):
        refuse("MNLM_SA_GROUP_UNSEALED", "duplicate-group files changed or were not retained by this run")
    if (paths.dedup / "spans.seal.json").exists():
        audit.require("spans")
        prior = json.loads((paths.dedup / "spans.summary.json").read_text(encoding="utf-8"))
        if prior.get("resolver") != RESOLVER or prior.get("threshold_bytes") != threshold:
            refuse("MNLM_SA_RESOLVER_MISMATCH", "accepted spans use another resolver; build a new generation")
        return prior
    parents = audit.begin("spans")
    mapped, stats = resolve_groups(sa / "corpus.bin", offsets, threshold, sa / "span-windows.sqlite")
    out = paths.dedup / "spans.jsonl.zst"
    corpus = np.memmap(sa / "corpus.bin", dtype=np.uint8, mode="r")
    with out.open("wb") as fh, zstandard.ZstdCompressor(level=6).stream_writer(fh) as zw:
        for a, start, end, b, dup in mapped.tolist():
            rec = {"doc": int(docs[a]), "ref": ref_of(meta, pop, int(docs[a])), "start": start, "end": end,
                   "dup_of": int(docs[b]), "dup_of_ref": ref_of(meta, pop, int(docs[b])), "dup_start": dup,
                   "resolved_by": RESOLVER, "cluster_b3": blake3.blake3(bytes(corpus[offsets[a]+start:offsets[a]+end])).hexdigest()}
            zw.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
    mapped[:, 0], mapped[:, 3] = docs[mapped[:, 0]], docs[mapped[:, 3]]
    save_npy(paths.dedup / "spans.npy", mapped)
    # Several provenance counterparts can cover the same source bytes. Actual
    # removal volume is the union per document, not the sum of annotation lengths.
    cut_bytes, last_doc, last_end = 0, None, 0
    for doc, start, end, _, _ in mapped:
        if doc != last_doc:
            last_doc, last_end = doc, 0
        cut_bytes += max(0, int(end) - max(int(start), last_end))
        last_end = max(last_end, int(end))
    summary = {"layer": "exact-substring", "threshold_bytes": threshold, "corpus_blake3": corpus_b3,
               "collected_ranges": len(r), **stats, "spans_cut": len(mapped), "unresolved_kept": 0,
               "bytes_cut": cut_bytes, "docs_affected": int(len(np.unique(mapped[:, 0]))),
               "corpus_docs": len(docs), "corpus_bytes": int(offsets[-1]),
               "secs": round(time.time() - t0, 1), "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "spans.summary.json", summary)
    digests = {"spans.npy": digest_file(paths.dedup / "spans.npy")[0], "spans.jsonl.zst": digest_file(out)[0], "corpus.json": digest_file(sa / "corpus.json")[0]}
    e = ledger_append(paths, f"sa_dedup@{HOST}", "dedup.spans", paths.version, {"summary": summary, "digests": digests})
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(paths.dedup / "spans.summary.json", summary)
    audit.finish("spans", parents)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--step", default="all", choices=["corpus", "table", "similar", "map", "all"])
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--part-bytes", type=int, default=1_500_000_000)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    steps = ["corpus", "table", "similar", "map"] if args.step == "all" else [args.step]
    for st in steps:
        t0 = time.time()
        if st == "corpus":
            r = build_corpus(paths)
        elif st == "table":
            r = build_table(paths, Path(args.tool), args.part_bytes, args.concurrent, args.threads)
        elif st == "similar":
            r = self_similar(paths, Path(args.tool), args.threshold, args.threads)
        else:
            r = map_spans(paths, args.threshold)
        print(f"{st}: {json.dumps(r)} [{round(time.time() - t0, 1)} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
