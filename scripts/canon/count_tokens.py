"""Stage 1 output (issue #8, spec §17 "expected yield ... the exact figure is a Stage 1 output"): measured token counts of the
unique prose population, whole documents and after the recorded span cuts, with the GPT-2 byte-level BPE tokenizer.

    python -I scripts/canon/count_tokens.py --root R --out O --version V [--workers 24] [--tokenizer-json PATH]

The tokenizer is the public `openai-community/gpt2` tokenizer.json (fetched once with huggingface_hub, sha256 recorded in
the summary). Per shard: every unique prose doc's text is encoded whole; the cut view removes the doc's recorded spans
(merged intervals) and is encoded separately. Output <out>/dedup/<v>/tokens.summary.json and one ledger entry. Cost:
O(bytes) tokenisation, one process per shard; nothing is written except the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, iter_canon_docs, ledger_append, now_iso, write_json  # noqa: E402
from dedup import load_population  # noqa: E402

TOKENIZER_REPO, TOKENIZER_FILE = "openai-community/gpt2", "tokenizer.json"
_tok = None


def tokenizer(path: str):
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer

        _tok = Tokenizer.from_file(path)
    return _tok


def cut_view(text: bytes, spans: np.ndarray) -> bytes:
    if len(spans) == 0:
        return text
    iv = spans[np.argsort(spans[:, 0])]
    out, pos = [], 0
    for a, b in iv.tolist():
        if a > pos:
            out.append(text[pos:a])
        pos = max(pos, b)
    out.append(text[pos:])
    return b"".join(out)


def pin_threads() -> None:
    """Pool initializer: one tokenizer thread per worker. encode_batch spawns a rayon pool sized to every core in each
    of the 24 workers (768 threads on 32 cores); measured 24 pinned workers 67.2 MB/s against 49.8 MB/s unpinned."""
    import os

    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def process_shard(args) -> dict:
    """One chunk of one shard: the docs of `rows_wanted` (ascending). A shard is split into chunks so that one large shard
    (wikidata5m_text: 4.8 M docs, 2,872 s alone at the tail of the p0-v1 run) is spread over the workers; the per-doc
    counts are independent, so chunking changes no number. The scan stops after the last wanted row."""
    tok_path, canon_dir, shard, rows_wanted, span_rows, span_bounds, src = args
    tok = tokenizer(tok_path)
    want = set(rows_wanted.tolist())
    last = int(rows_wanted[-1]) if len(rows_wanted) else -1
    by_row: dict[int, list] = {}
    for r, (a, b) in zip(span_rows.tolist(), span_bounds.tolist()):
        by_row.setdefault(r, []).append((a, b))
    t0 = time.time()
    docs = whole = cut = bytes_whole = bytes_cut = 0
    batch, batch_cut = [], []

    def flush():
        nonlocal whole, cut
        if batch:
            whole += sum(len(e.ids) for e in tok.encode_batch(batch, add_special_tokens=False))
            batch.clear()
        if batch_cut:
            cut += sum(len(e.ids) for e in tok.encode_batch(batch_cut, add_special_tokens=False))
            batch_cut.clear()

    for rec in iter_canon_docs(canon_dir, shard):
        if rec["row"] > last:
            break
        if rec["row"] not in want:
            continue
        docs += 1
        tb = rec["text"].encode("utf-8")
        bytes_whole += len(tb)
        batch.append(rec["text"])
        if rec["row"] in by_row:
            cb = cut_view(tb, np.asarray(by_row[rec["row"]], dtype=np.int64))
            bytes_cut += len(cb)
            batch_cut.append(cb.decode("utf-8"))  # strict: spans are character-aligned, so a cut view is valid text
        else:
            bytes_cut += len(tb)
            batch_cut.append(rec["text"])
        if len(batch) >= 256:
            flush()
    flush()
    return {"source": src, "shard": shard, "docs": docs, "tokens_whole": whole, "tokens_cut": cut, "bytes_whole": bytes_whole, "bytes_cut": bytes_cut,
            "secs": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tokenizer-json", default=None)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    if args.tokenizer_json:
        tok_path = args.tokenizer_json
    else:
        from huggingface_hub import hf_hub_download

        tok_path = hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE)
    tok_sha = hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
    from artifact_seals import Audit
    audit = Audit(paths)
    audit.require("spans")
    pop, meta = load_population(paths, audit)
    uniq = np.load(paths.dedup / "unique.npy")
    docs = uniq[pop["prose"][uniq]]
    spans = np.load(paths.dedup / "spans.npy")
    jobs = []
    for k, sh in enumerate(meta["shards"]):
        if sh["kind"] != PROSE_KIND:
            continue
        first = int(np.searchsorted(pop["shard"], k, side="left"))
        lo, hi = np.searchsorted(docs, [first, first + sh["docs"]])  # docs and spans[:, 0] are ascending: a bisect, not a
        d = docs[lo:hi]                                               # boolean mask over every doc per shard (9,770 x 5 M)
        if len(d) == 0:
            continue
        slo, shi = np.searchsorted(spans[:, 0], [first, first + sh["docs"]])
        sp = spans[slo:shi]
        jobs.append((tok_path, paths.canon(sh["source"]), sh["shard"], pop["row"][d], pop["row"][sp[:, 0]], sp[:, 1:3], sh["source"]))
    n_shards = len(jobs)
    chunk = max(20000, -(-sum(len(j[3]) for j in jobs) // (args.workers * 8)))  # about 8 chunks per worker; small shards stay whole
    chunks = []
    for tp, cd, sh_name, rows, srows, sb, src in jobs:
        for a in range(0, len(rows), chunk):
            rw = rows[a:a + chunk]
            m = (srows >= rw[0]) & (srows <= rw[-1])
            chunks.append((tp, cd, sh_name, rw, srows[m], sb[m], src))
    chunks.sort(key=lambda j: -len(j[3]))  # largest first
    print(f"{n_shards} shards in {len(chunks)} chunks of up to {chunk} docs, workers={args.workers}", flush=True)
    t0 = time.time()
    per_src: dict[str, dict] = {}
    tot = {"docs": 0, "tokens_whole": 0, "tokens_cut": 0, "bytes_whole": 0, "bytes_cut": 0}
    with mp.Pool(min(args.workers, len(chunks)), initializer=pin_threads) as pool:
        for r in pool.imap_unordered(process_shard, chunks):
            ps = per_src.setdefault(r["source"], {k: 0 for k in tot})
            for k in tot:
                ps[k] += r[k]
                tot[k] += r[k]
            print(f"  {r['source']}/{r['shard']}: {r['docs']} docs, {r['tokens_whole']} tokens, {r['tokens_cut']} after cuts, {r['secs']} s", flush=True)
    if tot["docs"] != len(docs):  # a shard rebuilt since the population was built yields fewer rows: an undercount, refused
        raise SystemExit(f"MNLM_TOKENS_DOC_COUNT: counted {tot['docs']} docs, the unique prose population has {len(docs)}")
    summary = {"version": paths.version, "tokenizer": {"repo": TOKENIZER_REPO, "file": TOKENIZER_FILE, "sha256": tok_sha}, **tot,
               "bytes_per_token_whole": round(tot["bytes_whole"] / tot["tokens_whole"], 4) if tot["tokens_whole"] else None,
               "per_source": per_src, "shards": n_shards, "secs": round(time.time() - t0, 1), "host": HOST, "ts": now_iso()}
    write_json(paths.dedup / "tokens.summary.json", summary)
    e = ledger_append(paths, f"count_tokens@{HOST}", "dedup.tokens", paths.version, {"summary": {k: summary[k] for k in ("docs", "tokens_whole", "tokens_cut", "bytes_whole", "bytes_cut")}, "tokenizer_sha256": tok_sha})
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(paths.dedup / "tokens.summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("docs", "tokens_whole", "tokens_cut", "bytes_per_token_whole", "secs", "ledger_seq")}))
    return 0


PROSE_KIND = "prose"

if __name__ == "__main__":
    sys.exit(main())
