"""Stage 1 op 3, part 1 (issue #8): MinHash signatures per canonical prose shard.

    python -I scripts/canon/minhash.py --root /zfs/archive/mnlm --out /zfs/hot/mnlm --version p0micro-v1 --source X [--workers 8]

Contract `minhash-v1` (frozen): lowercase `\\w+` words, word 5-gram shingles hashed with xxh3-64 per word and a wrapping
polynomial over the 5 word hashes (low 32 bits kept), 112 permutations h -> (a*h + b) mod 4294967311 (a, b from seed
1000003), 14 bands x 8 rows, band key = wrapping multiply-xor fold of the 8 rows and the band index. Docs with fewer than
5 words carry no signature (valid=False) and are kept, never deduplicated. Outputs per shard: <shard>.sig.npy
(uint32[n,112]), <shard>.bands.npy (uint64[n,14]), <shard>.valid.npy, and a manifest row keyed by the canon shard's
blake3 (skip-if-sealed, PC-07). Cost: O(words) hashing + O(unique shingles x 112) per doc; one process per shard.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import xxhash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, append_jsonl, iter_canon_docs, manifest_rows, now_iso, read_jsonl, save_npy, sealed_shards  # noqa: E402
from contracts import PROSE  # noqa: E402

CONTRACT_ID = "minhash-v1"
NGRAM, PERMS, BANDS, ROWS = 5, 112, 14, 8
PRIME = 4294967311
SEED = 1000003
MASK32 = np.uint64(0xFFFFFFFF)
MIX = np.uint64(0x9E3779B97F4A7C15)
WORD = re.compile(r"\w+")
_rng = np.random.default_rng(SEED)
A = _rng.integers(1, PRIME, size=PERMS, dtype=np.uint64)
B = _rng.integers(0, PRIME, size=PERMS, dtype=np.uint64)
POLY = np.array([31 ** i for i in range(NGRAM)], dtype=np.uint64)
CHUNK = 65536


def shingle_hashes(text: str) -> np.ndarray:
    words = WORD.findall(text.lower())
    if len(words) < NGRAM:
        return np.empty(0, dtype=np.uint64)
    wh = np.fromiter((xxhash.xxh3_64_intdigest(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words))
    n = len(words) - NGRAM + 1
    with np.errstate(over="ignore"):
        acc = np.zeros(n, dtype=np.uint64)
        for i in range(NGRAM):
            acc += wh[i:i + n] * POLY[i]
        acc ^= acc >> np.uint64(29)
        acc *= MIX
        acc ^= acc >> np.uint64(32)
    return np.unique(acc & MASK32)


def signature(h: np.ndarray) -> np.ndarray:
    sig = np.full(PERMS, 0xFFFFFFFF, dtype=np.uint64)
    for s in range(0, len(h), CHUNK):
        hh = h[s:s + CHUNK]
        with np.errstate(over="ignore"):
            v = (hh[:, None] * A[None, :] + B[None, :]) % PRIME
        np.minimum(sig, v.min(axis=0), out=sig)
    return (sig & MASK32).astype(np.uint32)


def band_keys(sig: np.ndarray) -> np.ndarray:
    """uint64[n,14] fold of each band's 8 rows (vectorised over docs)."""
    v = sig.reshape(len(sig), BANDS, ROWS).astype(np.uint64)
    with np.errstate(over="ignore"):
        k = np.zeros((len(sig), BANDS), dtype=np.uint64)
        for r in range(ROWS):
            k = (k ^ v[:, :, r]) * MIX
            k ^= k >> np.uint64(31)
        k ^= (np.arange(BANDS, dtype=np.uint64) + np.uint64(1)) * MIX
    return k


def sign_chunk(args) -> tuple[int, np.ndarray, np.ndarray]:
    """Signatures of docs [a, b) of one shard, in doc order. A shard is split into chunks so that a source with fewer
    shards than cores (FineWeb-Edu: 14 shards, 334 s wall on 32 cores) still uses every core; per-doc values are
    independent, so chunking changes no byte of the output. The parse of skipped docs costs O(bytes) and is small
    against the hashing (measured 0.33 ms per doc of hashing against about 3 us per doc of parse)."""
    canon_dir, shard, a, b = args
    sigs, valid = [], []
    for i, rec in enumerate(iter_canon_docs(canon_dir, shard)):
        if i < a:
            continue
        if i >= b:
            break
        h = shingle_hashes(rec["text"])
        if len(h) == 0:
            sigs.append(np.full(PERMS, 0xFFFFFFFF, dtype=np.uint32))
            valid.append(False)
        else:
            sigs.append(signature(h))
            valid.append(True)
    return a, (np.vstack(sigs) if sigs else np.empty((0, PERMS), dtype=np.uint32)), np.asarray(valid, dtype=bool)


def write_shard(src: str, sig_dir: Path, row: dict, parts: list[tuple[int, np.ndarray, np.ndarray]], t0: float) -> dict:
    from artifact_seals import invalidate, publish
    invalidate(sig_dir, row["shard"])
    parts.sort(key=lambda p: p[0])
    sig = np.vstack([p[1] for p in parts]) if parts else np.empty((0, PERMS), dtype=np.uint32)
    val = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, dtype=bool)
    if len(sig) != row["docs"]:
        raise RuntimeError(f"MNLM_MINHASH_DOC_COUNT {row['shard']}: signed {len(sig)} docs, manifest {row['docs']}")
    bands = band_keys(sig) if len(sig) else np.empty((0, BANDS), dtype=np.uint64)
    bands[~val] = 0
    sig_dir.mkdir(parents=True, exist_ok=True)
    save_npy(sig_dir / f"{row['shard']}.sig.npy", sig)
    save_npy(sig_dir / f"{row['shard']}.bands.npy", bands)
    save_npy(sig_dir / f"{row['shard']}.valid.npy", val)
    publish(sig_dir, row["shard"], signature_files(row["shard"]),
            {"canon_blake3": row["out_blake3"], "contract": CONTRACT_ID})
    return {"shard": row["shard"], "source": src, "contract": CONTRACT_ID, "canon_blake3": row["out_blake3"], "docs": int(len(sig)),
            "valid": int(val.sum()), "too_short": int((~val).sum()), "secs": round(time.time() - t0, 3), "host": HOST, "ts": now_iso()}


def signature_files(shard):
    return tuple(f"{shard}.{s}.npy" for s in ("sig", "bands", "valid"))


def accept_signature(directory, shard, canon_blake3):
    from artifact_seals import accept
    return accept(directory, shard, signature_files(shard),
                  {"canon_blake3": canon_blake3, "contract": CONTRACT_ID})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-docs", type=int, default=0, help="split shards into chunks of N docs (0 = whole shards; see the note in main)")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    canon_dir, sig_dir = paths.canon(args.source), paths.sig(args.source)
    rows = [r for r in sealed_shards(canon_dir) if r["kind"] == PROSE]
    # keyed by (shard, canon blake3): two shards with identical content must each be signed; all three files must exist
    have = {(r["shard"], r["canon_blake3"]) for r in read_jsonl(sig_dir / "manifest.jsonl") if not r.get("error")}
    todo = []
    for r in rows:
        if (r["shard"], r["out_blake3"]) in have and (sig_dir / (r["shard"] + ".seal.json")).exists():
            accept_signature(sig_dir, r["shard"], r["out_blake3"])
        else:
            todo.append(r)  # legacy unsealed signatures must run the real computation again
    print(f"{args.source}: {len(rows)} prose shards, {len(rows) - len(todo)} already signed, {len(todo)} to sign", flush=True)
    stats = {"shards": len(rows), "signed": 0, "docs": 0, "valid": 0, "too_short": 0}
    t0 = time.time()
    if todo:
        # Chunking is opt-in (--chunk-docs N): measured on FineWeb-Edu (14 shards of 0.7 M docs, 16 physical cores) it was
        # slower than whole shards (435 s and 501 s against 334 s), because the prefix re-parse and the extra processes
        # cost more than the idle cores gave back. It exists for the single-shard straggler (wikidata5m_text, 4.8 M docs).
        chunk_docs = args.chunk_docs if args.chunk_docs > 0 else max(r["docs"] for r in todo)
        jobs, pending = [], {}
        for row in todo:
            bounds = list(range(0, row["docs"], chunk_docs)) or [0]
            pending[row["shard"]] = {"row": row, "parts": [], "left": len(bounds), "t0": time.time()}
            jobs += [(canon_dir, row["shard"], a, min(a + chunk_docs, row["docs"])) for a in bounds]
        print(f"  {len(jobs)} chunks of up to {chunk_docs} docs over {len(todo)} shards", flush=True)
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            for job, part in zip(jobs, pool.imap(sign_chunk, jobs)):  # ordered: a shard's chunks arrive in order, memory bounded by one shard
                st = pending[job[1]]
                st["parts"].append(part)
                st["left"] -= 1
                if st["left"]:
                    continue
                r = write_shard(args.source, sig_dir, st["row"], st["parts"], st["t0"])
                del pending[job[1]]
                append_jsonl(sig_dir / "manifest.jsonl", r)
                stats["signed"] += 1
                for k in ("docs", "valid", "too_short"):
                    stats[k] += r[k]
                print(f"  {r['shard']}: {r['docs']} docs, {r['too_short']} too short, {r['secs']} s", flush=True)
    stats["secs"] = round(time.time() - t0, 3)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
