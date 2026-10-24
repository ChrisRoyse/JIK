"""Stage 1 op 1 driver (issue #8): raw shards -> canonical shards under <out>/canon/<version>/<source>/.

    python -I scripts/canon/canonicalise.py --root /zfs/archive/mnlm --out /zfs/hot/mnlm --version p0micro-v1 \
        --census p0-v1 --source fineweb-edu-sample-10BT [--shards 000_00000,...] [--workers 8]

Per output shard: <shard>.jsonl.zst (records {row, b3, n, kind, text, f}), sidecars <shard>.b3.bin / .rows.npy / .n.npy,
and one manifest.jsonl row {shard, raw_files, raw_sha256, raw_bytes, kind, docs, empty, canonical_bytes, out_blake3,
out_bytes, contract, secs}. A shard is a pure function of (raw sha256 list, contract id) and is skipped only after a cold
raw/canonical/sidecar digest check (PC-07). Per-shard failures are manifest rows with `error` and do
not abort the source (PC-32). Cost O(raw bytes) per shard; workers = concurrent shards (one process each).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import blake3
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts  # noqa: E402
from attestation import attest_raw, attest_shard, seal_sidecars
from plan_seals import accepted, group_rows, make_seal, plan_id, read_seals
from common import HOST, Paths, append_jsonl, digest_file, ledger_append, now_iso, read_jsonl, save_npy  # noqa: E402

CUYAHOGA_ROOT_DEFAULT = "/zfs/archive/calyx/freelaw-cuyahoga"


def load_receipts(raw_dir: Path) -> dict:
    out: dict = {}
    for r in read_jsonl(raw_dir / "receipts.jsonl"):
        out[r["path"]] = r  # last receipt per path wins (append-only file)
    return out


def write_shard(source: str, shard_name: str, kind: str, docs, out_dir: Path) -> dict:
    """Streams one output shard; returns the manifest counts. Written to .tmp and renamed only when complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{shard_name}.jsonl.zst"
    tmp = out_dir / f"{shard_name}.jsonl.zst.tmp.{os.getpid()}"  # process-unique: two workers never share a temporary
    b3s, rows, ns = bytearray(), [], []
    docs_n, empty, canonical_bytes, kinds = 0, 0, 0, set()
    cctx = zstandard.ZstdCompressor(level=3, threads=1)
    try:
        with tmp.open("wb") as fh, cctx.stream_writer(fh) as zw:
            for row, text, fields, k in docs:
                if text is None:
                    empty += 1
                    continue
                kinds.add(k)
                if len(kinds) > 1:  # refused before anything is committed (not after the rename, which left orphans)
                    raise contracts.ContractError("MNLM_CANON_MIXED_KINDS", f"{shard_name} mixes kinds {sorted(kinds)}")
                tb = text.encode("utf-8")
                h = blake3.blake3(tb).digest(length=16)
                rec = {"row": row, "b3": h.hex(), "n": len(tb), "kind": k, "text": text, "f": fields}
                zw.write((json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8"))
                b3s += h
                rows.append(row)
                ns.append(len(tb))
                docs_n += 1
                canonical_bytes += len(tb)
            zw.flush(zstandard.FLUSH_FRAME)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:  # a refused shard leaves no .tmp behind (readback finding F-6)
        tmp.unlink(missing_ok=True)
        raise
    # sidecars first (atomic each), the data file last: a crash leaves sidecars without a data file, which no reader
    # accepts (sealed_shards needs the data file at the recorded size AND complete sidecars), never the reverse
    b3p = out_dir / f"{shard_name}.b3.bin.tmp.{os.getpid()}"
    b3p.write_bytes(bytes(b3s))
    os.replace(b3p, out_dir / f"{shard_name}.b3.bin")
    save_npy(out_dir / f"{shard_name}.rows.npy", np.asarray(rows, dtype=np.int32))
    save_npy(out_dir / f"{shard_name}.n.npy", np.asarray(ns, dtype=np.int64))
    os.replace(tmp, final)
    out_b3, out_bytes = digest_file(final)
    return {"docs": docs_n, "empty": empty, "canonical_bytes": canonical_bytes, "out_blake3": out_b3, "out_bytes": out_bytes,
            "kind": next(iter(kinds)) if kinds else kind, "sidecars": seal_sidecars(out_dir, shard_name)}


def rel_names(files: list[Path], raw_dir: Path) -> list[str]:
    out = []
    for f in files:
        try:
            out.append(f.relative_to(raw_dir).as_posix())
        except ValueError:
            out.append(f.name)
    return out


def process_plan(args) -> list[dict]:
    source, plan, out_dir, raw_dir = args
    t0 = time.time()
    results = []
    names = rel_names(plan["files"], raw_dir)
    try:
        from training_scope import allowed
        if not allowed(source):
            raise contracts.ContractError("MNLM_SOURCE_EXCLUDED", f"{source}: excluded from MNLM training")
        raw_bytes = attest_raw(plan)
        for shard_name, kind, docs in contracts.open_shard(source, plan):
            counts = write_shard(source, shard_name, kind, docs, out_dir)
            if counts["docs"] + counts["empty"] == 0:  # a reader that yields nothing is a refusal, never a sealed empty shard
                for sfx in (".jsonl.zst", ".b3.bin", ".rows.npy", ".n.npy"):
                    (out_dir / f"{shard_name}{sfx}").unlink(missing_ok=True)
                raise contracts.ContractError("MNLM_CANON_SHARD_EMPTY", f"{shard_name}: the reader yielded no record")
            if "attested_rows" in plan and counts["docs"] + counts["empty"] != plan["attested_rows"]:
                raise contracts.ContractError("MNLM_CANON_ROWCOUNT_MISMATCH", f"{shard_name}: {counts['docs'] + counts['empty']} rows read, {plan['attested_rows']} attested")
            results.append({"shard": shard_name, "source": source, "contract": contracts.CONTRACT_ID, "raw_files": names,
                            "raw_sha256": plan["sha256"], "raw_bytes": raw_bytes, "plan_shard": plan["shard"], "secs": round(time.time() - t0, 3),
                            "host": HOST, "ts": now_iso(), **counts})
        if not results:
            raise contracts.ContractError("MNLM_CANON_PLAN_EMPTY", f"{plan['shard']}: extraction produced no shards")
    except contracts.ContractError as e:
        results.append({"shard": plan["shard"], "source": source, "contract": contracts.CONTRACT_ID, "raw_files": names,
                        "raw_sha256": plan["sha256"], "host": HOST, "ts": now_iso(), "error": {"code": e.code, "message": e.message, **e.detail}})
    except Exception as e:  # noqa: BLE001 - recorded as a typed row, never swallowed (a reader's own typed code is kept)
        code = e.code if isinstance(getattr(e, "code", None), str) and e.code.startswith("MNLM_") else "MNLM_CANON_SHARD_FAILED"
        results.append({"shard": plan["shard"], "source": source, "contract": contracts.CONTRACT_ID, "raw_files": names,
                        "raw_sha256": plan["sha256"], "host": HOST, "ts": now_iso(),
                        "error": {"code": code, "message": getattr(e, "message", f"{type(e).__name__}: {e}"), **getattr(e, "detail", {}), "trace": traceback.format_exc()[-2000:]}})
    for r in results:
        r["plan_seal_required"] = True
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--census", default="p0-v1")
    ap.add_argument("--source", required=True)
    ap.add_argument("--shards", default=None, help="comma list of plan shard names; default all verified")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cuyahoga-root", default=CUYAHOGA_ROOT_DEFAULT)
    ap.add_argument("--raw-dir", default=None, help="override <root>/raw/<source> (plant corpus)")
    args = ap.parse_args()
    from training_scope import require_training_source
    require_training_source(args.source)
    paths = Paths(args.root, args.out, args.version)
    raw_dir = Path(args.raw_dir) if args.raw_dir else paths.root / "raw" / contracts.scripture.RAW_DIR_ALIAS.get(args.source, args.source)
    out_dir = paths.canon(args.source)
    receipts = load_receipts(raw_dir)
    plan = contracts.list_shards(args.source, raw_dir, receipts, Path(args.cuyahoga_root))
    if args.shards:
        want = set(args.shards.split(","))
        plan = [p for p in plan if p["shard"] in want]
    names = [p["shard"] for p in plan]
    if len(set(names)) != len(names):  # two raw files that map to one shard name would overwrite each other's output
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"MNLM_CANON_SHARD_NAME_COLLISION {args.source}: {dup[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".canonicalise.lock"  # one canonicalise per source directory; a dead holder's lock is retired
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {HOST} {now_iso()}\n".encode())
        os.close(fd)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8").split()
        alive = False
        if holder and holder[1] == HOST and holder[0].isdigit():
            try:
                os.kill(int(holder[0]), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(f"MNLM_CANON_LOCKED {args.source}: held by pid {holder[0]} on {holder[1]} since {holder[2] if len(holder) > 2 else '?'}")
        lock.write_text(f"{os.getpid()} {HOST} {now_iso()} (retired stale lock of {' '.join(holder)})\n", encoding="utf-8")
    try:
        return run_source(args, paths, raw_dir, out_dir, plan)
    finally:
        lock.unlink(missing_ok=True)


def run_source(args, paths, raw_dir, out_dir, plan):
    by_plan = group_rows(read_jsonl(out_dir / "manifest.jsonl"))
    plan_seals = read_seals(out_dir)
    todo = []
    for p in plan:
        sealed = accepted(by_plan.get(p["shard"], []), plan_seals.get(p["shard"]))
        if sealed and all(r["raw_sha256"] == p["sha256"] and r["contract"] == contracts.CONTRACT_ID for r in sealed):
            attest_raw(p)
            for r in sealed:
                attest_shard(out_dir, r)
            continue
        if by_plan.get(p["shard"]) and sealed is None:
            print(f"REBUILD MNLM_CANON_PLAN_UNSEALED {p['shard']}: no cached extraction accepted; replaying pinned raw bytes", flush=True)
        todo.append(p)
    print(f"{args.source}: {len(plan)} planned shards, {len(plan) - len(todo)} already sealed, {len(todo)} to build, workers={args.workers}", flush=True)
    stats = {"planned": len(plan), "skipped": len(plan) - len(todo), "built": 0, "errors": 0, "docs": 0, "empty": 0, "canonical_bytes": 0, "raw_bytes": 0}
    t0 = time.time()
    jobs = [(args.source, p, out_dir, raw_dir) for p in todo]
    seen_plans: set[str] = set()
    if jobs:
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            for rows in pool.imap_unordered(process_plan, jobs):
                for r in rows:
                    append_jsonl(out_dir / "manifest.jsonl", r)
                    by_plan.setdefault(plan_id(r), []).append(r)
                    if r.get("error"):
                        stats["errors"] += 1
                        print(f"  REFUSED {r['shard']}: {r['error']['code']} {r['error'].get('message', '')[:160]}", flush=True)
                    else:
                        stats["built"] += 1
                        for k in ("docs", "empty", "canonical_bytes"):
                            stats[k] += r[k]
                        if r["plan_shard"] not in seen_plans:  # a multi-shard plan (an archive) reports its raw bytes once
                            seen_plans.add(r["plan_shard"])
                            stats["raw_bytes"] += r["raw_bytes"]
                        print(f"  {r['shard']}: {r['docs']} docs, {r['empty']} empty, {r['canonical_bytes']} canonical B, {r['secs']} s", flush=True)
                if rows and not any(r.get("error") for r in rows):
                    append_jsonl(out_dir / "plans.jsonl", make_seal(by_plan[plan_id(rows[0])], rows))
    stats["secs"] = round(time.time() - t0, 3)
    mpath = out_dir / "manifest.jsonl"
    if mpath.exists():
        mb3, _ = digest_file(mpath)
        e = ledger_append(paths, f"canonicalise@{HOST}", "canon.source", f"{args.version}/{args.source}",
                          {"manifest_blake3": mb3, "contract": contracts.CONTRACT_ID, **stats})
        print(f"ledger seq {e['seq']} head {e['hash'][:16]}")
    print(json.dumps(stats))
    return 0 if stats["errors"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
