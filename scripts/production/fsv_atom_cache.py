"""Cross-plan native cache FSV over the approved real trajectory/audio increment.

No synthetic records. The small altered views below are negative controls only.
Read source and native storage independently; never infer reuse from exit status.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
from senses.contract import canonical, digest, SenseError
from senses.store import Store
from senses.pipeline import ingest_file
from production.fsv_stream import prepare
from production.stream import run, paths, JobLock
from production.input_plan import sha, validate
from production.atom_cache import AtomCache, panel_key, selector, input_key
from production.session import prepare as visible
from universal import ChannelReader
from lenscommon import LEASE, LensError

CACHE = {"schema": "atom-cache-v1", "max_db_bytes": 128 << 20}


def write(path, value):
    path.write_bytes(canonical(value))


def cache_state(root):
    # SQLite must be able to roll back a hot journal left by the crash probe.
    db = sqlite3.connect((root / "index.sqlite").resolve().as_uri() + "?mode=rw", uri=True)
    row = db.execute("SELECT hash,body FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    result = {"head": row[0] if row else None, "entries": db.execute("SELECT count(*) FROM entries").fetchone()[0],
              "proofs": db.execute("SELECT count(*) FROM proofs").fetchone()[0],
              "instruments": db.execute("SELECT count(*) FROM instruments").fetchone()[0]}
    db.close()
    return result


def atom_arrays(arrays, i):
    """Independent row readback, not the production cache slicer."""
    out = {k: arrays[k][i:i + 1] for k in ("absent", "windows")}
    if "X" in arrays:
        out["X"] = arrays["X"][i:i + 1]
    elif "tokens" in arrays:
        lo, hi = arrays["offsets"][i:i + 2]
        out.update(tokens=arrays["tokens"][lo:hi], offsets=np.array([0, hi - lo], np.int64))
    else:
        lo, hi = arrays["indptr"][i:i + 2]
        out.update(data=arrays["data"][lo:hi], indices=arrays["indices"][lo:hi],
                   indptr=np.array([0, hi - lo], np.int64), shape=np.array([1, arrays["shape"][1]], np.int64))
    return out


def fingerprint(arrays):
    h = hashlib.sha256()
    for k, a in sorted(arrays.items()):
        h.update(canonical([k, a.dtype.str, list(a.shape)]))
        h.update(a.tobytes())
    return h.hexdigest()


def job_values(job):
    result, modes = {}, set()
    for cp in sorted((job / "slots").glob("*/*.checkpoint.json")):
        proof = json.loads(cp.read_bytes())
        slot = proof["slot"]
        path = cp.with_name(cp.name.replace(".checkpoint.json", ".npz"))
        assert sha(path) == slot["sha256"] and sha(path.with_suffix(".views.json")) == slot["views_sha256"]
        views = json.loads(path.with_suffix(".views.json").read_bytes())
        with np.load(path, allow_pickle=False) as z:
            arrays = dict(z)
        modes.add(slot["replay_mode"])
        for i, b in enumerate(views["bindings"]):
            result[proof["instrument"], b["atom"]] = fingerprint(atom_arrays(arrays, i))
    return result, modes


def audit_cache(root, jobs):
    """Every row, dependency, event and native BLOB, through SQLite/NumPy only."""
    db = sqlite3.connect((root / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    previous, entries, proofs, source_files = None, {}, {}, {}
    for seq, h, raw in db.execute("SELECT seq,hash,body FROM events ORDER BY seq"):
        e = json.loads(raw)
        assert digest(e) == h and e["seq"] == seq and e["parent"] == previous and e["before_count"] == len(entries)
        for key, value in e["entries"]:
            assert key not in entries
            entries[key] = value
        assert e["after_count"] == len(entries)
        previous = h
    for h, raw in db.execute("SELECT hash,body FROM proofs"):
        p = json.loads(raw)
        assert digest(p) == h
        job = jobs[p["plan_sha256"]]
        path, cp = paths(job, p["shard"], p["instrument"])
        assert json.loads(cp.read_bytes()) == p and sha(path) == p["slot"]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            source_files[h] = dict(z)
        proofs[h] = (p, json.loads(path.with_suffix(".views.json").read_bytes()))
    kinds, seen = set(), set()
    for key, h, raw, native in db.execute("SELECT key,hash,body,native FROM entries"):
        b = json.loads(raw)
        assert entries[key] == h == digest(b) and key == digest(b["key"])
        assert hashlib.sha256(native).hexdigest() == b["native_sha256"]
        p, v = proofs[b["proof_sha256"]]
        assert v["bindings"][b["row"]]["atom"] == b["key"]["atom"]
        assert b["detail"] == {k: x for k, x in v["details"][b["row"]].items() if k != "cache_origin"}
        with np.load(io.BytesIO(native), allow_pickle=False) as z:
            arrays = dict(z)
        assert fingerprint(arrays) == fingerprint(atom_arrays(source_files[b["proof_sha256"]], b["row"]))
        kind = "dense" if "X" in arrays else "multi" if "tokens" in arrays else "sparse"
        kinds.add(kind)
        ih, iraw = db.execute("SELECT hash,body FROM instruments WHERE key=?", (b["key"]["selector"],)).fetchone()
        info = json.loads(iraw)
        assert digest(info) == ih and info["lens_id"] == p["slot"]["lens_id"] and info["kind"] == kind
        seen.add(key)
    db.close()
    assert seen == set(entries) and kinds == {"dense", "sparse", "multi"}
    return {"entries": len(entries), "proofs": len(proofs), "head": previous, "kinds": sorted(kinds)}


def make_plan(base, selected, generation, root, batch=2, shard_size=4):
    root.mkdir()
    rows_dir = root / "rows"
    rows_dir.mkdir()
    plan = copy.deepcopy(base)
    plan.update(generation=generation, batch_rows=batch, cache=CACHE, shards={})
    for tier in sorted({i["tier"] for i in plan["instruments"].values()}):
        for start in range(0, len(selected), shard_size):
            key = f"{tier}-{start // shard_size}"
            rs = [{**r, "i": i} for i, r in enumerate(selected[start:start + shard_size])]
            path = rows_dir / (key + ".jsonl")
            path.write_bytes(b"".join(canonical(r) + b"\n" for r in rs))
            plan["shards"][key] = {"n": len(rs), "rows_sha256": sha(path), "rows_bytes": path.stat().st_size,
                                   "family": "stream-boundary", "tier": tier,
                                   "instruments": sorted(n for n, i in plan["instruments"].items() if i["tier"] == tier)}
    validate(plan)
    write(root / "plan.json", plan)
    return plan, rows_dir


def original_proof_boundary(out, plan, policies, check):
    reader = ChannelReader(out / "observations.sqlite", policies, generation=plan["generation"])
    cp = next((out / "read" / "job" / "slots").glob("*/ucr-fields-v1.checkpoint.json"))
    proof = json.loads(cp.read_bytes())
    name = proof["instrument"]
    rows = [json.loads(x) for x in (out / "read" / "rows" / (proof["shard"] + ".jsonl")).read_bytes().splitlines()]
    ps = visible(plan["instruments"][name], reader, rows, reader.policy("stream-boundary"))
    keys = [input_key(panel_key(plan), selector(plan, name), p) for p in ps]
    cache = AtomCache(out / "missing-original-cache", CACHE, plan["metadata_budget_bytes"], plan["slot_budget_bytes"])
    before, error = cache.state(), None
    try:
        cache.accept(plan, name, rows, keys, paths(out / "read" / "job", proof["shard"], name)[0], proof)
    except SenseError as exc:
        error = exc.code
    after = cache.state()
    cache.close()
    reader.close()
    check("cache_readback_cannot_replace_original_forward_proof", error == "MNLM_CACHE_PROVENANCE" and after == before,
          error=error, before=before, after=after)
    # Even a caller supplying a new outer file hash cannot overwrite an existing
    # atom's different native value. This is a deliberately altered control.
    original = paths(out / "read" / "job", proof["shard"], name)[0]
    altered = out / "conflicting.npz"
    with np.load(original, allow_pickle=False) as z:
        arrays = dict(z)
    arrays["tokens"].flat[0] += np.float32(.125)
    np.savez(altered, **arrays)
    shutil.copyfile(original.with_suffix(".views.json"), altered.with_suffix(".views.json"))
    changed = copy.deepcopy(proof)
    changed["slot"]["sha256"] = sha(altered)
    cache = AtomCache(out / "cache", CACHE, plan["metadata_budget_bytes"], plan["slot_budget_bytes"])
    before, error = cache.state(), None
    try:
        cache.accept(plan, name, rows, keys, altered, changed)
    except SenseError as exc:
        error = exc.code
    after = cache.state()
    cache.close()
    check("existing_atom_native_conflict_refuses", error == "MNLM_CACHE_CONFLICT" and before == after,
          error=error, before=before, after=after)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "blobs", "out"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--child", choices=("read", "crash"))
    a = ap.parse_args()
    out = a.out.resolve()
    if a.child:
        plan = json.loads((out / "plan.json").read_bytes())
        report = run(plan, a.store, out.parent / "inputs" / "policies.json", out / "rows", out / "job",
                     blobs_path=a.blobs, cache_root=out.parent / ("crash-cache" if a.child == "crash" else "cache"),
                     after_cache_insert=(lambda n: os._exit(95)) if a.child == "crash" else None)
        report["torch_imported"] = "torch" in sys.modules
        write(out / "receipt.json", report)
        return 0
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    source_sha = sha(a.store)
    base, policies, _ = prepare(a.store, a.blobs, out / "inputs")
    selected = json.loads((out / "inputs" / "input-census.json").read_bytes())["selected"]
    initial = [r for r in selected if r["source"] == "swe-agent-trajectories"]
    addition = [r for r in selected if r not in initial]
    source = Store(a.store)
    store = out / "observations.sqlite"
    target = Store(store, create=True, records_only=True)
    checks, jobs = [], {}

    def check(name, ok, **evidence):
        checks.append({"name": name, "pass": bool(ok), **evidence})
        print(json.dumps(checks[-1]), flush=True)

    def append(label, rows):
        path = out / (label + ".jsonl")
        path.write_bytes(b"".join(canonical(source.get(r["atom"], base["generation"])) + b"\n" for r in rows))
        manifest = {"schema": "ucr-export-v1", "sha256": sha(path), "count": len(rows),
                    "source_generation": base["generation"], "selected_atoms": [r["atom"] for r in rows]}
        write(path.with_suffix(".manifest.json"), manifest)
        ingest_file(target, path, manifest, target.head)
        return target.head

    def execute(label, plan, rows, cache=None):
        root = out / label
        report = run(plan, store, policies, rows, root / "job", blobs_path=a.blobs, cache_root=cache)
        write(root / "receipt.json", report)
        jobs[digest(plan)] = root / "job"
        return report

    generation = append("first-eight", initial)
    p1, rows1 = make_plan(base, initial, generation, out / "first")
    first = execute("first", p1, rows1, out / "cache")
    check("initial_96_real_atom_instrument_misses", first["cache"]["atom_misses"] == 96
          and first["cache"]["entries_added"] == 96 and first["cache"]["atom_hits"] == 0, cache=first["cache"])
    newer = append("next-four", addition)
    target.close()
    source.close()
    check("real_append_changes_generation", generation != newer, before=generation, after=newer)
    p2, rows2 = make_plan(base, list(reversed(initial + addition)), newer, out / "append", batch=3, shard_size=3)
    second = execute("append", p2, rows2, out / "cache")
    check("append_only_new_atoms_measured", second["cache"]["atom_hits"] == 96 and second["cache"]["atom_misses"] == 48
          and second["cache"]["entries_added"] == 48, cache=second["cache"])
    p3, rows3 = make_plan(base, selected[2:] + selected[:2], newer, out / "read", batch=1, shard_size=4)
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--store", str(store), "--blobs", str(a.blobs)]
    proc = subprocess.run(cmd + ["--out", str(out / "read"), "--child", "read"], capture_output=True)
    (out / "read.stdout").write_bytes(proc.stdout)
    (out / "read.stderr").write_bytes(proc.stderr)
    assert proc.returncode == 0, "cache read child failed; inspect read.stderr"
    third = json.loads((out / "read" / "receipt.json").read_bytes())
    jobs[digest(p3)] = out / "read" / "job"
    check("cache_only_no_model_no_torch_no_gpu", third["cache"]["atom_hits"] == 144 and third["cache"]["atom_misses"] == 0
          and third["loaded_instruments"] == [] and not third["torch_imported"] and third["resident_before"] is None
          and third["gpu_peak_allocated_bytes"] == 0 and not (out / "read" / "job" / "resident-before.json").exists())
    check("cache_only_no_cache_mutation", third["cache"]["before"] == third["cache"]["after"])
    values1, _ = job_values(out / "first" / "job")
    values2, modes2 = job_values(out / "append" / "job")
    values3, modes3 = job_values(out / "read" / "job")
    check("all_native_values_survive_append_reorder_reshard", values2 == values3 and all(values2[k] == v for k, v in values1.items()))
    check("replay_distinguishes_readback_from_forward", modes3 == {"cache_readback"} and "mixed_forward_and_cache" in modes2)
    # Fresh execution is an independent numerical oracle for the cache-only path.
    cold, cold_rows = make_plan(base, selected, newer, out / "cold", batch=2)
    del cold["cache"]
    write(out / "cold" / "plan.json", cold)
    fresh = execute("cold", cold, cold_rows)
    fresh_values, _ = job_values(out / "cold" / "job")
    check("cache_matches_fresh_native_execution", fresh_values == values3)
    check("complete_cache_physical_readback", True, census=audit_cache(out / "cache", jobs))
    original_proof_boundary(out, p3, policies, check)
    # Mask/context/contract changes do not reuse unchanged-looking native values.
    reader = ChannelReader(store, policies, generation=newer)
    cache = AtomCache(out / "cache", CACHE, p3["metadata_budget_bytes"], p3["slot_budget_bytes"])
    name = "ucr-fields-v1"
    prepared = visible(p3["instruments"][name], reader, [initial[0]], reader.policy("stream-boundary"))[0]
    key = input_key(panel_key(p3), selector(p3, name), prepared)
    check("bound_real_view_hits", cache.lookup(key) is not None)
    for field in ("input", "context", "binding"):
        altered = copy.deepcopy(prepared)
        if field == "input":
            altered["input"]["channels"]["fields"]["items"] = []
        elif field == "context":
            altered["context"] = [prepared["input"]]
        else:
            altered["binding"]["policy_digest"] = "0" * 64
        check("changed_" + field + "_misses", cache.lookup(input_key(panel_key(p3), selector(p3, name), altered)) is None)
    for field in ("code_sha256", "environment", "instruments"):
        altered = copy.deepcopy(p3)
        if field == "instruments":
            altered[field][name]["weights_sha256"] = "0" * 64
        else:
            altered[field] = {"negative_control": True}
        check("changed_" + field + "_misses", cache.lookup(input_key(panel_key(altered), selector(altered, name), prepared)) is None)
    cache.close()
    reader.close()
    # A corrupted hit cannot be mistaken for an absent entry or trigger a model.
    for label, column, payload in (("corrupt-native", "native", b"corrupted"), ("corrupt-json", "body", b"{")):
        bad = out / label
        shutil.copytree(out / "cache", bad)
        db = sqlite3.connect(bad / "index.sqlite")
        db.execute("DROP TRIGGER immutable_entries_UPDATE")
        db.execute(f"UPDATE entries SET {column}=? WHERE key=?", (payload, digest(key)))
        db.execute("CREATE TRIGGER immutable_entries_UPDATE BEFORE UPDATE ON entries "
                   "BEGIN SELECT RAISE(ABORT,'MNLM_CACHE_IMMUTABLE'); END")
        db.commit()
        db.close()
        before = cache_state(bad)
        damaged_sha = sha(bad / "index.sqlite")
        error = None
        try:
            run(p3, store, policies, rows3, out / (label + "-job"), blobs_path=a.blobs, cache_root=bad)
        except SenseError as exc:
            error = exc.code
        check(label + "_refuses_without_repair", error == "MNLM_CACHE_CORRUPT" and before == cache_state(bad)
              and sha(bad / "index.sqlite") == damaged_sha
              and not (out / (label + "-job") / "resident-before.json").exists(), error=error, before=before, after=cache_state(bad))
    # Kill after SQL insert, before COMMIT. Checkpoint survives; no cache row may.
    tiny = copy.deepcopy(base)
    tiny["instruments"] = {name: tiny["instruments"][name]}
    pc, rc = make_plan(tiny, initial[:2], newer, out / "crash")
    proc = subprocess.run(cmd + ["--out", str(out / "crash"), "--child", "crash"], capture_output=True)
    (out / "crash.stdout").write_bytes(proc.stdout)
    (out / "crash.stderr").write_bytes(proc.stderr)
    cps = list((out / "crash" / "job" / "slots").glob("*/*.checkpoint.json"))
    checkpoint_hashes = {str(p): sha(p) for p in cps}
    before = cache_state(out / "crash-cache")
    check("process_death_rolls_back_cache_transaction", proc.returncode == 95 and before["entries"] == before["proofs"] == before["instruments"] == 0
          and len(cps) == 1, exit=proc.returncode, state=before)
    resumed = execute("crash", pc, rc, out / "crash-cache")
    check("resume_backfills_checkpoint_without_forward", resumed["loaded_instruments"] == [] and resumed["checkpoints_computed"] == 0
          and resumed["checkpoints_reused"] == 1 and resumed["cache"]["entries_added"] == 2
          and all(sha(Path(p)) == h for p, h in checkpoint_hashes.items()))
    # Fixed page cap refuses a real large Multi checkpoint atomically.
    small = AtomCache(out / "small-cache", {**CACHE, "max_db_bytes": 65536}, p2["metadata_budget_bytes"], p2["slot_budget_bytes"])
    reader = ChannelReader(store, policies, generation=newer)
    cp = next(cp for cp in (out / "cold" / "job" / "slots").glob("*/ucr-whisper*.checkpoint.json")
              if json.loads(cp.read_bytes())["slot"]["seal"]["absent"] < json.loads(cp.read_bytes())["slot"]["seal"]["n"])
    checkpoint = json.loads(cp.read_bytes())
    rows = [json.loads(x) for x in (cold_rows / (checkpoint["shard"] + ".jsonl")).read_bytes().splitlines()]
    name = checkpoint["instrument"]
    ps = visible(cold["instruments"][name], reader, rows, reader.policy("stream-boundary"))
    keys = [input_key(panel_key(cold), selector(cold, name), p) for p in ps]
    error, before = None, small.state()
    try:
        small.accept(cold, name, rows, keys, paths(out / "cold" / "job", checkpoint["shard"], name)[0], checkpoint)
    except SenseError as exc:
        error = exc.code
    check("page_budget_rolls_back_without_raising_limit", error == "MNLM_CACHE_BUDGET" and small.state() == before
          and (out / "small-cache" / "index.sqlite").stat().st_size <= 65536, error=error)
    small.close()
    reader.close()
    lock = JobLock(out / "cache")
    error = None
    try:
        run(p3, store, policies, rows3, out / "locked-job", blobs_path=a.blobs, cache_root=out / "cache")
    except SenseError as exc:
        error = exc.code
    finally:
        lock.close()
    check("cache_writer_lock_refuses_before_output", error == "MNLM_STREAM_WRITER_BUSY" and not (out / "locked-job").exists())
    prod = copy.deepcopy(p3)
    prod["purpose"] = "production"
    error = None
    try:
        run(prod, store, policies, rows3, out / "production-job", blobs_path=a.blobs, cache_root=out / "production-cache")
    except (SenseError, LensError) as exc:
        error = exc.code
    check("cache_never_bypasses_admission", error is not None and error.startswith("MNLM_STAGE4_")
          and not (out / "production-job").exists() and not (out / "production-cache").exists(), error=error)
    check("source_unchanged_lease_released", sha(a.store) == source_sha and not LEASE.exists(), source_sha256=source_sha)
    check("phase_coverage_and_no_scientific_acceptance", all(r["timing_coverage"] >= .90 and not r["stage4_accepted"]
          for r in (first, second, third, fresh, resumed)))
    report = {"acceptance": "author-checked", "stage4_accepted": False, "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "wall_seconds": time.monotonic() - started, "source_sha256": source_sha}
    write(out / "report.json", report)
    files = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*"))
             if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    write(out / "SHA256.json", files)
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}), flush=True)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
