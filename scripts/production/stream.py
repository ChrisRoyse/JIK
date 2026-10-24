"""Stream frozen UCR instruments into bounded, restartable native tier shards.

Production admission is checked before output/lease/GPU use. Engineering runs
exercise the same source and native storage boundaries without claiming Stage 4
acceptance. Run with Python -I; all paths must be explicitly supplied.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import digest, require, SenseError
from production.input_plan import validate, read_rows, input_bindings
from production.publication import Publication, sync_dir
from production.session import Session
from production.stream_slots import durable, measure, audit_checkpoint
from universal import ChannelReader
from lenscommon import LensError, take_lease


class JobLock:
    """OS-owned lock; a terminated process releases it without a stale PID guess."""
    def __init__(self, root):
        self.file = (root / "writer.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.file.seek(0, 2) == 0:
                    self.file.write(b"\0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise SenseError("MNLM_STREAM_WRITER_BUSY", "another process owns this input plan") from exc

    def close(self):
        self.file.close()


def paths(root, key, name):
    base = root / "slots" / key
    return base / (name + ".npz"), base / (name + ".checkpoint.json")


def recover(root, path, checkpoint):
    """Discard only unaccepted, exactly named files in this locked job's scope."""
    require(not checkpoint.exists(), "MNLM_STREAM_IMMUTABLE", "completed checkpoint cannot be recovered by deletion")
    candidates = [path, path.with_name(path.name + ".tmp"), path.with_name(path.name + ".tokens.raw"),
                  path.with_suffix(".views.json"), path.with_suffix(".views.json.tmp"),
                  checkpoint.with_name(checkpoint.name + ".tmp")]
    removed = []
    for candidate in candidates:
        require(candidate.resolve().is_relative_to(root.resolve()) and not candidate.is_symlink(),
                "MNLM_STREAM_PATH", "recovery escaped the explicit output directory")
        if candidate.exists():
            removed.append({"file": str(candidate.relative_to(root)), "bytes": candidate.stat().st_size})
            candidate.unlink()
    return removed


def run(plan, store, policies, rows_root, output, *, blobs_path=None, admission_root=None, ledger=None,
        after_checkpoint=None, after_native=None, cache_root=None, after_cache_insert=None):
    started, phases = time.monotonic(), {}
    validate(plan, admission_root, ledger)
    output, rows_root = Path(output).resolve(), Path(rows_root).resolve()
    for source in (Path(store).resolve(), Path(policies).resolve(), rows_root):
        require(not source.is_relative_to(output) and not output.is_relative_to(source),
                "MNLM_STREAM_PATH", "output must be separate from source paths")
    require(("cache" in plan) == (cache_root is not None), "MNLM_CACHE_CONFIG", "cache plan and explicit cache directory must agree")
    if cache_root is not None:
        cache_root = Path(cache_root).resolve()
        for other in (Path(store).resolve(), Path(policies).resolve(), rows_root, output):
            require(not cache_root.is_relative_to(other) and not other.is_relative_to(cache_root),
                    "MNLM_STREAM_PATH", "cache must be separate from input and output paths")
    reader, lock, leased = ChannelReader(store, policies, generation=plan["generation"]), None, False
    cache, cache_lock, cache_before, cache_hits, cache_misses, cache_added = None, None, None, 0, 0, 0
    try:
        blobs = None
        if any(i["spec"]["module"] in ("audio_ucr", "image_ucr", "vision_ucr") for i in plan["instruments"].values()):
            from senses.blobs import Blobs
            blobs = Blobs(blobs_path, plan.get("blobs_manifest_sha256"))
        pairs = input_bindings(plan, reader, rows_root, blobs)
        phases["preflight"] = time.monotonic() - started
        at = time.monotonic()
        if cache_root is not None:
            from production.atom_cache import AtomCache
            from production.cached_session import CachedSession
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_lock = JobLock(cache_root)
            cache = AtomCache(cache_root, plan["cache"], plan["metadata_budget_bytes"], plan["slot_budget_bytes"])
            cache_before = cache.state()
        phases["cache_setup"] = time.monotonic() - at
        at = time.monotonic()
        output.mkdir(parents=True, exist_ok=True)
        lock = JobLock(output)
        frozen = output / "input-plan.json"
        if frozen.exists():
            require(json.loads(frozen.read_bytes()) == plan, "MNLM_STREAM_PLAN_DRIFT", "job belongs to another plan")
        else:
            require(set(p.name for p in output.iterdir()) <= {"writer.lock"},
                    "MNLM_STREAM_ORPHAN", "output lacks its frozen plan; inspect unowned files")
            durable(frozen, plan, plan["metadata_budget_bytes"])
        completed, missing, checkpoints = {}, {}, {}
        for key, shard in plan["shards"].items():
            _, raw = read_rows(rows_root / (key + ".jsonl"), shard)
            for name in shard["instruments"]:
                path, cp = paths(output, key, name)
                if cp.exists():
                    require(cp.stat().st_size <= plan["metadata_budget_bytes"],
                            "MNLM_STREAM_METADATA_BUDGET", "checkpoint exceeds metadata reservation")
                    checkpoint = json.loads(cp.read_bytes())
                    completed[key, name] = audit_checkpoint(checkpoint, path, plan, key, name, raw)
                    checkpoints[key, name] = checkpoint
                else:
                    missing.setdefault(name, []).append(key)
        reused = len(completed)
        phases["resume_audit"] = time.monotonic() - at
        at = time.monotonic()
        if cache is not None:
            for (key, name), checkpoint in checkpoints.items():
                rows, _ = read_rows(rows_root / (key + ".jsonl"), plan["shards"][key])
                helper = CachedSession(plan, name, reader, blobs, cache, None)
                cache_added += helper.backfill(rows, reader.policy(plan["shards"][key]["family"]),
                                              paths(output, key, name)[0], checkpoint)
        phases["cache_backfill"] = time.monotonic() - at
        recovery, loads, computed, resident = [], [], 0, None
        phases["gpu_setup"] = phases["load"] = phases["measure_and_checkpoint"] = phases["frozen_state"] = 0.0
        for name in sorted(missing, key=lambda n: (plan["instruments"][n]["spec"]["module"] != "universal", n)):
            item = plan["instruments"][name]
            def load():
                nonlocal leased, resident
                if item["spec"]["module"] in ("learned_ucr", "audio_ucr", "vision_ucr") and not leased:
                    setup = time.monotonic()
                    from measure import nvidia_smi
                    resident = nvidia_smi()
                    take_lease("16", "frozen input-plan streaming inference " + digest(plan)[:16])
                    leased = True
                    import torch
                    torch.use_deterministic_algorithms(True)
                    torch.cuda.reset_peak_memory_stats()
                    if not (output / "resident-before.json").exists():
                        durable(output / "resident-before.json", {"nvidia_smi": resident}, plan["metadata_budget_bytes"])
                    phases["gpu_setup"] += time.monotonic() - setup
                at = time.monotonic()
                value = Session(item, reader, plan["code_sha256"], blobs)
                loads.append(name)
                phases["load"] += time.monotonic() - at
                return value
            session = load() if cache is None else CachedSession(plan, name, reader, blobs, cache, load)
            try:
                for key in missing[name]:
                    at = time.monotonic()
                    shard = plan["shards"][key]
                    rows, raw = read_rows(rows_root / (key + ".jsonl"), shard)
                    path, cp = paths(output, key, name)
                    require(path.resolve().is_relative_to(output), "MNLM_STREAM_PATH", "slot directory escaped job root")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    recovery.extend(recover(output, path, cp))
                    nested_before = phases["gpu_setup"] + phases["load"]
                    slot = measure(session, rows, raw, reader.policy(shard["family"]), path,
                                   {**plan, "slot_budget_bytes": item["slot_budget_bytes"]})
                    if after_native is not None:
                        after_native(computed)
                    # A durable checkpoint must not survive a mutated resident instrument.
                    require(session.state_digest() == session.initial_state, "MNLM_LENS_FROZEN_VIOLATION", "weights changed at shard boundary")
                    checkpoint = {"plan_sha256": digest(plan), "shard": key, "instrument": name, "slot": slot}
                    durable(cp, checkpoint, plan["metadata_budget_bytes"])
                    completed[key, name] = audit_checkpoint(checkpoint, path, plan, key, name, raw)
                    if cache is not None:
                        cache_added += session.accept(rows, path, checkpoint, after_cache_insert)
                    computed += 1
                    nested = phases["gpu_setup"] + phases["load"] - nested_before
                    phases["measure_and_checkpoint"] += time.monotonic() - at - nested
                    if after_checkpoint is not None:  # process-crash boundary used by physical FSV
                        after_checkpoint(computed)
            finally:
                at = time.monotonic()
                session.close()
                if cache is not None:
                    cache_hits += session.hits
                    cache_misses += session.misses
                phases["frozen_state"] += time.monotonic() - at
        at = time.monotonic()
        publication = {"schema": "native-publication-plan-v1", "purpose": plan["purpose"], "storage": plan["storage"],
                       "training_scope_sha256": plan["training_scope_sha256"], "input_plan_sha256": digest(plan),
                       "blob_budget_bytes": plan["blob_budget_bytes"], "blobs": {}, "shards": {}}
        inputs = {}

        def blob(h, p):
            inputs[h] = str(p)
            publication["blobs"][h] = p.stat().st_size
            return h

        for key, shard in plan["shards"].items():
            record = {"key": key, "tier": shard["tier"], "n": shard["n"],
                      "rows_sha256": blob(shard["rows_sha256"], rows_root / (key + ".jsonl")), "slots": {}}
            for name in shard["instruments"]:
                slot = completed[key, name]
                path, _ = paths(output, key, name)
                blob(slot["sha256"], path)
                blob(slot["views_sha256"], path.with_suffix(".views.json"))
                record["slots"][name] = slot
            publication["shards"][key] = record
        pub = Publication(output / "publication", publication)
        try:
            before = pub.state()
            for key in publication["shards"]:
                pub.publish(key, inputs)
            after = pub.state()
        finally:
            pub.close()
        phases["publication"] = time.monotonic() - at
        report = {"acceptance": "author-checked", "stage4_accepted": False, "served": False,
                  "plan_sha256": digest(plan), "atom_instrument_pairs": pairs, "loaded_instruments": loads,
                  "checkpoints_computed": computed, "checkpoints_reused": reused, "recovered_unaccepted_files": recovery,
                  "resident_before": resident, "before": before, "after": after, "phases_seconds": phases,
                  "gpu_peak_allocated_bytes": sys.modules["torch"].cuda.max_memory_allocated() if leased else 0,
                  "cache": None if cache is None else {"before": cache_before, "after": cache.state(),
                                                      "atom_hits": cache_hits, "atom_misses": cache_misses, "entries_added": cache_added},
                  "wall_seconds": time.monotonic() - started}
        report["timing_coverage"] = sum(phases.values()) / report["wall_seconds"]
        require(report["timing_coverage"] >= .90, "MNLM_STREAM_TIMING", "phase instrumentation below 90%")
        return report
    finally:
        reader.close()
        if lock:
            lock.close()
        if cache is not None:
            cache.close()
        if cache_lock is not None:
            cache_lock.close()
        if leased:
            from lenscommon import release_lease
            release_lease()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "store", "policies", "rows", "out", "receipt"):
        ap.add_argument("--" + name, required=True, type=Path)
    for name in ("blobs", "admission-root", "ledger", "cache"):
        ap.add_argument("--" + name, type=Path)
    a = ap.parse_args()
    require(not a.receipt.exists(), "MNLM_STREAM_IMMUTABLE", "choose a new receipt path")
    plan = json.loads(a.plan.read_bytes())
    report = run(plan, a.store, a.policies, a.rows, a.out, blobs_path=a.blobs, admission_root=a.admission_root, ledger=a.ledger,
                 cache_root=a.cache)
    durable(a.receipt, report, plan["metadata_budget_bytes"])
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except (SenseError, LensError) as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}))
        sys.exit(2)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": "MNLM_STREAM_ARTIFACT", "message": "Inspect the declared input/checkpoint artifact: " + str(exc)}))
        sys.exit(2)
