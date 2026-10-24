"""Replay every recorded attempt of an environment-validated SWE task (decision #12 6240340899, A).

The task's environment was already witnessed: its reference passed every required test with nothing
uncollectable and its base established the FAIL_TO_PASS / PASS_TO_PASS roles. Each attempt adds only
an agent condition: pinned source + that attempt's own patch + the identical test patch, run in the
same pinned runtime and sandbox. Attempts are taken in fixed shard/row order with no selection; the
provider label is compared after the fact, never used to choose. The task stays the grouping unit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from oracle.container_run import execute
from oracle.prepare_swe import (MODES, apply_patch, check_modes, executables, inventory, patch_paths,
                                new_executables, reset_test_files, source_files)
from oracle.replay_swe import recount, verify_prepared
from oracle.test_contract import selectors
from senses.contract import SenseError, require
from senses.trajectory_population import open_bound, publish, read_pinned

CODE = "MNLM_ATTEMPT_REPLAY"
COLUMNS = ["instance_id", "generated_patch", "target", "model_name"]


def census(root, shards, tasks):
    """Outcome-blind: reads only instance_id, one hash-bound pass per shard."""
    found = {t: [] for t in tasks}
    for binding in shards:
        with open_bound(root, binding) as stream:
            offset = 0
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=4096, columns=["instance_id"]):
                for k, task in enumerate(batch.column(0).to_pylist()):
                    if task in found:
                        found[task].append({**binding, "row": offset + k})
                offset += batch.num_rows
    return found


def validated(prepared, prepared_sha, execution):
    spec, _ = verify_prepared(prepared, prepared_sha)
    ref = json.loads((execution / "reference" / "recount.json").read_bytes())
    base = json.loads((execution / "base" / "recount.json").read_bytes())
    require(ref["outcome"] == 1 and not ref.get("collection_failed"), CODE, "reference is not a clean witness")
    require(all(base["states"][n] == ("failed" if r == "FAIL_TO_PASS" else "passed")
                for n, r in spec["required_tests"].items()), CODE, "base does not establish roles")
    return spec


def one(root, plan, spec, files, execs, prepared, attempt, rows, out, fallback=False):
    """Prepare and run one attempt's agent condition; returns its record."""
    t0 = time.perf_counter()
    record = {"attempt": {k: attempt[k] for k in ("path", "row")}}
    a = rows.read(attempt, COLUMNS)
    require(a["instance_id"] == spec["task"], CODE, "attempt row is another task")
    record.update(target=int(a["target"]), model=a["model_name"])
    tree = out / "agent-tree"
    try:
        for path, data in files.items():
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            (tree / path).write_bytes(data)
        executables(tree, execs)
        text = a["generated_patch"] or ""
        if text.strip():
            record["patch_method"] = apply_patch(tree, out / "agent.patch", text, sorted(patch_paths(text)),
                                                 fallback=fallback)  # the benchmark harness's own fallback
        else:
            record["empty_submission"] = True  # agent condition = base by construction: no agent signal
        metadata = rows.read(plan["benchmark"], ["test_patch"])
        if fallback:  # SWE-bench evaluation: test files return to base before the test patch
            record["test_files_reset"] = reset_test_files(tree, files, plan["allowed_paths"]["test"])
        apply_patch(tree, out / "test.patch", metadata["test_patch"], plan["allowed_paths"]["test"])
        check_modes(tree, sorted(set(execs) | new_executables(text) | new_executables(metadata["test_patch"])))
        tests = {p: h for p, h in inventory(tree).items() if p in plan["allowed_paths"]["test"]}
        require(tests == {p: spec["conditions"]["base"][p] for p in plan["allowed_paths"]["test"] if p in spec["conditions"]["base"]}, CODE,
                "attempt condition used different test files")
        record["tree_inventory_sha256"] = hashlib.sha256(json.dumps(inventory(tree), sort_keys=True).encode()).hexdigest()
        execute(spec["image"], tree, prepared / "runtime", out / "agent", modules=selectors(spec["required_tests"]))
        stored = json.loads((out / "agent" / "execution.json").read_bytes())
        result = recount(stored, spec["required_tests"], allow_uncollectable=True)  # witness checked per task
        publish(out / "agent" / "recount.json", result)
        record.update(status="observed", outcome=result["outcome"], counts=result["counts"],
                      agrees_with_provider=result["outcome"] == record["target"],
                      collection_failed=len(result.get("collection_failed", [])))
    except SenseError as exc:
        record.update(status="gap", error=exc.envelope())
    finally:
        shutil.rmtree(tree, ignore_errors=True)  # reconstructible from pinned source + both patches
    record["seconds"] = time.perf_counter() - t0
    return record


def run(root, tasks, output, workers=1, scope=None, only_paths=None, fallback=False):
    """tasks: [{task, plan, prepared, prepared_sha256, execution}] (validated by the execution of record)."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    scope = scope or root / "evidence/fsv/12/family-audit-20261017/population-scope.json"
    shards = json.loads(Path(scope).read_bytes())["shards"]
    if only_paths:  # e.g. only newly acquired shards, so earlier replays are not repeated
        shards = [s for s in shards if s["path"] in set(only_paths)]
        require(len(shards) == len(set(only_paths)), CODE, "requested shard not in the scope")
    shards = [{**s, "path": "raw/swe-agent-trajectories/" + s["path"]} for s in shards]
    found = census(root, shards, {t["task"] for t in tasks})
    publish(output / "census.json", {"tasks": {k: [{"path": v["path"], "row": v["row"]} for v in vs]
                                                for k, vs in sorted(found.items())}, "columns_read": ["instance_id"]})
    censused = time.perf_counter()
    jobs, contexts = [], {}
    for t in tasks:
        plan = read_pinned(root / t["plan"]["path"], t["plan"]["sha256"])
        spec = validated(t["prepared"], t["prepared_sha256"], t["execution"])
        modes = {}
        files = source_files(root, plan, modes)
        execs = sorted(n for n, m in modes.items() if MODES[m])
        if plan.get("generated_files") is not None:  # mirror prepare(): declared install-time files
            from oracle.scm_version import generated
            extra = generated(files)
            require(plan["generated_files"] == [{"path": g["path"], "rule": g["rule"],
                                                 "sha256": hashlib.sha256(g["content"]).hexdigest()} for g in extra],
                    CODE, "declared generated files differ from the snapshot's own declaration")
            files = {**files, **{g["path"]: g["content"] for g in extra}}
            require(all(spec["conditions"]["base"].get(g["path"]) == hashlib.sha256(g["content"]).hexdigest()
                        for g in extra), CODE, "generated file differs from the validated base condition")
        contexts[t["task"]] = (plan, spec, files, execs, t["prepared"])
        for n, attempt in enumerate(found[t["task"]]):
            jobs.append((t["task"], n, attempt))
    requests = {}
    for task, _, attempt in jobs:
        requests[(attempt["path"], attempt["row"])] = (attempt, COLUMNS)
    for task, (plan, *_rest) in contexts.items():
        requests[("b", plan["benchmark"]["path"], plan["benchmark"]["row"])] = (plan["benchmark"], ["test_patch"])
    reqs = list(requests.values())
    binders = [BoundRows(root, reqs[j:j + 256]) for j in range(0, len(reqs), 256)]
    owner = {}
    for j, b in enumerate(binders):
        for binding, cols in reqs[j * 256:(j + 1) * 256]:
            owner[json.dumps([binding["path"], binding["row"], cols])] = b

    class Rows:  # route each read to the binder that hashed its source
        def read(self, binding, cols):
            return owner[json.dumps([binding["path"], binding["row"], cols])].read(binding, cols)
    bound = time.perf_counter()

    def work(job):
        task, n, attempt = job
        plan, spec, files, execs, prepared = contexts[task]
        out = output / task / str(n)
        out.mkdir(parents=True)
        rec = one(root, plan, spec, files, execs, prepared, attempt, Rows(), out, fallback)
        publish(out / "record.json", {"task": task, **rec})
        return {"task": task, "n": n, **rec}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(work, jobs))
    finished = time.perf_counter()
    return publish(output / "report.json", {"schema": "swe-attempt-replay-v1", "records": records, "workers": workers,
        "tasks": len(tasks), "attempts": len(jobs), "admitted": False,
        "phases": {"census": censused - started, "binding": bound - censused, "execution": finished - bound}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True, help="JSON list of validated tasks")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--scope", type=Path, help="trajectory shard scope file (default: original 3-shard scope)")
    p.add_argument("--only-shard", action="append", help="restrict the census to these scope paths")
    p.add_argument("--patch-fallback", action="store_true",
                   help="SWE-bench evaluation semantics: patch --fuzz=5 after git apply; test files reset to base")
    a = p.parse_args()
    tasks = [{**t, "prepared": Path(t["prepared"]), "execution": Path(t["execution"])}
             for t in json.loads(a.tasks.read_bytes())]
    print(json.dumps({"sha256": run(a.root, tasks, a.output, a.workers, a.scope, a.only_shard, a.patch_fallback)}))
