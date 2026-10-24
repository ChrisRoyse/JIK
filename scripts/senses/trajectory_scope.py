"""Outcome-blind sampling of task units from an already retained trajectory shard.

The initial prefix is useful for format discovery, not independent task support.
Read only instance identifiers here; choose one attempt per task by a fixed hash
order. Selected complete trajectories are decoded separately in bounded batches.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, digest, require
from senses.pipeline import file_hash

SAMPLER = "one-attempt-per-task-v1"
SEED = "mnlm-task-units-20261017"
MAX_TASKS = 128


def plan(path, sha256, limit=MAX_TASKS):
    require(type(limit) is int and 0 < limit <= MAX_TASKS, "MNLM_TRAJECTORY_SCOPE", "bounded pass requires 1..128 tasks")
    require(file_hash(path) == sha256, "MNLM_UCR_INPUT_DIGEST", "trajectory bytes differ from the acquired source")
    return metadata_plan(pq.ParquetFile(path), sha256, limit)


def metadata_plan(parquet, sha256, limit):
    """Caller has already bound the source bytes; metadata is invariant per run."""
    best, offset = {}, 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["instance_id"]):
        for j, task in enumerate(batch.column(0).to_pylist()):
            require(isinstance(task, str) and task, "MNLM_TRAJECTORY_SCOPE", "source task identity required")
            row = offset + j
            order = digest([SAMPLER, SEED, "attempt", task, row])
            if task not in best or order < best[task][0]:
                best[task] = order, row
        offset += batch.num_rows
    require(type(limit) is int and 0 < limit <= MAX_TASKS and offset == parquet.metadata.num_rows and len(best) >= limit,
            "MNLM_TRAJECTORY_SCOPE", "insufficient distinct retained task units or invalid scope")
    tasks = sorted(best, key=lambda t: (digest([SAMPLER, SEED, "task", t]), t))[:limit]
    chosen = sorted(({"row": best[t][1], "task": t} for t in tasks), key=lambda v: v["row"])
    return {"schema": "trajectory-scope-v1", "sampler": SAMPLER, "seed": SEED, "raw_sha256": sha256,
            "source_rows": offset, "source_task_units": len(best), "row_groups": parquet.metadata.num_row_groups,
            "selection": chosen, "count": len(chosen), "outcome_fields_read": [], "admitted": False}


def validate(selection, sha256, rows, limit):
    require(isinstance(selection, dict) and selection.get("schema") == "trajectory-scope-v1"
            and selection.get("sampler") == SAMPLER and selection.get("seed") == SEED
            and selection.get("raw_sha256") == sha256 and selection.get("source_rows") == rows
            and selection.get("outcome_fields_read") == [] and selection.get("admitted") is False,
            "MNLM_TRAJECTORY_SCOPE", "selection does not bind this shard and the outcome-blind sampler")
    chosen = selection.get("selection")
    require(isinstance(chosen, list) and 0 < len(chosen) <= min(limit, MAX_TASKS) and selection.get("count") == len(chosen),
            "MNLM_TRAJECTORY_SCOPE", "invalid selected task count")
    require(all(isinstance(c, dict) and set(c) == {"row", "task"} and type(c["row"]) is int
                and 0 <= c["row"] < rows and isinstance(c["task"], str) and c["task"] for c in chosen),
            "MNLM_TRAJECTORY_SCOPE", "invalid selected row identity")
    indices = [c["row"] for c in chosen]
    require(indices == sorted(set(indices)) and len({c["task"] for c in chosen}) == len(chosen),
            "MNLM_TRAJECTORY_SCOPE", "duplicate/out-of-order rows or repeated task units")
    return {c["row"]: c["task"] for c in chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    selected = plan(args.input, args.sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as dst:
        dst.write(canonical(selected) + b"\n")
        dst.flush()
        os.fsync(dst.fileno())
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output), "source_task_units": selected["source_task_units"],
                      "selected_tasks": selected["count"], "admitted": False}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
