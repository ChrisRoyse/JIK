"""Reconcile pinned SWE benchmark contracts with outcome-blind selected attempts."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.benchmark_contracts import load_contracts
from senses.benchmark_outcomes import reconcile
from senses.contract import SenseError, require
from senses.trajectory_outcomes import inspect_outcome
from senses.trajectory_population import open_bound, plan, publish, read_pinned, validate_selection

CODE = "MNLM_BENCHMARK_AUDIT"
OUTCOME_COLUMNS = ("instance_id", "target", "eval_logs")


def selected_logs(root, selection):
    for shard in selection["scope"]["shards"]:
        chosen = {r["row"]: r["task"] for r in selection["selection"] if r["shard"] == shard["path"]}
        indices, offset = sorted(chosen), 0
        with open_bound(root, shard) as stream:
            parquet = pq.ParquetFile(stream)
            require(parquet.metadata.num_rows == shard["rows"], CODE, "source row count changed")
            for batch in parquet.iter_batches(batch_size=64, columns=list(OUTCOME_COLUMNS)):
                end = offset + batch.num_rows
                wanted = indices[bisect_left(indices, offset):bisect_left(indices, end)]
                selected = batch.take([i - offset for i in wanted]).to_pylist() if wanted else []
                for index, raw in zip(wanted, selected):
                    require(raw["instance_id"] == chosen[index], CODE, "selected task binding differs")
                    yield {"task": chosen[index], "shard": shard["path"], "row": index,
                           "raw_sha256": shard["sha256"]}, raw
                offset = end
            require(offset == shard["rows"], CODE, "outcome scan ended early")


def run(root, metadata_root, selection_path, selection_sha256, scope_path, scope_sha256, output):
    started = time.perf_counter()
    selection = read_pinned(selection_path, selection_sha256)
    require(isinstance(selection, dict) and "scope" in selection, CODE,
            "pinned selection object with population scope required")
    validate_selection(selection, plan(root, selection["scope"]))
    scope = read_pinned(scope_path, scope_sha256)
    validated = time.perf_counter()
    contracts, census = load_contracts(metadata_root, scope, {r["task"] for r in selection["selection"]})
    loaded = time.perf_counter()
    rows = []
    for locator, raw in selected_logs(root, selection):
        old = inspect_outcome(raw)
        rows.append({**locator, **reconcile(raw, contracts[locator["task"]]),
                     "previous_audit": {k: old[k] for k in ("status", "outcome")}})
    require(len(rows) == selection["count"], CODE, "audit lost a selected task")
    result = {"schema": "trajectory-benchmark-audit-v1", "selection_sha256": selection_sha256,
              "metadata_scope_sha256": scope_sha256, "metadata_census": census,
              "outcome_columns_read": list(OUTCOME_COLUMNS), "rows": rows, "count": len(rows),
              "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
              "candidate_outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items())),
              "admitted": False, "iid_established": False, "independently_reexecuted": False,
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("audit_benchmark.py", "benchmark_contracts.py", "benchmark_outcomes.py",
                                           "trajectory_population.py", "trajectory_outcomes.py", "contract.py")}}
    reconciled = time.perf_counter()
    sha = publish(output, result)
    finished = time.perf_counter()
    return {"output": str(output), "sha256": sha, "count": len(rows),
            "status_counts": result["status_counts"], "candidate_outcomes": result["candidate_outcomes"],
            "timings": {"validation": validated - started, "metadata": loaded - validated,
                        "reconciliation": reconciled - loaded, "publication": finished - reconciled,
                        "wall": finished - started}, "admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "metadata-root", "selection", "scope", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("selection-sha256", "scope-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.metadata_root, args.selection, args.selection_sha256,
                         args.scope, args.scope_sha256, args.output)))


if __name__ == "__main__":
    try:
        main()
    except (SenseError, OSError, ValueError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
