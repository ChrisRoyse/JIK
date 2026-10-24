"""Audit provider-recorded outcomes separately from all encoder input channels.

This checks transcript/label consistency, not independent execution or admission.
No command in a transcript is executed. Missing/ambiguous transcripts stay gaps.
"""
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
from senses.contract import SenseError, require
from senses.trajectory_population import (CODE, open_bound, plan, publish, read_pinned,
                                          validate_selection)

START = "___TESTS_SECTION___"
END = "___TESTS_SECTION_END___"
SETUP = "___ENVIRONMENT_SET_UP_PATCH_APPLIED___"
VERDICTS = {"All tests passed.": True, "Some tests failed.": False}
# An explicit per-item parser budget, not an assumption that all logs fit.
MAX_LOG_BYTES = 256 * 1024


def inspect_outcome(row):
    target, log = row.get("target"), row.get("eval_logs")
    report = {"provider_target": target if type(target) is bool else None,
              "evidence_level": "provider_recorded_not_independently_reexecuted",
              "admitted": False, "outcome": None, "log_sha256": None,
              "log_bytes": None, "provider_verdict": None}

    def refuse(code):
        return {**report, "status": code}

    if type(target) is not bool:
        return refuse("MNLM_TRAJECTORY_TARGET_TYPE")
    if not isinstance(log, str) or not log.strip():
        return refuse("MNLM_TRAJECTORY_NO_EVALUATION")
    data = log.encode("utf-8")
    report.update(log_sha256=hashlib.sha256(data).hexdigest(), log_bytes=len(data))
    if len(data) > MAX_LOG_BYTES:
        return refuse("MNLM_TRAJECTORY_EVALUATION_BUDGET")
    lines = log.splitlines()
    if any(lines.count(marker) != 1 for marker in (SETUP, START, END)):
        return refuse("MNLM_TRAJECTORY_EXECUTION_BOUNDARY")
    setup, start, end = (lines.index(marker) for marker in (SETUP, START, END))
    if not setup < start < end or not any(line.strip() for line in lines[start + 1:end]):
        return refuse("MNLM_TRAJECTORY_EXECUTION_BOUNDARY")
    tail = [line.strip() for line in lines[end + 1:] if line.strip()]
    if len(tail) != 1 or tail[0] not in VERDICTS:
        return refuse("MNLM_TRAJECTORY_EVALUATION_VERDICT")
    verdict = VERDICTS[tail[0]]
    report["provider_verdict"] = verdict
    # These are observations in a provider transcript, never a replacement oracle.
    report["test_section_sha256"] = hashlib.sha256("\n".join(lines[start + 1:end]).encode()).hexdigest()
    report["test_section_lines"] = end - start - 1
    report["test_status_lines"] = dict(Counter(
        line.split(" ", 1)[0] for line in lines[start + 1:end]
        if line.startswith(("PASSED ", "FAILED ", "ERROR ", "SKIPPED ", "XFAIL ", "XPASS "))))
    statuses = report["test_status_lines"]
    failed = statuses.get("FAILED", 0) + statuses.get("ERROR", 0)
    if not statuses.get("PASSED", 0) and not failed:
        return refuse("MNLM_TRAJECTORY_NO_TEST_RESULT")
    if (verdict and failed) or (not verdict and not failed):
        # A benchmark may deliberately ignore baseline failures. Without that
        # baseline/required-test contract we cannot resolve this discrepancy.
        return refuse("MNLM_TRAJECTORY_TEST_VERDICT_CONFLICT")
    if verdict != target:
        return refuse("MNLM_TRAJECTORY_TARGET_CONFLICT")
    report["outcome"] = int(verdict)
    return {**report, "status": "provider_transcript_consistent"}


def selected_outcomes(root, selection):
    """Private audit path: outcomes never flow to UCR or an input instrument."""
    for shard in selection["scope"]["shards"]:
        chosen = {r["row"]: r["task"] for r in selection["selection"] if r["shard"] == shard["path"]}
        indices, offset = sorted(chosen), 0
        with open_bound(root, shard) as stream:
            parquet = pq.ParquetFile(stream)
            require(parquet.metadata.num_rows == shard["rows"], CODE, "Parquet row count changed")
            # Read only the outcome audit columns; no trajectory or final patch.
            for batch in parquet.iter_batches(batch_size=64, columns=["instance_id", "target", "eval_logs"]):
                end = offset + batch.num_rows
                wanted = indices[bisect_left(indices, offset):bisect_left(indices, end)]
                if wanted:
                    for index, raw in zip(wanted, batch.take([i - offset for i in wanted]).to_pylist()):
                        require(raw["instance_id"] == chosen[index], CODE, "outcome task binding differs")
                        yield {"shard": shard["path"], "row": index, "task": chosen[index],
                               "raw_sha256": shard["sha256"], **inspect_outcome(raw)}
                offset = end
            require(offset == shard["rows"], CODE, "outcome scan ended early")


def run(root, selection_path, selection_sha256, output):
    started = time.perf_counter()
    selection = read_pinned(selection_path, selection_sha256)
    require(isinstance(selection, dict) and "scope" in selection, CODE, "selection object required")
    validate_selection(selection, plan(root, selection["scope"]))
    selection_seconds = time.perf_counter() - started
    rows = list(selected_outcomes(root, selection))
    require(len(rows) == selection["count"], CODE, "outcome audit lost a selected task")
    # Include every selected task, including every refusal; no class balancing.
    result = {"schema": "trajectory-outcome-audit-v1", "selection_sha256": selection_sha256,
              "population_sha256": selection["scope_sha256"], "count": len(rows),
              "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
              "provider_targets": dict(sorted(Counter(str(r["provider_target"]) for r in rows).items())),
              "candidate_outcomes": dict(sorted(Counter(str(r["outcome"]) for r in rows).items())),
              "rows": rows, "admitted": False, "iid_established": False,
              "independently_reexecuted": False}
    sha = publish(output, result)
    return {"output": str(output), "sha256": sha, "count": len(rows),
            "status_counts": result["status_counts"], "candidate_outcomes": result["candidate_outcomes"],
            "timings": {"selection_readback_seconds": selection_seconds,
                        "outcome_readback_publish_seconds": time.perf_counter() - started - selection_seconds,
                        "wall_seconds": time.perf_counter() - started}, "admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.selection, args.selection_sha256, args.output)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SenseError, OSError, ValueError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
