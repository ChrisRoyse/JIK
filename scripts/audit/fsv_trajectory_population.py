"""Cold-source task/outcome readback and three-boundary negative controls (#12)."""
import argparse
from bisect import bisect_left
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, require
from senses.pipeline import file_hash
from senses.trajectory_population import (CODE, SAMPLER, SEED, open_bound, plan, publish,
                                          read_pinned, select_metadata, validate_scope, validate_selection)
from senses.trajectory_outcomes import inspect_outcome, run as audit_outcomes


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "pass": bool(passed), **details})
        require(passed, "MNLM_TRAJECTORY_FSV", name)

    def refused(name, action, expected):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == expected, code=exc.code)
        else:
            check(name, False)

    scope = read_pinned(args.scope, args.scope_sha256)
    selected = read_pinned(args.selection, args.selection_sha256)
    original = read_pinned(args.audit, args.audit_sha256)
    before = {str(p): file_hash(p) for p in (args.scope, args.selection, args.audit)}
    # B1: separately re-derive from ALL identifier rows, not the selector's output.
    attempts, ids, raw_hashes = [], {}, {}
    memberships = {}
    for shard in scope["shards"]:
        path = args.root / shard["path"]
        raw_hashes[shard["path"]] = file_hash(path)
        check("raw_digest_" + path.name, raw_hashes[shard["path"]] == shard["sha256"]
              and path.stat().st_size == shard["bytes"])
        identifiers = pq.read_table(path, columns=["instance_id"]).column(0).to_pylist()
        ids[shard["path"]] = identifiers
        check("raw_row_count_" + path.name, len(identifiers) == shard["rows"])
        for row, task in enumerate(identifiers):
            encoded = json.dumps([SAMPLER, SEED, "attempt", task, shard["sha256"], row],
                                 ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            attempts.append((hashlib.sha256(encoded).hexdigest(), shard["sha256"], row, task, shard["path"]))
            memberships.setdefault(task, set()).add(shard["path"])
    independent = {}
    for rank, sha, row, task, name in sorted(attempts):
        independent.setdefault(task, {"task": task, "shard": name, "row": row})
    expected = sorted(independent.values(), key=lambda r: (r["shard"], r["row"]))
    check("all_global_task_choices_rederived", selected["selection"] == expected
          and len(expected) == scope["distinct_tasks"], attempts=len(attempts), tasks=len(expected))
    check("cross_shard_duplicates_counted_once", selected["cross_shard_tasks"] ==
          sum(len(v) > 1 for v in memberships.values()), cross_shard_tasks=selected["cross_shard_tasks"])
    real_plan = plan(args.root, scope)
    check("production_plan_cold_replay", canonical(real_plan) == canonical(selected))
    permuted = copy.deepcopy(scope)
    permuted["shards"].reverse()
    check("input_shard_order_invariant", select_metadata(permuted, lambda s: [ids[s["path"]]]) == selected)
    check("batch_partition_invariant", select_metadata(scope, lambda s: (
        ids[s["path"]][i:i + 17] for i in range(0, len(ids[s["path"]]), 17))) == selected)
    check("selection_did_not_read_or_include_outcomes", selected["outcome_fields_read"] == []
          and all(set(r) == {"task", "shard", "row"} for r in selected["selection"]))

    for name, mutate in [
        ("escaping_path", lambda s: s["shards"][0].update(path="../escape.parquet")),
        ("absolute_path", lambda s: s["shards"][0].update(path="/escape.parquet")),
        ("duplicate_shard", lambda s: s["shards"].append(s["shards"][0])),
        ("aliased_shard_digest", lambda s: s["shards"][1].update(sha256=s["shards"][0]["sha256"])),
        ("unknown_size", lambda s: s["shards"][0].update(bytes=None)),
        ("mutable_revision", lambda s: s.update(revision="main")),
        ("boolean_task_count", lambda s: s.update(distinct_tasks=True)),
    ]:
        bad = copy.deepcopy(scope)
        mutate(bad)
        refused(name, lambda: validate_scope(bad), CODE)
    bad = copy.deepcopy(scope)
    bad["distinct_tasks"] += 1
    refused("false_task_count", lambda: select_metadata(bad, lambda s: [ids[s["path"]]]), CODE)
    bad_shard = {**scope["shards"][0], "sha256": "0" * 64}
    refused("raw_hash_mismatch", lambda: open_bound(args.root, bad_shard), "MNLM_UCR_INPUT_DIGEST")
    for name, mutate in [
        ("invented_task", lambda s: s["selection"][0].update(task="not-a-retained-task")),
        ("wrong_row", lambda s: s["selection"][0].update(row=-1)),
        ("duplicate_task", lambda s: s["selection"].append(s["selection"][0])),
        ("seed_changed", lambda s: s.update(seed="outcome-selected")),
        ("outcome_selected", lambda s: s.update(outcome_fields_read=["target"])),
        ("claimed_admission", lambda s: s.update(admitted=True)),
    ]:
        bad = copy.deepcopy(selected)
        mutate(bad)
        refused(name, lambda: validate_selection(bad, real_plan), CODE)
    refused("wrong_selection_pin", lambda: read_pinned(args.selection, "0" * 64), "MNLM_UCR_INPUT_DIGEST")
    b1_seconds = time.perf_counter() - started

    # B2: run the real consumer, then independently bind every result to source.
    replay = audit_outcomes(args.root, args.selection, args.selection_sha256, args.output / "outcome-replay.json")
    check("outcome_file_bit_identical", replay["sha256"] == args.audit_sha256)
    audited = {(r["shard"], r["row"]): r for r in original["rows"]}
    source_rows, good, failed_example = 0, None, None
    for shard in scope["shards"]:
        wanted = sorted(row for name, row in audited if name == shard["path"])
        offset = 0
        for batch in pq.ParquetFile(args.root / shard["path"]).iter_batches(
                batch_size=37, columns=["instance_id", "target", "eval_logs"]):
            end = offset + batch.num_rows
            indices = wanted[bisect_left(wanted, offset):bisect_left(wanted, end)]
            if indices:
                for index, raw in zip(indices, batch.take([i - offset for i in indices]).to_pylist()):
                    row = audited[(shard["path"], index)]
                    require(raw["instance_id"] == row["task"] and raw["target"] == row["provider_target"],
                            "MNLM_TRAJECTORY_FSV", "source/task/target binding")
                    log = raw["eval_logs"]
                    if isinstance(log, str) and log.strip():
                        require(hashlib.sha256(log.encode()).hexdigest() == row["log_sha256"],
                                "MNLM_TRAJECTORY_FSV", "retained transcript digest")
                    if row["status"] == "provider_transcript_consistent":
                        section = re.search(r"(?ms)^___TESTS_SECTION___\r?\n(.*?)^___TESTS_SECTION_END___\r?$", log)
                        require(section is not None, "MNLM_TRAJECTORY_FSV", "source execution section missing")
                        fails = re.findall(r"(?m)^(?:FAILED|ERROR) .+$", section[1])
                        passes = re.findall(r"(?m)^PASSED .+$", section[1])
                        require(row["outcome"] == int(raw["target"])
                                and (bool(passes) and not fails if raw["target"] else bool(fails)),
                                "MNLM_TRAJECTORY_FSV", "source result contradicts candidate outcome")
                        if raw["target"] and good is None:
                            good = raw
                        if not raw["target"] and failed_example is None:
                            failed_example = raw
                    else:
                        require(row["outcome"] is None, "MNLM_TRAJECTORY_FSV", "refusal became an outcome")
                    source_rows += 1
            offset = end
    check("all_outcomes_cold_bound_to_source", source_rows == len(expected), rows=source_rows)
    check("all_selected_tasks_including_gaps_retained", sorted(audited) ==
          sorted((r["shard"], r["row"]) for r in expected))
    check("no_independent_execution_or_admission_claim", original["independently_reexecuted"] is False
          and original["admitted"] is False and original["iid_established"] is False)
    require(good is not None and failed_example is not None, "MNLM_TRAJECTORY_FSV", "real positive and negative examples required")
    log = good["eval_logs"]
    for name, raw, status in [
        ("no_log", {**good, "eval_logs": None}, "MNLM_TRAJECTORY_NO_EVALUATION"),
        ("empty_log", {**good, "eval_logs": ""}, "MNLM_TRAJECTORY_NO_EVALUATION"),
        ("nonboolean_target", {**good, "target": 1}, "MNLM_TRAJECTORY_TARGET_TYPE"),
        ("label_flip", {**good, "target": False}, "MNLM_TRAJECTORY_TARGET_CONFLICT"),
        ("missing_start", {**good, "eval_logs": log.replace("___TESTS_SECTION___", "")}, "MNLM_TRAJECTORY_EXECUTION_BOUNDARY"),
        ("duplicate_end", {**good, "eval_logs": log + "\n___TESTS_SECTION_END___"}, "MNLM_TRAJECTORY_EXECUTION_BOUNDARY"),
        ("truncated_verdict", {**good, "eval_logs": log.replace("All tests passed.", "")}, "MNLM_TRAJECTORY_EVALUATION_VERDICT"),
        ("duplicated_verdict", {**good, "eval_logs": log + "\nAll tests passed."}, "MNLM_TRAJECTORY_EVALUATION_VERDICT"),
        ("unknown_verdict", {**good, "eval_logs": log.replace("All tests passed.", "probably passed")}, "MNLM_TRAJECTORY_EVALUATION_VERDICT"),
        ("empty_test_section", {**good, "eval_logs": re.sub(r"(?s)___TESTS_SECTION___.*?___TESTS_SECTION_END___", "___TESTS_SECTION___\n___TESTS_SECTION_END___", log)}, "MNLM_TRAJECTORY_EXECUTION_BOUNDARY"),
        ("no_test_statuses", {**good, "eval_logs": re.sub(r"(?m)^(PASSED|SKIPPED|XFAIL|XPASS) ", "UNRECOGNIZED ", log)}, "MNLM_TRAJECTORY_NO_TEST_RESULT"),
        ("contradictory_test_failure", {**good, "eval_logs": log.replace("___TESTS_SECTION_END___", "FAILED injected-negative-control\n___TESTS_SECTION_END___")}, "MNLM_TRAJECTORY_TEST_VERDICT_CONFLICT"),
        ("false_footer_on_failure", {**failed_example, "target": True, "eval_logs": failed_example["eval_logs"].replace("Some tests failed.", "All tests passed.")}, "MNLM_TRAJECTORY_TEST_VERDICT_CONFLICT"),
        ("over_budget", {**good, "eval_logs": log + " " * (256 * 1024)}, "MNLM_TRAJECTORY_EVALUATION_BUDGET"),
    ]:
        got = inspect_outcome(raw)
        check(name, got["status"] == status and got["outcome"] is None and got["admitted"] is False)
    b2_seconds = time.perf_counter() - started - b1_seconds

    # B3: persisted publication cannot be overwritten and failures leave no temp files.
    before_files = sorted(p.name for p in args.output.iterdir())
    refused("immutable_output", lambda: publish(args.output / "outcome-replay.json", {"replacement": True}),
            "MNLM_UCR_OUTPUT_EXISTS")
    check("publication_unchanged_after_refusal", before_files == sorted(p.name for p in args.output.iterdir())
          and file_hash(args.output / "outcome-replay.json") == args.audit_sha256)
    check("scope_selection_and_original_unchanged", before == {str(p): file_hash(p) for p in (args.scope, args.selection, args.audit)})
    check("all_raw_sources_unchanged", all(file_hash(args.root / s["path"]) == raw_hashes[s["path"]] for s in scope["shards"]))
    report = {"schema": "trajectory-population-fsv-v1", "author_checked": True,
              "independently_adjudicated": False, "admitted": False,
              "scope_sha256": args.scope_sha256, "selection_sha256": args.selection_sha256,
              "audit_sha256": args.audit_sha256, "checks": checks,
              "attempts": len(attempts), "tasks": len(expected),
              "status_counts": original["status_counts"], "candidate_outcomes": original["candidate_outcomes"],
              "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/"): file_hash(p)
                              for p in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "senses/trajectory_population.py",
                                        Path(__file__).resolve().parents[1] / "senses/trajectory_outcomes.py")},
              "timings": {"source_selection_seconds": b1_seconds, "outcome_audit_seconds": b2_seconds,
                          "publication_readback_seconds": time.perf_counter() - started - b1_seconds - b2_seconds,
                          "wall_seconds": time.perf_counter() - started}}
    report_sha = publish(args.output / "report.json", report)
    print(json.dumps({"report_sha256": report_sha, "checks": len(checks), "tasks": len(expected),
                      "timings": report["timings"], "admitted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "scope", "selection", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("scope", "selection", "audit"):
        parser.add_argument("--" + name + "-sha256", required=True)
    run(parser.parse_args())
