"""Physical benchmark-audit boundaries using retained source rows; no data execution."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.audit_benchmark import selected_logs
from senses.benchmark_contracts import COLUMNS, contract, load_contracts
from senses.benchmark_outcomes import reconcile
from senses.contract import SenseError, canonical, require
from senses.trajectory_population import open_bound, publish, read_pinned

CODE = "MNLM_BENCHMARK_FSV"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recount(report, logs, root, scope):
    """Read raw oracle projections and witness lines without the outcome parser.

    Recounts every reported required-test observation and every accepted outcome.
    Does not certify unobserved execution or the truth of provider transcripts.
    """
    projections = {}
    for file in scope["files"]:
        with open_bound(root, file) as stream:
            offset = 0
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=32, columns=list(COLUMNS)):
                for raw in batch.to_pylist():
                    if raw["instance_id"] in logs:
                        projections[raw["instance_id"]] = (file, offset, raw)
                    offset += 1
    witnesses, accepted = 0, 0
    require(len(report["rows"]) == len(logs) == report["count"], CODE, "report population changed")
    require(len({r["task"] for r in report["rows"]}) == len(logs), CODE, "duplicate report task")
    for row in report["rows"]:
        locator, raw_log = logs[row["task"]]
        require(all(row[k] == v for k, v in locator.items()), CODE, "attempt locator changed")
        log = raw_log["eval_logs"]
        actual_hash = hashlib.sha256(log.encode()).hexdigest() if isinstance(log, str) else None
        require(row["log_sha256"] == actual_hash and row["admitted"] is False, CODE,
                "log digest changed or audit was promoted")
        meta = row["metadata"]
        if row["task"] not in projections:
            require(meta["status"] == "MNLM_BENCHMARK_TASK_ABSENT" and row["outcome"] is None,
                    CODE, "missing metadata accepted")
            continue
        file, offset, source = projections[row["task"]]
        require(meta["source"]["sha256"] == file["sha256"] and meta["row"] == offset
                and meta["projection_sha256"] == hashlib.sha256(canonical(source)).hexdigest(),
                CODE, "oracle projection differs from physical source")
        if meta["status"] != "contract_present":
            require(row["outcome"] is None, CODE, "invalid metadata accepted")
            continue
        roles = {}
        for role in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            tests = json.loads(source[role]) if isinstance(source[role], str) else source[role]
            require(meta["tests"][role] == sorted(tests), CODE, "required-test list changed")
            roles.update((t, role) for t in tests)
        if "required_tests" not in row:
            require(row["outcome"] is None, CODE, "accepted without per-test witnesses")
            continue
        tests = row["required_tests"]
        require(len(tests) == len(roles) and {t["test"]: t["role"] for t in tests} == roles,
                CODE, "test inventory differs from benchmark")
        lines = log.splitlines()
        markers = ("___ENVIRONMENT_SET_UP_PATCH_APPLIED___", "___TESTS_SECTION___", "___TESTS_SECTION_END___")
        require(all(lines.count(m) == 1 for m in markers), CODE, "execution markers are ambiguous")
        setup, start, end = (lines.index(m) for m in markers)
        summary = [i for i in range(start + 1, end)
                   if lines[i].strip("=").strip() == "short test summary info"]
        require(setup < start < end and len(summary) == 1, CODE, "test section is ambiguous")
        all_states = []
        for test in tests:
            for obs in test["observations"]:
                require(summary[0] < obs["line"] < end, CODE, "witness is outside the test summary")
                line = lines[obs["line"]]
                expected = obs["status"] + " " + test["test"]
                require(line == expected or (obs["status"] in ("FAILED", "ERROR")
                                              and line.startswith(expected + " - ")),
                        CODE, "witness does not name exact required test")
                require(obs["line_sha256"] == hashlib.sha256(line.encode()).hexdigest(),
                        CODE, "witness hash differs")
                witnesses += 1
            all_states += [o["status"] for o in test["observations"]]
        if row["outcome"] is not None:
            require(all(len(t["observations"]) == 1 for t in tests)
                    and all(s in ("PASSED", "FAILED", "ERROR") for s in all_states),
                    CODE, "accepted incomplete/ambiguous test evidence")
            expected = int(all(s == "PASSED" for s in all_states))
            require(row["outcome"] == expected and raw_log["target"] == bool(expected),
                    CODE, "outcome recount differs")
            tail = [s.strip() for s in lines[end + 1:] if s.strip()]
            require(tail == ["All tests passed." if expected else "Some tests failed."],
                    CODE, "provider verdict does not match independently counted tests")
            accepted += 1
    counts = dict(sorted(Counter(str(r["outcome"]) for r in report["rows"]).items()))
    require(counts == report["candidate_outcomes"], CODE, "headline differs from rows")
    return {"metadata_rows_joined": len(projections), "test_witnesses": witnesses,
            "accepted_provider_outcomes": accepted, "candidate_outcomes": counts}


def run(root, metadata_root, selection_path, scope_path, report_path, report_sha256, output):
    report = read_pinned(report_path, report_sha256)
    require(report["admitted"] is False and report["iid_established"] is False
            and report["independently_reexecuted"] is False, CODE, "audit was promoted")
    selection = read_pinned(selection_path, report["selection_sha256"])
    scope = read_pinned(scope_path, report["metadata_scope_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    before = {str(p): file_hash(p) for p in (selection_path, scope_path, report_path)}
    logs = {loc["task"]: (loc, raw) for loc, raw in selected_logs(root, selection)}
    checks = []
    raw = logs["ppizarror__pygame-menu-357"][1]
    meta = next(r["metadata"] for r in report["rows"] if r["task"] == raw["instance_id"])
    baseline = reconcile(raw, meta)
    require(baseline["outcome"] == 1, CODE, "real control row is not complete positive")
    test = meta["tests"]["FAIL_TO_PASS"][0]
    line = "PASSED " + test

    def check(name, candidate, metadata, expected):
        result = reconcile(candidate, metadata)
        require(result["status"] == expected, CODE, f"{name}: {result['status']}")
        path = output / (name + ".json")
        sha = publish(path, {"before": baseline, "action": name, "raw": candidate,
                             "metadata": metadata, "after": result})
        checks.append({"name": name, "status": result["status"], "sha256": sha})

    def mutate(name, log, expected, target=True):
        check(name, {**raw, "eval_logs": log, "target": target}, meta, expected)

    good = "provider_required_tests_consistent"
    check("real-positive", raw, meta, good)
    check("missing-contract", raw, {"status": "MNLM_BENCHMARK_TASK_ABSENT"}, "MNLM_BENCHMARK_TASK_ABSENT")
    mutate("target-conflict", raw["eval_logs"], "MNLM_TRAJECTORY_TARGET_CONFLICT", False)
    mutate("required-failure", raw["eval_logs"].replace(line, "FAILED " + test),
           "MNLM_BENCHMARK_REQUIRED_VERDICT_CONFLICT")
    mutate("required-missing", raw["eval_logs"].replace(line + "\n", ""),
           "MNLM_BENCHMARK_REQUIRED_TEST_MISSING")
    mutate("prefix-impostor", raw["eval_logs"].replace(line, line + "_not_the_same_test"),
           "MNLM_BENCHMARK_REQUIRED_TEST_MISSING")
    mutate("repeat-result", raw["eval_logs"].replace(line, line + "\n" + line),
           "MNLM_BENCHMARK_AMBIGUOUS_TEST_RESULT")
    mutate("outside-section-impostor", line + "\n" + raw["eval_logs"].replace(line + "\n", ""),
           "MNLM_BENCHMARK_REQUIRED_TEST_MISSING")
    mutate("unrelated-failure", raw["eval_logs"].replace(line, line + "\nFAILED unrelated/test.py::test_no"), good)
    mutate("unsupported-state", raw["eval_logs"].replace(line, "XPASS " + test),
           "MNLM_BENCHMARK_UNSUPPORTED_TEST_STATE")
    mutate("duplicate-boundary", raw["eval_logs"].replace("___TESTS_SECTION___", "___TESTS_SECTION___\n___TESTS_SECTION___"),
           "MNLM_TRAJECTORY_EXECUTION_BOUNDARY")
    mutate("extra-tail", raw["eval_logs"] + "untrusted verdict\n", "MNLM_TRAJECTORY_EVALUATION_VERDICT")
    mutate("budget", raw["eval_logs"] + "x" * (256 * 1024), "MNLM_TRAJECTORY_EVALUATION_BUDGET")
    ambiguous_meta = deepcopy(meta)
    ambiguous_meta["tests"]["FAIL_TO_PASS"].append(test + " - ambiguous")
    check("ambiguous-detail-separator", {**raw, "eval_logs": raw["eval_logs"].replace(
        line, "FAILED " + test + " - ambiguous")}, ambiguous_meta,
        "MNLM_BENCHMARK_AMBIGUOUS_TEST_RESULT")
    spaced_meta = deepcopy(meta)
    spaced_meta["tests"]["FAIL_TO_PASS"][0] = test + "[literal spaces]"
    check("literal-spaces-in-id", {**raw, "eval_logs": raw["eval_logs"].replace(
        line, line + "[literal spaces]")}, spaced_meta, good)
    # Contract controls use an actual normalized row; never a training example.
    source = {"instance_id": raw["instance_id"], "repo": meta["repo"],
              "base_commit": meta["base_commit"], "environment_setup_commit": meta["environment_setup_commit"],
              **meta["tests"]}
    for name, replacement in (
        ("empty-required", {"FAIL_TO_PASS": []}),
        ("duplicate-required", {"FAIL_TO_PASS": [test, test]}),
        ("overlapping-roles", {"PASS_TO_PASS": [test]}),
        ("bad-commit", {"base_commit": "not-a-commit"}),
    ):
        check(name, raw, contract({**source, **replacement}, "arrow_list"), "MNLM_BENCHMARK_CONTRACT")
    check("wrong-list-encoding", raw, contract(source, "json_string"), "MNLM_BENCHMARK_CONTRACT")

    def refuses(name, action, code):
        try:
            action()
        except SenseError as exc:
            require(exc.code == code, CODE, f"{name}: wrong refusal {exc.code}")
            checks.append({"name": name, "status": exc.code})
        else:
            raise SenseError(CODE, f"{name}: accepted broken state")

    refuses("pinned-report-tamper", lambda: read_pinned(report_path, "0" * 64), "MNLM_UCR_INPUT_DIGEST")
    refuses("immutable-publication", lambda: publish(report_path, {"incomplete": True}), "MNLM_UCR_OUTPUT_EXISTS")
    damaged = deepcopy(report)
    next(r for r in damaged["rows"] if r["outcome"] == 1)["required_tests"].pop()
    refuses("missing-persisted-witness", lambda: recount(damaged, logs, metadata_root, scope), CODE)
    copied_root = output / "corrupted-source"
    copied_root.mkdir()
    smallest = min(scope["files"], key=lambda f: f["bytes"])
    copied = copied_root / "copy.parquet"
    shutil.copyfile(metadata_root / smallest["path"], copied)
    with copied.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    bad_scope = {**scope, "files": [{**smallest, "path": "copy.parquet"}]}
    refuses("raw-byte-tamper", lambda: load_contracts(copied_root, bad_scope, set(logs)), "MNLM_UCR_INPUT_DIGEST")
    duplicate = output / "duplicate.parquet"
    shutil.copyfile(metadata_root / smallest["path"], duplicate)
    # A second genuine source projection under another path is still ambiguous.
    shutil.copyfile(duplicate, output / "duplicate2.parquet")
    duplicate_scope = {**scope, "files": [{**smallest, "path": p} for p in ("duplicate.parquet", "duplicate2.parquet")]}
    refuses("duplicate-source-task", lambda: load_contracts(output, duplicate_scope, set(logs)), "MNLM_BENCHMARK_DUPLICATE_TASK")
    reread = recount(report, logs, metadata_root, scope)
    require(before == {str(p): file_hash(p) for p in (selection_path, scope_path, report_path)},
            CODE, "boundary checks mutated accepted state")
    return publish(output / "report.json", {"schema": "benchmark-fsv-v1", "checks": checks,
                                            "count": len(checks), "before": before, "after": before,
                                            "recount": reread, "admitted": False, "author_checked": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "metadata-root", "selection", "scope", "report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps({"sha256": run(args.root, args.metadata_root, args.selection, args.scope,
                                   args.report, args.report_sha256, args.output)}))
