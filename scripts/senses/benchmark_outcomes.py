"""Conservative exact required-test reconciliation of provider logs, not execution.

Only pytest short-summary rows are supported. No substring/alias matching, no
last-result-wins, no inference from absent tests, and no test text as input data.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from senses.trajectory_outcomes import END, MAX_LOG_BYTES, SETUP, START, VERDICTS

SUMMARY = re.compile(r"=+ short test summary info =+")
STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")


def required_trie(tests):
    tree = {}
    for test in tests:
        branch = tree
        for char in test:
            branch = branch.setdefault(char, {})
        branch[None] = test
    return tree


def failure_matches(node, tree):
    # Linear in the line, even with many " - " delimiters. Repeatedly slicing
    # and hashing every delimiter prefix would be quadratic in adversarial logs.
    matches = set()
    branch = tree
    for offset, char in enumerate(node, 1):
        branch = branch.get(char)
        if branch is None:
            break
        if None in branch and (offset == len(node) or node.startswith(" - ", offset)):
            matches.add(branch[None])
    return matches


def reconcile(raw, metadata):
    target, log = raw.get("target"), raw.get("eval_logs")
    result = {"outcome": None, "admitted": False, "independently_reexecuted": False,
              "evidence_level": "provider_recorded_not_independently_reexecuted",
              "provider_target": target if type(target) is bool else None,
              "provider_verdict": None, "required_outcome": None,
              "metadata": metadata, "log_sha256": None, "log_bytes": None}

    def gap(code):
        return {**result, "status": code}

    if isinstance(log, str):
        data = log.encode("utf-8")
        result.update(log_sha256=hashlib.sha256(data).hexdigest(), log_bytes=len(data))
    if metadata["status"] != "contract_present":
        return gap(metadata["status"])
    if type(target) is not bool:
        return gap("MNLM_TRAJECTORY_TARGET_TYPE")
    if not isinstance(log, str) or not log.strip():
        return gap("MNLM_TRAJECTORY_NO_EVALUATION")
    if result["log_bytes"] > MAX_LOG_BYTES:
        return gap("MNLM_TRAJECTORY_EVALUATION_BUDGET")
    lines = log.splitlines()
    if any(lines.count(marker) != 1 for marker in (SETUP, START, END)):
        return gap("MNLM_TRAJECTORY_EXECUTION_BOUNDARY")
    setup, start, end = (lines.index(marker) for marker in (SETUP, START, END))
    if not setup < start < end:
        return gap("MNLM_TRAJECTORY_EXECUTION_BOUNDARY")
    tail = [line.strip() for line in lines[end + 1:] if line.strip()]
    if len(tail) != 1 or tail[0] not in VERDICTS:
        return gap("MNLM_TRAJECTORY_EVALUATION_VERDICT")
    result["provider_verdict"] = VERDICTS[tail[0]]
    headers = [i for i in range(start + 1, end) if SUMMARY.fullmatch(lines[i])]
    if len(headers) != 1:
        return gap("MNLM_BENCHMARK_TEST_FORMAT")
    roles = {test: role for role, tests in metadata["tests"].items() for test in tests}
    trie = required_trie(roles)
    observations = {test: [] for test in roles}
    unmatched = Counter()
    ambiguous = []
    for i in range(headers[0] + 1, end):
        status, sep, node = lines[i].partition(" ")
        if not sep or status not in STATUSES:
            continue
        # Preserve spaces in parameter IDs. A failure detail separator is allowed
        # only where it leaves a unique exact required node ID, never a prefix.
        candidates = {node} if node in roles else set()
        if status in ("FAILED", "ERROR"):
            candidates = failure_matches(node, trie)
        if len(candidates) > 1:
            ambiguous.append(i)
        elif candidates:
            test = next(iter(candidates))
            observations[test].append({"status": status, "line": i,
                                       "line_sha256": hashlib.sha256(lines[i].encode()).hexdigest()})
        else:
            unmatched[status] += 1
    result["required_tests"] = [{"test": test, "role": roles[test], "observations": observations[test]}
                                for test in sorted(roles)]
    result["unmatched_status_lines"] = dict(sorted(unmatched.items()))
    result["ambiguous_lines"] = ambiguous
    result["required_count"] = len(roles)
    result["missing_count"] = sum(not rows for rows in observations.values())
    result["multiple_count"] = sum(len(rows) > 1 for rows in observations.values())
    result["test_section_sha256"] = hashlib.sha256("\n".join(lines[start + 1:end]).encode()).hexdigest()
    if ambiguous or result["multiple_count"]:
        return gap("MNLM_BENCHMARK_AMBIGUOUS_TEST_RESULT")
    if result["missing_count"]:
        return gap("MNLM_BENCHMARK_REQUIRED_TEST_MISSING")
    states = [rows[0]["status"] for rows in observations.values()]
    if any(state not in ("PASSED", "FAILED", "ERROR") for state in states):
        return gap("MNLM_BENCHMARK_UNSUPPORTED_TEST_STATE")
    result["required_outcome"] = int(all(state == "PASSED" for state in states))
    if bool(result["required_outcome"]) != result["provider_verdict"]:
        return gap("MNLM_BENCHMARK_REQUIRED_VERDICT_CONFLICT")
    if result["provider_verdict"] != target:
        return gap("MNLM_TRAJECTORY_TARGET_CONFLICT")
    return {**result, "outcome": result["required_outcome"],
            "status": "provider_required_tests_consistent"}
