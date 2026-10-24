"""Private benchmark oracle metadata; never an input-channel adapter (#12).

Only declared columns are decoded. Gold/test patches remain outside this path.
One streaming scan builds a selected-task index, preserving missing/invalid rows.
"""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
import re

import pyarrow.parquet as pq

from senses.contract import digest, is_sha256, require
from senses.trajectory_population import open_bound

CODE = "MNLM_BENCHMARK_CONTRACT"
COLUMNS = ("instance_id", "repo", "base_commit", "environment_setup_commit",
           "FAIL_TO_PASS", "PASS_TO_PASS")
FORMATS = {"nebius/SWE-bench-extra": ("train", "arrow_list"),
           "princeton-nlp/SWE-bench": ("dev", "json_string")}
SHA1 = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def validate_scope(scope):
    require(isinstance(scope, dict) and set(scope) == {"schema", "approval", "files"}
            and scope["schema"] == "trajectory-benchmark-scope-v1"
            and isinstance(scope["approval"], str) and bool(scope["approval"])
            and isinstance(scope["files"], list) and bool(scope["files"]), CODE,
            "pin the benchmark scope and decision")
    seen = set()
    for file in scope["files"]:
        require(isinstance(file, dict) and set(file) == {
            "repo", "revision", "split", "provider_path", "path", "bytes", "sha256", "receipt"
        }, CODE, "benchmark file contract keys differ")
        require(isinstance(file["repo"], str) and file["repo"] in FORMATS, CODE,
                "unsupported benchmark schema; declare a source adapter")
        require(file["split"] == FORMATS[file["repo"]][0]
                and isinstance(file["revision"], str) and SHA1.fullmatch(file["revision"])
                and is_sha256(file["sha256"]) and type(file["bytes"]) is int
                and file["bytes"] > 0, CODE, "invalid split, revision or byte binding")
        for key in ("path", "provider_path"):
            name = file[key]
            require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
                    CODE, "portable relative benchmark path required")
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and str(path) == name
                    and path.suffix == ".parquet", CODE, "invalid benchmark path")
        require(file["path"] not in seen, CODE, "duplicate metadata file")
        seen.add(file["path"])
    return scope


def contract(raw, encoding):
    """Item errors are explicit gaps; neither fallback coercion nor partial lists."""
    import json

    report = {"status": CODE, "projection_sha256": digest(raw)}
    if not isinstance(raw["repo"], str) or not REPO.fullmatch(raw["repo"]):
        return {**report, "reason": "repository identity missing or malformed"}
    for key in ("base_commit", "environment_setup_commit"):
        if not isinstance(raw[key], str) or not SHA1.fullmatch(raw[key]):
            return {**report, "reason": f"{key} missing or malformed"}
    tests = {}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = raw[key]
        if encoding == "json_string":
            if not isinstance(value, str):
                return {**report, "reason": f"{key} is not the declared JSON string"}
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                return {**report, "reason": f"{key} invalid JSON"}
        elif encoding != "arrow_list":
            return {**report, "reason": "undeclared test-list encoding"}
        if not isinstance(value, list) or any(
            not isinstance(t, str) or not t or t != t.strip() or any(ord(c) < 32 for c in t)
            for t in value
        ):
            return {**report, "reason": f"{key} invalid test identities"}
        if len(value) != len(set(value)):
            return {**report, "reason": f"{key} duplicate test identities"}
        tests[key] = sorted(value)
    if not tests["FAIL_TO_PASS"] or set(tests["FAIL_TO_PASS"]) & set(tests["PASS_TO_PASS"]):
        return {**report, "reason": "empty fail-to-pass set or overlapping test roles"}
    return {**report, "status": "contract_present", "repo": raw["repo"],
            "base_commit": raw["base_commit"],
            "environment_setup_commit": raw["environment_setup_commit"], "tests": tests}


def load_contracts(root, scope, tasks):
    scope = validate_scope(scope)
    selected, seen, census = {}, set(), []
    for file in scope["files"]:
        offset = 0
        with open_bound(root, file) as stream:
            parquet = pq.ParquetFile(stream)
            require(set(COLUMNS) <= set(parquet.schema_arrow.names), CODE,
                    "required oracle columns missing")
            for batch in parquet.iter_batches(batch_size=32, columns=list(COLUMNS)):
                for raw in batch.to_pylist():
                    task = raw["instance_id"]
                    require(isinstance(task, str) and task, CODE, "metadata has no task identity")
                    require(task not in seen, "MNLM_BENCHMARK_DUPLICATE_TASK",
                            "metadata identity is not unique; resolve the source scope")
                    seen.add(task)
                    if task in tasks:
                        selected[task] = {**contract(raw, FORMATS[file["repo"]][1]),
                                          "source": {k: file[k] for k in
                                                     ("repo", "revision", "path", "sha256")},
                                          "row": offset}
                    offset += 1
            require(offset == parquet.metadata.num_rows, CODE, "metadata scan ended early")
        census.append({"path": file["path"], "rows": offset, "sha256": file["sha256"]})
    for task in sorted(tasks - selected.keys()):
        selected[task] = {"status": "MNLM_BENCHMARK_TASK_ABSENT",
                          "reason": "no exact identity in the pinned metadata scope"}
    return selected, {"files": census, "selected": len(selected),
                      "status_counts": dict(sorted(Counter(r["status"] for r in selected.values()).items())),
                      "columns_read": list(COLUMNS)}
