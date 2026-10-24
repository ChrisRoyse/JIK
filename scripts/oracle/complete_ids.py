"""Complete whitespace-truncated parametrized test IDs from the reference tree's own collection (#12 D).

The benchmark metadata cut some parametrized IDs at their first space (`test_x[a` for `test_x[a b]`).
The full ID is taken only from pytest's own collection of the reference tree, inside the sandbox and
in the plan's runtime, and only when exactly one collected ID starts with `truncated + " "`. Anything
else stays a gap. The map and its evidence are bound into a new plan; prepare() re-checks both.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from oracle import declared_deps
from oracle.container_run import execute
from oracle.prepare_swe import (MODES, apply_patch, bytes_bound, executables, row_bound, source_files)
from oracle.runtime_site import plugins, unpack
from oracle.test_contract import selectors
from senses.contract import SenseError, require
from senses.trajectory_population import publish

CODE = "MNLM_TEST_ID_COMPLETION"
RULE = "unique collected id starting with truncated + ' ' in the reference tree, plan runtime, sandbox"


def truncated(node):
    try:
        selectors({node: "FAIL_TO_PASS"})
        return False
    except SenseError:
        return "[" in node and not node.endswith("]")


def check(collected, mapping):
    """Shared by complete() and prepare(): every completion unique, prefix-exact and collected."""
    for short, full in mapping.items():
        require(truncated(short) and full.startswith(short + " ") and full in collected
                and sum(c.startswith(short + " ") for c in collected) == 1, CODE,
                f"completion of {short!r} is not a unique collected identity")
        selectors({full: "FAIL_TO_PASS"})


def complete(root, plan, required, work):
    """required: benchmark test IDs (may include truncated ones). Returns (mapping, evidence binding)."""
    short = sorted(n for n in required if truncated(n))
    require(bool(short), CODE, "no truncated identities to complete")
    work.mkdir(parents=True, exist_ok=False)
    modes = {}
    files = source_files(root, plan, modes)
    tree = work / "reference"
    for path, data in files.items():
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_bytes(data)
    executables(tree, sorted(n for n, m in modes.items() if MODES[m]))
    meta = row_bound(root, plan["benchmark"], ["patch", "test_patch"])
    apply_patch(tree, work / "reference.patch", meta["patch"], plan["allowed_paths"]["reference"])
    apply_patch(tree, work / "test.patch", meta["test_patch"], plan["allowed_paths"]["test"])
    runtime = work / "runtime"; site = runtime / "site"; site.mkdir(parents=True)
    owners, skipped = {}, []
    for b in plan["wheels"]:
        unpack(bytes_bound(root, b), Path(b["path"]).name, site, owners, skipped, plan.get("interpreter"))
    record = declared_deps.read(files)
    spec = {"schema": "swe-runtime-layout-v1", "layout": plan["runtime_layout"], "site": "site",
            "source_roots": [r["path"] for r in record["source_roots"]], "plugins": plugins(site),
            "skipped_data_entries": sorted(skipped)}
    (runtime / "mnlm-runtime.json").write_text(json.dumps(spec, sort_keys=True))
    if os.name == "posix":
        (runtime / "mnlm-runtime.json").chmod(0o644)
    modules = sorted({n.split("::", 1)[0] for n in required})
    result = execute(plan["image"], tree, runtime, work / "collect", "collect-only", modules=modules)
    require(result["status"] == "observed" and result["result"] and "collected" in result["result"], CODE,
            "collection did not produce an envelope")
    collected = result["result"]["collected"]
    mapping = {}
    for s in short:
        hits = [c for c in collected if c.startswith(s + " ")]
        require(len(hits) == 1, CODE, f"{s!r}: {len(hits)} collected completions; an identity needs exactly one")
        mapping[s] = hits[0]
    require(not (set(mapping.values()) & set(required)), CODE, "a completion duplicates a listed identity")
    check(collected, mapping)
    shutil.rmtree(tree); shutil.rmtree(runtime)  # reconstructible; evidence is the collection record
    evidence = work / "collect" / "execution.json"
    data = evidence.read_bytes()
    return mapping, {"path": evidence.relative_to(root).as_posix(), "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()}
