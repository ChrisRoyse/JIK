"""Export only the base repository and submitted patch before external evaluation.

This is a different prediction cutoff from the early four-exchange forecast.
No benchmark columns, reference solution, test patch or execution artifacts are read.
The output contract is source-neutral: two byte-exact file snapshots and their order.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.perf_counter()

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.prepare_swe import apply_patch, inventory, row_bound, source_files
from senses.contract import SenseError, digest, require
from senses.trajectory_population import publish, read_pinned

SCHEMA = "swe-pretest-input-plan-v1"
CUTOFF = "submitted_patch_before_external_evaluation"
CODE = "MNLM_PRETEST_INPUT"
KEYS = {"schema", "cutoff", "task", "base_commit", "archive", "tree", "commit", "attempt", "licence"}


def input_plan(execution_plan, *, licence=None):
    """Explicit projection: private oracle fields do not affect input identity."""
    require(isinstance(licence, str) and bool(licence.strip()), CODE,
            "explicit source licence attestation required; never inherit a demo repository's licence")
    return {"schema": SCHEMA, "cutoff": CUTOFF,
            **{k: execution_plan[k] for k in ("task", "base_commit", "archive", "tree", "commit", "attempt")},
            "licence": licence,
            **({"exact_blobs": execution_plan["exact_blobs"]} if execution_plan.get("exact_blobs") else {})}


def validate(plan):
    require(isinstance(plan, dict) and set(plan) - {"exact_blobs"} == KEYS and plan["schema"] == SCHEMA
            and plan["cutoff"] == CUTOFF, CODE, "explicit pre-external-test cutoff and only input bindings required")
    require(all(isinstance(plan[k], str) and plan[k] for k in ("task", "base_commit", "licence")),
            CODE, "task, base commit and licence required")
    for name in ("archive", "tree", "commit", "attempt"):
        allowed = {"path", "bytes", "sha256"} | ({"row", "rows"} if name == "attempt" else set())
        require(isinstance(plan[name], dict) and set(plan[name]) == allowed,
                CODE, "input binding contains unknown or missing fields")
    require(plan.get("exact_blobs", [None]) != [], CODE, "empty exact blob list must be omitted")
    for binding in plan.get("exact_blobs", []):
        require(isinstance(binding, dict) and set(binding) == {"name", "path", "bytes", "sha256"},
                CODE, "exact blob binding contains unknown or missing fields")


def build(root, plan_path, plan_sha256, output, *, entry_started=None):
    start = time.perf_counter()
    beginning = start if entry_started is None else entry_started
    plan = read_pinned(plan_path, plan_sha256)
    validate(plan)
    require(not output.exists(), CODE, "immutable output already exists")
    files = source_files(root, plan)
    source_done = time.perf_counter()
    # Deliberately no trajectory, target, eval_logs, exit_status or benchmark read.
    attempt = row_bound(root, plan["attempt"], ["instance_id", "model_name", "generated_patch"])
    require(attempt["instance_id"] == plan["task"] and isinstance(attempt["model_name"], str)
            and attempt["model_name"], CODE, "selected attempt identity/model differ")
    patch = attempt["generated_patch"]
    from oracle.prepare_swe import patch_paths
    paths = patch_paths(patch)
    input_done = time.perf_counter()
    output.mkdir(parents=True)
    for name in ("base", "submitted"):
        tree = output / name
        tree.mkdir()
        for path, data in files.items():
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    apply_patch(output / "submitted", output / "submitted.patch", patch, sorted(paths))
    before, after = inventory(output / "base"), inventory(output / "submitted")
    require(not (before.keys() - after.keys()), CODE, "deletion not supported by this snapshot adapter")
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    # Include byte counts alongside exact SHA bindings for a bounded cold reader.
    snapshots = {name: {p: {"sha256": sha, "bytes": (output / name / p).stat().st_size}
                        for p, sha in inv.items()}
                 for name, inv in (("base", before), ("submitted", after))}
    result = {"schema": "code-snapshot-transition-v1", "cutoff": CUTOFF,
              "unit": digest(["swe-task", plan["task"]]), "source": "swe-submitted-code",
              "source_binding": {"input_plan_sha256": plan_sha256, "base_commit": plan["base_commit"],
                                 "attempt": plan["attempt"], "task": plan["task"]},
              "model": attempt["model_name"], "licence": plan["licence"], "snapshots": snapshots,
              "changed": changed, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "order": ["base", "submitted"], "clock": "declared_snapshot_order_not_wall_time",
              "evidence": "local_patch_application; not provider execution or external test outcome",
              "admitted": False}
    built = time.perf_counter()
    sha = publish(output / "snapshot.json", result)
    finish = time.perf_counter()
    publish(output / "receipt.json", {"manifest_sha256": sha, "source_files": len(files),
            "source_bytes": sum(map(len, files.values())), "submitted_files": len(after),
            "changed": changed, "wall_seconds": finish - beginning,
            "scope": "function" if entry_started is None else "cli_including_imports",
            "phases": {"imports": start - beginning, "source": source_done - start, "attempt": input_done - source_done,
                       "build": built - input_done, "publication": finish - built}})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    a = p.parse_args()
    try:
        result = build(a.root, a.plan, a.plan_sha256, a.output, entry_started=ENTRY_STARTED)
        print(json.dumps({"changed": result["changed"], "admitted": False}))
    except (SenseError, OSError, ValueError, KeyError, TypeError) as exc:
        err = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(err.envelope()), file=sys.stderr)
        sys.exit(2)
