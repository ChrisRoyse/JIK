"""Prepare an explicit SWE cohort with one hash/projection pass per source."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from oracle.prepare_swe import ATTEMPT_COLUMNS, BENCHMARK_COLUMNS, prepare, relative
from senses.contract import SenseError, require
from senses.trajectory_population import bound_path, publish, read_pinned

CODE = "MNLM_EXECUTION_COHORT"


def run(root, manifest, manifest_sha256, output):
    started = time.perf_counter()
    scope = read_pinned(manifest, manifest_sha256)
    require(scope["schema"] == "swe-execution-cohort-v1" and not output.exists()
            and 0 < len(scope["plans"]) <= 128, CODE, "explicit bounded cohort/output required")
    plans, requests, tasks = [], [], set()
    for binding in scope["plans"]:
        relative(binding["path"])
        path = bound_path(root, binding)
        plan = read_pinned(path, binding["sha256"])
        require(plan["schema"] == "swe-execution-plan-v1" and plan["task"] not in tasks,
                CODE, "duplicate task or unsupported plan")
        tasks.add(plan["task"])
        plans.append((path, binding["sha256"], plan))
        requests += [(plan["benchmark"], BENCHMARK_COLUMNS), (plan["attempt"], ATTEMPT_COLUMNS)]
    validated = time.perf_counter()
    rows = BoundRows(root, requests)
    bound = time.perf_counter()
    output.mkdir(parents=True)
    publish(output / "row-bindings.json", {"sources": rows.receipts, "admitted": False})
    results = []
    for i, (path, sha, plan) in enumerate(plans):
        dest = output / str(i)
        try:
            prepare(root, path, sha, dest, rows=rows)
            result = {"task": plan["task"], "status": "prepared", "directory": str(i),
                      "plan_sha256": sha}
        except (SenseError, OSError, ValueError, KeyError, TypeError, tarfile.TarError,
                zipfile.BadZipFile, subprocess.SubprocessError) as exc:
            error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
            result = {"task": plan["task"], "status": "gap", "error": error.envelope(),
                      "directory": str(i), "plan_sha256": sha}
        results.append(result)
    finished = time.perf_counter()
    return publish(output / "report.json", {"schema": "swe-cohort-preparation-v1",
        "manifest_sha256": manifest_sha256, "results": results, "admitted": False,
        "wall_seconds": finished - started, "phases": {"plan_validation": validated - started,
        "row_binding": bound - validated, "preparation": finished - bound}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    try:
        print(json.dumps({"sha256": run(a.root, a.manifest, a.manifest_sha256, a.output)}))
    except (SenseError, OSError, ValueError, KeyError, TypeError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
