"""Commission independent execution of one pinned task; no scientific admission."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.container_run import execute
from oracle.prepare_swe import check_modes, inventory
from oracle.test_contract import selectors, junit_identity
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_EXECUTION_EVIDENCE"
ENVIRONMENT = "MNLM_EXECUTION_ENVIRONMENT"


def recount(execution, required, allow_uncollectable=False):
    """Uncollectable required modules count as failed tests only when the caller has an environment
    witness (the reference condition collected and passed everything in the same runtime)."""
    selectors(required)
    require(execution["status"] == "observed" and not execution["oom_killed"], CODE, "execution incomplete")
    r = execution["result"]
    unc = r.get("uncollectable") or {}
    lost = sorted(n for n in required if n.split("::", 1)[0] in unc)
    if lost and not allow_uncollectable:
        raise SenseError("MNLM_EXECUTION_NOT_EVALUATED",
                         f"{len(lost)} required tests are in modules that fail to import: {sorted(unc)[:5]}",
                         "Without a reference witness an import failure is an environment gap, not a task failure.")
    full = required
    required = {n: v for n, v in required.items() if n not in lost}
    if not required:
        require(r["exitcode"] == 1 and r.get("main_exitcode") is None and not r["phases"], CODE,
                "no runnable required tests but a main run was reported")
        return {"outcome": 0, "states": {n: "failed" for n in lost}, "collection_failed": lost,
                "counts": {"failed": len(lost)}, "independently_executed": True,
                "scope": "this_pinned_runtime_only", "admitted": False}
    if r["exitcode"] not in (0, 1):
        raise SenseError("MNLM_EXECUTION_NOT_EVALUATED",
                         f"pytest exited {r['exitcode']} with {len(r['collected'])} collected tests; no outcome label",
                         "Inspect the retained condition stdout.bin and stderr.bin; pin missing dependencies or "
                         "repair the explicit runtime/test contract. Never count setup or collection errors as task failure.")
    require(len(r["collected"]) == len(required) and set(r["collected"]) == set(required),
            CODE, f"collected {len(r['collected'])} tests versus {len(required)} required; "
            f"missing={len(set(required) - set(r['collected']))}, extra={len(set(r['collected']) - set(required))}; "
            "inspect exact collected IDs; do not infer aliases")
    phases = {node: [] for node in required}
    for phase in r["phases"]:
        require(phase["nodeid"] in phases, CODE, "undeclared test report")
        phases[phase["nodeid"]].append((phase["when"], phase["outcome"]))
    states = {}
    for node, values in phases.items():
        require(len(values) == 3 and values[0] == ("setup", "passed")
                and values[2] == ("teardown", "passed") and values[1][0] == "call"
                and values[1][1] in ("passed", "failed"), CODE, "missing/duplicate/abnormal test phase")
        states[node] = values[1][1]
    require(isinstance(r["junit"], str) and "<!" not in r["junit"], CODE, "JUnit evidence missing/unsupported")
    xml = ET.fromstring(r["junit"])
    cases = list(xml.iter("testcase"))
    require(len(cases) == len(required) and not list(xml.iter("error")) and not list(xml.iter("skipped")),
            CODE, "JUnit inventory or execution errors")
    actual = {}
    for c in cases:
        key = (c.attrib["classname"], c.attrib["name"])
        require(key not in actual, CODE, "duplicate JUnit result")
        actual[key] = "failed" if c.find("failure") is not None else "passed"
    expected = {}
    for node, state in states.items():
        key = junit_identity(node)
        require(key not in expected, CODE, "required tests collide in JUnit identity")
        expected[key] = state
    require(actual == expected, CODE, "JUnit disagrees with hook observations")
    states.update({n: "failed" for n in lost})
    states = {n: states[n] for n in full}
    failed = sum(v == "failed" for v in states.values())
    require(execution["exitcode"] == r["exitcode"] == int(failed > 0), CODE, "process exit contradicts tests")
    return {"outcome": int(not failed), "states": states, **({"collection_failed": lost} if lost else {}),
            "counts": dict(sorted(Counter(states.values()).items())), "independently_executed": True,
            "scope": "this_pinned_runtime_only", "admitted": False}


def verify_prepared(prepared, prepared_sha256):
    spec = read_pinned(prepared / "prepared.json", prepared_sha256)
    require(spec["admitted"] is False, CODE, "preparation is not admission")
    expected = {"runtime": spec["runtime"], **spec["conditions"]}
    before = {name: inventory(prepared / name) for name in expected}
    require(before == expected, CODE, "prepared bytes changed")
    for name in spec["conditions"]:
        check_modes(prepared / name, spec.get("condition_executables", {}).get(name, spec.get("executables", [])))
    return spec, before


def run(prepared, prepared_sha256, output):
    started = time.perf_counter()
    spec, before = verify_prepared(prepared, prepared_sha256)
    output.mkdir(parents=True, exist_ok=False)
    validated = time.perf_counter()
    probe = execute(spec["image"], prepared / "base", prepared / "runtime", output / "probe", "probe")
    require(probe["status"] == "observed" and probe["exitcode"] == 0, CODE, "sandbox probe failed")
    checks = probe["result"]["probe"]
    require(checks["uid"] == 65534 and all(v is True for k, v in checks.items() if k != "uid"),
            CODE, "sandbox isolation failed")
    probed = time.perf_counter()
    rows = {}

    def condition(name, witness):
        execute(spec["image"], prepared / name, prepared / "runtime", output / name,
                modules=selectors(spec["required_tests"]))
        # Reopen durable output instead of accepting a child function's return.
        stored = json.loads((output / name / "execution.json").read_bytes())
        rows[name] = recount(stored, spec["required_tests"], allow_uncollectable=witness)
        publish(output / name / "recount.json", rows[name])

    # Environment stage: the reference is the witness, the base must establish the benchmark's roles.
    # The agent condition is not executed until this stage passes, so it can never select a runtime.
    try:
        condition("reference", False)
        require(rows["reference"]["outcome"] == 1, CODE, "reference fix failed under this runtime")
        condition("base", True)
        require(all(rows["base"]["states"][n] == ("failed" if role == "FAIL_TO_PASS" else "passed")
                    for n, role in spec["required_tests"].items()), CODE, "baseline does not establish required test roles")
    except SenseError as exc:
        raise SenseError(ENVIRONMENT, f"{exc.code}: {exc.message}", "This runtime does not reproduce the benchmark "
                         "contract; a later predeclared candidate may be tried. No agent condition was run.") from exc
    condition("agent", True)
    rows = {k: rows[k] for k in ("base", "reference", "agent")}
    require(rows["agent"]["outcome"] == int(spec["provider_target"]), CODE,
            "independent agent outcome differs from provider; retain the discrepancy")
    conditions_done = time.perf_counter()
    controls = {}
    for mode, expected_code, wall in (("timeout-control", "MNLM_EXECUTION_TIMEOUT", 2),
                                      ("output-control", "MNLM_EXECUTION_OUTPUT_BUDGET", 30)):
        record = execute(spec["image"], prepared / "base", prepared / "runtime", output / mode, mode, wall=wall)
        require(record["status"] == expected_code, CODE, "resource negative control failed")
        controls[mode] = record["status"]
    controls_done = time.perf_counter()
    after = {name: inventory(prepared / name) for name in before}
    require(before == after, CODE, "execution mutated source or runtime payload")
    finished = time.perf_counter()
    result = {"schema": "swe-reexecution-v1", "task": spec["task"], "prepared_sha256": prepared_sha256,
              "image": spec["image"], "conditions": rows, "controls": controls, "probe": checks,
              "before": before, "after": after, "wall_seconds": finished - started,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob("*.py")},
              "admitted": False, "original_provider_runtime": False, "author_checked": True}
    sha = publish(output / "report.json", result)
    published = time.perf_counter()
    publish(output / "receipt.json", {"report_sha256": sha, "wall_seconds": published - started,
        "phases": {"validation": validated - started, "probe": probed - validated,
                   "conditions": conditions_done - probed, "resource_controls": controls_done - conditions_done,
                   "final_readback": finished - controls_done, "publication": published - finished}})
    return {"task": spec["task"], "conditions": {k: v["counts"] for k, v in rows.items()},
            "controls": controls, "wall_seconds": result["wall_seconds"], "admitted": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--prepared-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(run(args.prepared, args.prepared_sha256, args.output)))
    except (SenseError, OSError, ValueError, KeyError, TypeError, ET.ParseError,
            subprocess.SubprocessError) as exc:
        error = exc if isinstance(exc, SenseError) else SenseError(CODE, str(exc))
        print(json.dumps(error.envelope()), file=sys.stderr)
        sys.exit(2)
