"""Cold source/report readback and broken-state controls for a real SWE execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.container_run import validate_config
from oracle.prepare_swe import inventory, patch_paths
from oracle.replay_swe import recount, verify_prepared
from oracle.test_contract import junit_identity
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_REEXECUTION_FSV"


def run(prepared, prepared_sha256, execution, report_sha256, output):
    spec, before = verify_prepared(prepared, prepared_sha256)
    report = read_pinned(execution / "report.json", report_sha256)
    require(report["prepared_sha256"] == prepared_sha256 and report["admitted"] is False,
            CODE, "report binding or admission differs")
    output.mkdir(parents=True, exist_ok=False)
    independent = {}
    for name in ("base", "reference", "agent"):
        record = json.loads((execution / name / "execution.json").read_bytes())
        text = (execution / name / "stdout.bin").read_text(encoding="utf-8")
        envelopes = [json.loads(s.split("=", 1)[1]) for s in text.splitlines() if s.startswith("MNLM_EXECUTION_RESULT=")]
        require(envelopes == [record["result"]], CODE, "durable stdout differs from stored envelope")
        reported = report["conditions"][name]["states"]
        lost = set(report["conditions"][name].get("collection_failed", []))
        unc = record["result"].get("uncollectable") or {}
        require(name != "reference" or not lost, CODE, "reference witness cannot have collection failures")
        require(all(n.split("::", 1)[0] in unc and reported[n] == "failed" for n in lost)
                and lost == {n for n in reported if n.split("::", 1)[0] in unc}, CODE,
                "collection-failed tests differ from the envelope's uncollectable modules")
        junit = record["result"]["junit"]
        cases = list(ET.fromstring(junit).iter("testcase")) if junit else []
        states = {(c.attrib["classname"], c.attrib["name"]):
                  ("failed" if c.find("failure") is not None else "passed") for c in cases}
        expected = {junit_identity(n): v for n, v in reported.items() if n not in lost}
        require(states == expected and len(cases) == len(spec["required_tests"]) - len(lost), CODE,
                "independent JUnit recount differs")
        independent[name] = {"tests": len(cases), "collection_failed": len(lost),
                             "failed": sum(v == "failed" for v in states.values()) + len(lost)}
    checks = []

    def refuses(name, action, code):
        try:
            action()
        except SenseError as exc:
            require(exc.code == code, CODE, name + " wrong refusal")
            checks.append({"name": name, "code": exc.code})
        else:
            raise SenseError(CODE, name + " accepted broken state")

    good = json.loads((execution / "reference" / "execution.json").read_bytes())
    for name, edit in (
        ("missing-test", lambda r: r["result"]["collected"].pop()),
        ("duplicate-test", lambda r: r["result"]["collected"].append(r["result"]["collected"][0])),
        ("missing-phase", lambda r: r["result"]["phases"].pop()),
        ("failed-setup", lambda r: r["result"]["phases"][0].update(outcome="failed")),
        ("junit-mismatch", lambda r: r["result"].update(junit="<testsuites/>")),
        ("exit-mismatch", lambda r: r.update(exitcode=1)),
        ("timeout-not-failure", lambda r: r.update(status="MNLM_EXECUTION_TIMEOUT")),
    ):
        changed = deepcopy(good)
        edit(changed)
        publish(output / (name + ".json"), changed)
        refuses(name, lambda: recount(changed, spec["required_tests"]), "MNLM_EXECUTION_EVIDENCE")
    unwitnessed = deepcopy(good)
    unwitnessed["result"]["uncollectable"] = {sorted(spec["required_tests"])[0].split("::", 1)[0]: "control"}
    publish(output / "unwitnessed-import-failure.json", unwitnessed)
    refuses("unwitnessed-import-failure", lambda: recount(unwitnessed, spec["required_tests"]),
            "MNLM_EXECUTION_NOT_EVALUATED")
    for name, text in (
        ("patch-traversal", "diff --git a/../outside.py b/../outside.py\n"),
        ("patch-symlink", "diff --git a/link b/link\nnew file mode 120000\n"),
    ):
        refuses(name, lambda: patch_paths(text), "MNLM_EXECUTION_PREPARATION")
    copied = output / "payload-control"
    shutil.copytree(prepared, copied)
    source = copied / "base" / sorted(spec["conditions"]["base"])[0]
    data = source.read_bytes()
    source.write_bytes(data + b"\n# altered control\n")
    refuses("source-mutated", lambda: verify_prepared(copied, prepared_sha256), "MNLM_EXECUTION_EVIDENCE")
    source.write_bytes(data)
    wheel = next((copied / "runtime").glob("pytest-*.whl"), None) or         next(p for p in sorted((copied / "runtime" / "site").rglob("*.py")))  # site-v1 layout
    wheel.write_bytes(wheel.read_bytes() + b"control")
    refuses("runtime-mutated", lambda: verify_prepared(copied, prepared_sha256), "MNLM_EXECUTION_EVIDENCE")
    config = json.loads((execution / "probe" / "before.json").read_bytes())
    mounts = {m["Destination"]: m["Source"] for m in config["Mounts"]}
    label = config["Config"]["Labels"]["mnlm.oracle.run"]
    # The writable checkout tmpfs exists only for site-v1 runtimes; a zip-layout run must never carry it.
    work_tmpfs = "/work-rw" in config["HostConfig"]["Tmpfs"]
    require(not work_tmpfs or spec.get("runtime_layout") is not None, CODE, "work tmpfs on a non-site runtime")
    for name, edit in (
        ("network-enabled", lambda r: r["HostConfig"].update(NetworkMode="host")),
        ("root-writable", lambda r: r["HostConfig"].update(ReadonlyRootfs=False)),
        ("bind-writable", lambda r: r["Mounts"][0].update(RW=True)),
        ("image-changed", lambda r: r.update(Image="sha256:" + "0" * 64)),
    ):
        changed = deepcopy(config)
        edit(changed)
        refuses(name, lambda: validate_config(changed, spec["image"], label, mounts,
                                              work_tmpfs), "MNLM_EXECUTION_SANDBOX")
    refuses("changed-report-digest", lambda: read_pinned(execution / "report.json", "0" * 64), "MNLM_UCR_INPUT_DIGEST")
    refuses("immutable-report", lambda: publish(execution / "report.json", {}), "MNLM_UCR_OUTPUT_EXISTS")
    require(before == {n: inventory(prepared / n) for n in before}, CODE, "controls changed authoritative payload")
    require(hashlib.sha256((execution / "report.json").read_bytes()).hexdigest() == report_sha256,
            CODE, "controls changed authoritative report")
    return publish(output / "report.json", {"checks": checks, "count": len(checks), "independent": independent,
                  "before": before, "after": before, "admitted": False, "author_checked": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared-sha256", "report-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps({"sha256": run(a.prepared, a.prepared_sha256, a.execution, a.report_sha256, a.output)}))
