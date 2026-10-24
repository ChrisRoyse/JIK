"""Exercise exact selectors against retained tests and explicit boundary controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from oracle.container_run import execute
from oracle.prepare_swe import inventory, ATTEMPT_COLUMNS
from oracle.replay_swe import recount, verify_prepared
from oracle.test_contract import selectors
from senses.contract import SenseError, require
from senses.trajectory_population import publish, read_pinned

CODE = "MNLM_EXECUTION_SELECTION_FSV"


def run(root, prepared, prepared_sha256, plan_path, plan_sha256, output):
    spec, before = verify_prepared(prepared, prepared_sha256)
    plan = read_pinned(plan_path, plan_sha256)
    require(spec["plan_sha256"] == plan_sha256, CODE, "plan differs from prepared source")
    output.mkdir(parents=True, exist_ok=False)
    # Select a proper subset of the real retained module, without relabeling it.
    required = {next(n for n, r in spec["required_tests"].items() if r == role): role
                for role in ("FAIL_TO_PASS", "PASS_TO_PASS")}
    require(len(required) < len(spec["required_tests"]), CODE, "proper subset required")
    execute(spec["image"], prepared / "reference", prepared / "runtime", output / "subset",
            modules=selectors(required))
    record = json.loads((output / "subset/execution.json").read_bytes())
    result = recount(record, required)
    require(result["outcome"] == 1 and result["counts"] == {"passed": 2}, CODE, "real subset changed")
    publish(output / "subset/recount.json", result)
    checks = []

    def refuses(name, action, code):
        try:
            action()
        except SenseError as exc:
            require(exc.code == code, CODE, name + " refused for wrong reason")
            checks.append({"name": name, "code": exc.code})
        else:
            raise SenseError(CODE, name + " accepted broken state")

    refuses("subset-is-not-full-coverage", lambda: recount(record, spec["required_tests"]),
            "MNLM_EXECUTION_EVIDENCE")
    incomplete = deepcopy(record)
    incomplete["result"]["exitcode"] = 4
    publish(output / "collection-error.json", incomplete)
    refuses("collection-error-is-not-failure", lambda: recount(incomplete, required),
            "MNLM_EXECUTION_NOT_EVALUATED")
    for i, node in enumerate(("-p::plugin", "../escape.py::test_a", "/tmp/a.py::test_a",
                              "tests/a.py", "tests/a.py::test_a[truncated", "tests/a.py::test_a\n")):
        payload = {node: "FAIL_TO_PASS"}
        publish(output / ("invalid-selector-" + str(i) + ".json"), payload)
        refuses("invalid-selector-" + str(i), lambda: selectors(payload), "MNLM_EXECUTION_TEST_CONTRACT")
    # This is a named test-harness control, never corpus input or an outcome label.
    control = output / "identity-control"
    control.mkdir()
    (control / "test_identity.py").write_text(
        'import pytest\n@pytest.mark.parametrize("x", [1, 2, 3], ids=["left::right", "nested[x]", "-p no:terminal"])\n'
        'def test_parameter(x):\n    assert x > 0\n'
        'class TestA:\n    def test_same(self):\n        assert True\n'
        'class TestB:\n    def test_same(self):\n        assert True\n', encoding="utf-8")
    control_before = inventory(control)
    identities = {"test_identity.py::test_parameter[" + s + "]": "PASS_TO_PASS"
                  for s in ("left::right", "nested[x]", "-p no:terminal")}
    identities.update({"test_identity.py::TestA::test_same": "FAIL_TO_PASS",
                       "test_identity.py::TestB::test_same": "PASS_TO_PASS"})
    execute(spec["image"], control, prepared / "runtime", output / "parameters", modules=selectors(identities))
    parameter_record = json.loads((output / "parameters/execution.json").read_bytes())
    parameter_result = recount(parameter_record, identities)
    require(parameter_result["counts"] == {"passed": 5} and inventory(control) == control_before,
            CODE, "parameter identity control changed or failed")
    publish(output / "parameters/recount.json", {**parameter_result, "control_only": True})
    changed = deepcopy(parameter_record)
    changed["result"]["junit"] = changed["result"]["junit"].replace("left::right", "left:right")
    publish(output / "parameter-mismatch.json", changed)
    refuses("parameter-identity-change", lambda: recount(changed, identities), "MNLM_EXECUTION_EVIDENCE")
    binding = plan["attempt"]
    rows = BoundRows(root, [(binding, ATTEMPT_COLUMNS), (binding, ATTEMPT_COLUMNS)])
    require(len(rows.receipts) == 1 and rows.receipts[0]["hash_passes"] == 1
            and rows.receipts[0]["selected_rows"] == 1, CODE, "duplicate binding reread source")
    a = rows.read(binding, ATTEMPT_COLUMNS)
    a["instance_id"] = "mutated-consumer-copy"
    require(rows.read(binding, ATTEMPT_COLUMNS)["instance_id"] == plan["task"], CODE, "cache mutation leaked")
    refuses("unbound-column", lambda: rows.read(binding, ["target"]), "MNLM_EXECUTION_ROW_BINDING")
    for name, edit, error in (
        ("bad-source-digest", {"sha256": "0" * 64}, "MNLM_UCR_INPUT_DIGEST"),
        ("row-outside-source", {"row": binding["rows"]}, "MNLM_EXECUTION_ROW_BINDING"),
    ):
        bad = {**binding, **edit}
        publish(output / (name + ".json"), bad)
        refuses(name, lambda: BoundRows(root, [(bad, ATTEMPT_COLUMNS)]), error)
    refuses("conflicting-source-binding", lambda: BoundRows(root, [
        (binding, ATTEMPT_COLUMNS), ({**binding, "sha256": "0" * 64}, ATTEMPT_COLUMNS)]),
        "MNLM_EXECUTION_ROW_BINDING")
    require(before == {k: inventory(prepared / k) for k in before}, CODE, "source/runtime mutated")
    return publish(output / "report.json", {"checks": checks, "count": len(checks),
        "real_subset": result["counts"], "control_only_parameters": parameter_result["counts"],
        "row_bindings": rows.receipts, "before": before, "after": before,
        "admitted": False, "author_checked": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prepared", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared-sha256", "plan-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps({"sha256": run(a.root, a.prepared, a.prepared_sha256,
                                    a.plan, a.plan_sha256, a.output)}))
