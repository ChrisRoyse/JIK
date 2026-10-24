"""Real pre-execution condition commitments, outcome joins and durable correction."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import Memory, canonical, digest
from claims.event_journal import EventJournal
from claims.process_conditions import ConditionInputs, FILES, condition_input, publish
from claims.process_observation import ProcessObservations
from claims.tool_evidence import ToolEvidence
from claims.tool_temporal_source import ToolSource
from model.tool_execution import condition_attachment
from oracle.prepare_swe import inventory


PREPARED = "8594049aaa0ce03da33e3af647cb01a1338475e55ff9d8244e7ab088ba1fcf4c"
OLD = "1fae3f91fb2ab923262e79ded1fbce9390951d9889b4ea4dfce601df91176c4f"


def put(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def restore(root, source, obs, saved):
    memory = Memory(ToolEvidence(source, obs))
    journal = EventJournal(root/"experience.sqlite3", saved["contract"], create=False,
                           payload_bound=1 << 20)
    def apply(body):
        record = body["record"]
        command = {k:v for k,v in record["event"].items() if k not in ("result", "affected")}
        check(memory.apply(command) == record, "CONDITION_MEMORY_REPLAY")
    journal.restore(apply, head=saved["journal_head"], contract=digest(saved["contract"]))
    check(memory.snapshot() == saved["memory"], "CONDITION_RESTART")
    journal.close()
    return dict(events=len(memory.events), memory_head=memory.head, exact=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "prepared", "old", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--cold", action="store_true")
    a = p.parse_args()
    start = time.perf_counter()
    source = ToolSource(a.raw)
    old = ProcessObservations(a.old, OLD, source)
    check(sha((a.prepared/"prepared.json").read_bytes()) == PREPARED, "CONDITION_PREPARED")
    spec = json.loads((a.prepared/"prepared.json").read_bytes())
    before = inventory(a.prepared)
    old_bytes = inventory(a.old)
    phases = dict(source=time.perf_counter()-start)
    tick = time.perf_counter()
    if a.cold:
        report = json.loads((a.output/"report.json").read_bytes())
        obs = ProcessObservations(a.output/"observations", report["observation_pin"], source)
        saved = json.loads((a.output/"state.json").read_bytes())
        result = restore(a.output, source, obs, saved)
        check(inventory(a.prepared) == report["prepared_before"] == before
              and inventory(a.old) == report["old_before"] == old_bytes, "CONDITION_SOURCE_RETAINED")
        phases["cold_readback"] = time.perf_counter()-tick
        wall = time.perf_counter()-start
        put(a.output/"cold.json", dict(complete=True, result=result, phases=phases,
                                     wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
        print(json.dumps(result))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    put(a.output/"before.json", dict(prepared=before, old=old_bytes))
    with (a.output/"execution.log").open("xb") as log:
        result = subprocess.run([sys.executable, "-I", "scripts/oracle/observe_tool_process.py",
            "--prepared", str(a.prepared), "--pin", PREPARED,
            "--output", str(a.output/"observations")], stdout=log, stderr=subprocess.STDOUT,
            check=False)
    check(result.returncode == 0, "CONDITION_EXECUTION_DRIVER")
    phases["actual_execution_and_commitment"] = time.perf_counter()-tick
    tick = time.perf_counter()
    obs_pin = (a.output/"observations/COMMITTED").read_text()
    obs = ProcessObservations(a.output/"observations", obs_pin, source)
    inputs = obs.condition_inputs()
    controls = []
    def refuse(label, code, operation):
        try:
            operation()
        except ValueError as e:
            check(str(e) == "MNLM_CODATA_"+code, "CONDITION_CONTROL_CODE")
            controls.append(dict(name=label, refused=str(e)))
        else:
            raise AssertionError(label+" did not refuse")
    readings = {}
    for key, entry in obs.entries.items():
        measured = obs.read(key)
        readings[key] = dict(condition=entry["condition"], input=inputs.read(entry["condition"]),
            returncode=measured["result"]["returncode"], captured=measured["captured"],
            binding=obs.execution_conditions(key), container=measured["container"])
        wrong = next(k for k in inputs.inputs if k != entry["condition"])
        refuse("swap_condition_"+key, "CONDITION_OUTCOME_MISMATCH",
               lambda: inputs.match(wrong, entry, measured))
        wrong_time = dict(measured, started="2000-01-01T00:00:00Z")
        refuse("late_commitment_"+key, "CONDITION_NOT_PREEXECUTION",
               lambda: inputs.match(entry["condition"], entry, wrong_time))
    refuse("historical_receipt_no_commitment", "PROCESS_PRECONDITION_UNAVAILABLE",
           lambda: old.execution_conditions(next(iter(old.entries))))
    poisoned = copy.deepcopy(spec)
    for item in poisoned["conditions"].values():
        item["target"] = dict(untrusted="not a model input", expected_returncode=-999)
    put(a.output/"poisoned-private-targets.json", poisoned)
    harness = sha((a.output/"observations/process_inside.py").read_bytes())
    for name, key in inputs.commitment["routing"].items():
        program = (a.prepared/name/"work/program.py").read_bytes()
        check(condition_input(poisoned, name, program, harness) == inputs.read(key),
              "CONDITION_PRIVATE_TARGET_LEAK")
        renamed = copy.deepcopy(spec)
        renamed["conditions"]["opaque"] = renamed["conditions"].pop(name)
        check(condition_input(renamed, "opaque", program, harness) == inputs.read(key),
              "CONDITION_LABEL_LEAK")
        bad = a.output/(name+"-changed-program.py")
        bad.write_bytes(program+b"\n")
        refuse("changed_program_"+name, "CONDITION_PROGRAM",
               lambda: condition_input(spec, name, bad.read_bytes(), harness))
    controls.append(dict(name="private_targets_and_condition_labels_excluded", invariant=True))
    corrupt = a.output/"corrupt-input"
    corrupt.mkdir()
    for name in FILES:
        shutil.copyfile(a.output/"observations"/name, corrupt/name)
    (corrupt/FILES[0]).write_bytes((corrupt/FILES[0]).read_bytes()+b" ")
    refuse("corrupt_persisted_input", "CONDITION_INPUT_PIN",
           lambda: ConditionInputs(corrupt, inputs.pin))
    bad_runtime = a.output/"changed-runtime"
    shutil.copytree(a.prepared, bad_runtime)
    wheel = next((bad_runtime/"with_dependency/runtime").glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes()+b"changed")
    refuse("changed_runtime_wheel", "CONDITION_RUNTIME_BYTES",
           lambda: publish(bad_runtime, spec, PREPARED, harness))
    phases["actual_outcome_and_negative_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    memory = Memory(ToolEvidence(source, obs))
    session = SimpleNamespace(memory=memory, observations=obs)
    contract = dict(schema="execution-condition-memory-proof-v1", observation_set=obs_pin,
                    condition_commitment=inputs.pin, purpose="preconditions_to_observed_outcome")
    journal = EventJournal(a.output/"experience.sqlite3", contract, create=True, payload_bound=1 << 20)
    def append(record):
        journal.append(dict(kind="memory", record=record))
    links = {}
    for key in obs.entries:
        condition, records = condition_attachment(session, key)
        for record in records:
            append(record)
        ep = obs.episode(key, [condition])
        append(memory.apply(dict(op="intake", episode=ep)))
        checked = memory.apply(dict(op="check_source", episode=ep["id"]))
        append(checked)
        append(memory.apply(dict(op="promote", episode=ep["id"], receipt=checked["event"]["result"]["id"])))
        links[key] = dict(condition=condition, occurrence=ep["id"])
    midpoint = memory.snapshot()
    first, second = list(links)
    append(memory.apply(dict(op="invalidate", episode=links[first]["condition"],
                             reason="controlled withdrawal of this runtime attestation")))
    check(memory.state[links[first]["occurrence"]]["grounded_scope"] is None
          and memory.state[links[second]["occurrence"]]["grounded_scope"] == "observed_process_in_pinned_runtime",
          "CONDITION_DEPENDENT_INVALIDATION")
    snapshot = memory.snapshot()
    refuse("withdrawn_precondition", "TOOL_PRECONDITION_UNAVAILABLE",
           lambda: condition_attachment(session, first))
    check(memory.snapshot() == snapshot, "CONDITION_REFUSAL_MUTATION")
    saved = dict(contract=contract, journal_head=journal.head, memory=snapshot)
    journal.close()
    put(a.output/"state.json", saved)
    replay = restore(a.output, source, obs, saved)
    check(inventory(a.prepared) == before and inventory(a.old) == old_bytes, "CONDITION_OLD_ARTIFACT_DRIFT")
    phases["durable_memory_and_correction"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema="process-conditions-fsv-v1", complete=True, observation_pin=obs_pin,
        commitment=inputs.pin, readings=readings, controls=controls, links=links,
        midpoint=midpoint, final=snapshot, replay=replay, prepared_before=before, old_before=old_bytes,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        learned_skill=False, model_consumption_established=False, scientific_admission=False,
        evidence_status="author_checked", next_dependency="native_condition_input_to_predictor")
    check(report["phase_coverage"] >= .9, "CONDITION_PHASE_COVERAGE")
    put(a.output/"report.json", report)
    print(json.dumps(dict(complete=True, observations=len(readings), controls=len(controls),
                         wall_seconds=wall, report=sha((a.output/"report.json").read_bytes()))))


if __name__ == "__main__":
    main()
