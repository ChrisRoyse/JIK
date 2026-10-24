"""Continue the first prospective cycle in place, preserving its native evidence."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import learn, write
from audit.tool_execution_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims import forecast_commitment
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.process_batches import append
from model.tool_growth import GrowingToolLoop, parameters
from model.tool_kernel import ToolKernel
from model.tool_learning import ToolLearningData
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=("forecast", "finish", "cold", "growth"), required=True)
    a = p.parse_args()
    begin = time.perf_counter()
    c = json.loads(a.config.read_bytes())
    root, stage = Path(c["output"]), Path(c["output"])/"cycle-2"
    gpu = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "successive-execution-"+a.stage)
    configure()
    source = ToolSource(Path(c["raw"]))
    inputs = ToolInputs(source, Path(c["population"]), Path(c["native"]),
        population_pin=c["population_pin"], native_pin=c["native_pin"])
    conditions = ConditionNative(ConditionInputs(Path(c["conditions"]), c["condition_pin"]), inputs,
        Path(c["condition_population"]), Path(c["condition_native"]),
        population_pin=c["condition_population_pin"], native_pin=c["condition_native_pin"])
    parent_root = Path(c["parent"])
    check(sha((parent_root/"run.json").read_bytes()) == c["parent_pin"], "SUCCESSIVE_PARENT")
    parent = json.loads((parent_root/"run.json").read_bytes())
    routing = ToolKernel(inputs, Path(c["kernel"]), (Path(c["kernel"])/"COMMITTED").read_text())
    first = json.loads((root/"snapshot.json").read_bytes())
    contexts = first["contexts"]
    def open_session(path=None, **kw):
        return GrowingToolLoop(path or root/"experience.sqlite3", source, inputs, parent_root/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts=contexts, device="cuda", condition_inputs=conditions, routing=routing,
            growth_root=root/"growth", candidate_root=root/"candidates", **kw)
    saved = first if a.stage == "forecast" else json.loads(
        (stage/("snapshot.json" if a.stage in ("cold", "growth") else "pending.json")).read_bytes())
    session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
    check(session.snapshot() == saved, "SUCCESSIVE_REOPEN")
    phases = dict(source_native_reopen=time.perf_counter()-begin)
    tick, controls = time.perf_counter(), []
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), "SUCCESSIVE_CONTROL_CODE_"+name+":"+str(error))
        else:
            raise AssertionError("SUCCESSIVE_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "SUCCESSIVE_CONTROL_STATE_"+name)
        controls.append(dict(name=name, code=code))
    first_event = first["execution_enrollment"]
    first_key, first_observation = next(iter(first_event["links"].items()))
    first_binding = copy.deepcopy(session.execution_inputs.checked(first_observation))
    first_execution = copy.deepcopy(session.executions[first_key])
    first_episode = copy.deepcopy(session.memory.episodes[first_execution["episode"]])
    initial = parameters(session.model)
    if a.stage == "forecast":
        check(not Path(c["observations"]).exists(), "SUCCESSIVE_OUTCOME_ALREADY_PRESENT")
        stage.mkdir(exist_ok=False)
        (stage/"gpu-before.txt").write_text(gpu)
        growth = None
        if c.get("grow_between", False):
            before = session.snapshot()
            growth = session.grow("between-cycles", [first_key])
            check(parameters(session.model) == initial, "SUCCESSIVE_GROWTH_PARAMETER_DRIFT")
            write(stage/"between-cycle-growth.json", dict(before=before, action=growth,
                after=session.snapshot(), recall={n:v["recall"] for n,v in session.routing.calibration.items()}))
        for sequence in range(15, 23):
            for context, task in contexts.items():
                check(session.states[context].sequence == sequence, "SUCCESSIVE_SEQUENCE")
                if context == "train:1" and sequence == 21:
                    session.bind_execution_conditions(context, conditions.conditions.commitment["routing"]["with_dependency"])
                session.advance(context, receipt=source.receipt(task, 3+2*(sequence//5)) if sequence % 5 == 3 else None)
        key = session.pending["train:1"]
        history = session.forecasts[key]["record"]["region_context"]["definition"]["observed_echoes"]
        check(history["7"]["binding"] == first_binding and history["7"]["episode"] == first_episode["id"],
              "SUCCESSIVE_HISTORY_LOST")
        refuse("already_observed_seal", "FORECAST_SEAL_PRIOR_OUTCOME",
               lambda: forecast_commitment.publish(session, stage/"invalid-prior", [first_key]))
        pin = forecast_commitment.publish(session, stage/"forecast", [key])
        saved = session.snapshot()
        session.close()
        session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
        check(session.snapshot() == saved and len(session.execution_enrollments) == 1,
              "SUCCESSIVE_PENDING_REPLAY")
        check(not Path(c["observations"]).exists(), "SUCCESSIVE_OUTCOME_ALREADY_PRESENT")
        session.close()
        write(stage/"pending.json", saved)
        phases["forecast_seal_replay"] = time.perf_counter()-tick
        wall = time.perf_counter()-begin
        write(stage/"forecast-stage.json", dict(complete=True, pin=pin, forecast=key, controls=controls,
            first_binding=first_binding, first_episode=first_episode, first_execution=first_execution,
            between_cycle_growth=growth,
            phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
        print(json.dumps(dict(complete=True, stage=a.stage, pin=pin)))
        return
    roots = [Path(first_event["spec"]["observations"]), Path(c["observations"])]
    if a.stage == "growth":
        from model import tool_observed_inputs
        path = stage/"growth-branch.sqlite3"
        with sqlite3.connect(path) as db:
            session.journal.db.backup(db)
        original = session
        session = open_session(path, expected_head=saved["head"], expected_contract=saved["contract"])
        train, evaluate = session.decisions[-1]["train"], session.decisions[-1]["evaluate"]
        boundaries = []
        existing = {g["keys"][0] for g in session.growths}
        for number, key in enumerate((k for k in train if k not in existing), 1):
            before = session.snapshot()
            refuse("withheld_growth_"+str(number), "TOOL_GROWTH_SPLIT",
                   lambda: session.grow("invalid-withheld", [evaluate[-1]]))
            growth = session.grow("catalogue-"+str(number), [key])
            unit = session.inputs.world_units[-1]
            pair = session.routing.pairs[str(unit)]
            response = pair["response"]
            check(response["native"] == session.executions[key]["native"]
                  and parameters(session.model) == initial, "SUCCESSIVE_GROWTH_SOURCE")
            for item in session.inputs.observed_regions:
                old_pair = session.routing.pairs[item]
                tool_observed_inputs.validate(session.inputs, item, old_pair["response"], old_pair["action"])
            bad = copy.deepcopy(response)
            bad["native"]["observations"] = "0"*64
            changed = copy.copy(session.inputs)
            changed.observed_regions = {**session.inputs.observed_regions, str(unit):bad}
            refuse("growth_binding_substitution_"+str(number), "TOOL_KERNEL_OBSERVED_BINDING",
                lambda: tool_observed_inputs.validate(changed, str(unit), bad, pair["action"]))
            boundaries.append(dict(before=before, action=growth, after=session.snapshot(),
                recall={name:v["recall"] for name,v in session.routing.calibration.items()}))
        branch = session.snapshot()
        session.close()
        session = open_session(path, expected_head=branch["head"], expected_contract=branch["contract"])
        check(session.snapshot() == branch, "SUCCESSIVE_GROWTH_REPLAY")
        direct = readback(path, inputs, branch, roots, session.execution_inputs)
        session.close()
        session = original
        check(session.snapshot() == saved, "SUCCESSIVE_GROWTH_BRANCH_ISOLATION")
        session.close()
        phases["growth_readback_replay"] = time.perf_counter()-tick
        wall = time.perf_counter()-begin
        write(stage/"growth-snapshot.json", branch)
        write(stage/"growth-report.json", dict(complete=True, boundaries=boundaries, readback=direct,
            controls=controls, phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
            main_unchanged=True, scientific_admission=False, acceptance="author-checked only"))
        print(json.dumps(dict(complete=True, stage=a.stage, controls=len(controls))))
        return
    if a.stage == "cold":
        direct = readback(root/"experience.sqlite3", inputs, saved, roots, session.execution_inputs)
        check(direct == json.loads((stage/"report.json").read_bytes())["readback"], "SUCCESSIVE_COLD_READBACK")
        session.close()
        phases["cold_readback"] = time.perf_counter()-tick
        wall = time.perf_counter()-begin
        write(stage/"cold.json", dict(complete=True, readback=direct, phases=phases,
            wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
        print(json.dumps(dict(complete=True, stage=a.stage)))
        return
    sealed = json.loads((stage/"forecast-stage.json").read_bytes())
    check(first_binding == sealed["first_binding"] and first_execution == sealed["first_execution"]
          and first_episode == sealed["first_episode"], "SUCCESSIVE_PRIOR_EVIDENCE_CHANGED")
    old_provider = session.execution_inputs
    refuse("batch_collision", "PROCESS_BATCH_COLLISION", lambda: append(old_provider, old_provider))
    spec = json.loads(Path(c["enrollment"]).read_bytes())
    with sqlite3.connect(stage/"stale-enrollment.sqlite3") as db:
        session.journal.db.backup(db)
    original = session
    session = open_session(stage/"stale-enrollment.sqlite3", expected_head=saved["head"], expected_contract=saved["contract"])
    session.advance("withheld:0", receipt=source.receipt(contexts["withheld:0"], 11))
    refuse("stale_seal_head", "EXECUTION_ENROLLMENT_HEAD", lambda: session.enroll_execution(spec))
    stale = session.snapshot()
    session.close()
    session = original
    check(session.snapshot() == saved, "SUCCESSIVE_STALE_BRANCH_ISOLATION")
    write(stage/"stale-enrollment-snapshot.json", stale)
    event = session.enroll_execution(spec)
    check(session.execution_inputs.for_observation(first_observation) is old_provider
          and session.execution_inputs.checked(first_observation) == first_binding
          and session.executions[first_key] == first_execution
          and session.memory.episodes[first_episode["id"]] == first_episode, "SUCCESSIVE_APPEND_CHANGED_PRIOR")
    corrupt = copy.copy(old_provider)
    corrupt.native = stage/"corrupt-earlier-native"
    shutil.copytree(old_provider.native, corrupt.native)
    target = corrupt.native/old_provider.specs[0].name/"slots.npz"
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    refuse("corrupt_earlier_batch", "PROCESS_NATIVE_BYTES", lambda: corrupt.checked(first_observation))
    key, observation = next(iter(event["links"].items()))
    refuse("duplicate_enrollment", "EXECUTION_ALREADY_ENROLLED", lambda: session.enroll_execution(spec))
    refuse("enrolled_pending_seal", "FORECAST_SEAL_PRIOR_OUTCOME",
           lambda: forecast_commitment.publish(session, stage/"invalid-enrolled", [key]))
    refuse("wrong_batch_echo", "TOOL_EXECUTION_FORECAST", lambda: session.advance(
        "train:1", receipt=source.receipt(contexts["train:1"], 11), execution=first_observation))
    enrolled = session.snapshot()
    session.close()
    session = open_session(expected_head=enrolled["head"], expected_contract=enrolled["contract"])
    check(session.snapshot() == enrolled, "SUCCESSIVE_ENROLLMENT_REPLAY")
    withheld_key = session.pending["withheld:0"]
    for phase in (3, 4):
        for context, task in contexts.items():
            session.advance(context, receipt=source.receipt(task, 11) if phase == 3 else None,
                            execution=observation if context == "train:1" and phase == 3 else None)
    phases["append_echo_correction_replay"] = time.perf_counter()-tick
    tick = time.perf_counter()
    previous = session.decisions[-1]
    train, evaluate = previous["train"]+[key], previous["evaluate"]+[withheld_key]
    training = learn(session, root, 2, train, evaluate, refuse)
    check(training["continuity"] is not None and training["optimizer_steps"] == 16,
          "SUCCESSIVE_OPTIMIZER_CONTINUITY")
    check(parameters(session.model) == initial, "SUCCESSIVE_ACTIVE_CHANGED")
    snapshot = session.snapshot()
    with sqlite3.connect(stage/"withdrawal.sqlite3") as db:
        session.journal.db.backup(db)
    original = session
    session = open_session(stage/"withdrawal.sqlite3", expected_head=snapshot["head"], expected_contract=snapshot["contract"])
    condition_episode = session.forecasts[first_key]["record"]["request"]["execution_conditions"]["episode"]
    withdrawal = session.invalidate(condition_episode, reason="FSV earlier-batch precondition withdrawal")
    check(all(session.forecasts[k]["invalidated"] for k in train), "SUCCESSIVE_WITHDRAWAL_DEPENDENCY")
    refuse("withdrawn_continuation", "TOOL_EXECUTION_UNAVAILABLE" if session.growths else "TOOL_LEARNING_INELIGIBLE",
           lambda: ToolLearningData(session, train=train, evaluate=evaluate, continuation="increment-2"))
    branch = session.snapshot()
    session.close()
    repair = None
    if c.get("grow_between", False):
        session = original
        refuse("withdrawn_bank_cold", "TOOL_EXECUTION_UNAVAILABLE", lambda: open_session(
            stage/"withdrawal.sqlite3", expected_head=branch["head"], expected_contract=branch["contract"]))
        session = open_session(stage/"withdrawal.sqlite3", expected_head=branch["head"],
                               expected_contract=branch["contract"], recovery=True)
        check(session.snapshot() == branch, "SUCCESSIVE_WITHDRAWAL_RECOVERY")
        write(stage/"withdrawn-before-repair.json", branch)
        repair = session.repair_bank("withdrawal-retire")
        check(parameters(session.model) == initial and len(repair["removed"]) == 1,
              "SUCCESSIVE_WITHDRAWAL_RETIRED")
        refuse("withdrawn_context_after_repair", "TOOL_CONTEXT_REVIEW", lambda: session.advance("train:1"))
        branch = session.snapshot()
        session.close()
    session = open_session(stage/"withdrawal.sqlite3", expected_head=branch["head"], expected_contract=branch["contract"])
    check(session.snapshot() == branch, "SUCCESSIVE_WITHDRAWAL_REPLAY")
    session.close()
    session = original
    check(session.snapshot() == snapshot, "SUCCESSIVE_BRANCH_ISOLATION")
    direct = readback(root/"experience.sqlite3", inputs, snapshot, roots, session.execution_inputs)
    session.close()
    write(stage/"snapshot.json", snapshot)
    write(stage/"withdrawal-snapshot.json", branch)
    write(stage/"training.json", training)
    phases["retained_learning_withdrawal_readback"] = time.perf_counter()-tick
    wall = time.perf_counter()-begin
    write(stage/"report.json", dict(complete=True, enrollment=event, withdrawal=withdrawal,
        repair=repair,
        controls=controls, readback=direct, phases=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, retained_binding=first_binding,
        batches=2, new_occurrences_after_forecast=2, unseen_task_skill_established=False,
        useful_capability=False, scientific_admission=False, acceptance="author-checked only"))
    print(json.dumps(dict(complete=True, stage=a.stage, controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
