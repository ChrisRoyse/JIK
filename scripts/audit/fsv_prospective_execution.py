"""Seal first, execute externally, then enroll native feedback and learn."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.fsv_process_feedback import learn, write
from audit.tool_execution_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims import forecast_commitment
from claims.process_conditions import ConditionInputs
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.tool_growth import GrowingToolLoop, parameters
from model.tool_kernel import ToolKernel
from model.tool_learning import ToolLearningData
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=("forecast", "finish", "cold"), required=True)
    a = p.parse_args()
    begin = time.perf_counter()
    c = json.loads(a.config.read_bytes())
    root = Path(c["output"])
    gpu = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "prospective-process-"+a.stage)
    configure()
    source = ToolSource(Path(c["raw"]))
    inputs = ToolInputs(source, Path(c["population"]), Path(c["native"]),
                        population_pin=c["population_pin"], native_pin=c["native_pin"])
    conditions = ConditionNative(ConditionInputs(Path(c["conditions"]), c["condition_pin"]), inputs,
        Path(c["condition_population"]), Path(c["condition_native"]),
        population_pin=c["condition_population_pin"], native_pin=c["condition_native_pin"])
    parent_root = Path(c["parent"])
    check(sha((parent_root/"run.json").read_bytes()) == c["parent_pin"], "PROSPECTIVE_PARENT")
    parent = json.loads((parent_root/"run.json").read_bytes())
    routing = ToolKernel(inputs, Path(c["kernel"]), (Path(c["kernel"])/"COMMITTED").read_text())
    withheld = next(e["task"] for e in source.episodes if e["partition"] == "withheld" and e["status"] == "ready")
    contexts = {"train:1":"tonioo__sievelib-79", "withheld:0":withheld}
    def open_session(path=None, **kw):
        return GrowingToolLoop(path or root/"experience.sqlite3", source, inputs, parent_root/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts=contexts, device="cuda", condition_inputs=conditions, routing=routing,
            growth_root=root/"growth", candidate_root=root/"candidates", **kw)
    phases = dict(source_native=time.perf_counter()-begin)
    tick = time.perf_counter()
    controls = []
    session = None
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), "PROSPECTIVE_CONTROL_CODE_"+name+":"+str(error))
        else:
            raise AssertionError("PROSPECTIVE_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "PROSPECTIVE_CONTROL_STATE_"+name)
        controls.append(dict(name=name, code=code))
    if a.stage == "forecast":
        check(not Path(c["observations"]).exists(), "PROSPECTIVE_OUTCOME_ALREADY_PRESENT")
        root.mkdir(parents=True, exist_ok=False)
        (root/"gpu-before.txt").write_text(gpu)
        write(root/"config.json", c)
        session = open_session(create=True)
        for sequence in range(13):
            phase, exchange = sequence % 5, sequence // 5
            for context, task in contexts.items():
                if context == "train:1" and sequence == 11:
                    session.bind_execution_conditions(context, conditions.conditions.commitment["routing"]["without_dependency"])
                session.advance(context, receipt=source.receipt(task, 3+2*exchange) if phase == 3 else None)
        key = session.pending["train:1"]
        refuse("unconditioned_forecast", "FORECAST_SEAL_PENDING",
               lambda: forecast_commitment.publish(session, root/"invalid-unconditioned", [session.pending["withheld:0"]]))
        pin = forecast_commitment.publish(session, root/"forecast", [key])
        refuse("duplicate_seal", "FORECAST_SEAL_EXISTS", lambda: forecast_commitment.publish(session, root/"forecast", [key]))
        refuse("wrong_seal_pin", "FORECAST_SEAL_PIN", lambda: forecast_commitment.read(root/"forecast", "0"*64))
        corrupt = root/"corrupt-forecast"
        shutil.copytree(root/"forecast", corrupt)
        body = json.loads((corrupt/"forecast.json").read_bytes())
        body["forecasts"][key]["region"] += "_altered"
        raw = canonical(body)
        (corrupt/"forecast.json").write_bytes(raw)
        (corrupt/"COMMITTED").write_text(sha(raw))
        refuse("altered_prediction", "FORECAST_SEAL_RECORD", lambda: forecast_commitment.read(corrupt, sha(raw)))
        saved = session.snapshot()
        session.close()
        session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
        check(session.snapshot() == saved and session.observations is None and session.execution_inputs is None,
              "PROSPECTIVE_PENDING_REPLAY")
        session.close()
        check(not Path(c["observations"]).exists(), "PROSPECTIVE_OUTCOME_ALREADY_PRESENT")
        write(root/"pending.json", saved)
        phases["forecast_seal_controls_replay"] = time.perf_counter()-tick
        wall = time.perf_counter()-begin
        write(root/"forecast-stage.json", dict(complete=True, pin=pin, forecast=key, controls=controls,
            outcome_available=False, phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
        print(json.dumps(dict(complete=True, stage=a.stage, pin=pin, controls=len(controls))))
        return
    saved = json.loads((root/("snapshot.json" if a.stage == "cold" else "pending.json")).read_bytes())
    session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
    check(session.snapshot() == saved, "PROSPECTIVE_REOPEN")
    if a.stage == "cold":
        direct = readback(root/"experience.sqlite3", inputs, saved, Path(c["observations"]), session.execution_inputs)
        report = json.loads((root/"report.json").read_bytes())
        check(direct == report["readback"], "PROSPECTIVE_COLD_READBACK")
        session.close()
        phases["cold_replay_readback"] = time.perf_counter()-tick
        wall = time.perf_counter()-begin
        write(root/"cold.json", dict(complete=True, readback=direct, phases=phases, wall_seconds=wall,
                                    phase_coverage=sum(phases.values())/wall))
        print(json.dumps(dict(complete=True, stage=a.stage, wall_seconds=wall)))
        return
    spec = json.loads(Path(c["enrollment"]).read_bytes())
    initial = parameters(session.model)
    bad = dict(spec, forecast_pin="0"*64)
    refuse("unsealed_enrollment", "FORECAST_SEAL_PIN", lambda: session.enroll_execution(bad))
    event = session.enroll_execution(spec)
    refuse("duplicate_enrollment", "EXECUTION_ALREADY_ENROLLED", lambda: session.enroll_execution(spec))
    after = session.snapshot()
    session.close()
    session = open_session(expected_head=after["head"], expected_contract=after["contract"])
    check(session.snapshot() == after, "PROSPECTIVE_ENROLLMENT_REPLAY")
    key, observation = next(iter(event["links"].items()))
    seal = forecast_commitment.read(Path(spec["forecast_root"]), spec["forecast_pin"])
    late = dict(seal, sealed_at="2099-01-01T00:00:00+00:00")
    entry = session.observations.entries[observation]
    actual = session.observations.read(observation)
    refuse("late_commitment", "FORECAST_NOT_PREEXECUTION",
        lambda: forecast_commitment.match(late, session.forecasts[key]["record"], entry, actual))
    wrong = dict(entry, condition="0"*64)
    refuse("wrong_condition", "FORECAST_SEAL_OUTCOME",
        lambda: forecast_commitment.match(seal, session.forecasts[key]["record"], wrong, actual))
    withheld_key = session.pending["withheld:0"]
    for phase in (3,4):
        for context, task in contexts.items():
            session.advance(context, receipt=source.receipt(task,7) if phase == 3 else None,
                            execution=observation if context == "train:1" and phase == 3 else None)
    check(session.execution_evidence(key)["native_feedback_available"], "PROSPECTIVE_NATIVE_FEEDBACK")
    phases["enrollment_echo_correction"] = time.perf_counter()-tick
    tick = time.perf_counter()
    training = learn(session, root, 1, [key], [withheld_key], refuse)
    check(parameters(session.model) == initial, "PROSPECTIVE_ACTIVE_CHANGED")
    snapshot = session.snapshot()
    with sqlite3.connect(root/"withdrawal.sqlite3") as db:
        session.journal.db.backup(db)
    original = session
    session = open_session(root/"withdrawal.sqlite3", expected_head=snapshot["head"], expected_contract=snapshot["contract"])
    condition_episode = session.forecasts[key]["record"]["request"]["execution_conditions"]["episode"]
    withdrawal = session.invalidate(condition_episode, reason="FSV precondition withdrawal")
    refuse("withdrawn_learning", "TOOL_LEARNING_INELIGIBLE",
           lambda: ToolLearningData(session, train=[key], evaluate=[withheld_key]))
    branch = session.snapshot()
    session.close()
    session = open_session(root/"withdrawal.sqlite3", expected_head=branch["head"], expected_contract=branch["contract"])
    check(session.snapshot() == branch, "PROSPECTIVE_WITHDRAWAL_REPLAY")
    session.close()
    session = original
    check(session.snapshot() == snapshot, "PROSPECTIVE_BRANCH_ISOLATION")
    direct = readback(root/"experience.sqlite3", inputs, snapshot, Path(c["observations"]), session.execution_inputs)
    session.close()
    write(root/"snapshot.json", snapshot)
    write(root/"withdrawal-snapshot.json", branch)
    write(root/"training.json", [training])
    phases["learning_withdrawal_readback"] = time.perf_counter()-tick
    wall = time.perf_counter()-begin
    write(root/"report.json", dict(complete=True, enrollment=event, withdrawal=withdrawal, controls=controls,
        readback=direct, phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        new_occurrences_after_forecast=1, unseen_task_skill_established=False, useful_capability=False,
        scientific_admission=False, acceptance="author-checked only"))
    print(json.dumps(dict(complete=True, stage=a.stage, controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
