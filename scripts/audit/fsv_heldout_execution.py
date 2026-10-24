"""Two programs, separately sealed executions, and native held-out learning feedback."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

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
from model.condition_catalog import NativeConditions
from model.tool_growth import GrowingToolLoop, parameters
from model.tool_kernel import ToolKernel
from model.tool_learning import ToolLearningData
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=("forecast", "enroll-train", "finish", "cold"), required=True)
    a = p.parse_args()
    started = time.perf_counter()
    c = json.loads(a.config.read_bytes())
    root = Path(c["output"])
    gpu = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "heldout-process-"+a.stage)
    configure()
    source = ToolSource(Path(c["raw"]))
    inputs = ToolInputs(source, Path(c["population"]), Path(c["native"]),
                        population_pin=c["population_pin"], native_pin=c["native_pin"])
    providers = [ConditionNative(ConditionInputs(Path(v["conditions"]), v["condition_pin"]), inputs,
        Path(v["condition_population"]), Path(v["condition_native"]),
        population_pin=v["condition_population_pin"], native_pin=v["condition_native_pin"])
        for v in c["providers"]]
    conditions = NativeConditions(providers)
    parent_root = Path(c["parent"])
    check(sha((parent_root/"run.json").read_bytes()) == c["parent_pin"], "HELDOUT_PARENT")
    parent = json.loads((parent_root/"run.json").read_bytes())
    routing = ToolKernel(inputs, Path(c["kernel"]), (Path(c["kernel"])/"COMMITTED").read_text())
    contexts = {v["context"]:v["task"] for v in c["providers"]}
    check(len(contexts) == 2 and c["providers"][0]["context"] == "train:1"
          and c["providers"][1]["context"] == "withheld:0", "HELDOUT_CONTEXTS")
    def open_session(path=None, **kw):
        return GrowingToolLoop(path or root/"experience.sqlite3", source, inputs, parent_root/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts=contexts, device="cuda", condition_inputs=conditions, routing=routing,
            growth_root=root/"growth", candidate_root=root/"candidates", **kw)
    phases = dict(source_native=time.perf_counter()-started)
    tick = time.perf_counter()
    controls, session = [], None
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), "HELDOUT_CONTROL_CODE_"+name+":"+str(error))
        else:
            raise AssertionError("HELDOUT_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "HELDOUT_CONTROL_STATE_"+name)
        controls.append(dict(name=name, code=code))
    def snapshot(name):
        saved = session.snapshot()
        write(root/name, saved)
        return saved
    if a.stage == "forecast":
        check(not any(Path(v["observations"]).exists() for v in c["providers"]), "HELDOUT_ALREADY_OBSERVED")
        root.mkdir(parents=True, exist_ok=False)
        (root/"gpu-before.txt").write_text(gpu)
        write(root/"config.json", c)
        session = open_session(create=True)
        refuse("duplicate_provider", "CONDITION_CATALOG_COLLISION", lambda: NativeConditions(providers*2))
        for sequence in range(13):
            for v, provider in zip(c["providers"], providers):
                context, task = v["context"], v["task"]
                if sequence == 11:
                    other = providers[1] if context == "train:1" else providers[0]
                    refuse("wrong_program_"+context, "TOOL_CONDITION_SOURCE",
                        lambda: session.bind_execution_conditions(context, next(iter(other.conditions.inputs))))
                    session.bind_execution_conditions(context, provider.conditions.commitment["routing"][v["condition"]])
                session.advance(context, receipt=source.receipt(task, 3+2*(sequence//5))
                                if sequence % 5 == 3 else None)
        pin = forecast_commitment.publish(session, root/"forecast-train", [session.pending["train:1"]])
        saved = snapshot("pending-1.json")
        session.close()
        session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
        check(session.snapshot() == saved and session.observations is None, "HELDOUT_PENDING_REPLAY")
        check(not any(Path(v["observations"]).exists() for v in c["providers"]), "HELDOUT_ALREADY_OBSERVED")
        result = dict(pin=pin, forecasts=dict(session.pending), actual_outcomes_available=False)
    else:
        name = {"enroll-train":"pending-1.json", "finish":"pending-2.json", "cold":"snapshot.json"}[a.stage]
        saved = json.loads((root/name).read_bytes())
        session = open_session(expected_head=saved["head"], expected_contract=saved["contract"])
        check(session.snapshot() == saved, "HELDOUT_REOPEN")
        if a.stage == "cold":
            direct = readback(root/"experience.sqlite3", inputs, saved,
                             [Path(v["observations"]) for v in c["providers"]], session.execution_inputs)
            report = json.loads((root/"finish.json").read_bytes())
            check(direct == report["result"]["readback"], "HELDOUT_COLD_READBACK")
            result = dict(readback=direct)
        else:
            index = 0 if a.stage == "enroll-train" else 1
            v = c["providers"][index]
            spec = json.loads(Path(v["enrollment"]).read_bytes())
            parameters_before = parameters(session.model)
            event = session.enroll_execution(spec)
            refuse("duplicate_enrollment", "EXECUTION_ALREADY_ENROLLED", lambda: session.enroll_execution(spec))
            if a.stage == "enroll-train":
                check(not Path(c["providers"][1]["observations"]).exists(), "HELDOUT_ALREADY_OBSERVED")
                key = session.pending["withheld:0"]
                pin = forecast_commitment.publish(session, root/"forecast-withheld", [key])
                snapshot("pending-2.json")
                result = dict(enrollment=event, pin=pin, heldout_outcome_available=False)
            else:
                train_key, eval_key = session.pending["train:1"], session.pending["withheld:0"]
                obs = {k:ob for e in session.execution_enrollments for k,ob in e["links"].items()}
                check(len(obs) == 2 and len(session.observations.condition_inputs().inputs) == 4,
                      "HELDOUT_PROGRAM_CATALOG")
                refuse("swapped_execution", "TOOL_EXECUTION_FORECAST", lambda: session.advance("withheld:0",
                    receipt=source.receipt(contexts["withheld:0"],7), execution=obs[train_key]))
                for phase in (3,4):
                    for context, key in (("train:1",train_key),("withheld:0",eval_key)):
                        session.advance(context, receipt=source.receipt(contexts[context],7) if phase == 3 else None,
                                        execution=obs[key] if phase == 3 else None)
                refuse("heldout_training_leak", "TOOL_LEARNING_SPLIT",
                    lambda: ToolLearningData(session, train=[eval_key], evaluate=[train_key]))
                data = ToolLearningData(session, train=[train_key], evaluate=[eval_key])
                actual_targets = {}
                for key in (train_key, eval_key):
                    e = data.examples[key]
                    provider = session.execution_inputs.for_observation(obs[key])
                    check(e["target_inputs"] is provider and data.binding["evidence"][key]["execution"] ==
                          session.executions[key], "HELDOUT_ACTUAL_TARGET")
                    lanes = {}
                    for s in session.model.specs:
                        slot = e["target"][s.name]
                        values = slot.values[0, ~slot.padding[0]].cpu().numpy()
                        native, ptr = provider.arrays[s.name]
                        unit = provider.lookup[obs[key]]
                        check(np.array_equal(values,native[ptr[unit]:ptr[unit+1]]), "HELDOUT_TARGET_WINDOWS")
                        lanes[s.name] = dict(shape=list(values.shape), sha256=sha(values.tobytes()))
                    actual_targets[key] = dict(binding=provider.checked(obs[key]), slots=lanes)
                training = learn(session, root, 1, [train_key], [eval_key], refuse)
                write(root/"learning.json", training)
                check(parameters(session.model) == parameters_before, "HELDOUT_ACTIVE_CHANGED")
                saved = snapshot("snapshot.json")
                with sqlite3.connect(root/"withdrawal.sqlite3") as db:
                    session.journal.db.backup(db)
                original = session
                session = open_session(root/"withdrawal.sqlite3", expected_head=saved["head"],
                                       expected_contract=saved["contract"])
                episode = session.forecasts[eval_key]["record"]["request"]["execution_conditions"]["episode"]
                withdrawal = session.invalidate(episode, reason="FSV held-out precondition withdrawal")
                session.eligible(train_key)
                refuse("withdrawn_evaluation", "TOOL_LEARNING_INELIGIBLE",
                       lambda: ToolLearningData(session, train=[train_key], evaluate=[eval_key]))
                branch = snapshot("withdrawal-snapshot.json")
                session.close()
                session = open_session(root/"withdrawal.sqlite3", expected_head=branch["head"],
                                       expected_contract=branch["contract"])
                check(session.snapshot() == branch, "HELDOUT_WITHDRAWAL_REPLAY")
                session.close()
                session = original
                check(session.snapshot() == saved, "HELDOUT_BRANCH_ISOLATION")
                direct = readback(root/"experience.sqlite3", inputs, saved,
                    [Path(v["observations"]) for v in c["providers"]], session.execution_inputs)
                result = dict(enrollment=event, actual_targets=actual_targets, training=training,
                              withdrawal=withdrawal, readback=direct, useful_capability_established=False)
    session.close()
    phases[a.stage] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    write(root/(a.stage+".json"), dict(complete=True, result=result, controls=controls,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall))
    print(json.dumps(dict(complete=True, stage=a.stage, controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    run_cli(main)
