"""Real constellation forecasts, new execution facts, cold replay and withdrawal."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_execution_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_observation import ProcessObservations
from claims.tool_evidence import ToolEvidence
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.tool_growth import GrowingToolLoop, parameters, state_pin
from model.tool_kernel import ToolKernel
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "kernel", "observations", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin", "observation-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cold", action="store_true")
    args = p.parse_args()
    start = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "tool-execution-integration")
    configure()
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    observations = ProcessObservations(args.observations, args.observation_pin, source)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "EXEC_FSV_PARENT")
    parent = json.loads((args.parent/"run.json").read_bytes())
    routing = ToolKernel(inputs, args.kernel, (args.kernel/"COMMITTED").read_text())
    phases = dict(source_native=time.perf_counter()-start)
    task, context = "tonioo__sievelib-79", "train:1"
    def open_session(**kw):
        return GrowingToolLoop(args.output/"experience.sqlite3", source, inputs, args.parent/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts={context:task}, device="cuda", observations=observations,
            routing=routing, growth_root=args.output/"growth", **kw)
    tick = time.perf_counter()
    if args.cold:
        original = json.loads((args.output/"report.json").read_bytes())
        snapshot = json.loads((args.output/"snapshot.json").read_bytes())
        session = open_session(expected_head=snapshot["head"], expected_contract=snapshot["contract"])
        check(session.snapshot() == snapshot, "EXEC_FSV_COLD")
        direct = readback(args.output/"experience.sqlite3", inputs, snapshot, args.observations)
        check(direct == original["readback"], "EXEC_FSV_COLD_READBACK")
        session.close()
        phases["cold_replay_readback"] = time.perf_counter()-tick
        write(args.output/"cold.json", dict(complete=True, phases=phases,
              wall_seconds=time.perf_counter()-start, readback=direct))
        print(json.dumps(dict(complete=True, cold=True, observations=2)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"gpu-before.txt").write_text(resident)
    session = open_session(create=True)
    initial_parameters = parameters(session.model)
    controls, boundaries = [], []
    def refuse(name, code, action):
        before = digest(session.snapshot())
        try:
            action()
        except (ValueError, SenseError) as error:
            check(code in str(error), "EXEC_FSV_CONTROL_CODE_"+name)
        else:
            raise AssertionError("EXEC_FSV_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "EXEC_FSV_CONTROL_STATE_"+name)
        controls.append(dict(name=name, code=code))
    refuse("forged_registry", "PROCESS_MANIFEST_PIN",
           lambda: ProcessObservations(args.observations, "0"*64, source))
    by_turn = {v["target"]["turn"]: k for k,v in observations.entries.items()}
    for exchange in range(5):
        for phase in range(5):
            turn = 3+exchange*2
            session.advance(context, receipt=source.receipt(task, turn) if phase == 3 else None)
            if exchange in (2, 4) and phase in (2, 3):
                key = next(reversed(session.forecasts))
                if phase == 2:
                    refuse("premature_"+str(turn), "TOOL_LEARNING_INELIGIBLE",
                           lambda: session.observe_execution(key, by_turn[turn]))
                else:
                    wrong = by_turn[11 if turn == 7 else 7]
                    refuse("wrong_forecast_"+str(turn), "TOOL_EXECUTION_FORECAST",
                           lambda: session.observe_execution(key, wrong))
                    before = session.snapshot()
                    hidden = state_pin(session.states)
                    body = session.observe_execution(key, by_turn[turn])
                    checked = session.execution_evidence(key)
                    check(checked["comparison"]["concordant"] and
                          checked["native_feedback_available"] is False
                          and state_pin(session.states) == hidden and parameters(session.model) == initial_parameters,
                          "EXEC_FSV_ATTACHMENT_EFFECT")
                    boundaries.append(dict(before=digest(before), event=body, after=digest(session.snapshot()),
                                           unchanged_intrinsic_state=hidden))
                    refuse("duplicate_"+str(turn), "TOOL_DUPLICATE_EXECUTION",
                           lambda: session.observe_execution(key, by_turn[turn]))
    phases["experience"] = time.perf_counter()-tick
    tick = time.perf_counter()
    before = session.snapshot()
    check(len(before["memory"]["grounded_world_claims"]) == 2, "EXEC_FSV_OBSERVED_ROSTER")
    first, second = list(session.executions)
    ep = session.memory.episodes[session.executions[first]["episode"]]
    refuse("no_occurrence_checker", "TOOL_NO_EXECUTION_CHECKER",
           lambda: ToolEvidence(source).validate_episode(ep))
    damaged = copy.deepcopy(ep)
    damaged["independent_occurrence"] = False
    damaged["id"] = digest({k:v for k,v in damaged.items() if k != "id"})
    refuse("mislabelled_occurrence", "PROCESS_EPISODE",
           lambda: session.memory.apply(dict(op="intake", episode=damaged)))
    corrupt_root = args.output/"corrupt-observations"
    shutil.copytree(args.observations, corrupt_root)
    entry = observations.entries[by_turn[7]]
    broken = corrupt_root/entry["execution"]/"stdout.bin"
    raw = broken.read_bytes()
    broken.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    refuse("corrupted_capture", "PROCESS_ARTIFACT_PIN",
           lambda: ProcessObservations(corrupt_root, observations.pin, source))
    direct_before = readback(args.output/"experience.sqlite3", inputs, before, args.observations)
    write(args.output/"before-withdrawal.json", before)
    key = session.executions[first]["episode"]
    invalidation = session.invalidate(key, reason="FSV observation receipt withdrawn for review")
    refuse("withdrawn_execution", "TOOL_EXECUTION_UNAVAILABLE", lambda: session.execution_evidence(first))
    check(session.execution_evidence(second)["comparison"]["concordant"]
          and len(session.snapshot()["memory"]["grounded_world_claims"]) == 1,
          "EXEC_FSV_WITHDRAWAL_SCOPE")
    boundaries.append(dict(before=digest(before), event=invalidation, after=digest(session.snapshot())))
    snapshot = session.snapshot()
    session.close()
    reopened = open_session(expected_head=snapshot["head"], expected_contract=snapshot["contract"])
    check(reopened.snapshot() == snapshot and parameters(reopened.model) == initial_parameters,
          "EXEC_FSV_REOPEN")
    reopened.close()
    direct = readback(args.output/"experience.sqlite3", inputs, snapshot, args.observations)
    phases["controls_replay_readback"] = time.perf_counter()-tick
    report = dict(complete=True, controls=controls, boundaries=boundaries,
        readback=direct, before_withdrawal=direct_before, phases=phases,
        wall_seconds=time.perf_counter()-start, observation_pin=observations.pin,
        model_parameters_unchanged=initial_parameters, scientific_admission=False,
        useful_capability=False, native_feedback_available=False)
    report["timing_coverage"] = sum(phases.values())/report["wall_seconds"]
    write(args.output/"snapshot.json", snapshot)
    write(args.output/"report.json", report)
    print(json.dumps(dict(complete=True, controls=len(controls), observations=2,
                          wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
