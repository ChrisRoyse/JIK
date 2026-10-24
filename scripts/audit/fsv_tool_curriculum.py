"""Fixed ready-partition curriculum: typed forecasts, retained learning and replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_growth_learning import learn
from audit.tool_loop_readback import readback
from audit.tool_private_regions import controls as private_controls
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.tool_temporal_source import ToolSource, write
from lens.lenscommon import run_cli, take_lease
from model.tool_growth import GrowingToolLoop, parameters, state_pin
from model.tool_kernel import ToolKernel, build
from model.tool_temporal_input import ToolInputs
from model.world_checkpoint import load as load_world
from senses.contract import SenseError


def additions(session, keys):
    """Use source order; duplicate occurrences remain training examples."""
    seen = set(session.inputs.world_units)
    selected, skipped = [], []
    for key in keys:
        item = session.eligible(key)
        request = item["record"]["request"]
        check(session.source.by_task[request["task"]]["partition"] == "train",
              "CURRICULUM_TRAIN_PARTITION")
        unit = item["target_unit"]
        if unit in seen:
            skipped.append(dict(forecast=key, unit=unit, reason="region_already_present"))
        else:
            selected.append(key)
            seen.add(unit)
    check(selected, "CURRICULUM_NO_NEW_REGIONS")
    return selected, skipped


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--run-pin")
    args = parser.parse_args()
    started = time.perf_counter()
    resident = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"
    ]).decode()
    take_lease("28", "tool-fixed-curriculum")
    configure()
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "CURRICULUM_PARENT_PIN")
    parent = json.loads((args.parent/"run.json").read_bytes())
    phases = dict(source_native=time.perf_counter()-started)
    tick = time.perf_counter()
    if not args.cold:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"gpu-before.txt").write_text(resident)
        kernel_pin = build(inputs, args.output/"initial-kernel")
    else:
        kernel_pin = (args.output/"initial-kernel"/"COMMITTED").read_text()
    routing = ToolKernel(inputs, args.output/"initial-kernel", kernel_pin)
    contexts = {}
    for partition, count in (("train", 30), ("withheld", 15)):
        tasks = [e["task"] for e in source.episodes
                 if e["status"] == "ready" and e["partition"] == partition]
        check(len(tasks) == count, "CURRICULUM_ROSTER")
        contexts.update({f"{partition}:{i}": task for i, task in enumerate(tasks)})
    kw = dict(model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
              contexts=contexts, device="cuda", growth_root=args.output/"growth", routing=routing,
              candidate_root=args.output/"candidates")

    def open_session(**extra):
        return GrowingToolLoop(args.output/"experience.sqlite3", source, inputs,
                               args.parent/"initial", **kw, **extra)

    controls = []
    session = None

    def refuse(label, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as exc:
            expected = code if code.startswith("MNLM_") else "MNLM_CODATA_"+code
            check(getattr(exc, "code", str(exc)) == expected, "CURRICULUM_CONTROL_"+label)
        else:
            raise AssertionError(label+" did not refuse")
        check(digest(session.snapshot()) == before, "CURRICULUM_CONTROL_MUTATED")
        controls.append(dict(name=label, code=code))

    if args.cold:
        with (args.output/"gpu-before-cold.txt").open("x") as f:
            f.write(resident)
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "CURRICULUM_RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        session = open_session(expected_head=run["head"], expected_contract=run["contract"])
        snapshot = json.loads((args.output/"snapshot.json").read_bytes())
        check(session.snapshot() == snapshot, "CURRICULUM_COLD_STATE")
        result = readback(args.output/"experience.sqlite3", inputs, snapshot)
        current = session.growths[-1]
        root = args.output/"growth"/current["name"]/"model"

        def read_world(path=root, **override):
            contract = dict(expected=current["checkpoint"], standard=session.model.standard_id,
                            world=session.model.world_id, device="cuda")
            return load_world(path, **(contract | override))

        refuse("old_checkpoint", "MNLM_WORLD_CHECKPOINT_PIN",
               lambda: read_world(expected=current["old_checkpoint"]))
        refuse("old_world", "MNLM_WORLD_CHECKPOINT_CONTRACT",
               lambda: read_world(world=current["old_world"]))
        broken = args.output/"integrity-controls"/"corrupt-world"
        shutil.copytree(root, broken)
        raw = bytearray((broken/"world.pt").read_bytes())
        raw[-1] ^= 1
        (broken/"world.pt").write_bytes(raw)
        refuse("corrupt_world", "MNLM_WORLD_CHECKPOINT_BYTES", lambda: read_world(broken))
        private_controls(session, refuse)
        write(args.output/"cold-readback.json", dict(complete=True, result=result,
            replayed=session.replayed, controls=controls, contexts=contexts,
            wall_seconds=time.perf_counter()-started))
        session.close()
        print(json.dumps(dict(cold=True, complete=True, controls=len(controls))))
        return
    session = open_session(create=True)
    initial_parameters = parameters(session.model)
    phases["initial_kernel_journal"] = time.perf_counter()-tick
    learning, boundaries, growth_selection = [], [], []
    for generation, exchanges in enumerate((range(3), range(3, 5)), 1):
        tick = time.perf_counter()
        for exchange in exchanges:
            for phase in range(5):
                for context, task in contexts.items():
                    session.advance(context, receipt=source.receipt(task, 3+2*exchange)
                                    if phase == 3 else None)
                if phase == 2 and exchange == exchanges.stop-1:
                    refuse(f"pending_growth_{generation}", "TOOL_GROWTH_PENDING",
                           lambda: session.grow("pending", []))
        phases[f"experience_{generation}"] = time.perf_counter()-tick
        train = [k for k, v in session.forecasts.items()
                 if v["record"]["request"]["context"].startswith("train:")]
        evaluate = [k for k in session.forecasts if k not in train]
        check(len(train) == 30*generation and len(evaluate) == 15*generation,
              "CURRICULUM_FORECAST_ROSTER")
        tick = time.perf_counter()
        if generation == 1:
            private_controls(session, refuse)
        selected, skipped = additions(session, train)
        refuse(f"foreign_split_{generation}", "TOOL_GROWTH_SPLIT",
               lambda: session.grow("foreign", [evaluate[-1]]))
        refuse(f"duplicate_keys_{generation}", "TOOL_GROWTH_KEYS",
               lambda: session.grow("duplicate", [selected[0], selected[0]]))
        refuse(f"future_forecast_{generation}", "TOOL_FORECAST_UNKNOWN",
               lambda: session.grow("future", ["0"*64]))
        prior, prior_state = session.snapshot(), state_pin(session.states)
        body = session.grow(f"generation-{generation}", selected)
        check(parameters(session.model) == initial_parameters and state_pin(session.states) == prior_state,
              "CURRICULUM_RETAINED_STATE")
        check(len(session.inputs.world_units) == len(prior["region_units"])+len(selected),
              "CURRICULUM_REGION_COUNT")
        refuse(f"already_present_{generation}", "TOOL_GROWTH_ALREADY_PRESENT",
               lambda: session.grow("repeated", selected))
        growth_selection.append(dict(generation=generation, selected=selected, skipped=skipped,
                                     regions=len(session.inputs.world_units)))
        boundaries.append(dict(before=digest(prior), action=body, after=digest(session.snapshot())))
        phases[f"growth_{generation}"] = time.perf_counter()-tick
        print(json.dumps(dict(generation=generation, stage="learning", train=len(train),
                              evaluate=len(evaluate), added=len(selected))), flush=True)
        tick = time.perf_counter()
        before = digest(session.snapshot())
        proof = learn(session, args.output, generation, train, evaluate, refuse)
        learning.append(proof)
        boundaries.append(dict(before=before, action=proof["decision"], after=digest(session.snapshot())))
        phases[f"learning_{generation}"] = time.perf_counter()-tick
        tick = time.perf_counter()
        head, contract, expected = session.head, session.journal.pin, session.snapshot()
        session.close()
        session = open_session(expected_head=head, expected_contract=contract)
        check(session.snapshot() == expected, "CURRICULUM_MIDPOINT_REPLAY")
        phases[f"replay_{generation}"] = time.perf_counter()-tick
        print(json.dumps(dict(generation=generation, stage="replayed",
                              optimizer_steps=proof["optimizer_steps"])), flush=True)
    tick = time.perf_counter()
    snapshot = session.snapshot()
    result = readback(args.output/"experience.sqlite3", inputs, snapshot)
    check(result["transitions"] == 1125 and len(snapshot["forecasts"]) == 90,
          "CURRICULUM_COMPLETE_PREFIX")
    run = dict(parent=args.parent_pin, head=session.head, contract=session.journal.pin)
    session.close()
    write(args.output/"snapshot.json", snapshot)
    write(args.output/"training.json", learning)
    write(args.output/"run.json", run)
    phases["readback_publication"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    check(sum(phases.values())/wall >= .9, "CURRICULUM_TIMING")
    write(args.output/"report.json", dict(complete=True, acceptance="author-checked", result=result,
        boundaries=boundaries, controls=controls, growth_selection=growth_selection, contexts=contexts,
        roster=source.episodes, phases=phases, wall_seconds=wall, timing_coverage=sum(phases.values())/wall,
        parameters_unchanged=True, scientific_admission=False, predictive_gain=False,
        files={n: sha((args.output/n).read_bytes()) for n in ("run.json", "snapshot.json", "training.json")}))
    print(json.dumps(dict(complete=True, contexts=len(contexts), regions=len(snapshot["region_units"]),
                         controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
