"""Two checked region-growth boundaries, subsequent forecasts and exact cold replay."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_loop_readback import readback
from audit.tool_growth_learning import learn
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
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv"]).decode()
    take_lease("28", "tool-checked-growth")
    configure()
    import torch
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                       population_pin=args.population_pin, native_pin=args.native_pin)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "GROWTH_PARENT_PIN")
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
    for partition in ("train", "withheld"):
        tasks = [e["task"] for e in source.episodes if e["status"] == "ready"
                 and e["partition"] == partition][:2]
        contexts.update({f"{partition}:{i}": task for i, task in enumerate(tasks)})
    kw = dict(model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
              contexts=contexts, device="cuda", growth_root=args.output/"growth", routing=routing,
              candidate_root=args.output/"candidates")
    def open_session(path=None, **extra):
        return GrowingToolLoop(path or args.output/"experience.sqlite3", source, inputs,
                               args.parent/"initial", **kw, **extra)
    if args.cold:
        with (args.output/"gpu-before-cold.txt").open("x") as f:
            f.write(resident)
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "GROWTH_RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        session = open_session(expected_head=run["head"], expected_contract=run["contract"])
        check(session.snapshot() == json.loads((args.output/"snapshot.json").read_bytes()),
              "GROWTH_COLD_STATE")
        result = readback(args.output/"experience.sqlite3", inputs, session.snapshot())
        controls = []
        current = session.growths[-1]
        current_root = args.output/"growth"/current["name"]
        audit_root = args.output/"integrity-controls"
        audit_root.mkdir(exist_ok=False)
        def refused(name, code, fn):
            before = digest(session.snapshot())
            try:
                fn()
            except (ValueError, SenseError) as exc:
                check(getattr(exc, "code", str(exc)) == code, "GROWTH_INTEGRITY_"+name)
            else:
                raise AssertionError(name+" did not refuse")
            check(digest(session.snapshot()) == before, "GROWTH_INTEGRITY_MUTATED")
            controls.append(dict(name=name, code=code))
        def read_world(root=current_root/"model", **overrides):
            contract = dict(expected=current["checkpoint"], standard=session.model.standard_id,
                            world=session.model.world_id, device="cuda")
            return load_world(root, **(contract | overrides))
        refused("old_checkpoint", "MNLM_WORLD_CHECKPOINT_PIN",
                lambda: read_world(expected=current["old_checkpoint"]))
        refused("old_world", "MNLM_WORLD_CHECKPOINT_CONTRACT",
                lambda: read_world(world=current["old_world"]))
        refused("old_kernel", "MNLM_CODATA_TOOL_KERNEL_PIN",
                lambda: ToolKernel(session.inputs, current_root/"kernel", current["old_kernel"],
                                   expected_pairs=session.routing.pairs))
        broken_world = audit_root/"corrupt-world"
        shutil.copytree(current_root/"model", broken_world)
        raw = bytearray((broken_world/"world.pt").read_bytes())
        raw[-1] ^= 1
        (broken_world/"world.pt").write_bytes(raw)
        refused("corrupt_world_bytes", "MNLM_WORLD_CHECKPOINT_BYTES", lambda: read_world(broken_world))
        broken_kernel = audit_root/"corrupt-kernel"
        shutil.copytree(current_root/"kernel", broken_kernel)
        with (broken_kernel/"manifest.json").open("ab") as f:
            f.write(b"\n")
        refused("corrupt_kernel_bytes", "MNLM_CODATA_TOOL_KERNEL_PIN",
                lambda: ToolKernel(session.inputs, broken_kernel, current["kernel"],
                                   expected_pairs=session.routing.pairs))
        incomplete = audit_root/"incomplete-world"
        shutil.copytree(current_root/"model", incomplete)
        (incomplete/"COMMITTED").unlink()
        refused("uncommitted_world", "MNLM_WORLD_CHECKPOINT_INCOMPLETE", lambda: read_world(incomplete))
        # A publication interrupted before journal selection must never be adopted.
        orphan = args.output/"growth"/"uncommitted-control"
        orphan.mkdir(exist_ok=False)
        (orphan/"INTERRUPTED").write_text("FSV: artifact directory exists, no journal transition.")
        keys = [k for k, v in session.forecasts.items()
                if v["record"]["request"]["context"].startswith("train:")
                and v["record"]["request"]["turn"] == 14]
        refused("interrupted_publication", "MNLM_CODATA_TOOL_GROWTH_ARTIFACT_EXISTS",
                lambda: session.grow("uncommitted-control", keys))
        expected = session.snapshot()
        session.close()
        session = open_session(expected_head=run["head"], expected_contract=run["contract"])
        check(session.snapshot() == expected, "GROWTH_UNCOMMITTED_IGNORED")
        write(args.output/"cold-readback.json", dict(complete=True, result=result,
            replayed=session.replayed, growths=len(session.growths), controls=controls,
            ignored_uncommitted_artifact=True, wall_seconds=time.perf_counter()-started))
        session.close()
        print(json.dumps(dict(cold=True, complete=True, transitions=result["transitions"], growths=2)))
        return
    session = open_session(create=True)
    initial_parameters = parameters(session.model)
    phases["initial_kernel_journal"] = time.perf_counter()-tick
    controls, boundaries, learning = [], [], []
    def refuse(label, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as exc:
            expected_code = code if code.startswith("MNLM_") else "MNLM_CODATA_"+code
            check(getattr(exc, "code", str(exc)) == expected_code, "GROWTH_CONTROL_"+label)
        else:
            raise AssertionError(label+" did not refuse")
        check(digest(session.snapshot()) == before, "GROWTH_CONTROL_MUTATED")
        controls.append(dict(name=label, code=code))
    for generation, exchanges in enumerate((range(3), range(3, 5), range(5, 7)), 1):
        tick = time.perf_counter()
        for exchange in exchanges:
            for phase in range(5):
                for context, task in contexts.items():
                    session.advance(context, receipt=source.receipt(task, 3+2*exchange) if phase == 3 else None)
                if phase == 2 and exchange == exchanges.stop-1:
                    refuse(f"pending_growth_{generation}", "TOOL_GROWTH_PENDING",
                           lambda: session.grow("pending", []))
        phases[f"experience_{generation}"] = time.perf_counter()-tick
        keys = list(session.forecasts)
        train = [k for k in keys if session.forecasts[k]["record"]["request"]["context"].startswith("train:")]
        evaluation = [k for k in keys if k not in train]
        if generation == 1:
            private_controls(session, refuse)
        if generation <= 2:
            tick = time.perf_counter()
            fresh = train[-2:]
            refuse(f"foreign_split_{generation}", "TOOL_GROWTH_SPLIT",
                   lambda: session.grow("foreign", [evaluation[-1]]))
            refuse(f"duplicate_keys_{generation}", "TOOL_GROWTH_KEYS",
                   lambda: session.grow("duplicate", [fresh[0], fresh[0]]))
            refuse(f"future_forecast_{generation}", "TOOL_FORECAST_UNKNOWN",
                   lambda: session.grow("future", ["0"*64]))
            prior, prior_state = session.snapshot(), state_pin(session.states)
            body = session.grow(f"generation-{generation}", fresh)
            check(parameters(session.model) == initial_parameters and state_pin(session.states) == prior_state,
                  "GROWTH_RETAINED_STATE")
            check(len(session.inputs.world_units) == 60+2*generation, "GROWTH_REGION_COUNT")
            refuse(f"already_present_{generation}", "TOOL_GROWTH_ALREADY_PRESENT",
                   lambda: session.grow("repeated", fresh))
            boundaries.append(dict(before=digest(prior), action=body, after=digest(session.snapshot())))
            learning.append(learn(session, args.output, generation, train, evaluation, refuse))
            head, contract, expected = session.head, session.journal.pin, session.snapshot()
            session.close()
            session = open_session(expected_head=head, expected_contract=contract)
            check(session.snapshot() == expected, "GROWTH_MIDPOINT_REPLAY")
            phases[f"growth_{generation}_learning_replay"] = time.perf_counter()-tick
    tick = time.perf_counter()
    snapshot = session.snapshot()
    run = dict(parent=args.parent_pin, head=session.head, contract=session.journal.pin)
    # A private withdrawal must affect only its context, never the shared bank.
    private_path = args.output/"private-withdrawal.sqlite3"
    with sqlite3.connect(private_path) as target:
        session.journal.db.backup(target)
    private = open_session(private_path, expected_head=session.head, expected_contract=session.journal.pin)
    own = [k for k, v in private.forecasts.items()
           if v["record"]["request"]["context"] == "withheld:0"]
    foreign = [k for k in private.forecasts if k not in own]
    before_private = digest(private.snapshot())
    episode = private.forecasts[own[0]]["record"]["support"]["3"]
    private.invalidate(episode, reason="FSV private observation withdrawal")
    check(all(private.forecasts[k]["invalidated"] for k in own) and
          all(not private.forecasts[k]["invalidated"] for k in foreign), "PRIVATE_WITHDRAWAL_ISOLATION")
    private._bank_checked()
    from model.tool_context_regions import historical
    original = session
    session = private
    refuse("withdrawn_private_history", "TOOL_EVIDENCE_UNAVAILABLE",
           lambda: historical(private, private.forecasts[own[0]]["record"]))
    refuse("withdrawn_private_context", "TOOL_CONTEXT_REVIEW", lambda: private.advance("withheld:0"))
    from model.tool_learning import ToolLearningData
    latest = private.decisions[-1]
    refuse("withdrawn_learner_evidence", "TOOL_LEARNING_INELIGIBLE",
           lambda: ToolLearningData(private, train=latest["train"], evaluate=latest["evaluate"],
                                    continuation=latest["candidate"]))
    private.advance("withheld:1")
    private_head, private_contract, private_snapshot = private.head, private.journal.pin, private.snapshot()
    private.close()
    private = open_session(private_path, expected_head=private_head, expected_contract=private_contract)
    check(private.snapshot() == private_snapshot, "PRIVATE_WITHDRAWAL_REPLAY")
    boundaries.append(dict(before=before_private, action="withdraw private observation",
                           after=digest(private.snapshot()), affected=len(own), unaffected=len(foreign)))
    private.close()
    session = original
    # Withdraw a new bank dependency in its own real persisted branch.
    branch_path = args.output/"withdrawal.sqlite3"
    with sqlite3.connect(branch_path) as target:
        session.journal.db.backup(target)
    branch = open_session(branch_path, expected_head=session.head, expected_contract=session.journal.pin)
    original = session
    session = branch
    support = next(iter(branch.growths[0]["supports"].values()))["response"]
    earlier = [k for k, v in branch.forecasts.items()
               if v["record"]["request"]["context"].startswith("withheld:") and
               v["record"]["request"]["turn"] == 6]
    later = [k for k, v in branch.forecasts.items()
             if v["record"]["request"]["context"].startswith("withheld:") and
             v["record"]["request"]["turn"] > 6]
    before = digest(branch.snapshot())
    branch.invalidate(support, reason="FSV explicit withdrawal of a newly added source dependency")
    check(all(not branch.forecasts[k]["invalidated"] for k in earlier)
          and all(branch.forecasts[k]["invalidated"] for k in later), "GROWTH_WITHDRAWAL_SCOPE")
    branch.eligible(earlier[0])
    refuse("withdrawn_new_bank", "TOOL_EVIDENCE_UNAVAILABLE",
           branch._bank_checked)
    refuse("withdrawn_new_bank_context", "TOOL_CONTEXT_REVIEW",
           lambda: branch.advance("withheld:0"))
    refuse("withdrawn_dependent_forecast", "TOOL_LEARNING_INELIGIBLE", lambda: branch.eligible(later[0]))
    boundaries.append(dict(before=before, action="withdraw new region support on persisted branch",
                           after=digest(branch.snapshot()), original_unchanged=original.snapshot() == snapshot))
    branch.close()
    session = original
    result = readback(args.output/"experience.sqlite3", inputs, snapshot)
    session.close()
    phases["withdrawal_readback"] = time.perf_counter()-tick
    tick = time.perf_counter()
    write(args.output/"snapshot.json", snapshot)
    write(args.output/"run.json", run)
    write(args.output/"training.json", learning)
    phases["publication"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    check(sum(phases.values())/wall >= .9, "GROWTH_TIMING")
    write(args.output/"report.json", dict(complete=True, acceptance="author-checked",
        result=result, boundaries=boundaries, controls=controls, phases=phases,
        wall_seconds=wall, timing_coverage=sum(phases.values())/wall, regions=64,
        # The active model is unchanged; candidate deltas are in training.json.
        parameters_unchanged=True, scientific_admission=False, predictive_gain=False,
        files={n: sha((args.output/n).read_bytes()) for n in
               ("run.json", "snapshot.json", "training.json")}))
    print(json.dumps(dict(complete=True, regions=64, growths=2, transitions=result["transitions"],
                         controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
