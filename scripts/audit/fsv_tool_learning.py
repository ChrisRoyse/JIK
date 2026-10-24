"""Real durable tool experience, two readout increments, withdrawal and cold replay."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_loop_readback import readback
from audit.tool_journal_controls import corruptions, runtime
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource, write
from lens.lenscommon import take_lease, run_cli
from model import constellation_training_checkpoint as checkpoint
from model.constellation_learning import frozen_pin, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.tool_learning import ToolLearningData
from model.tool_loop import ToolLoop
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("raw", "population", "native", "parent", "output"):
        parser.add_argument("--"+n, type=Path, required=True)
    for n in ("population-pin", "native-pin", "parent-pin"):
        parser.add_argument("--"+n, required=True)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--run-pin")
    parser.add_argument("--with-kernel", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv"]).decode()
    take_lease("28", "tool-durable-learning")
    configure()
    import torch
    source = ToolSource(args.raw)
    inputs = ToolInputs(source, args.population, args.native,
                       population_pin=args.population_pin, native_pin=args.native_pin)
    check(sha((args.parent/"run.json").read_bytes()) == args.parent_pin, "TOOL_PARENT_PIN")
    parent = json.loads((args.parent/"run.json").read_bytes())
    phases = dict(source_native=time.perf_counter()-started)
    if not args.cold:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"candidates").mkdir()
        (args.output/"gpu-before.txt").write_text(resident, encoding="utf-8")
    routing = None
    if args.with_kernel:
        from model.tool_kernel import ToolKernel, build
        tick = time.perf_counter()
        root = args.output/"kernel"
        pin = (root/"COMMITTED").read_text() if args.cold else build(inputs, root)
        routing = ToolKernel(inputs, root, pin)
        phases["kernel_build_readback"] = time.perf_counter()-tick
    contexts = {}
    for partition in ("train", "withheld"):
        tasks = [e["task"] for e in source.episodes if e["status"] == "ready" and e["partition"] == partition][:2]
        contexts.update({f"{partition}:{i}": t for i, t in enumerate(tasks)})
    kw = dict(model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
              contexts=contexts, device="cuda", candidate_root=args.output/"candidates", routing=routing)
    tick = time.perf_counter()
    def open_session(**extra):
        return ToolLoop(args.output/"experience.sqlite3", source, inputs, args.parent/"initial", **kw, **extra)
    if args.cold:
        check(sha((args.output/"run.json").read_bytes()) == args.run_pin, "TOOL_RUN_PIN")
        run = json.loads((args.output/"run.json").read_bytes())
        check(run["parent"] == args.parent_pin, "TOOL_RUN_PARENT")
        session = open_session(expected_head=run["head"], expected_contract=run["contract"])
        expected = json.loads((args.output/"snapshot.json").read_bytes())
        check(session.snapshot() == expected, "TOOL_COLD_SNAPSHOT")
        result = readback(args.output/"experience.sqlite3", inputs, expected)
        replayed = session.replayed
        session.close()
        write(args.output/"cold-readback.json", dict(complete=True, result=result, replayed=replayed,
              wall_seconds=time.perf_counter()-started, acceptance="author-checked"))
        print(json.dumps(dict(cold=True, complete=True, replayed=replayed)))
        return
    session = open_session(create=True)
    phases["initial_journal"] = time.perf_counter()-tick
    controls, boundaries, training = [], [], []
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as exc:
            check(getattr(exc, "code", str(exc)) == "MNLM_CODATA_"+code, "TOOL_CONTROL_"+name)
        else:
            raise AssertionError(name+" did not refuse")
        check(digest(session.snapshot()) == before, "TOOL_CONTROL_MUTATED")
        controls.append(dict(name=name, code=code))
    first = "train:0"
    task = contexts[first]
    if routing is not None:
        refuse("kernel_wrong_pin", "TOOL_KERNEL_PIN",
               lambda: ToolKernel(inputs, args.output/"kernel", "0"*64))
        refuse("kernel_response_as_query", "TOOL_KERNEL_QUERY",
               lambda: routing.route(inputs.population.lookup[task, 3]))
        changed_ref = dict(inputs.population.lookup[task, 2], unit=-1)
        refuse("kernel_changed_query", "TOOL_KERNEL_QUERY", lambda: routing.route(changed_ref))
        key = next(iter(session.route_memory.values()))
        saved = session.memory.state[key]["active"]
        session.memory.state[key]["active"] = False
        try:
            refuse("kernel_withdrawn_action_support", "TOOL_EVIDENCE_UNAVAILABLE",
                   lambda: session.advance(first))
        finally:
            session.memory.state[key]["active"] = saved
        grouped = {}
        for ref in routing.action_references():
            grouped.setdefault(ref["unit"], []).append(ref)
        shared = [r for group in grouped.values() if len(group) > 1 for r in group]
        check(len(session.route_memory) == len(routing.action_references()) and shared,
              "TOOL_KERNEL_OCCURRENCE_SUPPORT")
        for ref in shared:
            key = session.route_memory[digest(ref)]
            session.memory.state[key]["active"] = False
            try:
                refuse("kernel_shared_reading_"+key[:8], "TOOL_EVIDENCE_UNAVAILABLE",
                       lambda: session.advance(first))
            finally:
                session.memory.state[key]["active"] = True
    refuse("future_input", "TOOL_NEXT_REFERENCE", lambda: session.advance(first,
        reference=inputs.population.lookup[task, 2]))
    refuse("cross_task_input", "TOOL_NEXT_REFERENCE", lambda: session.advance(first,
        reference=inputs.population.lookup[contexts["train:1"], 1]))
    candidate = copy.deepcopy(session.model)
    parent_pin = parent["initial"]
    for increment, exchanges in enumerate((range(3), range(3, 5)), 1):
        tick = time.perf_counter()
        before = digest(session.snapshot())
        for exchange in exchanges:
            for phase in range(5):
                if exchange == exchanges.stop-1 and phase == 3:
                    keys = list(session.forecasts)
                    train = [k for k in keys if session.forecasts[k]["record"]["request"]["context"].startswith("train:")]
                    evaluate = [k for k in keys if k not in train]
                    refuse(f"pending_learning_{increment}", "TOOL_LEARNING_INELIGIBLE",
                           lambda: ToolLearningData(session, train=train, evaluate=evaluate))
                    head, contract = session.head, session.journal.pin
                    snapshot = session.snapshot()
                    session.close()
                    session = open_session(expected_head=head, expected_contract=contract)
                    check(session.snapshot() == snapshot, "TOOL_PENDING_RECOVERY")
                    refuse(f"self_echo_{increment}", "TOOL_FEEDBACK_RECEIPT",
                           lambda: session.advance(first, receipt=dict(id="model-confidence")))
                for context, current_task in contexts.items():
                    receipt = source.receipt(current_task, 3+2*exchange) if phase == 3 else None
                    session.advance(context, receipt=receipt)
        boundaries.append(dict(name=f"increment_{increment}_committed_then_observed",
                               before=before, after=digest(session.snapshot())))
        phases[f"experience_{increment}"] = time.perf_counter()-tick
        tick = time.perf_counter()
        keys = list(session.forecasts)
        train = [k for k in keys if session.forecasts[k]["record"]["request"]["context"].startswith("train:")]
        evaluate = [k for k in keys if k not in train]
        refuse(f"wrong_partition_{increment}", "TOOL_LEARNING_SPLIT",
               lambda: ToolLearningData(session, train=evaluate, evaluate=train))
        data = ToolLearningData(session, train=train, evaluate=evaluate)
        before_weights = tensor_fingerprint(dict(candidate.named_parameters()))
        optimizer = checkpoint.optimizer_for(candidate, data.plan)
        losses, mid_pin = [], None
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            loss, details = data.loss(model)
            check(bool(torch.isfinite(loss)), "TOOL_LOSS_NONFINITE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable(model)], 1.)
            opt.step()
            return dict(loss=float(loss.detach()), details=details)
        for step_number in range(1, data.plan["steps"]+1):
            losses.append(step(candidate, optimizer))
            if step_number == 4:
                mid_pin = checkpoint.save(args.output/f"mid-{increment}", candidate, optimizer, data,
                                          step=4, parent=parent_pin)
        recovered, recovered_optimizer, _ = checkpoint.load(args.output/f"mid-{increment}", data, expected=mid_pin)
        for step_number in range(4, data.plan["steps"]):
            check(step(recovered, recovered_optimizer) == losses[step_number], "TOOL_OPTIMIZER_CONTINUATION")
        check(fingerprint(recovered, recovered_optimizer) == fingerprint(candidate, optimizer),
              "TOOL_OPTIMIZER_RECOVERY")
        after_weights = tensor_fingerprint(dict(candidate.named_parameters()))
        changed = [name for name in before_weights if before_weights[name] != after_weights[name]]
        check(changed and all(n.startswith(("query.", "place.")) for n in changed)
              and frozen_pin(candidate) == data.frozen, "TOOL_FROZEN_RETENTION")
        name = f"increment-{increment}"
        pin = checkpoint.save(args.output/"candidates"/name, candidate, optimizer, data,
                              step=data.plan["steps"], parent=parent_pin)
        decision = session.consider(name, pin=pin, train=train, evaluate=evaluate, parent=parent_pin)
        check(not decision["result"]["selected"], "TOOL_UNADMITTED_SELECTION")
        training.append(dict(name=name, parent=parent_pin, pin=pin, losses=losses, changed=changed,
                             before_parameters=digest(before_weights), after_parameters=digest(after_weights),
                             exact_optimizer_recovery=True, decision=decision))
        refuse(f"stale_training_binding_{increment}", "TOOL_LEARNING_STALE", data.fresh)
        parent_pin = pin
        phases[f"learning_{increment}"] = time.perf_counter()-tick
    tick = time.perf_counter()
    stale = open_session(expected_head=session.head, expected_contract=session.journal.pin)
    before = digest(session.snapshot())
    key = session.support[first][1]
    session.invalidate(key, reason="FSV explicit withdrawal of source support; not a claim the source is false")
    check(session.needs_review(first) and not session.needs_review("withheld:0"), "TOOL_INVALIDATION_SCOPE")
    refuse("withdrawn_context", "TOOL_CONTEXT_REVIEW", lambda: session.advance(first))
    refuse("withdrawn_learning", "TOOL_LEARNING_INELIGIBLE",
           lambda: ToolLearningData(session, train=train, evaluate=evaluate))
    session.eligible(evaluate[0])
    refuse("stale_writer", "JOURNAL_STALE", lambda: stale.advance("withheld:0"))
    stale.close()
    runtime(session, refuse)
    boundaries.append(dict(name="withdrawal_invalidates_dependents_preserves_other_context",
                           before=before, after=digest(session.snapshot())))
    snapshot = session.snapshot()
    run = dict(parent=args.parent_pin, head=session.head, contract=session.journal.pin)
    session.close()
    result = readback(args.output/"experience.sqlite3", inputs, snapshot)
    def open_at(path, head, contract):
        return ToolLoop(path, source, inputs, args.parent/"initial", **kw,
                        expected_head=head, expected_contract=contract)
    controls.extend(corruptions(args.output/"experience.sqlite3", open_at,
                               head=run["head"], contract=run["contract"], output=args.output))
    phases["withdrawal_and_independent_readback"] = time.perf_counter()-tick
    tick = time.perf_counter()
    write(args.output/"run.json", run)
    write(args.output/"snapshot.json", snapshot)
    write(args.output/"training.json", training)
    phases["publication"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    coverage = sum(phases.values())/wall
    check(coverage >= .9, "TOOL_TIMING_COVERAGE")
    write(args.output/"report.json", dict(complete=True, acceptance="author-checked", result=result,
        controls=controls, boundaries=boundaries, phases_seconds=phases, timing_coverage=coverage,
        wall_seconds=wall, training_increments=len(training), serving=False,
        kernel=None if routing is None else dict(identity=routing.identity,
            recall={n: r["recall"] for n, r in routing.calibration.items()},
            action_windows={n: len(l.original) for n, l in routing.lanes.items()}),
        files={n: sha((args.output/n).read_bytes()) for n in ("run.json", "snapshot.json", "training.json")}))
    print(json.dumps(dict(complete=True, events=result["events"], transitions=result["transitions"],
                         controls=len(controls), wall_seconds=wall)))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
