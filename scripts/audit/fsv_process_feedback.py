"""Captured output -> native Echo -> two retained learning increments -> cold replay."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.tool_execution_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_observation import ProcessObservations
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model import constellation_training_checkpoint as checkpoint
from model.constellation_learning import frozen_pin, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.process_input import ProcessInputs
from claims.process_conditions import ConditionInputs
from model.condition_input import ConditionNative
from model.tool_growth import GrowingToolLoop, parameters, state_pin
from model.tool_kernel import ToolKernel, build
from model.tool_learning import ToolLearningData
from model.tool_learner import start as start_learner
from model.tool_temporal_input import ToolInputs
from senses.contract import SenseError


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value))


def learn(session, output, number, train, evaluate, refuse):
    continuation = session.decisions[-1]["candidate"] if session.decisions else None
    data = ToolLearningData(session, train=train, evaluate=evaluate, continuation=continuation)
    model, optimizer, continuity = start_learner(data)
    parent = session.active_checkpoint
    before = data.evaluate(model)
    target_checks = {}
    for key in train:
        e = data.examples[key]
        if "execution_conditions" in e["request"]:
            actual_input = session.native_action(e["request"]).slots
            check(all(torch.equal(e["native"][s.name].values, actual_input[s.name].values)
                      and e["native"][s.name].rows == actual_input[s.name].rows for s in model.specs),
                  "CONDITION_LEARNER_INPUT")
        observation = session.forecasts[key]["execution"]["observation"]
        provider = session.execution_inputs.for_observation(observation)
        check(e["target_inputs"] is provider
              and data.binding["evidence"][key]["execution"] == session.executions[key],
              "FEEDBACK_LEARNER_SOURCE")
        target_checks[key] = dict(binding=session.execution_inputs.checked(observation), slots={})
        for spec in model.specs:
            slot = e["target"][spec.name]
            actual = slot.values[0, ~slot.padding[0]].cpu().numpy()
            values, ptr = provider.arrays[spec.name]
            unit = provider.lookup[observation]
            expected = values[ptr[unit]:ptr[unit+1]]
            check(np.array_equal(actual, expected), "FEEDBACK_LEARNER_WINDOWS")
            target_checks[key]["slots"][spec.name] = dict(shape=list(actual.shape), sha256=sha(actual.tobytes()))
    original = tensor_fingerprint(dict(model.named_parameters()))
    losses = []
    def step(candidate, opt):
        opt.zero_grad(set_to_none=True)
        loss, details = data.loss(candidate)
        check(bool(torch.isfinite(loss)), "FEEDBACK_FINITE_LOSS")
        loss.backward()
        gradients = [n for n,p in candidate.named_parameters() if p.grad is not None and bool((p.grad != 0).any())]
        check(gradients and all(n.startswith(("query.", "place.")) for n in gradients),
              "FEEDBACK_GRADIENT_SCOPE")
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable(candidate)], 1.)
        opt.step()
        return dict(loss=float(loss.detach()), details=details, gradients=gradients)
    checkpoint.save(output/f"start-{number}", model, optimizer, data, step=0, parent=parent)
    for cursor in range(1, data.plan["steps"]+1):
        losses.append(step(model, optimizer))
        if cursor == 4:
            mid = checkpoint.save(output/f"mid-{number}", model, optimizer, data, step=cursor, parent=parent)
    recovered, recovered_opt, _ = checkpoint.load(output/f"mid-{number}", data, expected=mid)
    for cursor in range(4, data.plan["steps"]):
        check(step(recovered, recovered_opt) == losses[cursor], "FEEDBACK_OPTIMIZER_REPLAY")
    check(fingerprint(model, optimizer) == fingerprint(recovered, recovered_opt), "FEEDBACK_OPTIMIZER_RESTART")
    after = data.evaluate(model)
    current = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in original if original[n] != current[n]]
    check(changed and all(n.startswith(("query.", "place.")) for n in changed)
          and frozen_pin(model) == data.frozen, "FEEDBACK_FROZEN_CORE")
    name = f"increment-{number}"
    pin = checkpoint.save(output/"candidates"/name, model, optimizer, data, step=data.plan["steps"], parent=parent)
    decision = session.consider(name, pin=pin, train=train, evaluate=evaluate, parent=parent, continuation=continuation)
    refuse("stale_binding_"+str(number), "TOOL_LEARNING_STALE", data.fresh)
    refuse("unadmitted_selection_"+str(number), "TOOL_CANDIDATE_NOT_SELECTED", lambda: session.require_selection(name))
    return dict(increment=number, targets=target_checks, before=before, after=after, losses=losses,
                changed=changed, continuity=continuity, optimizer_steps=data.optimizer_offset+data.plan["steps"],
                exact_optimizer_recovery=True, decision=decision)


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "parent", "kernel", "observations",
                 "execution-population", "execution-native", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "parent-pin", "observation-pin",
                 "execution-population-pin", "execution-native-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--grow-observed", action="store_true")
    p.add_argument("--condition-population", type=Path)
    p.add_argument("--condition-native", type=Path)
    p.add_argument("--condition-population-pin")
    p.add_argument("--condition-native-pin")
    a = p.parse_args()
    begin = time.perf_counter()
    gpu = subprocess.check_output(["nvidia-smi"]).decode()
    take_lease("28", "process-constellation-feedback")
    configure()
    source = ToolSource(a.raw)
    inputs = ToolInputs(source, a.population, a.native, population_pin=a.population_pin, native_pin=a.native_pin)
    observations = ProcessObservations(a.observations, a.observation_pin, source)
    native = ProcessInputs(observations, inputs, a.execution_population, a.execution_native,
                           population_pin=a.execution_population_pin, native_pin=a.execution_native_pin)
    conditions = None
    if a.condition_population is not None:
        conditions = ConditionNative(ConditionInputs(a.observations, observations.manifest["condition_commitment"]),
            inputs, a.condition_population, a.condition_native,
            population_pin=a.condition_population_pin, native_pin=a.condition_native_pin)
    check(sha((a.parent/"run.json").read_bytes()) == a.parent_pin, "FEEDBACK_PARENT")
    parent = json.loads((a.parent/"run.json").read_bytes())
    if not a.cold:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output/"gpu-before.txt").write_text(gpu)
    kernel_root = a.output/"initial-kernel" if a.grow_observed else a.kernel
    if a.grow_observed and not a.cold:
        build(inputs, kernel_root)
    routing = ToolKernel(inputs, kernel_root, (kernel_root/"COMMITTED").read_text())
    withheld = next(e["task"] for e in source.episodes if e["partition"] == "withheld" and e["status"] == "ready")
    contexts = {"train:1":"tonioo__sievelib-79", "withheld:0":withheld}
    def open_session(path=None, **kw):
        return GrowingToolLoop(path or a.output/"experience.sqlite3", source, inputs, a.parent/"initial",
            model_pin=parent["initial"], standard=parent["standard"], world=parent["world"],
            contexts=contexts, device="cuda", observations=observations, execution_inputs=native,
            condition_inputs=conditions,
            routing=routing, growth_root=a.output/"growth", candidate_root=a.output/"candidates", **kw)
    phases = dict(source_native=time.perf_counter()-begin)
    tick = time.perf_counter()
    if a.cold:
        report = json.loads((a.output/"report.json").read_bytes())
        snapshot = json.loads((a.output/"snapshot.json").read_bytes())
        session = open_session(expected_head=snapshot["head"], expected_contract=snapshot["contract"])
        check(session.snapshot() == snapshot, "FEEDBACK_COLD")
        direct = readback(a.output/"experience.sqlite3", inputs, snapshot, a.observations, native)
        check(direct == report["readback"], "FEEDBACK_COLD_READBACK")
        if a.grow_observed:
            from audit.process_growth_readback import readback as read_growth
            check(read_growth(inputs, native, a.output, a.parent) == report["growth_readback"],
                  "FEEDBACK_GROWTH_COLD_READBACK")
        session.close()
        phases["replay_readback"] = time.perf_counter()-tick
        write(a.output/"cold.json", dict(complete=True, phases=phases, readback=direct,
                                         wall_seconds=time.perf_counter()-begin))
        print(json.dumps(dict(complete=True, cold=True)))
        return
    session = open_session(create=True)
    initial_parameters = parameters(session.model)
    phases["initial_journal"] = time.perf_counter()-tick
    controls, boundaries, training = [], [], []
    def refuse(name, code, action):
        before = digest(session.snapshot())
        try:
            action()
        except (ValueError, SenseError) as error:
            check(code in str(error), "FEEDBACK_CONTROL_CODE_"+name)
        else:
            raise AssertionError("FEEDBACK_ACCEPTED_"+name)
        check(digest(session.snapshot()) == before, "FEEDBACK_CONTROL_STATE_"+name)
        controls.append(dict(name=name, code=code))
    by_turn = {entry["target"]["turn"]:key for key,entry in observations.entries.items()}
    if conditions is not None:
        refuse("conditions_wrong_phase", "TOOL_CONDITION_PHASE",
               lambda: session.bind_execution_conditions("train:1", conditions.keys[0]))
    refuse("wrong_phase", "TOOL_EXECUTION_ECHO_PHASE", lambda: session.advance("train:1", execution=by_turn[7]))
    damaged = a.output/"corrupt-native"
    shutil.copytree(a.execution_native, damaged)
    data_path = damaged/native.specs[0].name/"slots.npz"
    raw = data_path.read_bytes()
    data_path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    refuse("corrupt_native", "MNLM_WINDOW_READBACK_FILE_HASH",
           lambda: ProcessInputs(observations, inputs, a.execution_population, damaged,
                                 population_pin=a.execution_population_pin, native_pin=a.execution_native_pin))
    for number, exchanges in enumerate((range(3), range(3,5)), 1):
        tick = time.perf_counter()
        for exchange in exchanges:
            turn = 3+2*exchange
            for phase in range(5):
                for context, task in contexts.items():
                    if conditions is not None and phase == 1 and context == "train:1" and turn in by_turn:
                        key = observations.entries[by_turn[turn]]["condition"]
                        saved_conditions = session.condition_inputs
                        session.condition_inputs = None
                        try:
                            refuse("missing_condition_native_"+str(turn), "TOOL_CONDITION_NATIVE_MISSING",
                                   lambda: session.bind_execution_conditions(context, key))
                        finally:
                            session.condition_inputs = saved_conditions
                        refuse("foreign_condition_context_"+str(turn), "TOOL_CONDITION_SOURCE",
                               lambda: session.bind_execution_conditions("withheld:0", key))
                        before = digest(session.snapshot())
                        bound = session.bind_execution_conditions(context, key)
                        refuse("duplicate_conditions_"+str(turn), "TOOL_CONDITION_DUPLICATE",
                               lambda: session.bind_execution_conditions(context, key))
                        boundaries.append(dict(before=before, action=bound, after=digest(session.snapshot())))
                    if conditions is not None and phase == 2 and context == "train:1" and turn in by_turn:
                        request = session._request(context, None, None)
                        actual = session.native_action(request)
                        plain = inputs.observed(request["reference"], session.model.config, session.device)
                        other = next(k for k in conditions.keys if k != request["execution_conditions"]["native"]["condition"])
                        alternative = conditions.merge(request["reference"], conditions.checked(other),
                                                       session.model.config, session.device)
                        result, _, _ = session._prediction(request, actual)
                        absent, _, _ = session._prediction(request, plain)
                        different, _, _ = session._prediction(request, alternative)
                        signatures = {n: digest(tensor_fingerprint(dict(
                            hidden=o["states"][0].hidden, predictions=o["slots"])))
                            for n,o in (("actual", result), ("absent", absent), ("other", different))}
                        check(len(set(signatures.values())) == 3, "CONDITION_MODEL_INSENSITIVE")
                        before = digest(session.snapshot())
                        body = session.advance(context)
                        check(body["hidden_sha256"] == sha(result["states"][0].hidden.cpu().numpy().tobytes()),
                              "CONDITION_FORWARD_BINDING")
                        boundaries.append(dict(before=before, action=body, after=digest(session.snapshot()),
                                               same_action_condition_sensitivity=signatures))
                        continue
                    receipt = source.receipt(task, turn) if phase == 3 else None
                    execution = by_turn.get(turn) if phase == 3 and context == "train:1" else None
                    if execution is not None:
                        wrong = by_turn[11 if turn == 7 else 7]
                        refuse("wrong_forecast_"+str(turn), "TOOL_EXECUTION_FORECAST",
                               lambda: session.advance(context, receipt=receipt, execution=wrong))
                        saved_native = session.execution_inputs
                        session.execution_inputs = None
                        try:
                            refuse("missing_native_"+str(turn), "TOOL_EXECUTION_NATIVE_MISSING",
                                   lambda: session.advance(context, receipt=receipt, execution=execution))
                        finally:
                            session.execution_inputs = saved_native
                        request = session._request(context, receipt, None, execution)
                        original = inputs.observed(request["reference"], session.model.config, session.device)
                        archived, _, _ = session._prediction(request, original)
                        actual = native.observed(execution, session.model.config, session.device)
                        measured, _, _ = session._prediction(request, actual)
                        old_hidden = sha(archived["states"][0].hidden.cpu().numpy().tobytes())
                        new_hidden = sha(measured["states"][0].hidden.cpu().numpy().tobytes())
                        check(old_hidden != new_hidden, "FEEDBACK_ECHO_INSENSITIVE")
                        before = digest(session.snapshot())
                        body = session.advance(context, receipt=receipt, execution=execution)
                        check(body["hidden_sha256"] == new_hidden and session.execution_evidence(
                            next(reversed(session.executions)))["native_feedback_available"], "FEEDBACK_ECHO_NATIVE")
                        boundaries.append(dict(before=before, action=body, after=digest(session.snapshot()),
                                               archived_hidden=old_hidden, observed_hidden=new_hidden))
                    elif phase == 4 and context == "train:1" and turn in by_turn:
                        request = session._request(context, None, None)
                        check(request["execution_context"] == native.checked(by_turn[turn]),
                              "FEEDBACK_CORRECTION_BINDING")
                        original = inputs.observed(request["reference"], session.model.config, session.device)
                        archived, _, _ = session._prediction(request, original)
                        measured, _, _ = session._prediction(request, native.observed(
                            by_turn[turn], session.model.config, session.device))
                        old_hidden = sha(archived["states"][0].hidden.cpu().numpy().tobytes())
                        new_hidden = sha(measured["states"][0].hidden.cpu().numpy().tobytes())
                        check(old_hidden != new_hidden, "FEEDBACK_CORRECTION_INSENSITIVE")
                        before = digest(session.snapshot())
                        body = session.advance(context)
                        check(body["hidden_sha256"] == new_hidden, "FEEDBACK_CORRECTION_NATIVE")
                        boundaries.append(dict(before=before, action=body, after=digest(session.snapshot()),
                                               archived_hidden=old_hidden, observed_hidden=new_hidden))
                    else:
                        session.advance(context, receipt=receipt)
        phases[f"experience_{number}"] = time.perf_counter()-tick
        train = [k for k,v in session.forecasts.items() if v["record"]["request"]["context"] == "train:1"]
        evaluate = [k for k in session.forecasts if k not in train]
        tick = time.perf_counter()
        if a.grow_observed:
            from model import tool_context_regions, tool_observed_inputs
            historical = {k: tool_context_regions.historical(session, v["record"])[0].world_id
                          for k,v in session.forecasts.items()}
            before = session.snapshot()
            before_state = state_pin(session.states)
            refuse("withheld_growth_"+str(number), "TOOL_GROWTH_SPLIT",
                   lambda: session.grow("invalid-withheld", [evaluate[-1]]))
            growth = session.grow("observed-"+str(number), [train[-1]])
            unit = session.inputs.world_units[-1]
            response = session.routing.pairs[str(unit)]["response"]
            check(response["kind"] == "captured_process_output" and
                  session.bank[unit] == session.executions[train[-1]]["episode"]
                  and parameters(session.model) == initial_parameters
                  and state_pin(session.states) == before_state, "FEEDBACK_GROWTH_BINDING")
            for key,item in session.forecasts.items():
                view, units = tool_context_regions.historical(session, item["record"])
                check(view.world_id == historical[key] and unit not in units,
                      "FEEDBACK_GROWTH_NO_FUTURE")
            refuse("duplicate_growth_"+str(number), "TOOL_GROWTH_ALREADY_PRESENT",
                   lambda: session.grow("invalid-duplicate", [train[-1]]))
            bad = copy.deepcopy(response)
            bad["native"]["native"] = "0"*64
            changed = copy.copy(session.inputs)
            changed.observed_regions = {**session.inputs.observed_regions, str(unit):bad}
            refuse("binding_substitution_"+str(number), "TOOL_KERNEL_OBSERVED_BINDING",
                   lambda: tool_observed_inputs.validate(changed, str(unit), bad,
                               session.routing.pairs[str(unit)]["action"]))
            boundaries.append(dict(before=before, action=growth, after=session.snapshot()))
        training.append(learn(session, a.output, number, train, evaluate, refuse))
        snapshot = session.snapshot()
        session.close()
        session = open_session(expected_head=snapshot["head"], expected_contract=snapshot["contract"])
        check(session.snapshot() == snapshot, "FEEDBACK_REOPEN")
        phases[f"learning_replay_{number}"] = time.perf_counter()-tick
    check(parameters(session.model) == initial_parameters, "FEEDBACK_ACTIVE_MODEL_CHANGED")
    later = session.forecasts[train[-1]]["record"]["region_context"]["definition"]
    check(set(later["observed_echoes"]) == {"7"}, "FEEDBACK_CAUSAL_HISTORY")
    if a.grow_observed:
        tick = time.perf_counter()
        # Read subsequent actual source turns after both bank versions were committed.
        for exchange in range(5,7):
            for phase in range(5):
                for context, task in contexts.items():
                    session.advance(context, receipt=source.receipt(task, 3+2*exchange)
                                    if phase == 3 else None)
        phases["subsequent_forecasts"] = time.perf_counter()-tick
    tick = time.perf_counter()
    snapshot = session.snapshot()
    direct = readback(a.output/"experience.sqlite3", inputs, snapshot, a.observations, native)
    # A real persisted branch preserves the main learner while demonstrating retraction.
    branch_path = a.output/"withdrawal.sqlite3"
    with sqlite3.connect(branch_path) as db:
        session.journal.db.backup(db)
    original = session
    session = open_session(branch_path, expected_head=snapshot["head"], expected_contract=snapshot["contract"])
    before = digest(session.snapshot())
    withdrawal = (session.forecasts[train[0]]["record"]["request"]["execution_conditions"]["episode"]
                  if conditions is not None else session.executions[train[0]]["episode"])
    event = session.invalidate(withdrawal, reason="FSV condition or captured output withdrawn")
    affected = set(event["record"]["event"]["affected"])
    for item in session.forecasts.values():
        expected = bool(affected.intersection((*session.record_support(item["record"]),
                           item.get("target_episode"), item.get("execution", {}).get("episode"))))
        check(item["invalidated"] == expected, "FEEDBACK_WITHDRAWAL_SCOPE")
    refuse("withdrawn_context", "TOOL_CONTEXT_REVIEW", lambda: session.advance("train:1"))
    refuse("withdrawn_training", "TOOL_EXECUTION_UNAVAILABLE" if a.grow_observed else "TOOL_LEARNING_INELIGIBLE",
           lambda: ToolLearningData(session, train=train, evaluate=evaluate))
    branch_snapshot = session.snapshot()
    boundaries.append(dict(before=before, action=event, after=digest(branch_snapshot)))
    session.close()
    if a.grow_observed:
        session = original
        refuse("withdrawn_bank_cold", "TOOL_EXECUTION_UNAVAILABLE",
               lambda: open_session(branch_path, expected_head=branch_snapshot["head"],
                                    expected_contract=branch_snapshot["contract"]))
    else:
        session = open_session(branch_path, expected_head=branch_snapshot["head"], expected_contract=branch_snapshot["contract"])
        check(session.snapshot() == branch_snapshot, "FEEDBACK_WITHDRAWAL_REPLAY")
        session.close()
    session = original
    check(session.snapshot() == snapshot, "FEEDBACK_BRANCH_ISOLATION")
    session.close()
    phases["withdrawal_readback"] = time.perf_counter()-tick
    write(a.output/"snapshot.json", snapshot)
    write(a.output/"withdrawal-snapshot.json", branch_snapshot)
    write(a.output/"training.json", training)
    report = dict(complete=True, controls=controls, boundaries=boundaries, readback=direct, phases=phases,
        wall_seconds=time.perf_counter()-begin, scientific_admission=False, useful_capability=False,
        native_feedback_available=True, observed_training_outcomes=2, independently_executed_withheld_outcomes=0)
    if conditions is not None:
        report["native_condition_inputs"] = conditions.identity
    if a.grow_observed:
        from audit.process_growth_readback import readback as read_growth
        tick = time.perf_counter()
        report["growth_readback"] = read_growth(inputs, native, a.output, a.parent)
        phases["growth_readback"] = time.perf_counter()-tick
        report["wall_seconds"] = time.perf_counter()-begin
    report["timing_coverage"] = sum(phases.values())/report["wall_seconds"]
    check(report["timing_coverage"] >= .9, "FEEDBACK_TIMING")
    write(a.output/"report.json", report)
    print(json.dumps(dict(complete=True, controls=len(controls), wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
