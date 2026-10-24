"""Actual readout updates and conditional-placement boundaries in the growing loop."""
import base64

import numpy as np
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model import constellation_training_checkpoint as checkpoint
from model.constellation_learning import frozen_pin, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.tool_learning import ToolLearningData
from model.tool_learner import start
from audit.tool_native_recognition import controls as recognition_controls, inspect_start
from audit import tool_shared_objective
from audit import tool_causal_history
from audit import tool_region_risk
from model import retained_update
from audit import tool_retained_update
from audit.tool_retention_controls import controls as retention_controls


def learn(session, root, generation, train, evaluate, refuse):
    continuation = session.decisions[-1]["candidate"] if session.decisions else None
    data = ToolLearningData(session, train=train, evaluate=evaluate, continuation=continuation)
    model, optimizer, continuity = start(data)
    parent = session.active_checkpoint
    before = tensor_fingerprint(dict(model.named_parameters()))
    start_pin = checkpoint.save(root/f"start-{generation}", model, optimizer, data,
                                step=0, parent=parent)
    refuse(f"unknown_predecessor_{generation}", "TOOL_LEARNER_PREDECESSOR",
           lambda: ToolLearningData(session, train=train, evaluate=evaluate, continuation="unknown"))
    if continuation:
        previous = session.decisions[-1]
        for split, current in (("train", train), ("evaluate", evaluate)):
            reduced = [k for k in current if k != previous[split][0]]
            arguments = dict(train=train, evaluate=evaluate, continuation=continuation)
            arguments[split] = reduced
            refuse(f"lost_{split}_retention_{generation}", "TOOL_LEARNER_RETENTION",
                   lambda: ToolLearningData(session, **arguments))
        first_state = next(iter(optimizer.state.values()))
        saved_cursor = first_state["step"].clone()
        first_state["step"].zero_()
        refuse(f"reset_optimizer_cursor_{generation}", "LEARNING_CURSOR",
               lambda: checkpoint.validate(model, optimizer, data, 0))
        first_state["step"].copy_(saved_cursor)
        checkpoint.validate(model, optimizer, data, 0)
    supervision, recognition, objective, history, risks = [], [], [], [], []
    recognition_controls(data, model, train[0], generation, refuse)
    for key, e in data.examples.items():
        record = session.forecasts[key]["record"]
        recorded = list(range(len(e["units"]))) if "region_context" in record else record["routing"]["allowed"]
        check(e["allowed"].nonzero().flatten().tolist() == recorded, "GROWTH_HINDSIGHT_MASK")
    for key in train:
        e = data.examples[key]
        out = data.forward(model, key)
        recognition.append(inspect_start(data, model, key, out))
        objective.append(tool_shared_objective.inspect_start(data, model, key, out))
        history.append(tool_causal_history.inspect(data, model, key, out))
        risks.append(tool_region_risk.inspect(data, model, key, out))
        if key == train[0]:
            tool_shared_objective.controls(data, model, key, out, generation, refuse)
            tool_causal_history.controls(data, model, key, out, generation, refuse)
            tool_region_risk.controls(data, model, key, out, generation, refuse)
        view = data.prediction_model(model, key)
        label = torch.tensor([e["region"]], dtype=torch.long, device=session.device)
        ids = [out["states"][0].last_event]
        conditional = view.supervised_placement(out, label, allowed=e["allowed"], event_ids=ids)
        unit = e["units"][e["region"]]
        errors, sizes = {}, {}
        for lane, spec in enumerate(model.specs):
            x, ptr = session.inputs.arrays[spec.name]
            book = x[ptr[unit]:ptr[unit+1]].astype(np.float64)
            native = view.references()[spec.name]
            check(np.array_equal(book, native.values[e["region"], ~native.padding[e["region"]]]
                                 .detach().cpu().numpy()), "SUPERVISED_SOURCE_REGION")
            array = lambda t: t.detach().cpu().numpy().astype(np.float64)
            hidden = array(out["placement_hidden"][0, lane])
            projector, place = model.core.projectors[spec.name], model.place[spec.name]
            conditioned = hidden+book@array(projector.weight).T+array(projector.bias)
            residual = np.tanh(conditioned@array(place.weight).T+array(place.bias))
            residual = .1*residual/np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1.)
            expected = book/np.linalg.norm(book, axis=1, keepdims=True)+residual
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            slot = conditional.slots[spec.name]
            actual = array(slot.values[0, ~slot.padding[0]])
            check(actual.shape == expected.shape, "SUPERVISED_WINDOW_ROSTER")
            error = float(np.max(np.abs(expected-actual)))
            check(error < 1e-6, "SUPERVISED_INDEPENDENT_PLACEMENT")
            errors[spec.name], sizes[spec.name] = error, len(actual)
        supervision.append(dict(forecast=key, label=unit,
            free_region=out["predicted_constellation"].regions[0], windows=sizes,
            independent_max_error=errors, predictions=conditional.pack(model.specs, model.config),
            hidden=base64.b64encode(out["states"][0].hidden.detach().cpu().numpy().tobytes()).decode()))
        # The supervised path cannot change the already computed target-free output.
        before_output = out["predicted_constellation"].pack(model.specs, model.config)
        with torch.no_grad():
            repeat = data.forward(model, key)
            check(repeat["predicted_constellation"].pack(model.specs, model.config) == before_output,
                  "SUPERVISED_INFERENCE_UNCHANGED")
            refuse(f"no_grad_supervision_{generation}_{key[:8]}", "CONSTELLATION_SUPERVISION_TRAINING_ONLY",
                   lambda: view.supervised_placement(out, label, allowed=e["allowed"], event_ids=ids))
        mask = e["allowed"].clone()
        mask[label] = False
        refuse(f"masked_supervision_{generation}_{key[:8]}", "CONSTELLATION_SUPERVISION_CUTOFF",
               lambda: view.supervised_placement(out, label, allowed=mask, event_ids=ids))
        invalid = torch.tensor([len(e["units"])], dtype=torch.long, device=session.device)
        refuse(f"foreign_supervision_{generation}_{key[:8]}", "CONSTELLATION_SUPERVISION_REGION",
               lambda: view.supervised_placement(out, invalid, allowed=e["allowed"], event_ids=ids))
    losses = []
    def step(candidate, opt, control=False):
        opt.zero_grad(set_to_none=True)
        loss, details = data.loss(candidate)
        check(bool(torch.isfinite(loss)), "SUPERVISED_LOSS_NONFINITE")
        loss.backward()
        gradients = [n for n, p in candidate.named_parameters()
                     if p.grad is not None and bool((p.grad != 0).any())]
        check(gradients and all(n.startswith(("query.", "place.")) for n in gradients),
              "SUPERVISED_GRADIENT_BOUNDARY")
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable(candidate)], 1.)
        protected_gradients = retained_update.prepare(data, candidate)
        before_step = {n: p.detach().clone() for n, p in trainable(candidate)}
        if control:
            tool_retained_update.controls(data, candidate, before_step, protected_gradients, generation, refuse)
        opt.step()
        update = retained_update.apply(data, candidate, before_step, protected_gradients)
        return dict(loss=float(loss.detach()), details=details, gradients=gradients, retained_update=update)
    for number in range(1, data.plan["steps"]+1):
        losses.append(step(model, optimizer, control=number == 1))
        if number == 4:
            mid_pin = checkpoint.save(root/f"mid-{generation}", model, optimizer, data, step=4, parent=parent)
    recovered, recovered_opt, _ = checkpoint.load(root/f"mid-{generation}", data, expected=mid_pin)
    for number in range(4, data.plan["steps"]):
        check(step(recovered, recovered_opt) == losses[number], "SUPERVISED_OPTIMIZER_STEPS")
    check(fingerprint(model, optimizer) == fingerprint(recovered, recovered_opt), "SUPERVISED_OPTIMIZER_RESTART")
    after = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in before if before[n] != after[n]]
    accepted = sum(row["retained_update"]["accepted"] for row in losses)
    check((len(changed) == 24 if accepted else not changed)
          and all(n.startswith(("query.", "place.")) for n in changed)
          and frozen_pin(model) == data.frozen, "SUPERVISED_FROZEN_RETENTION")
    name = f"increment-{generation}"
    pin = checkpoint.save(root/"candidates"/name, model, optimizer, data, step=data.plan["steps"], parent=parent)
    decision = session.consider(name, pin=pin, train=train, evaluate=evaluate, parent=parent,
                                continuation=continuation)
    check(not decision["result"]["selected"], "SUPERVISED_UNADMITTED_SELECTION")
    current_payload = session.journal.payload(decision["payload"])
    previous = session.decisions[-2] if continuation else decision
    previous_payload = session.journal.payload(previous["payload"])
    retention_controls(previous_payload["candidate"], current_payload["candidate"], previous["evaluate"],
                       {s.name: s.identity for s in model.specs}, refuse, f"retention_{generation}")
    refuse(f"stale_training_{generation}", "TOOL_LEARNING_STALE", data.fresh)
    refuse(f"unadmitted_candidate_{generation}", "TOOL_CANDIDATE_NOT_SELECTED",
           lambda: session.require_selection(name))
    if continuation:
        refuse(f"nonlatest_predecessor_{generation}", "TOOL_LEARNER_PREDECESSOR",
               lambda: ToolLearningData(session, train=train, evaluate=evaluate,
                                        continuation=continuation))
    return dict(generation=generation, name=name, parent=parent, pin=pin, losses=losses,
                start_pin=start_pin, continuity=continuity, optimizer_offset=data.optimizer_offset,
                optimizer_steps=checkpoint.optimizer_steps(data, data.plan["steps"]),
                supervision=supervision, changed=changed, exact_optimizer_recovery=True,
                recognition=recognition, objective=objective, history=history,
                region_risk_objective=data.plan["region_risk"], region_risks=risks,
                accepted_parameter_updates=accepted, optimizer_attempts=len(losses),
                before_parameters=digest(before), after_parameters=digest(after), decision=decision)
