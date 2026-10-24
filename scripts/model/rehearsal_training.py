"""Fixed-step rehearsal with retained updates and exact Adam checkpoint replay."""
import time

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import frozen_pin, trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.tool_rehearsal import RehearsalData
from model import constellation_training_checkpoint as checkpoint
from model import rehearsal_retention, tool_learner


def increment(session, root, *, name, train, evaluate, continuation=None):
    started = time.perf_counter()
    data = RehearsalData(session, train=train, evaluate=evaluate, continuation=continuation)
    model, optimizer, continuity = tool_learner.start(data)
    parent = session.active_checkpoint if continuation is None else session.decisions[-1]["pin"]
    before = data.evaluate(model)
    original = tensor_fingerprint(dict(model.named_parameters()))
    checkpoint.save(root/(name+"-start"), model, optimizer, data, step=0, parent=parent)
    phases = dict(bind_and_baseline=time.perf_counter()-started)

    def step(candidate, opt):
        opt.zero_grad(set_to_none=True)
        protected = rehearsal_retention.prepare(data, candidate)
        loss, details = data.loss(candidate)
        check(bool(torch.isfinite(loss)), "REHEARSAL_NONFINITE_LOSS")
        loss.backward()
        gradients = [n for n,p in candidate.named_parameters()
                     if p.grad is not None and bool((p.grad != 0).any())]
        check(gradients and all(n.startswith(("query.", "place.")) for n in gradients),
              "REHEARSAL_GRADIENT_SCOPE")
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable(candidate)], 1.)
        saved = {n:p.detach().clone() for n,p in trainable(candidate)}
        opt.step()
        update = rehearsal_retention.apply(data, candidate, saved, protected)
        return dict(loss=float(loss.detach()), details=details, gradients=gradients, update=update)

    tick = time.perf_counter()
    losses = []
    for cursor in range(1,data.plan["steps"]+1):
        losses.append(step(model, optimizer))
        if cursor == 4:
            mid = checkpoint.save(root/(name+"-mid"), model, optimizer, data, step=cursor, parent=parent)
    phases["fixed_training"] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered, recovered_opt, _ = checkpoint.load(root/(name+"-mid"), data, expected=mid)
    for cursor in range(4,data.plan["steps"]):
        check(step(recovered, recovered_opt) == losses[cursor], "REHEARSAL_OPTIMIZER_REPLAY")
    check(fingerprint(model, optimizer) == fingerprint(recovered, recovered_opt),
          "REHEARSAL_OPTIMIZER_RECOVERY")
    phases["optimizer_recovery"] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    current = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in original if current[n] != original[n]]
    check(all(n.startswith(("query.", "place.")) for n in changed)
          and frozen_pin(model) == data.frozen, "REHEARSAL_FROZEN_DRIFT")
    pin = checkpoint.save(session.candidate_root/name, model, optimizer, data,
                          step=data.plan["steps"], parent=parent)
    decision = session.consider_rehearsal(name, pin=pin, train=train, evaluate=evaluate,
                                         parent=parent, continuation=continuation)
    phases["candidate_readback_and_decision"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(name=name, binding=data.binding, before=before, after=after, losses=losses,
        continuity=continuity, optimizer_steps=data.optimizer_offset+data.plan["steps"],
        changed=changed, exact_optimizer_recovery=True, decision=decision,
        source_head=data.head, current_head=session.head, phases=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, binding_pin=digest(data.binding))
