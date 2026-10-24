"""Bounded correction from a saved pre-failure state, with exact recovery and rollback."""
import time

import torch

from claims.codata_source import check
from model.constellation_learning import trainable
from model.controlled_session import candidate_path
from model.corrective_learning import CorrectiveData
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model import corrective_step, field_checkpoint, rehearsal_retention


def increment(session, root, *, name, spec):
    started = time.perf_counter()
    data = CorrectiveData(session,spec)
    model, optimizer, migration = field_checkpoint.start(data)
    before = data.evaluate(model)
    check(before == data.baseline_scores, 'CORRECTION_START_CONTINUITY')
    parameters = tensor_fingerprint(dict(model.named_parameters()))
    initial = field_checkpoint.save(root/(name+'-start'),model,optimizer,data,step=0,parent=data.parent_pin)
    phases = dict(binding_and_baseline=time.perf_counter()-started)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        protected = rehearsal_retention.prepare(data,model)
        loss, details = data.loss(model)
        check(bool(torch.isfinite(loss)), 'CORRECTION_NONFINITE_LOSS')
        loss.backward()
        named = trainable(model)
        active = [n for n,p in named if p.grad is not None and bool((p.grad != 0).any())]
        check(active and all(n.startswith(('query.','place.')) for n in active), 'CORRECTION_GRADIENT_SCOPE')
        gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                              for _,p in named])
        torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
        old = {n:p.detach().clone() for n,p in named}
        optimizer.step()
        update = corrective_step.apply(data,model,old,protected,float(loss.detach()),gradient)
        return dict(loss=float(loss.detach()),details=details,gradients=active,update=update)

    tick = time.perf_counter()
    losses = []
    for cursor in range(1,data.plan['steps']+1):
        losses.append(step(model,optimizer))
        if cursor == 4:
            mid = field_checkpoint.save(root/(name+'-mid'),model,optimizer,data,step=cursor,parent=data.parent_pin)
    phases['fixed_updates'] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered, recovered_optimizer, _ = field_checkpoint.load(root/(name+'-mid'),data,expected=mid)
    for cursor in range(4,data.plan['steps']):
        check(step(recovered,recovered_optimizer) == losses[cursor], 'CORRECTION_UPDATE_REPLAY')
    check(fingerprint(recovered,recovered_optimizer) == fingerprint(model,optimizer), 'CORRECTION_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in parameters if parameters[n] != now[n]]
    check(all(n.startswith(('query.','place.')) for n in changed), 'CORRECTION_FROZEN_DRIFT')
    pin = field_checkpoint.save(candidate_path(session,name),model,optimizer,data,step=data.plan['steps'],parent=data.parent_pin)
    rollback, rollback_optimizer, _ = field_checkpoint.load(root/(name+'-start'),data,expected=initial)
    check(data.evaluate(rollback) == before, 'CORRECTION_ROLLBACK_BEHAVIOUR')
    result = session.consider_corrective(name,pin,spec)
    phases['readback_rollback_and_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,binding=data.binding,before=before,after=after,losses=losses,
        migration=migration,changed=changed,decision=result,exact_optimizer_recovery=True,
        rollback_fingerprint=fingerprint(rollback,rollback_optimizer),rollback_behaviour_equal=True,
        start_pin=initial,mid_pin=mid,candidate_pin=pin,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_count=sum(p.numel() for _,p in trainable(model)),
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall)
