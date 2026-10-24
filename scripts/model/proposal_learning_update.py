"""Bounded retained weight correction, optimizer recovery and exact rollback."""
import time

import torch

from claims.codata_source import check
from model.constellation_learning import trainable
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.proposal_learning import ProposalData
from model import corrective_step, field_checkpoint, rehearsal_retention


def step(data, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    protected = rehearsal_retention.prepare(data,model)
    loss,details = data.loss(model)
    check(bool(torch.isfinite(loss)),'PROPOSAL_LEARNING_NONFINITE')
    loss.backward()
    named = trainable(model)
    active = [n for n,p in named if p.grad is not None and bool((p.grad != 0).any())]
    check(active and all(n.startswith(('query.','place.')) for n in active),'PROPOSAL_LEARNING_GRADIENTS')
    gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                          for _,p in named])
    torch.nn.utils.clip_grad_norm_([p for _,p in named],1.)
    old = {n:p.detach().clone() for n,p in named}
    optimizer.step()
    update = corrective_step.apply(data,model,old,protected,float(loss.detach()),gradient)
    return dict(loss=float(loss.detach()),details=details,gradients=active,update=update)


def increment(session, root, *, name, spec):
    started = time.perf_counter()
    data = ProposalData(session,session.learning_base,spec)
    model,optimizer,migration = field_checkpoint.start(data)
    before = data.evaluate(model)
    check(all(r['retained'] for r in rehearsal_retention.comparisons(data,
        rehearsal_retention.scores(data,model))),'PROPOSAL_LEARNING_INITIAL_RETENTION')
    parameters = tensor_fingerprint(dict(model.named_parameters()))
    initial = field_checkpoint.save(root/(name+'-start'),model,optimizer,data,step=0,parent=data.parent_pin)
    phases = dict(binding_and_baseline=time.perf_counter()-started)
    losses = []
    tick = time.perf_counter()
    for cursor in range(1,data.plan['steps']+1):
        row = step(data,model,optimizer)
        losses.append(row)
        print(__import__('json').dumps(dict(step=cursor,accepted=row['update']['accepted'],
              objective=row['update']['objective_after'],retained=row['update']['retained'])),flush=True)
        if cursor == 4:
            mid = field_checkpoint.save(root/(name+'-mid'),model,optimizer,data,step=cursor,parent=data.parent_pin)
    phases['updates'] = time.perf_counter()-tick
    tick = time.perf_counter()
    recovered,recovered_optimizer,_ = field_checkpoint.load(root/(name+'-mid'),data,expected=mid)
    for cursor in range(4,data.plan['steps']):
        check(step(data,recovered,recovered_optimizer) == losses[cursor],'PROPOSAL_LEARNING_UPDATE_REPLAY')
    check(fingerprint(recovered,recovered_optimizer) == fingerprint(model,optimizer),
          'PROPOSAL_LEARNING_OPTIMIZER_RECOVERY')
    phases['optimizer_recovery'] = time.perf_counter()-tick
    tick = time.perf_counter()
    after = data.evaluate(model)
    now = tensor_fingerprint(dict(model.named_parameters()))
    changed = [n for n in parameters if parameters[n] != now[n]]
    check(all(n.startswith(('query.','place.')) for n in changed),'PROPOSAL_LEARNING_FROZEN_DRIFT')
    pin = field_checkpoint.save(root/name,model,optimizer,data,step=data.plan['steps'],parent=data.parent_pin)
    rollback,rollback_optimizer,_ = field_checkpoint.load(root/(name+'-start'),data,expected=initial)
    check(data.evaluate(rollback) == before,'PROPOSAL_LEARNING_ROLLBACK_BEHAVIOUR')
    binding = data.binding
    result = session.consider_proposal_learning(name,pin,spec)
    phases['readback_rollback_decision'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,binding=binding,before=before,after=after,losses=losses,migration=migration,
        changed=changed,decision=result,exact_optimizer_recovery=True,rollback_behaviour_equal=True,
        rollback_fingerprint=fingerprint(rollback,rollback_optimizer),start_pin=initial,mid_pin=mid,
        candidate_pin=pin,parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_count=sum(p.numel() for _,p in trainable(model)),phases=phases,wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall)
