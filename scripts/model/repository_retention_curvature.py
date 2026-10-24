"""Curvature-contracted repository updates; same projection and exact retention gates."""
import copy
import math
import time

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.loop_training_checkpoint import fingerprint
from model.repository_retention import Retention as BaseRetention, compare, measurements
from model.repository_retention import POLICY as BASE_POLICY
from model.retained_update import SCALES
from model.retention_projection import project

POLICY = dict(BASE_POLICY, rule='repository_curvature_retention_v1',
              scales='measured_curvature_contraction', trials=len(SCALES), safety_fraction=.5)


class Retention(BaseRetention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contract['policy'] = POLICY
        self.pin = digest(self.contract)


def contract(before,after,linear,scale):
    """field_retention.contract's rule applied to the retained named repository checks."""
    roots=[]
    for key in compare(before,after):
        if key not in linear:
            return None  # A discrete placement failure has no smooth contraction bound.
        slope=linear[key]
        curvature=(after[key]-before[key]-scale*slope)/(scale*scale)
        if not math.isfinite(curvature) or curvature<=0 or not math.isfinite(slope) or slope>=0:
            return None
        root=-slope/curvature  # Zero slack: compare against the pre-step value.
        if not math.isfinite(root) or root<=0:
            return None
        roots.append(root)
    check(roots,'RETENTION_CONTRACTION_WITHOUT_VIOLATION')
    next_scale=.5*min(scale,*roots)
    return next_scale if 0<next_scale<scale else None


def update(retention, model, optimizer, *, force_reject=False):
    started = time.perf_counter()
    params = tuple(p for p in model.parameters() if p.requires_grad)
    before = [p.detach().clone() for p in params]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state(model.core.initial.device)
    initial = fingerprint(model,optimizer)
    scores = retention.scores(model)
    with torch.no_grad(): objective = retention.objective(model)
    measured = time.perf_counter()
    gradients,order = retention.prepare(model,params)
    prepared = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for key in retention.new:
        loss = retention.data.losses(model,key,'learn_2')/len(retention.new)
        check(bool(torch.isfinite(loss)), 'RETENTION_LOSS_FINITE')
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params,POLICY['gradient_clip'])
    check(bool(torch.isfinite(norm)), 'RETENTION_GRADIENT_FINITE')
    accepted = False
    trials, projection = [], None
    try:
        optimizer.step()
        proposal_fingerprint = fingerprint(model,optimizer)
        displacement = torch.cat([(p.detach()-b).reshape(-1).double() for p,b in zip(params,before,strict=True)])
        delta,projection = project(gradients,displacement)
        linear = None if delta is None else dict(zip(order,(gradients@delta).tolist(),strict=True))
        projected = time.perf_counter()
        if delta is not None:
            with torch.no_grad():
                scale = 1.
                for _ in SCALES:
                    offset = 0
                    for p,b in zip(params,before,strict=True):
                        p.copy_((b.double()+scale*delta[offset:offset+p.numel()].reshape(p.shape)).to(p.dtype))
                        offset += p.numel()
                    actual = measurements(retention.data.evaluate(model,'learn_1'))
                    regressions = compare(scores,actual)
                    new_loss = retention.objective(model)
                    changed = any(not torch.equal(p,b) for p,b in zip(params,before,strict=True))
                    passed = not regressions and math.isfinite(new_loss) and new_loss < objective and changed
                    trials.append(dict(scale=scale,scores=actual,regressions=regressions,new_loss=new_loss,
                                       gate_passed=passed,forced_rejection=force_reject))
                    if passed and not force_reject:
                        accepted = True
                        break
                    next_scale = contract(scores,actual,linear,scale) if regressions else scale*.5
                    trials[-1]['next_scale'] = next_scale
                    if next_scale is None:
                        break
                    scale = next_scale
    finally:
        if not accepted:
            with torch.no_grad():
                for p,b in zip(params,before,strict=True):p.copy_(b)
            optimizer.load_state_dict(saved_optimizer)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng,model.core.initial.device)
        optimizer.zero_grad(set_to_none=True)
    after = fingerprint(model,optimizer)
    check(accepted or (after == initial and retention.scores(model) == scores), 'RETENTION_ATOMIC_ROLLBACK')
    if not accepted:
        check(torch.equal(cpu_rng,torch.get_rng_state()) and
              torch.equal(cuda_rng,torch.cuda.get_rng_state(model.core.initial.device)), 'RETENTION_RNG_ROLLBACK')
    final_scores = retention.scores(model)
    check(not compare(scores,final_scores), 'RETENTION_UPDATE_REGRESSION')
    ended = time.perf_counter()
    return dict(accepted=accepted,code='RETENTION_ACCEPTED' if accepted else 'RETENTION_NO_FEASIBLE_UPDATE',
        before=initial,after=after,before_scores=scores,after_scores=final_scores,new_loss_before=objective,
        proposal_fingerprint=proposal_fingerprint,directional=linear,
        constraints=order,gradient_bytes=gradients.numel()*gradients.element_size(),projection=projection,trials=trials,
        gradient_norm=float(norm),wall_seconds=ended-started,phases=dict(baseline=measured-started,
        gradients=prepared-measured,proposal_projection=projected-prepared,replay_rollback=ended-projected))
