"""Bounded objective descent preserving retained branch responses algebraically."""
import math

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model import field_retention, rehearsal_retention as retained, response_basis


def step(data, model, optimizer):
    data.fresh()
    basis = data.response_basis
    scores, drift = basis.measure(data,model)
    check(all(r['retained'] for r in retained.comparisons(data,scores)), 'RESPONSE_INITIAL_RETENTION')
    optimizer.zero_grad(set_to_none=True)
    loss, details = data.loss(model)
    check(bool(torch.isfinite(loss)), 'RESPONSE_NONFINITE_LOSS')
    loss.backward()
    params = dict(trainable(model))
    before = {n:p.detach().clone() for n,p in params.items()}
    projectors = basis.projectors_on(next(iter(params.values())).device)
    gradients = {n:params[n].grad.detach().double().clone() for n in basis.names}
    projected = {n:g@projectors[n] for n,g in gradients.items()}
    length = torch.sqrt(sum(v.square().sum() for v in projected.values()))
    torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
    optimizer.step()
    adam_norm = torch.sqrt(sum((params[n].detach().double()-before[n].double()).square().sum()
                              for n in basis.names))
    check(bool(torch.isfinite(length)) and bool(torch.isfinite(adam_norm)), 'RESPONSE_DIRECTION_FINITE')
    delta = ({n:-v*(adam_norm/length) for n,v in projected.items()} if float(length) > 0 else None)
    slope = None if delta is None else float(sum((gradients[n]*v).sum() for n,v in delta.items()))
    objective = float(loss.detach())
    trials, best, scale, lower, upper = [], None, 1., None, None

    def place(value):
        with torch.no_grad():
            for n,p in params.items():
                p.copy_((before[n].double()+value*delta[n]).to(p.dtype)
                        if value and n in delta else before[n])

    if delta is not None and slope < 0:
        for _ in range(field_retention.POLICY['trials']):
            place(scale)
            measured, response = basis.measure(data,model)
            current_loss, current_details = data.loss(model)
            current = float(current_loss.detach())
            check(math.isfinite(current), 'RESPONSE_OBJECTIVE_FINITE')
            kept = all(r['retained'] for r in retained.comparisons(data,measured))
            trial = dict(scale=scale,scores=measured,response_drift=response,retained=kept,
                objective=current,objective_decreased=current < objective,
                regions={k:next(iter(v.values()))['inference_region'] for k,v in current_details.items()})
            trials.append(trial)
            if kept and current < objective:
                if best is None or current < trials[best]['objective']:
                    best = len(trials)-1
                lower = scale
                if scale == 1.:
                    break
            else:
                upper = scale
            if len(trials) == field_retention.POLICY['trials']:
                break
            if lower is not None:
                value, rule = math.sqrt(lower*upper), 'geometric_bracket'
            else:
                values = [field_retention.POLICY['safety_fraction']*scale]
                if current >= objective:
                    curvature = (current-objective-scale*slope)/(scale*scale)
                    if math.isfinite(curvature) and curvature > 0:
                        values.append(.5*min(scale,-slope/curvature))
                value, rule = min(values), 'bounded_contraction'
            check(0 < value < (upper if lower is not None else scale), 'RESPONSE_SEARCH_BRACKET')
            trial.update(next_scale=value,next_rule=rule)
            scale = value
    scale = 0. if best is None else trials[best]['scale']
    place(scale)
    accepted = best is not None and any(not torch.equal(p,before[n]) for n,p in params.items())
    after, after_drift = basis.measure(data,model)
    current = float(data.loss(model)[0].detach())
    kept = all(r['retained'] for r in retained.comparisons(data,after))
    check(kept and (current < objective if accepted else current == objective), 'RESPONSE_ACTUAL_ACCEPTANCE')
    check((after == trials[best]['scores'] and after_drift == trials[best]['response_drift']
           and current == trials[best]['objective']) if accepted else after == scores, 'RESPONSE_RESTORE')
    check(all(torch.equal(p,before[n]) for n,p in params.items() if n not in basis.names),
          'RESPONSE_UPDATE_SCOPE')
    projection = dict(rule=response_basis.RULE, eligible=basis.names, adam_norm=float(adam_norm),
        projected_gradient_norm=float(length), objective_direction=slope,
        norm=0. if delta is None else float(torch.sqrt(sum(v.square().sum() for v in delta.values()))),
        algebraic_response={n:float(torch.max(torch.abs(
            torch.from_numpy(basis.arrays[f'h_{i}']).to(params[n].device).double()@delta[n].T)))
            for i,n in enumerate(basis.names)} if delta is not None else {},
        status='projected_objective' if delta is not None else 'zero_projected_gradient')
    update = dict(rule=response_basis.RULE,accepted=accepted,scale=scale,selected_trial=best,
        projection=projection,trials=trials,trial_limit=field_retention.POLICY['trials'],
        constraint_order=retained.ordered(data),before_scores=scores,after_scores=after,
        before_response_drift=drift,after_response_drift=after_drift,retained=kept,
        objective_before=objective,objective_after=current,
        before=digest(tensor_fingerprint(before)),after=digest(tensor_fingerprint(params)),
        reason='accepted_retained_descent' if accepted else 'rejected_no_parameter_change')
    return dict(loss=objective,details=details,update=update)
