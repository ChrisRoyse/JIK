"""Retain every grounded training slot and require actual objective decrease."""
import math

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model.corrective_projection import project
from model import field_retention, rehearsal_retention as retained


def apply(data, model, before, prepared, objective, gradient):
    data.fresh()
    params = dict(trainable(model))
    check(set(before) == set(params) and digest(tensor_fingerprint(before)) == prepared['before']
          and prepared['order'] == retained.ordered(data), 'CORRECTION_UPDATE_BINDING')
    g = torch.cat((prepared['gradients'], gradient[None,:]), dim=0)
    displacement = torch.cat([(p.detach()-before[n]).reshape(-1).double() for n,p in params.items()])
    check(g.shape == (len(prepared['order'])+1,len(displacement)), 'CORRECTION_GRADIENT_SHAPE')
    gradient_norm = torch.linalg.vector_norm(gradient)
    check(bool(gradient_norm > 0), 'CORRECTION_ZERO_OBJECTIVE_GRADIENT')
    proposal = -gradient*(torch.linalg.vector_norm(displacement)/gradient_norm)
    delta, projection = project(g, proposal)
    linear = None if delta is None else (g@delta).tolist()
    trials, best, scale, lower, upper = [], None, 1., None, None

    def place(value):
        cursor = 0
        with torch.no_grad():
            for n,p in params.items():
                if value == 0:
                    p.copy_(before[n])
                else:
                    p.copy_((before[n].double()+value*delta[cursor:cursor+p.numel()].reshape(p.shape)).to(p.dtype))
                cursor += p.numel()

    if delta is not None:
        for _ in range(field_retention.POLICY['trials']):
            place(scale)
            measured = retained.scores(data, model)
            loss, details = data.loss(model)
            current = float(loss.detach())
            check(math.isfinite(current), 'CORRECTION_OBJECTIVE_FINITE')
            kept = all(r['retained'] for r in retained.comparisons(data, measured))
            trial = dict(scale=scale, scores=measured, retained=kept, objective=current,
                         objective_decreased=current < objective,
                         regions={k:next(iter(v.values()))['inference_region'] for k,v in details.items()})
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
                check(upper is not None and 0 < lower < upper <= 1., 'CORRECTION_SEARCH_BRACKET')
                next_scale = math.sqrt(lower*upper)
                if not lower < next_scale < upper:
                    break
                trial.update(next_scale=next_scale,next_rule='geometric_bracket',lower=lower,upper=upper)
                scale = next_scale
                continue
            proposals = []
            if not kept:
                contraction = field_retention.contract(data,prepared['scores'],measured,linear[:-1],scale)
                if contraction is None:
                    break
                proposals.append(contraction)
            if current >= objective:
                slope = linear[-1]
                curvature = (current-objective-scale*slope)/(scale*scale)
                if slope >= 0 or not math.isfinite(curvature) or curvature <= 0:
                    break
                proposals.append(.5*min(scale,-slope/curvature))
            next_scale = min(proposals)
            if not 0 < next_scale < scale:
                break
            trial.update(next_scale=next_scale,next_rule='curvature_contraction')
            scale = next_scale
    scale = 0. if best is None else trials[best]['scale']
    place(scale)
    accepted = best is not None and any(not torch.equal(p,before[n]) for n,p in params.items())
    if not accepted:
        check(retained.scores(data,model) == prepared['scores'], 'CORRECTION_REJECT_RESTORE')
    after = retained.scores(data,model)
    after_objective = float(data.loss(model)[0].detach())
    kept = all(r['retained'] for r in retained.comparisons(data,after))
    check(kept and (after_objective < objective if accepted else after_objective == objective),
          'CORRECTION_ACCEPTANCE')
    if accepted:
        check(after == trials[best]['scores'] and after_objective == trials[best]['objective'],
              'CORRECTION_BEST_TRIAL_RESTORE')
    return dict(accepted=accepted, retained=kept, scale=scale, projection=projection,
        proposal_rule='common_descent_at_inherited_adam_proposal_norm',
        search_rule='bounded_geometric_feasibility_bracket_v1',selected_trial=best,
        trial_limit=field_retention.POLICY['trials'],
        constraints=len(prepared['order']), objective_before=objective, objective_after=after_objective,
        raw_objective_direction=float(gradient@displacement), projected_direction=linear,
        before_scores=prepared['scores'], after_scores=after, trials=trials,
        before=prepared['before'], after=digest(tensor_fingerprint(params)),
        gradient_bytes=g.numel()*g.element_size(), reason='accepted_retained_descent' if accepted else 'rejected_no_parameter_change')
