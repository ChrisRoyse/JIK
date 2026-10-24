"""Objective projection into available retention slack; actual ceilings stay exact."""
import math

import numpy as np
from scipy.optimize import minimize
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model import field_retention, rehearsal_retention as retained

RULE = 'objective_projection_fixed_ceiling_slack_v1'


def refine(gram, rhs, weights, tolerance, budget):
    weights = weights.copy()
    weights[weights <= tolerance] = 0.
    active = list(np.flatnonzero(weights > 0))
    trace = []
    for _ in range(budget):
        if not active:
            if np.max(rhs) <= tolerance:
                return weights, trace, 'certified_face'
            active.append(int(np.argmax(rhs)))
        solution, _, rank, _ = np.linalg.lstsq(gram[np.ix_(active, active)], rhs[active], rcond=None)
        candidate = np.zeros(len(weights))
        candidate[active] = solution
        negative = np.flatnonzero(candidate < 0)
        trace.append(dict(active=len(active), rank=int(rank), negative=len(negative)))
        if len(negative):
            ratios = weights[negative]/(weights[negative]-candidate[negative])
            dropped = int(negative[np.argmin(ratios)])
            weights += float(ratios.min())*(candidate-weights)
            weights[dropped] = 0.
            active.remove(dropped)
            if np.min(weights) < 0:
                return weights, trace, 'negative_multiplier'
            continue
        weights = candidate
        residual = gram@weights-rhs
        if np.min(residual) >= -tolerance and np.max(np.abs(weights*residual)) <= tolerance:
            return weights, trace, 'certified_face'
        entered = int(np.argmin(residual))
        if entered in active:
            return weights, trace, 'face_residual'
        active.append(entered)
    return weights, trace, 'iteration_limit'


def project(gradients, proposal, slack):
    check(gradients.ndim == 2 and proposal.ndim == slack.ndim == 1
          and gradients.shape == (len(slack), len(proposal))
          and all(bool(torch.isfinite(v).all()) for v in (gradients, proposal, slack))
          and bool((slack >= 0).all()), 'SLACK_PROJECTION_INPUT')
    norms = torch.linalg.vector_norm(gradients, dim=1)
    active = norms > 0
    g = gradients[active]/norms[active, None]
    bounds = field_retention.POLICY['safety_fraction']*slack[active]/norms[active]
    gram = (g@g.T).cpu().numpy()
    raw = g@proposal
    rhs = (raw-bounds).cpu().numpy()
    count = len(g)
    report = dict(rule=RULE, active=active.tolist(), norms=norms.tolist(), slack=slack.tolist(),
        bounds=bounds.tolist(), gram=gram.tolist(), rhs=rhs.tolist(), raw_direction=raw.tolist(),
        proposal_norm=float(torch.linalg.vector_norm(proposal)), iterations_limit=3*count)
    if not count:
        return proposal, dict(report, status='unconstrained')
    tolerance = 64*np.finfo(np.float64).eps*max(1., float(np.linalg.norm(gram, np.inf)), float(np.max(np.abs(rhs))))
    solved = minimize(lambda w:.5*float(w@gram@w)-float(rhs@w), np.zeros(count),
        jac=lambda w:gram@w-rhs, method='SLSQP', bounds=[(0., None)]*count,
        options=dict(maxiter=3*count, ftol=tolerance))
    weights, trace, status = refine(gram, rhs, solved.x, tolerance, max(0, 3*count-int(solved.nit)))
    residual = gram@weights-rhs
    report.update(weights=weights.tolist(), residual=residual.tolist(), tolerance=tolerance,
        solver_success=bool(solved.success), solver_message=str(solved.message),
        iterations=int(solved.nit)+len(trace), solver_iterations=int(solved.nit), refinement=trace,
        refinement_status=status)
    if not solved.success or status != 'certified_face':
        return None, dict(report, status='solver_failed')
    if not (np.isfinite(weights).all() and np.min(weights) >= 0 and np.min(residual) >= -tolerance
            and np.max(np.abs(weights*residual)) <= tolerance):
        return None, dict(report, status='dual_certificate_failed')
    delta = proposal-g.T@torch.from_numpy(weights).to(proposal.device)
    linear = g@delta
    length = float(torch.linalg.vector_norm(delta))
    report.update(direction=linear.tolist(), norm=length)
    if not bool(torch.isfinite(delta).all()) or bool((linear > bounds+tolerance).any()):
        return None, dict(report, status='primal_certificate_failed')
    if length > report['proposal_norm']+tolerance:
        return None, dict(report, status='norm_certificate_failed')
    return delta, dict(report, status='certified_slack_projection')


def contract(data, before, measured, linear, scale):
    roots = []
    ceilings = data.binding['grounded_retention']['ceilings']
    for (key, kind, lane), slope in zip(retained.ordered(data), linear):
        start, end = before[key][kind][lane], measured[key][kind][lane]
        ceiling = ceilings[key][kind][lane]
        if end <= ceiling:
            continue
        slack = ceiling-start
        curvature = (end-start-scale*slope)/(scale*scale)
        if slack < 0 or not math.isfinite(curvature) or curvature <= 0:
            return None
        discriminant = math.sqrt(slope*slope+4*curvature*slack)
        root = (2*slack/(discriminant+slope) if slope >= 0 and discriminant+slope > 0
                else (discriminant-slope)/(2*curvature))
        if not math.isfinite(root) or root <= 0:
            return None
        roots.append(root)
    check(roots, 'SLACK_CONTRACTION_WITHOUT_VIOLATION')
    value = field_retention.POLICY['safety_fraction']*min(scale, *roots)
    return value if 0 < value < scale else None


def apply(data, model, before, prepared, objective, gradient):
    data.fresh()
    params = dict(trainable(model))
    check(set(params) == set(before) and digest(tensor_fingerprint(before)) == prepared['before']
          and prepared['order'] == retained.ordered(data), 'SLACK_UPDATE_BINDING')
    rows = retained.comparisons(data, prepared['scores'])
    check(all(r['retained'] for r in rows), 'SLACK_INITIAL_RETENTION')
    displacement = torch.cat([(p.detach()-before[n]).reshape(-1).double() for n,p in params.items()])
    norm = torch.linalg.vector_norm(gradient)
    check(bool(norm > 0), 'SLACK_ZERO_GRADIENT')
    proposal = -gradient*(torch.linalg.vector_norm(displacement)/norm)
    slack = gradient.new_tensor([r['ceiling']-r['value'] for r in rows])
    delta, projection = project(prepared['gradients'], proposal, slack)
    linear = None if delta is None else (prepared['gradients']@delta).tolist()
    slope = None if delta is None else float(gradient@delta)
    trials, best, scale, lower, upper = [], None, 1., None, None

    def place(value):
        cursor = 0
        with torch.no_grad():
            for n,p in params.items():
                p.copy_(before[n] if value == 0 else (before[n].double()+
                    value*delta[cursor:cursor+p.numel()].reshape(p.shape)).to(p.dtype))
                cursor += p.numel()

    if delta is not None and slope < 0:
        for _ in range(field_retention.POLICY['trials']):
            place(scale)
            measured = retained.scores(data, model)
            loss, details = data.loss(model)
            current = float(loss.detach())
            check(math.isfinite(current), 'SLACK_OBJECTIVE_FINITE')
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
                check(upper is not None and 0 < lower < upper <= 1., 'SLACK_SEARCH_BRACKET')
                value = math.sqrt(lower*upper)
                rule = 'geometric_bracket'
            else:
                values = []
                if not kept:
                    value = contract(data, prepared['scores'], measured, linear, scale)
                    if value is None:
                        break
                    values.append(value)
                if current >= objective:
                    curvature = (current-objective-scale*slope)/(scale*scale)
                    if not math.isfinite(curvature) or curvature <= 0:
                        break
                    values.append(.5*min(scale, -slope/curvature))
                value, rule = min(values), 'curvature_contraction'
            if not 0 < value < (upper if lower is not None else scale):
                break
            trial.update(next_scale=value, next_rule=rule)
            scale = value
    scale = 0. if best is None else trials[best]['scale']
    place(scale)
    accepted = best is not None and any(not torch.equal(p, before[n]) for n,p in params.items())
    after = retained.scores(data, model)
    current = float(data.loss(model)[0].detach())
    kept = all(r['retained'] for r in retained.comparisons(data, after))
    check(kept and (current < objective if accepted else current == objective), 'SLACK_ACTUAL_ACCEPTANCE')
    check((after == trials[best]['scores'] and current == trials[best]['objective']) if accepted
          else after == prepared['scores'], 'SLACK_RESTORE')
    return dict(rule=RULE, accepted=accepted, retained=kept, scale=scale, projection=projection,
        selected_trial=best, trial_limit=field_retention.POLICY['trials'], constraints=len(rows),
        constraint_order=prepared['order'],
        objective_before=objective, objective_after=current, projected_direction=linear,
        objective_direction=slope, before_scores=prepared['scores'], after_scores=after, trials=trials,
        before=prepared['before'], after=digest(tensor_fingerprint(params)),
        reason='accepted_retained_descent' if accepted else 'rejected_no_parameter_change')
