"""Measured-curvature step contraction; exact retained checks remain authoritative."""
import math

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model.retention_projection import project
from model import rehearsal_retention as retained, retained_update


POLICY = dict(rule='retained_curvature_contraction_v1',trials=len(retained_update.SCALES),
              safety_fraction=.5,acceptance='original_exact_checks')


def validate(data):
    retained.validate(data)
    check(data.binding.get('field_update') == POLICY,'FIELD_UPDATE_POLICY')


def contract(data,before,measured,linear,scale):
    roots = []
    ceilings = data.binding['grounded_retention']['ceilings']
    for (key,kind,lane),slope in zip(retained.ordered(data),linear):
        start,end,ceiling = before[key][kind][lane],measured[key][kind][lane],ceilings[key][kind][lane]
        if end <= ceiling or (start > ceiling and end < start):
            continue
        slack = max(0.,ceiling-start)
        curvature = (end-start-scale*slope)/(scale*scale)
        if not all(math.isfinite(v) for v in (slope,curvature)) or curvature <= 0:
            return None
        discriminant = slope*slope+4*curvature*slack
        root = (-slope+math.sqrt(discriminant))/(2*curvature)
        if not math.isfinite(root) or root <= 0:
            return None
        roots.append(root)
    check(roots,'FIELD_UPDATE_CONTRACTION_WITHOUT_VIOLATION')
    next_scale = .5*min(scale,*roots)
    return next_scale if 0 < next_scale < scale else None


def apply(data,model,before,prepared):
    validate(data)
    params = dict(trainable(model))
    check(set(before) == set(params) and digest(tensor_fingerprint(before)) == prepared['before']
          and prepared['order'] == retained.ordered(data),'FIELD_UPDATE_PARAMETERS')
    check(data.retained_keys,'FIELD_UPDATE_RETENTION_REQUIRED')
    report = dict(rule=retained.RULE,policy=dict(POLICY),protected=len(prepared['order']),
        before=prepared['before'],parameter_coordinates=sum(p.numel() for p in params.values()),
        constraint_order=prepared['order'],before_scores=prepared['scores'],
        before_comparisons=retained.comparisons(data,prepared['scores']))
    g = prepared['gradients']
    check(g.shape == (report['protected'],report['parameter_coordinates']),'FIELD_UPDATE_GRADIENTS')
    displacement = torch.cat([(p.detach()-before[n]).reshape(-1).double() for n,p in params.items()])
    delta,projection = project(g,displacement)
    linear = None if delta is None else (g@delta).tolist()
    trials,accepted,scale = [],False,1.
    with torch.no_grad():
        if delta is not None:
            for _ in range(POLICY['trials']):
                start = 0
                for n,p in params.items():
                    p.copy_((before[n].double()+scale*delta[start:start+p.numel()].reshape(p.shape)).to(p.dtype))
                    start += p.numel()
                measured = retained.scores(data,model)
                passed = retained.advances(data,prepared['scores'],measured)
                trial = dict(scale=scale,scores=measured,advances=passed,
                             retained=all(r['retained'] for r in retained.comparisons(data,measured)))
                trials.append(trial)
                if passed:
                    accepted = any(not torch.equal(p,before[n]) for n,p in params.items())
                    break
                next_scale = contract(data,prepared['scores'],measured,linear,scale)
                trial['next_scale'] = next_scale
                if next_scale is None:
                    break
                scale = next_scale
        if not accepted:
            scale = 0.
            for n,p in params.items():
                p.copy_(before[n])
            check(retained.scores(data,model) == prepared['scores'],'FIELD_UPDATE_RESTORE')
    after = retained.scores(data,model)
    kept = all(r['retained'] for r in retained.comparisons(data,after))
    return dict(report,accepted=accepted,scale=scale,projections=projection,directional=linear,trials=trials,
        gradient_bytes=g.numel()*g.element_size(),after_scores=after,retained=kept,
        after=digest(tensor_fingerprint(params)),
        reason=('accepted_retained' if kept else 'accepted_repair_pending') if accepted else 'rejected_no_parameter_change')
