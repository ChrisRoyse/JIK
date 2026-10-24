"""Disjoint native-instrument corrections with exact observed behavior retention."""
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model import field_retention, rehearsal_retention, precise_region_solve
from model import brier_region_targets as targets


def blocks(data, model):
    result = {s.name:[f'query.{b}.{s.name}.up.weight' for b in ('_native','_fields')]
              for s in model.specs}
    names = [n for block in result.values() for n in block]
    check(len(names) == len(set(names)) and set(names) == set(data.response_basis.names),
          'TYPED_CORRECTION_PARAMETER_PARTITION')
    return result


def cache_key(data, model):
    check(not any(m.training for m in model.modules()), 'BRIER_REGION_CACHE_REQUIRES_EVAL')
    check(digest(data.binding) == data.binding_pin, 'BRIER_REGION_CACHE_BINDING')
    return digest(dict(binding=data.binding_pin, rows=data.typed_row_pin,
                       state=tensor_fingerprint(model.state_dict())))


def directions(data, model, params, partition, values):
    key = cache_key(data,model)
    margins = {n:v.detach().clone() for n,v in values.items()}
    cached = data.direction_cache
    if cached is not None and cached['key'] == key:
        check(all(torch.equal(v,cached['margins'][n]) for n,v in margins.items()),
              'BRIER_REGION_CACHE_OUTPUT_CHANGED')
        data.cache_trace.append(dict(key=key, hit=True, gradients=cached['pins']))
        return cached
    data.direction_cache = None
    cached = None
    projectors = data.response_basis.projectors_on(next(iter(values.values())).device)
    matrices, certificates, deltas, deficits, pieces = {}, {}, {}, {}, {}
    last = list(partition)[-1]
    for lane, names in partition.items():
        rows = []
        for index, value in enumerate(values[lane]):
            gradients = torch.autograd.grad(value,tuple(params[n] for n in names),
                retain_graph=lane != last or index+1 < len(values[lane]))
            rows.append(torch.cat([(g.detach().double()@projectors[n]).flatten()
                                   for n,g in zip(names,gradients)]))
        matrix = torch.stack(rows)
        deficit = data.typed_floors[lane]-values[lane].detach()
        delta, certificate = precise_region_solve.solve(matrix,deficit)
        matrices[lane], deficits[lane] = matrix, deficit
        certificates[lane], deltas[lane] = certificate, delta
        offset = 0
        if delta is not None:
            for n in names:
                pieces[n] = delta[offset:offset+params[n].numel()].reshape(params[n].shape)
                offset += params[n].numel()
    pins = {n:digest(tensor_fingerprint(g)) for n,g in matrices.items()}
    cached = dict(key=key, margins=margins, matrices=matrices, certificates=certificates,
                  deltas=deltas, deficits=deficits, pieces=pieces, pins=pins)
    data.direction_cache = cached
    data.cache_trace.append(dict(key=key, hit=False, gradients=pins))
    return cached


def step(data, model, optimizer):
    data.fresh()
    scores, drift = data.response_basis.measure(data,model)
    check(all(v['retained'] for v in rehearsal_retention.comparisons(data,scores)),
          'TYPED_CORRECTION_INITIAL_RETENTION')
    original = targets.measurements(data,model)
    params = dict(trainable(model))
    before = {n:p.detach().clone() for n,p in params.items()}
    partition = blocks(data,model)
    values = targets.constraints(data,model)
    objective = targets.objective(data,{n:v.detach() for n,v in values.items()})
    direction = directions(data,model,params,partition,values)
    matrices, certificates, deltas, deficits, pieces = [direction[n] for n in
        ('matrices','certificates','deltas','deficits','pieces')]
    feasible = all(c['status'] in ('certified_minimum_norm','margins_satisfied')
                   for c in certificates.values())
    full, details = data.loss(model)
    optimizer.zero_grad(set_to_none=True)
    full.backward()
    torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    optimizer.step()  # Preserve all inherited clocks; replace the proposed parameter move.
    def place(scale):
        with torch.no_grad():
            for n,p in params.items():
                p.copy_((before[n].double()+scale*pieces[n]).to(p.dtype)
                        if scale and n in pieces else before[n])
    trials, best = [], None
    if feasible and pieces:
        for attempt in range(field_retention.POLICY['trials']):
            scale = field_retention.POLICY['safety_fraction']**attempt
            place(scale)
            actual, response = data.response_basis.measure(data,model)
            measured = targets.measurements(data,model)
            with torch.no_grad():
                current_values = targets.constraints(data,model)
            current = targets.objective(data,current_values)
            kept = all(v['retained'] for v in rehearsal_retention.comparisons(data,actual))
            new_kept = {k:targets.nonincreasing(original[k],measured[k]) for k in original}
            trials.append(dict(scale=scale,scores=actual,response_drift=response,
                measurements=measured,retained=kept,new_retained=new_kept,objective=current,
                objective_decreased=current < objective,
                margins={n:v.tolist() for n,v in current_values.items()}))
            if kept and all(new_kept.values()) and current < objective:
                best = len(trials)-1
                break
    scale = 0. if best is None else trials[best]['scale']
    place(scale)
    after, after_drift = data.response_basis.measure(data,model)
    measured = targets.measurements(data,model)
    accepted = best is not None and any(not torch.equal(p,before[n]) for n,p in params.items())
    with torch.no_grad():
        final_values = targets.constraints(data,model)
    current = targets.objective(data,final_values)
    check(all(v['retained'] for v in rehearsal_retention.comparisons(data,after))
          and all(targets.nonincreasing(original[k],measured[k]) for k in original),
          'TYPED_CORRECTION_ACTUAL_RETENTION')
    check((current < objective and after == trials[best]['scores']
           and measured == trials[best]['measurements']) if accepted else
          (current == objective and after == scores and measured == original),
          'TYPED_CORRECTION_ACTUAL_ACCEPTANCE')
    check(all(torch.equal(p,before[n]) for n,p in params.items() if n not in data.response_basis.names),
          'TYPED_CORRECTION_PARAMETER_SCOPE')
    recorder = getattr(data,'record_typed_correction',None)
    if recorder is not None:
        recorder(matrices,deficits,deltas,before,params,partition)
    update = dict(rule=targets.RULE,accepted=accepted,scale=scale,selected_trial=best,trials=trials,
        trial_limit=field_retention.POLICY['trials'],constraint_order=rehearsal_retention.ordered(data),
        before_scores=scores,after_scores=after,before_response_drift=drift,after_response_drift=after_drift,
        retained=True,objective_before=objective,objective_after=current,
        margins_before={n:v.detach().tolist() for n,v in values.items()},
        margins_after={n:v.tolist() for n,v in final_values.items()},certificates=certificates,
        gradients=direction['pins'],cache_key=direction['key'],
        before_measurements=original,after_measurements=measured,partition=partition,
        before=digest(tensor_fingerprint(before)),after=digest(tensor_fingerprint(params)),
        reason='accepted_typed_region_correction' if accepted else 'rejected_no_parameter_change')
    return dict(loss=float(full.detach()),details=details,update=update)
