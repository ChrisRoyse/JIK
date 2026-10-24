"""Correct selected meaning regions while retaining exact prior behavior."""
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model import field_retention, rehearsal_retention, region_correction_solve

RULE = 'native_cost_region_margin_correction_v1'


def margins(data, model):
    gaps,rows = [],[]
    for key in data.new:
        e = data.examples[key]
        output = data.forward(model,key)
        scores = output['constellation_regions'][0]
        costs = torch.stack([e['region_costs'][s.name][1] for s in model.specs]).max(0).values
        for rival in e['allowed'].nonzero().flatten().tolist():
            if rival == e['region']:
                continue
            gaps.append(scores[e['region']]-scores[rival])
            rows.append(dict(key=key,target=e['region'],rival=rival,cost=float(costs[rival])))
    return torch.stack(gaps),rows


def native(data, model):
    evaluated = data.evaluate(model)
    return {key:{n:v['prediction_loss'] for n,v in evaluated[key]['assessment']['instruments'].items()}
            for key in data.new}


def violation(gaps, rows):
    costs = gaps.new_tensor([r['cost'] for r in rows])
    return float((costs-gaps).clamp_min(0).sum())


def step(data, model, optimizer):
    data.fresh()
    scores,drift = data.response_basis.measure(data,model)
    check(all(v['retained'] for v in rehearsal_retention.comparisons(data,scores)),'REGION_INITIAL_RETENTION')
    original_native = native(data,model)
    params = dict(trainable(model))
    names = data.response_basis.names
    before = {n:p.detach().clone() for n,p in params.items()}
    gaps,rows = margins(data,model)
    objective = violation(gaps.detach(),rows)
    matrices = []
    projectors = data.response_basis.projectors_on(gaps.device)
    for i,gap in enumerate(gaps):
        values = torch.autograd.grad(gap,tuple(params[n] for n in names),retain_graph=i+1 < len(gaps))
        matrices.append(torch.cat([(g.detach().double()@projectors[n]).flatten() for n,g in zip(names,values)]))
    gradients = torch.stack(matrices)
    deficit = gaps.detach().double().new_tensor([r['cost'] for r in rows])-gaps.detach().double()
    delta,certificate = region_correction_solve.solve(gradients,deficit)
    full,details = data.loss(model)
    optimizer.zero_grad(set_to_none=True)
    full.backward()
    torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    optimizer.step()  # Preserve the inherited full optimizer clock; correction replaces its proposal.
    pieces,offset = {},0
    if delta is not None:
        for n in names:
            pieces[n] = delta[offset:offset+params[n].numel()].reshape(params[n].shape)
            offset += params[n].numel()
    def place(scale):
        with torch.no_grad():
            for n,p in params.items():
                p.copy_((before[n].double()+scale*pieces[n]).to(p.dtype) if scale and n in pieces else before[n])
    trials,best = [],None
    if delta is not None:
        for attempt in range(field_retention.POLICY['trials']):
            scale = field_retention.POLICY['safety_fraction']**attempt
            place(scale)
            actual,response = data.response_basis.measure(data,model)
            current_native = native(data,model)
            with torch.no_grad():
                current_gaps,current_rows = margins(data,model)
            check(current_rows == rows,'REGION_MARGIN_ROSTER_CHANGED')
            current = violation(current_gaps,rows)
            kept = all(v['retained'] for v in rehearsal_retention.comparisons(data,actual))
            new_kept = all(v <= original_native[k][n] for k,lanes in current_native.items() for n,v in lanes.items())
            trials.append(dict(scale=scale,scores=actual,response_drift=response,native=current_native,
                retained=kept,new_native_retained=new_kept,objective=current,objective_decreased=current < objective,
                margins=current_gaps.tolist()))
            if kept and new_kept and current < objective:
                best = len(trials)-1
                break
    scale = 0. if best is None else trials[best]['scale']
    place(scale)
    after,after_drift = data.response_basis.measure(data,model)
    actual_native = native(data,model)
    accepted = best is not None and any(not torch.equal(p,before[n]) for n,p in params.items())
    with torch.no_grad():
        final_gaps,final_rows = margins(data,model)
    current = violation(final_gaps,final_rows)
    check(all(v['retained'] for v in rehearsal_retention.comparisons(data,after))
          and all(v <= original_native[k][n] for k,lanes in actual_native.items() for n,v in lanes.items()),
          'REGION_ACTUAL_RETENTION')
    check((current < objective and after == trials[best]['scores'] and actual_native == trials[best]['native'])
          if accepted else (current == objective and after == scores and actual_native == original_native),
          'REGION_ACTUAL_ACCEPTANCE')
    check(all(torch.equal(p,before[n]) for n,p in params.items() if n not in names),'REGION_PARAMETER_SCOPE')
    recorder = getattr(data,'record_correction',None)
    if recorder is not None:
        recorder(gradients,deficit,delta,before,params,names)
    update = dict(rule=RULE,accepted=accepted,scale=scale,selected_trial=best,trials=trials,
        trial_limit=field_retention.POLICY['trials'],constraint_order=rehearsal_retention.ordered(data),
        before_scores=scores,after_scores=after,before_response_drift=drift,after_response_drift=after_drift,
        retained=True,objective_before=objective,objective_after=current,rows=rows,margins_before=gaps.detach().tolist(),
        margins_after=final_gaps.tolist(),certificate=certificate,gradient=digest(tensor_fingerprint(gradients)),
        before_native=original_native,after_native=actual_native,
        before=digest(tensor_fingerprint(before)),after=digest(tensor_fingerprint(params)),
        reason='accepted_native_region_correction' if accepted else 'rejected_no_parameter_change')
    return dict(loss=float(full.detach()),details=details,update=update)
