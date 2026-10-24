"""Real target/rival margins and their derivatives in retained-response directions."""
import numpy as np
import torch

from audit.proposal_recognition import trace
from claims.codata_source import check
from claims.episodic_memory import digest
from model.loop_training_checkpoint import fingerprint


def diagnose(data, model, optimizer, prior, stage, arrays):
    data.fresh()
    before = fingerprint(model,optimizer)
    scores = trace(data,model,prior,arrays,stage)
    basis = data.response_basis
    params = dict(model.named_parameters())
    names = basis.names
    projectors = basis.projectors_on(model.core.initial.device)
    layout, cursor = [], 0
    for n in names:
        p = params[n]
        layout.append(dict(name=n,start=cursor,stop=cursor+p.numel(),shape=list(p.shape)))
        cursor += p.numel()

    def vector(term, label):
        gradients = torch.autograd.grad(term,tuple(params[n] for n in names),allow_unused=False)
        raw = torch.cat([g.detach().reshape(-1).double() for g in gradients])
        projected = torch.cat([(g.detach().double()@projectors[n]).reshape(-1) for n,g in zip(names,gradients)])
        arrays[stage+'/gradient/'+label+'/raw'] = raw.cpu().numpy()
        arrays[stage+'/gradient/'+label+'/projected'] = projected.cpu().numpy()
        return raw,projected

    objective, _ = data.loss(model)
    raw_objective, projected_objective = vector(objective,'objective')
    norm = torch.linalg.vector_norm(projected_objective)
    check(bool(norm > 0), 'RESPONSE_MARGIN_ZERO_OBJECTIVE')
    direction = -projected_objective/norm
    margins = {}
    for key in data.new:
        e = data.examples[key]
        others = [k for k in data.new if k != key]
        check(len(others) == 1, 'RESPONSE_MARGIN_RIVAL_ROSTER')
        other = others[0]
        rival = data.examples[other]['region']
        output = data.forward(model,key)
        term = output['constellation_regions'][0,e['region']]-output['constellation_regions'][0,rival]
        raw, projected = vector(term,key)
        projected_norm = torch.linalg.vector_norm(projected)
        derivative = float(raw@direction)
        margins[key] = dict(target=e['region'],rival=rival,rival_key=other,margin=float(term.detach()),
            norm=float(torch.linalg.vector_norm(raw)),projected_norm=float(projected_norm),
            objective_descent_derivative=derivative,
            objective_descent_cosine=None if float(projected_norm) == 0 else derivative/float(projected_norm),
            input=scores[key]['input'],prestate=scores[key]['prestate'])
    check(fingerprint(model,optimizer) == before, 'RESPONSE_MARGIN_MUTATION')
    kept, response = basis.measure(data,model)
    return dict(stage=stage,trace=scores,layout=layout,margins=margins,objective=float(objective.detach()),
        objective_norm=float(torch.linalg.vector_norm(raw_objective)),projected_objective_norm=float(norm),
        retained_scores=kept,response_drift=response,fingerprint=before,binding=digest(data.binding),
        scope='local_derivative_on_exposed_feedback_not_transfer')
