"""Transactional repository updates constrained by each retained native/outcome score."""
import copy
import math
import time

import torch
from torch.nn import functional as F

from claims.codata_source import check
from claims.episodic_memory import digest
from model.loop_training_checkpoint import fingerprint, tensor_fingerprint
from model.retained_update import MAX_RETAINED, SCALES
from model.retention_projection import project

POLICY = dict(rule='repository_exact_retention_v1', attempts=8, scales=list(SCALES),
    proposal='AdamW', learning_rate=.001, weight_decay=.01, gradient_clip=1.,
    projection='retention_projection.project', acceptance='all_exact_checks_and_new_loss_decrease',
    retained_partition='learn_1', new_partition='learn_2', tolerance=0.)
METRICS = ('brier', 'native_expected_loss', 'native_placement_loss')


def measurements(rows):
    result = {}
    for key, row in rows.items():
        for name, lane in row['lanes'].items():
            for metric in METRICS:
                if metric in lane:
                    value = lane[metric]
                    check(type(value) in (float, int) and math.isfinite(value), 'RETENTION_SCORE_FINITE')
                    result[key+'|'+name+'|'+metric] = float(value)
    return result


def compare(ceiling, measured):
    check(ceiling and set(ceiling) == set(measured), 'RETENTION_SCORE_ROSTER')
    check(all(math.isfinite(v) for v in (*ceiling.values(), *measured.values())), 'RETENTION_SCORE_FINITE')
    return {k:dict(before=v, after=measured[k], delta=measured[k]-v)
            for k,v in ceiling.items() if measured[k] > v}


class Retention:
    def __init__(self, data, model, reference, predecessor):
        self.data = data
        self.keys = data.stage('learn_1', model)['keys']
        self.new = [k for k in data.stage('learn_2', model)['keys'] if k not in self.keys]
        check(0 < len(self.keys) <= MAX_RETAINED and self.new, 'RETENTION_SPLIT')
        check(all(data.examples[k]['partition'] == 'learn_2' for k in self.new), 'RETENTION_SPLIT')
        current = data.evaluate(model, 'learn_1')
        check(current == reference, 'RETENTION_BASELINE_REPLAY')
        self.ceiling = measurements(reference)
        self.contract = dict(policy=POLICY, binding=data.binding_pin, predecessor=predecessor,
            keys=self.keys, new=self.new, ceiling=self.ceiling, baseline=digest(reference))
        self.pin = digest(self.contract)

    def checked(self):
        check(digest(self.contract) == self.pin and self.contract['ceiling'] == self.ceiling
              and self.contract['keys'] == self.keys and self.contract['new'] == self.new,
              'RETENTION_CONTRACT_CHANGED')

    def scores(self, model):
        self.checked()
        result = measurements(self.data.evaluate(model, 'learn_1'))
        check(not compare(self.ceiling, result), 'RETENTION_EXISTING_REGRESSION')
        return result

    def objective(self, model):
        return sum(float(self.data.losses(model,k,'learn_2')) for k in self.new)/len(self.new)

    def prepare(self, model, parameters):
        self.checked()
        stage = self.data.stage('learn_1',model)
        count = len(self.keys)*(len(model.specs)+1)+sum(not self.data.examples[k]['absent']
                  for k in self.keys)*len(model.specs)
        coordinates = sum(p.numel() for p in parameters)
        # This is optimizer-coordinate storage, never concatenated native instrument readings.
        needed = count*coordinates*8
        free,_ = torch.cuda.mem_get_info(model.core.initial.device)
        check(needed*4 < free, 'RETENTION_GRADIENT_MEMORY')
        matrix = torch.empty((count,coordinates),device=model.core.initial.device,dtype=torch.float64)
        order = []
        for key in self.keys:
            output = self.data.forward(model,key,'learn_1')
            e = self.data.examples[key]
            truth = torch.tensor([float(i==e['outcome']) for i in range(3)],device=self.data.device)
            lanes = dict(output['regions'],common=output['constellation_regions']/len(model.specs))
            terms = []
            for name,score in lanes.items():
                p = F.softmax(score[0],dim=-1)
                mass = torch.stack([p[stage['labels']==i].sum() for i in range(3)])
                terms.append((name,'brier',(mass-truth).square().sum()))
                if name != 'common' and not e['absent']:
                    terms.append((name,'native_expected_loss',(p*stage['costs'][key][name]).sum()))
            for j,(name,metric,term) in enumerate(terms):
                gradients = torch.autograd.grad(term,parameters,retain_graph=j+1<len(terms),allow_unused=True)
                offset = 0
                for p,g in zip(parameters,gradients,strict=True):
                    matrix[len(order),offset:offset+p.numel()] = 0 if g is None else g.detach().reshape(-1)
                    offset += p.numel()
                order.append(key+'|'+name+'|'+metric)
        check(len(order)==count and bool(torch.isfinite(matrix).all()), 'RETENTION_GRADIENT_ROSTER')
        return matrix,order


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
        projected = time.perf_counter()
        if delta is not None:
            with torch.no_grad():
                for scale in SCALES:
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
        proposal_fingerprint=proposal_fingerprint,
        constraints=order,gradient_bytes=gradients.numel()*gradients.element_size(),projection=projection,trials=trials,
        gradient_norm=float(norm),wall_seconds=ended-started,phases=dict(baseline=measured-started,
        gradients=prepared-measured,proposal_projection=projected-prepared,replay_rollback=ended-projected))
