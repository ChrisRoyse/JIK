"""Repair bank-growth deficits while preserving every old check and new learning."""
import copy
import math
import time

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.loop_training_checkpoint import fingerprint
from model.repository_bank_retention import Retention as BankRetention, POLICY as BANK_POLICY
from model.repository_retention import compare, measurements
from model.repository_retention_curvature import contract
from model.retained_update import SCALES
from model.retention_projection import project

POLICY = dict(BANK_POLICY, rule='repository_migration_repair_v1',
              proposal_objective='sum_positive_original_floor_deficits',
              additional_constraint='new_example_objective',
              acceptance='all_old_checks_and_strict_repair_and_new_loss_decrease')
NEW_LOSS = '__new_example_objective__'


class Retention(BankRetention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract['policy'] = POLICY
        self.pin = digest(self.contract)

    def checked(self):
        # The bank implementation pins its own policy. Check all the same
        # immutable fields without temporarily rewriting the sealed contract.
        check(digest(self.contract) == self.pin and self.contract['policy'] == POLICY,
              'REPAIR_CONTRACT_CHANGED')
        check(self.contract['ceiling'] == self.ceiling and self.contract['keys'] == self.keys
              and self.contract['new'] == self.new and
              self.contract['original_ceiling'] == self.original_ceiling and
              self.contract['banks'] == self.banks(self.data.parent), 'REPAIR_CONTRACT_CHANGED')

    def deficits(self, scores):
        self.checked()
        check(set(scores) == set(self.ceiling), 'RETENTION_SCORE_ROSTER')
        result = {}
        for key, floor in self.original_ceiling.items():
            name = 'learn_2|'+key
            check(math.isfinite(scores[name]), 'RETENTION_SCORE_FINITE')
            if scores[name] > floor:
                result[name] = scores[name]-floor
        return result


def accepted(before, after, old_loss, new_loss, old_debt, new_debt, changed):
    """Conjunctive actual-value gate; projection arithmetic never grants success."""
    regressions = compare(before, after)
    finite = all(math.isfinite(v) for v in (old_loss, new_loss, old_debt, new_debt))
    return (not regressions and finite and changed and new_loss < old_loss
            and new_debt < old_debt), regressions


def update(retention, model, optimizer, *, force_reject=False):
    started = time.perf_counter()
    params = tuple(p for p in model.parameters() if p.requires_grad)
    before = [p.detach().clone() for p in params]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(model.core.initial.device)
    initial = fingerprint(model, optimizer)
    scores = retention.scores(model)
    deficits = retention.deficits(scores)
    debt = sum(deficits.values())
    check(debt > 0, 'REPAIR_NO_MIGRATION_DEFICIT')
    with torch.no_grad():
        objective = retention.objective(model)
    measured = time.perf_counter()
    gradients, order = retention.prepare(model, params)
    check(set(deficits) <= set(order), 'REPAIR_NONDIFFERENTIABLE_DEFICIT')
    optimizer.zero_grad(set_to_none=True)
    for key in retention.new:
        loss = retention.data.losses(model, key, 'learn_2')/len(retention.new)
        check(bool(torch.isfinite(loss)), 'RETENTION_LOSS_FINITE')
        loss.backward()
    new_gradient = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad)
                              .detach().reshape(-1).double() for p in params])
    repair_gradient = gradients[[j for j, key in enumerate(order) if key in deficits]].sum(0)
    offset = 0
    for parameter in params:
        parameter.grad = repair_gradient[offset:offset+parameter.numel()].reshape(parameter.shape).to(parameter.dtype)
        offset += parameter.numel()
    norm = torch.nn.utils.clip_grad_norm_(params, POLICY['gradient_clip'])
    check(bool(torch.isfinite(norm)), 'RETENTION_GRADIENT_FINITE')
    gradients = torch.cat((gradients, new_gradient[None]), dim=0)
    order = order+[NEW_LOSS]
    prepared = time.perf_counter()
    did_accept, trials, projection = False, [], None
    try:
        optimizer.step()
        proposal_fingerprint = fingerprint(model, optimizer)
        displacement = torch.cat([(p.detach()-b).reshape(-1).double()
                                  for p, b in zip(params, before, strict=True)])
        delta, projection = project(gradients, displacement)
        linear = None if delta is None else dict(zip(order, (gradients@delta).tolist(), strict=True))
        projected = time.perf_counter()
        if delta is not None:
            with torch.no_grad():
                scale = 1.
                for _ in SCALES:
                    offset = 0
                    for parameter, saved in zip(params, before, strict=True):
                        parameter.copy_((saved.double()+scale*delta[offset:offset+parameter.numel()]
                                         .reshape(parameter.shape)).to(parameter.dtype))
                        offset += parameter.numel()
                    actual = measurements(retention.data.evaluate(model, 'learn_1'))
                    new_loss = retention.objective(model)
                    remaining = retention.deficits(actual)
                    new_debt = sum(remaining.values())
                    changed = any(not torch.equal(p, b) for p, b in zip(params, before, strict=True))
                    passed, regressions = accepted(scores, actual, objective, new_loss, debt, new_debt, changed)
                    trials.append(dict(scale=scale, scores=actual, regressions=regressions,
                        new_loss=new_loss, migration_deficits=remaining, migration_debt=new_debt,
                        gate_passed=passed, forced_rejection=force_reject))
                    if passed and not force_reject:
                        did_accept = True
                        break
                    prior = dict(scores, **{NEW_LOSS:objective})
                    candidate = dict(actual, **{NEW_LOSS:new_loss})
                    violations = compare(prior, candidate)
                    next_scale = contract(prior, candidate, linear, scale) if violations else scale*.5
                    trials[-1]['next_scale'] = next_scale
                    if next_scale is None:
                        break
                    scale = next_scale
    finally:
        if not did_accept:
            with torch.no_grad():
                for parameter, saved in zip(params, before, strict=True):
                    parameter.copy_(saved)
            optimizer.load_state_dict(saved_optimizer)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, model.core.initial.device)
        optimizer.zero_grad(set_to_none=True)
    after = fingerprint(model, optimizer)
    check(did_accept or (after == initial and retention.scores(model) == scores), 'RETENTION_ATOMIC_ROLLBACK')
    if not did_accept:
        check(torch.equal(cpu_rng, torch.get_rng_state()) and
              torch.equal(cuda_rng, torch.cuda.get_rng_state(model.core.initial.device)), 'RETENTION_RNG_ROLLBACK')
    final_scores = retention.scores(model)
    check(not compare(scores, final_scores), 'RETENTION_UPDATE_REGRESSION')
    final_debt = sum(retention.deficits(final_scores).values())
    check(final_debt < debt if did_accept else final_debt == debt, 'REPAIR_DEFICIT_REGRESSION')
    ended = time.perf_counter()
    return dict(accepted=did_accept, code='REPAIR_ACCEPTED' if did_accept else 'REPAIR_NO_FEASIBLE_UPDATE',
        before=initial, after=after, before_scores=scores, after_scores=final_scores, new_loss_before=objective,
        migration_deficits_before=deficits, migration_debt_before=debt, migration_debt_after=final_debt,
        proposal_fingerprint=proposal_fingerprint, directional=linear, constraints=order,
        gradient_bytes=gradients.numel()*gradients.element_size(), projection=projection, trials=trials,
        gradient_norm=float(norm), new_gradient_norm=float(torch.linalg.vector_norm(new_gradient)),
        wall_seconds=ended-started, phases=dict(baseline=measured-started, gradients=prepared-measured,
        proposal_projection=projected-prepared, replay_rollback=ended-projected))
