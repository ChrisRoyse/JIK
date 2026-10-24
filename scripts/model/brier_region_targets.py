"""Sufficient region-score floors derived from unchanged observed Brier ceilings."""
import json
import math
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.typed_correction_targets import constraints, objective, measurements, nonincreasing

RULE = 'native_region_margins_with_sufficient_brier_bounds_v1'


def bound(ceiling, regions):
    check(math.isfinite(ceiling) and 0 < ceiling <= 2 and regions > 1,
          'BRIER_REGION_FINITE_BOUND_REQUIRED')
    if ceiling == 2:
        return dict(ceiling=ceiling, probability_floor=0., margin=0., vacuous=True)
    residual = math.sqrt(ceiling/2)
    margin = math.log(regions-1)+math.log1p(-residual)-math.log(residual)
    check(math.isfinite(margin), 'BRIER_REGION_NONFINITE_BOUND')
    return dict(ceiling=ceiling, probability_floor=1-residual, margin=margin, vacuous=False)


def bind(data, spec):
    root = Path(spec['root'])
    raw, reader_raw = (root/'learning.json').read_bytes(), Path(spec['readback']).read_bytes()
    check(sha(raw) == spec['pin'] and sha(reader_raw) == spec['readback_pin'],
          'BRIER_REGION_SOURCE_PIN')
    source, reader = json.loads(raw), json.loads(reader_raw)
    check(source['binding'] == data.binding and source['parent'] == data.parent_pin
          and reader['learning']['sha256'] == sha(raw) and not source['changed']
          and source['before'] == source['after'] and reader['accepted_updates'] == 0,
          'BRIER_REGION_SOURCE_BINDING')
    rows, bounds = {}, {}
    lanes = len(data.model.specs)
    for key in data.new:
        e = data.examples[key]
        allowed = e['allowed'].nonzero().flatten().tolist()
        truth = data.outcomes[key]
        check(bool(e['allowed'][e['region']]) and int(data.labels[key][e['region']]) == truth,
              'BRIER_REGION_TARGET_OUTCOME')
        before = source['before'][key]['termination']
        check(before['score']['outcome'] == truth, 'BRIER_REGION_OUTCOME')
        scores = dict(common=before['score']['common'], **before['score']['instruments'])
        bounds[key] = {n:bound(v['brier'], len(allowed)) for n,v in scores.items()}
        for lane in scores.keys()-{'common'}:
            values = rows.setdefault(lane, [])
            lane_margin = bounds[key][lane]['margin']
            common_share = bounds[key]['common']['margin']/lanes
            for rival in allowed:
                if rival == e['region']:
                    continue
                cost = float(e['region_costs'][lane][1][rival])
                values.append(dict(kind='native', key=key, target=e['region'], rival=rival,
                    native_cost=cost, lane_margin=lane_margin, common_share=common_share,
                    floor=max(cost, lane_margin, common_share)))
    # Model order is part of the immutable partition; do not depend on set ordering.
    rows = {s.name:rows[s.name] for s in data.model.specs}
    data.typed_rows, data.typed_row_pin = rows, digest(rows)
    data.typed_floors = {n:torch.tensor([r['floor'] for r in rr], dtype=torch.float64,
                        device=data.session.device) for n,rr in rows.items()}
    data.typed_native = {n:torch.ones(len(rr), dtype=torch.bool, device=data.session.device)
                        for n,rr in rows.items()}
    return dict(rule=RULE, source=spec, rows=rows, bounds=bounds, lanes=lanes,
        derivation='Brier<=2*(1-p_true)^2; p_true>=p_target>=1/(1+(R-1)*exp(-margin))',
        objective='sum_joint_native_and_confidence_region_margin_violations',
        labels_used_only_as_training_targets=True, forward_unchanged=True,
        acceptance='strict_old_checks_and_stepwise_new_native_and_brier_retention',
        original_brier_ceilings_unchanged=True, scope='exposed_feedback_not_transfer')
