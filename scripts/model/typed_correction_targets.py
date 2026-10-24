"""Bind per-instrument native margins and original observed-class odds floors."""
import json
import math
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest

RULE = 'typed_native_region_and_outcome_correction_v1'


def bind(data, spec):
    raw = Path(spec['path']).read_bytes()
    check(sha(raw) == spec['pin'], 'TYPED_CORRECTION_SOURCE_PIN')
    source = json.loads(raw)
    check(source['binding'] == data.binding and source['parent'] == data.parent_pin,
          'TYPED_CORRECTION_SOURCE_BINDING')
    rows, masks = {}, {}
    for key in data.new:
        e = data.examples[key]
        labels = data.labels[key]
        masks[key] = [e['allowed'] & (labels == i) for i in range(3)]
        check(all(bool(m.any()) for m in masks[key]), 'TYPED_CORRECTION_CLASS_COVERAGE')
        before = source['before'][key]['termination']
        check(before['score']['outcome'] == data.outcomes[key], 'TYPED_CORRECTION_OUTCOME')
        for lane, probability in before['prediction']['instruments'].items():
            values = rows.setdefault(lane, [])
            costs = e['region_costs'][lane][1]
            for rival in e['allowed'].nonzero().flatten().tolist():
                if rival != e['region']:
                    values.append(dict(kind='native', key=key, target=e['region'], rival=rival,
                                       floor=float(costs[rival])))
            p = probability['probabilities']
            check(len(p) == 3 and all(math.isfinite(x) and x > 0 for x in p),
                  'TYPED_CORRECTION_PROBABILITIES')
            actual = data.outcomes[key]
            for other in range(3):
                if other != actual:
                    values.append(dict(kind='odds', key=key, target=actual, rival=other,
                                       floor=math.log(p[actual])-math.log(p[other])))
    check(set(rows) == {s.name for s in data.model.specs}, 'TYPED_CORRECTION_LANES')
    data.typed_rows, data.typed_masks = rows, masks
    data.typed_row_pin = digest(rows)
    data.typed_floors = {n:torch.tensor([r['floor'] for r in rr], dtype=torch.float64,
                                       device=data.session.device) for n,rr in rows.items()}
    data.typed_native = {n:torch.tensor([r['kind'] == 'native' for r in rr], dtype=torch.bool,
                                       device=data.session.device) for n,rr in rows.items()}
    return dict(rule=RULE, source=spec, rows=rows,
                acceptance='strict_old_native_and_outcome_retention_plus_new_per_lane_and_common_brier',
                objective='sum_per_instrument_native_margin_violations',
                labels_used_only_as_training_targets=True, forward_unchanged=True,
                scope='exposed_feedback_not_transfer')


def constraints(data, model):
    check(digest(data.typed_rows) == data.typed_row_pin, 'TYPED_CORRECTION_ROSTER_CHANGED')
    outputs = {key:data.forward(model,key) for key in data.new}
    result = {}
    for lane, rows in data.typed_rows.items():
        values = []
        for r in rows:
            scores = outputs[r['key']]['regions'][lane][0].double()
            if r['kind'] == 'native':
                values.append(scores[r['target']]-scores[r['rival']])
            else:
                masks = data.typed_masks[r['key']]
                values.append(torch.logsumexp(scores[masks[r['target']]],0)
                              -torch.logsumexp(scores[masks[r['rival']]],0))
        result[lane] = torch.stack(values)
    return result


def objective(data, values):
    return sum(float((data.typed_floors[n]-v)[data.typed_native[n]].clamp_min(0).sum())
               for n,v in values.items())


def measurements(data, model):
    evaluated = data.evaluate(model)
    native, brier = {}, {}
    for key in data.new:
        item = evaluated[key]
        native[key] = {n:v['prediction_loss'] for n,v in item['assessment']['instruments'].items()}
        score = item['termination']['score']
        brier[key] = {n:v['brier'] for n,v in score['instruments'].items()}
        brier[key]['common'] = score['common']['brier']
    return dict(native=native, brier=brier)


def nonincreasing(before, after):
    check(before.keys() == after.keys() and all(before[k].keys() == after[k].keys() for k in before),
          'TYPED_CORRECTION_MEASUREMENT_ROSTER')
    return all(v <= before[k][n] for k,lanes in after.items() for n,v in lanes.items())
