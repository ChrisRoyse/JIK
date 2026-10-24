"""CPU arithmetic readback of typed retention, native targets and constrained updates."""
import math
from pathlib import Path

import numpy as np
import torch

from audit.proposal_learning_readback import fingerprint
from audit.proposal_recognition import read, evidence
from audit.region_correction_readback import certificate
from audit.region_selection_readback import native_error
from claims.codata_source import check, sha
from claims.episodic_memory import digest


def measures(learning, stage):
    native, brier = {}, {}
    for key in learning['binding']['fresh_feedback_learning']['pairs']:
        item = learning[stage][key]
        native[key] = {n:v['prediction_loss'] for n,v in item['assessment']['instruments'].items()}
        value = item['termination']
        truth = value['score']['outcome']
        prediction = value['prediction']
        labels = [r['outcome'] for r in prediction['bank']['rows']]
        brier[key] = {}
        for name,p in dict(common=prediction['common'],**prediction['instruments']).items():
            mass = [sum(v for c,v in zip(labels,p['region_probabilities']) if c == k) for k in range(3)]
            check(np.allclose(mass,p['probabilities'],rtol=0,atol=1e-12), 'TYPED_READER_CLASS_MASS')
            score = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
            recorded = value['score']['common'] if name == 'common' else value['score']['instruments'][name]
            check(score == recorded['brier'], 'TYPED_READER_BRIER')
            brier[key][name] = score
    return dict(native=native,brier=brier)


def retained(learning):
    ceilings = learning['binding']['grounded_retention']['ceilings']
    rows = learning['binding']['typed_correction']['rows']
    count = 0
    previous = measures(learning,'before')
    for step in learning['losses']:
        u = step['update']
        check(len(u['trials']) <= u['trial_limit'], 'TYPED_READER_TRIAL_CAP')
        check(u['before_measurements'] == previous, 'TYPED_READER_SCORE_CONTINUITY')
        for key,kinds in u['after_scores'].items():
            for kind,lanes in kinds.items():
                for lane,value in lanes.items():
                    check(value <= ceilings[key][kind][lane], 'TYPED_READER_RETENTION')
                    count += 1
        for kind,keys in u['after_measurements'].items():
            check(kind in previous and keys.keys() == previous[kind].keys(), 'TYPED_READER_NEW_ROSTER')
            for key,lanes in keys.items():
                check(lanes.keys() == previous[kind][key].keys(), 'TYPED_READER_NEW_ROSTER')
                check(all(v <= previous[kind][key][n] for n,v in lanes.items()), 'TYPED_READER_NEW_RETENTION')
        for stage in ('before','after'):
            total = sum(sum(max(0.,r['floor']-v) for r,v in zip(rr,u['margins_'+stage][n])
                            if r['kind'] == 'native') for n,rr in rows.items())
            check(abs(total-u['objective_'+stage]) <= 1e-11*max(1.,abs(total)), 'TYPED_READER_MARGIN_SUM')
        check(u['objective_after'] < u['objective_before'] if u['accepted'] else
              u['objective_after'] == u['objective_before'], 'TYPED_READER_ADVANCE')
        if u['accepted']:
            trial = u['trials'][u['selected_trial']]
            check(trial['retained'] and all(trial['new_retained'].values())
                  and trial['scores'] == u['after_scores'] and trial['measurements'] == u['after_measurements']
                  and trial['margins'] == u['margins_after'], 'TYPED_READER_SELECTED_TRIAL')
        else:
            check(u['scale'] == 0 and u['before'] == u['after'], 'TYPED_READER_REJECTED_PARAMETERS')
        previous = u['after_measurements']
    check(previous == measures(learning,'after'), 'TYPED_READER_FINAL_MEASUREMENTS')
    return count


def floors(learning, config):
    spec = config['spec']['typed']
    check(sha(Path(spec['path']).read_bytes()) == spec['pin'], 'TYPED_READER_SOURCE_PIN')
    source = read(Path(spec['path']))
    check(learning['before'] == source['before'], 'TYPED_READER_ORIGINAL_BASELINE')
    root = Path(config['spec']['correction']['root'])
    audit = read(root/'report.json')
    check(evidence(root/'arrays.npz') == audit['arrays'], 'TYPED_READER_NATIVE_ARRAYS')
    with np.load(root/'arrays.npz',allow_pickle=False) as z:
        for lane,rows in learning['binding']['typed_correction']['rows'].items():
            for r in rows:
                if r['kind'] == 'odds':
                    p = source['before'][r['key']]['termination']['prediction']['instruments'][lane]['probabilities']
                    expected = math.log(p[r['target']])-math.log(p[r['rival']])
                    check(expected == r['floor'], 'TYPED_READER_ODDS_FLOOR')
                else:
                    i,j = audit['examples'].index(r['key']),audit['instruments'].index(lane)
                    prefix = f'start_{i}_{j}'
                    costs = z[prefix+'_costs']
                    check(float(costs[r['rival']]) == r['floor'], 'TYPED_READER_NATIVE_FLOOR')
                    expected = native_error(z[prefix+'_reference_'+str(r['rival'])],z[prefix+'_target'])
                    check(abs(expected-r['floor']) < 1e-6, 'TYPED_READER_NATIVE_DISTANCE')


def first_solve(root, learning, start):
    first = read(root/'first-correction.json')
    check(evidence(root/'first-correction.npz') == first['file'], 'TYPED_READER_FIRST_BYTES')
    with np.load(root/'first-correction.npz',allow_pickle=False) as z:
        arrays = {n:z[n] for n in z.files}
    u = learning['losses'][0]['update']
    check(first['partition'] == u['partition'] and first['certificates'] == u['certificates'],
          'TYPED_READER_FIRST_BINDING')
    audit = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')
    order = audit['instruments']
    def inspect(c):
        result = certificate(c)
        if 'gram' in c:
            gram,rhs,w = [np.array(c[n]) for n in ('gram','rhs','weights')]
            residual = gram@w-rhs
            check(np.allclose(residual,c['residual'],rtol=0,atol=1e-11), 'TYPED_READER_RESIDUAL_READBACK')
            result.update(minimum_residual=float(residual.min()),
                complementarity=float(np.max(np.abs(w*residual))),tolerance=c['tolerance'],
                maximum_multiplier=float(w.max()),refinement_status=c['refinement_status'])
        return result
    certificates = [{n:inspect(c) for n,c in row['update']['certificates'].items()}
                    for row in learning['losses']]
    example = next(iter(u['certificates'].values()))
    for i,lane in enumerate(order):
        g,deficit,delta = [arrays[n+'_'+str(i)] for n in ('gradients','deficit','delta')]
        check(digest(fingerprint(torch.from_numpy(g))) == u['gradients'][lane], 'TYPED_READER_GRADIENT_PIN')
        rows = learning['binding']['typed_correction']['rows'][lane]
        expected = np.array([r['floor'] for r in rows])-u['margins_before'][lane]
        check(np.array_equal(expected,deficit), 'TYPED_READER_DEFICIT')
        c = u['certificates'][lane]
        if c['status'] == 'certified_minimum_norm':
            example = c
            norms = np.linalg.norm(g,axis=1)
            active = np.array(c['active'])
            normalized = g[active]/norms[active,None]
            check(np.allclose(normalized@normalized.T,c['gram'],rtol=1e-12,atol=1e-12)
                  and np.allclose(deficit[active]/norms[active],c['rhs'],rtol=1e-12,atol=1e-12)
                  and np.allclose(normalized.T@np.array(c['weights']),delta,rtol=1e-11,atol=1e-11),
                  'TYPED_READER_LINEAR_SOLVE')
        offset = 0
        for j,n in enumerate(u['partition'][lane]):
            before,after = arrays[f'before_{i}_{j}'],arrays[f'after_{i}_{j}']
            part = delta[offset:offset+before.size].reshape(before.shape)
            check(np.array_equal(before,start[0][n].numpy()) and
                  np.array_equal(after,(before.astype(np.float64)+u['scale']*part).astype(np.float32)),
                  'TYPED_READER_FIRST_STEP')
            offset += before.size
        check(offset == len(delta), 'TYPED_READER_DELTA_COVERAGE')
    return certificates,example
