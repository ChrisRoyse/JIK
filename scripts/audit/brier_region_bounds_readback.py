"""Independent probability-bound arithmetic and strict actual retention readback."""
import copy
from pathlib import Path

import mpmath as mp
import numpy as np

from audit.proposal_recognition import read, evidence
from audit.typed_correction_readback_math import floors as original_floors, retained as original_retained
from claims.codata_source import check


def retained(learning):
    value = dict(learning, binding=copy.deepcopy(learning['binding']))
    value['binding']['typed_correction']['rows'] = value['binding']['brier_region']['rows']
    return original_retained(value)


def floors(learning, config):
    original_floors(learning,config)
    spec = config['spec']['brier']
    source_path = Path(spec['root'])/'learning.json'
    check(evidence(source_path)['sha256'] == spec['pin']
          and evidence(Path(spec['readback']))['sha256'] == spec['readback_pin'],
          'BRIER_READER_SOURCE')
    source = read(source_path)
    check(source['before'] == source['after'] == learning['before'], 'BRIER_READER_BASELINE')
    description = learning['binding']['brier_region']
    rows, bounds = description['rows'], description['bounds']
    check(len(rows) == description['lanes'], 'BRIER_READER_LANES')
    root = Path(config['spec']['correction']['root'])
    audit = read(root/'report.json')
    with np.load(root/'arrays.npz',allow_pickle=False) as z, mp.workdps(80):
        for key, lane_bounds in bounds.items():
            i = audit['examples'].index(key)
            allowed = set(np.flatnonzero(z[f'start_{i}_allowed']).tolist())
            labels = z[f'start_{i}_labels']
            before = source['before'][key]['termination']
            truth = before['score']['outcome']
            scores = dict(common=before['score']['common'], **before['score']['instruments'])
            check(lane_bounds.keys() == scores.keys(), 'BRIER_READER_BOUND_ROSTER')
            for lane, b in lane_bounds.items():
                ceiling = scores[lane]['brier']
                check(0 < ceiling <= 2 and b['ceiling'] == ceiling, 'BRIER_READER_CEILING')
                residual = mp.sqrt(mp.mpf(ceiling)/2)
                q = 1-residual
                margin = mp.mpf(0) if ceiling == 2 else mp.log((len(allowed)-1)*q/residual)
                check(abs(b['probability_floor']-float(q)) < 1e-15
                      and abs(b['margin']-float(margin)) < 1e-12
                      and b['vacuous'] == (ceiling == 2), 'BRIER_READER_BOUND')
                lower = 1/(1+(len(allowed)-1)*mp.exp(-margin))
                check(2*(1-lower)**2 <= mp.mpf(ceiling)+mp.mpf('1e-70'),
                      'BRIER_READER_SUFFICIENCY')
            for lane, all_rows in rows.items():
                rr = [r for r in all_rows if r['key'] == key]
                check(len(rr) == len(allowed)-1 and len({r['target'] for r in rr}) == 1,
                      'BRIER_READER_CONSTRAINT_ROSTER')
                target = rr[0]['target']
                check(target in allowed and labels[target] == truth
                      and {r['rival'] for r in rr} == allowed-{target}, 'BRIER_READER_TARGET')
                old = {r['rival']:r['floor'] for r in learning['binding']['typed_correction']['rows'][lane]
                       if r['kind'] == 'native' and r['key'] == key}
                for r in rr:
                    check(r['kind'] == 'native' and r['native_cost'] == old[r['rival']]
                          and r['lane_margin'] == lane_bounds[lane]['margin']
                          and r['common_share'] == lane_bounds['common']['margin']/len(rows)
                          and r['floor'] == max(r['native_cost'],r['lane_margin'],r['common_share']),
                          'BRIER_READER_FLOOR')


def cache_readback(root, learning):
    trace = read(root/'cache-trace.json')
    updates = [s['update'] for s in learning['losses']]
    sequence = updates+updates[len(updates)//2:]
    check(len(trace) == len(sequence), 'BRIER_READER_CACHE_LENGTH')
    last = None
    for item,u in zip(trace,sequence):
        check(item['key'] == u['cache_key'] and item['gradients'] == u['gradients'],
              'BRIER_READER_CACHE_BINDING')
        hit = last is not None and last['cache_key'] == u['cache_key']
        check(item['hit'] == hit, 'BRIER_READER_CACHE_HIT')
        if hit:
            check(u['before'] == last['before'] and u['gradients'] == last['gradients']
                  and u['certificates'] == last['certificates']
                  and u['margins_before'] == last['margins_before'], 'BRIER_READER_CACHE_STATE')
        last = u
    return dict(entries=1, calls=len(trace), hits=sum(t['hit'] for t in trace),
                misses=sum(not t['hit'] for t in trace), source=evidence(root/'cache-trace.json'))
