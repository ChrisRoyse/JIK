"""Independent optimality and actual parameter readback for the precise solver."""
from pathlib import Path

import mpmath as mp
import numpy as np
import torch

from audit.proposal_learning_readback import fingerprint
from audit.proposal_recognition import read,evidence
from audit.region_correction_readback import certificate as legacy_certificate
from audit.brier_region_bounds_readback import retained,floors
from audit.typed_numeric_readback_v2 import certificate as parameter_certificate
from claims.codata_source import check
from claims.episodic_memory import digest


def certificate(c):
    if c['status'] != 'certified_minimum_norm':
        return dict(status=c['status'],method=c['method'])
    if c['method'] == 'original_fp64':
        return legacy_certificate(c)
    with mp.workdps(80):
        gram = [[mp.mpf(v) for v in row] for row in c['precise_gram']]
        weights = [mp.mpf(v) for v in c['weights']]
        rhs = [mp.mpf(float(v)) for v in c['rhs']]
        residual = [mp.fsum(v*w for v,w in zip(row,weights))-b for row,b in zip(gram,rhs)]
        tolerance = c['tolerance']
        check(min(weights) >= 0 and min(residual) >= -tolerance
              and max(abs(w*r) for w,r in zip(weights,residual)) <= tolerance
              and c['iterations'] <= c['iteration_limit'],'PRECISE_READER_DUAL')
        actual = c['actual']
        check(c['actual_certified'] and actual['minimum_multiplier'] >= 0
              and actual['minimum_residual'] >= -tolerance
              and actual['complementarity'] <= tolerance
              and actual['stationarity_max'] <= tolerance
              and actual['tolerance'] == tolerance,'PRECISE_READER_ACTUAL_CERTIFICATE')
        return dict(status=c['status'],method=c['method'],minimum_residual=float(min(residual)),
            complementarity=float(max(abs(w*r) for w,r in zip(weights,residual))),
            tolerance=tolerance,actual=actual,iterations=c['iterations'],iteration_limit=c['iteration_limit'])


def first_solve(root, learning, start):
    first = read(root/'first-correction.json')
    check(evidence(root/'first-correction.npz') == first['file'],'PRECISE_READER_FIRST_BYTES')
    with np.load(root/'first-correction.npz',allow_pickle=False) as z:
        arrays = {n:z[n] for n in z.files}
    u = learning['losses'][0]['update']
    check(first['partition'] == u['partition'] and first['certificates'] == u['certificates'],
          'PRECISE_READER_FIRST_BINDING')
    audit = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')
    order = audit['instruments']
    results = [{n:certificate(c) for n,c in row['update']['certificates'].items()} for row in learning['losses']]
    example = next(iter(u['certificates'].values()))
    for i,lane in enumerate(order):
        g,deficit,delta = [arrays[n+'_'+str(i)] for n in ('gradients','deficit','delta')]
        check(digest(fingerprint(torch.from_numpy(g))) == u['gradients'][lane],'PRECISE_READER_GRADIENT_PIN')
        rows = learning['binding']['brier_region']['rows'][lane]
        check(np.array_equal(np.array([r['floor'] for r in rows])-u['margins_before'][lane],deficit),
              'PRECISE_READER_DEFICIT')
        c = u['certificates'][lane]
        if c['status'] == 'certified_minimum_norm':
            example = c
            active = np.array(c['active'])
            normalized = g[active]/np.array(c['norms'])[active,None]
            rhs = deficit[active]/np.array(c['norms'])[active]
            check(np.array_equal(rhs,c['rhs']),'PRECISE_READER_RHS')
            if c['method'] == 'original_fp64':
                check(np.allclose(normalized.T@np.array(c['weights']),delta,rtol=1e-11,atol=1e-11),
                      'PRECISE_READER_ORIGINAL_SOLVE')
            else:
                results[0][lane]['parameter_readback'] = parameter_certificate(normalized,rhs,
                    c['weights'],delta,c['tolerance'])
        offset = 0
        for j,n in enumerate(u['partition'][lane]):
            before,after = arrays[f'before_{i}_{j}'],arrays[f'after_{i}_{j}']
            part = delta[offset:offset+before.size].reshape(before.shape)
            check(np.array_equal(before,start[0][n].numpy()) and
                  np.array_equal(after,(before.astype(np.float64)+u['scale']*part).astype(np.float32)),
                  'PRECISE_READER_FIRST_STEP')
            offset += before.size
        check(offset == len(delta),'PRECISE_READER_DELTA_COVERAGE')
    return results,example
