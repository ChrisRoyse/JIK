"""Diagnose certificates with pairwise extended-precision reductions of actual gradients."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_recognition import read, evidence
from claims.codata_source import check, sha
from claims.process_conditions import write_committed


def decimal(value):
    return np.format_float_scientific(value,precision=25,unique=False)


def fit(gram, rhs, face):
    with mp.workdps(80):
        a = mp.matrix([[mp.mpf(decimal(gram[i,j])) for j in face] for i in face])
        b = mp.matrix([mp.mpf(float(rhs[i])) for i in face])
        fitted = mp.lu_solve(a,b)
        weights = [mp.mpf(0) for _ in rhs]
        for i,value in zip(face,fitted):
            weights[i] = value
        residual = [sum(mp.mpf(decimal(gram[i,j]))*w for j,w in enumerate(weights))
                    -mp.mpf(float(rhs[i])) for i in range(len(rhs))]
        return [mp.nstr(w,75) for w in weights],dict(
            minimum_multiplier=float(min(weights)),minimum_residual=float(min(residual)),
            complementarity=float(max(abs(w*r) for w,r in zip(weights,residual))))


def assess(g, rhs, weights, delta, tolerance):
    extended = g.astype(np.longdouble)
    w = np.array(weights,dtype=np.longdouble)
    residual = np.sum(extended*delta.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs.astype(np.longdouble)
    stationarity = delta.astype(np.longdouble)-extended.T@w
    result = dict(minimum_multiplier=float(w.min()),minimum_residual=float(residual.min()),
        complementarity=float(np.max(np.abs(w*residual))),
        stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
    result['passed'] = bool(np.isfinite(delta).all() and w.min() >= 0
        and residual.min() >= -tolerance and np.max(np.abs(w*residual)) <= tolerance
        and np.max(np.abs(stationarity)) <= tolerance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(),'TYPED_NUMERIC_EXISTS')
    first,learning = read(a.source/'first-correction.json'),read(a.source/'learning.json')
    check(sha((a.source/'learning.json').read_bytes()) ==
          '2ad5e296870664eb51026d93ad5c9ce0b6e56b85cfe4c143ffa016ed68214a29','TYPED_NUMERIC_LEARNING_PIN')
    check(evidence(a.source/'first-correction.npz') == first['file'],'TYPED_NUMERIC_SOURCE_BYTES')
    audit = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')
    with np.load(a.source/'first-correction.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n] for n in archive.files}
    phases = dict(authenticate_load=time.perf_counter()-started)
    results,deltas,controls = {},{},[]
    for index,lane in enumerate(audit['instruments']):
        tick = time.perf_counter()
        c = first['certificates'][lane]
        active = np.array(c['active'])
        g = arrays[f'gradients_{index}'][active]/np.array(c['norms'])[active,None]
        rhs = arrays[f'deficit_{index}'][active]/np.array(c['norms'])[active]
        check(np.array_equal(rhs,c['rhs']),'TYPED_NUMERIC_RHS')
        face = np.flatnonzero(np.array(c['weights']) > 0).tolist()
        old = np.array(c['gram'])
        raw = np.array(c['weights'])
        original = dict(minimum_residual=float((old@raw-rhs).min()),
                        complementarity=float(np.max(np.abs(raw*(old@raw-rhs)))))
        extended = g.astype(np.longdouble)
        precise = np.stack([np.sum(extended*row,axis=1,dtype=np.longdouble) for row in extended])
        row = dict(shape=list(g.shape),face=face,tolerance=c['tolerance'],original=original,
                   gram_difference=float(np.max(np.abs(precise-old))),methods={})
        for name,gram in (('saved_gram',old),('extended_gram',precise)):
            try:
                weights,dual = fit(gram,rhs,face)
                delta = (extended.T@np.array(weights,dtype=np.longdouble)).astype(np.float64)
                actual = assess(g,rhs,weights,delta,c['tolerance'])
                row['methods'][name] = dict(weights=weights,dual=dual,actual=actual,
                    gram=[[decimal(v) for v in r] for r in gram])
                if name == 'extended_gram':
                    deltas[lane] = delta
                    perturbed = delta+g[face[0]]
                    check(not assess(g,rhs,weights,perturbed,c['tolerance'])['passed'],
                          'TYPED_NUMERIC_PERTURBATION_ACCEPTED')
                    negative = copy.copy(weights)
                    negative[face[0]] = '-1'
                    check(not assess(g,rhs,negative,delta,c['tolerance'])['passed'],
                          'TYPED_NUMERIC_NEGATIVE_ACCEPTED')
                    controls.extend([dict(lane=lane,control='perturbed_displacement',rejected=True),
                                     dict(lane=lane,control='negative_multiplier',rejected=True)])
            except (ValueError,ZeroDivisionError) as error:
                row['methods'][name] = dict(error=str(error),passed=False)
        phases[lane] = time.perf_counter()-tick
        results[lane] = row
        print(json.dumps(dict(lane=lane,methods={n:v.get('actual',v) for n,v in row['methods'].items()})),flush=True)
    tick = time.perf_counter()
    a.output.mkdir()
    np.savez(a.output/'displacements.npz',**deltas)
    with np.load(a.output/'displacements.npz',allow_pickle=False) as saved:
        check(all(np.array_equal(saved[n],v) for n,v in deltas.items()),'TYPED_NUMERIC_READBACK')
    phases['persist_readback'] = time.perf_counter()-tick
    elapsed = time.perf_counter()-started
    report = dict(schema='typed-numerical-pairwise-diagnosis-v1',source=str(a.source),
        source_learning=evidence(a.source/'learning.json'),source_arrays=first['file'],
        code=evidence(Path(__file__)),results=results,controls=controls,
        arrays=evidence(a.output/'displacements.npz'),phases=phases,wall_seconds=elapsed,
        coverage=sum(phases.values())/elapsed,precision=dict(mpmath_digits=80,
        extended_mantissa=np.finfo(np.longdouble).nmant),
        scope='actual_linearized_constraints_not_model_acceptance',acceptance='author-checked',
        serving=False,scientific_admission=False)
    write_committed(a.output/'report.json',report)
    print('report '+sha((a.output/'report.json').read_bytes()),flush=True)


if __name__ == '__main__':
    main()
