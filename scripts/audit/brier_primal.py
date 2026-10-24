"""One direct primal residual correction under the original numerical certificate."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_recognition import read, evidence
from audit.typed_numeric_readback_v2 import certificate
from claims.codata_source import check
from claims.process_conditions import write_committed


def actual(g, rhs, weights, delta, tolerance):
    extended = g.astype(np.longdouble)
    w = np.array(weights,dtype=np.longdouble)
    residual = np.sum(extended*delta.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs
    stationarity = delta.astype(np.longdouble)-extended.T@w
    result = dict(minimum_multiplier=float(w.min()),minimum_residual=float(residual.min()),
        complementarity=float(np.max(np.abs(w*residual))),
        stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
    result['passed'] = bool(w.min() >= 0 and residual.min() >= -tolerance
        and np.max(np.abs(w*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
    return result,residual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(), 'BRIER_PRIMAL_EXISTS')
    learning, first = read(a.root/'learning.json'), read(a.root/'first-correction.json')
    check(evidence(a.root/'first-correction.npz') == first['file'], 'BRIER_PRIMAL_INPUT_BYTES')
    audit = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')
    with np.load(a.root/'first-correction.npz',allow_pickle=False) as z:
        arrays = {n:z[n] for n in z.files}
    parsed = time.perf_counter()
    results, saved, controls = {}, {}, []
    for i,lane in enumerate(audit['instruments']):
        tick = time.perf_counter()
        c = learning['losses'][0]['update']['certificates'][lane]
        check(c['method'] == 'pairwise_extended_gram_high_precision_face_v1'
              and c['precise_refinement_status'] == 'certified_face', 'BRIER_PRIMAL_FACE')
        active = np.array(c['active'])
        g = arrays[f'gradients_{i}'][active]/np.array(c['norms'])[active,None]
        rhs = arrays[f'deficit_{i}'][active]/np.array(c['norms'])[active]
        check(np.array_equal(rhs,c['rhs']), 'BRIER_PRIMAL_RHS')
        weights = c['weights']
        delta = (g.astype(np.longdouble).T@np.array(weights,dtype=np.longdouble)).astype(np.float64)
        before,residual = actual(g,rhs,weights,delta,c['tolerance'])
        check(before == dict(c['actual'],passed=c['actual_certified']), 'BRIER_PRIMAL_REPRODUCTION')
        corrected = delta.copy()
        action = dict(applied=False,iterations=c['iterations'])
        if not before['passed']:
            support = np.array([float(w) > 0 for w in weights])
            check(c['iterations'] < c['iteration_limit'], 'BRIER_PRIMAL_ITERATION_CAP')
            correction,_,rank,sv = np.linalg.lstsq(g[support],-residual[support].astype(np.float64),rcond=None)
            corrected += correction
            action = dict(applied=True,rows=int(support.sum()),rank=int(rank),
                singular_values=sv.tolist(),correction_norm=float(np.linalg.norm(correction)),
                represented_norm=float(np.linalg.norm(corrected-delta)),iterations=c['iterations']+1)
        after,_ = actual(g,rhs,weights,corrected,c['tolerance'])
        try:
            independent = dict(passed=True,certificate=certificate(g,rhs,weights,corrected,c['tolerance']))
        except ValueError as error:
            check(str(error).endswith('TYPED_NUMERIC_KKT'), 'BRIER_PRIMAL_READER_ERROR')
            independent = dict(passed=False,code=str(error))
        check(independent['passed'] == after['passed'], 'BRIER_PRIMAL_READER_DISAGREEMENT')
        for name,bad_weights,bad_delta in (
            ('negative_multiplier',[-1.]+list(weights[1:]),corrected),
            ('altered_displacement',weights,corrected+1.)):
            try:
                certificate(g,rhs,bad_weights,bad_delta,c['tolerance'])
            except ValueError as error:
                check(str(error).endswith('TYPED_NUMERIC_KKT'), 'BRIER_PRIMAL_CONTROL_CODE')
            else:
                raise AssertionError('BRIER_PRIMAL_CONTROL_ACCEPTED')
            controls.append(dict(lane=lane,name=name))
        saved[f'original_{i}'],saved[f'corrected_{i}'] = delta,corrected
        results[lane] = dict(before=before,action=action,after=after,independent=independent,
                            iteration_limit=c['iteration_limit'],seconds=time.perf_counter()-tick)
    computed = time.perf_counter()
    a.output.mkdir()
    np.savez(a.output/'displacements.npz',**saved)
    finished = time.perf_counter()
    phases = dict(input=parsed-started,computation_controls=computed-parsed,output=finished-computed)
    report = dict(schema='brier-primal-diagnostic-v1',source=str(a.root),learning=evidence(a.root/'learning.json'),
        first=first['file'],displacements=evidence(a.output/'displacements.npz'),results=results,controls=controls,
        phases=phases,wall_seconds=finished-started,coverage=sum(phases.values())/(finished-started),
        code=evidence(Path(__file__)),number_status='measured',acceptance='author-checked',
        actual_learning_run=False,serving=False,scientific_admission=False)
    write_committed(a.output/'report.json',report)
    print({n:dict(before=v['before']['passed'],after=v['after']['passed'],action=v['action'])
           for n,v in results.items()})


if __name__ == '__main__':
    main()
