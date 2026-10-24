"""Independently check parameter-space certificates using blocked high-precision sums."""
import argparse
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_recognition import evidence,read
from claims.codata_source import check,sha
from claims.process_conditions import write_committed


def certificate(g, rhs, weights, delta, tolerance):
    with mp.workdps(80):
        w = [mp.mpf(s) for s in weights]
        products = [[] for _ in rhs]
        for begin in range(0,g.shape[1],256):
            partial = np.sum(g[:,begin:begin+256].astype(np.longdouble)
                *delta[begin:begin+256].astype(np.longdouble),axis=1,dtype=np.longdouble)
            for i,value in enumerate(partial):
                products[i].append(mp.mpf(np.format_float_scientific(value,precision=25,unique=False)))
        residual = [mp.fsum(v)-mp.mpf(float(b)) for v,b in zip(products,rhs)]
        stationarity = delta.astype(np.longdouble)-g.astype(np.longdouble).T@np.array(weights,dtype=np.longdouble)
        check(min(w) >= 0 and min(residual) >= -tolerance
              and max(abs(v*r) for v,r in zip(w,residual)) <= tolerance
              and np.max(np.abs(stationarity)) <= tolerance,'TYPED_NUMERIC_KKT')
        return dict(minimum_residual=float(min(residual)),
                    complementarity=float(max(abs(v*r) for v,r in zip(w,residual))),
                    stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    report = read(a.root/'report.json')
    check(sha((a.root/'report.json').read_bytes()) ==
          '7bb81eb7b80df8b47809625f3f0a0f0539075f9796d7491f5bf97e67ed012db0','TYPED_NUMERIC_REPORT_PIN')
    source = Path(report['source'])
    check(evidence(source/'first-correction.npz') == report['source_arrays']
          and evidence(source/'learning.json') == report['source_learning'],'TYPED_NUMERIC_SOURCE_PIN')
    first = read(source/'first-correction.json')
    learning = read(source/'learning.json')
    audit = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')
    check(evidence(a.root/'displacements.npz') == report['arrays'],'TYPED_NUMERIC_DISPLACEMENT_PIN')
    checked,controls = {},[]
    with np.load(source/'first-correction.npz',allow_pickle=False) as raw, \
         np.load(a.root/'displacements.npz',allow_pickle=False) as saved:
        for i,lane in enumerate(audit['instruments']):
            c = first['certificates'][lane]
            active = np.array(c['active'])
            g = raw[f'gradients_{i}'][active]/np.array(c['norms'])[active,None]
            rhs = raw[f'deficit_{i}'][active]/np.array(c['norms'])[active]
            value = report['results'][lane]['methods']['extended_gram']
            check(c['tolerance'] == value['actual']['tolerance'] and value['actual']['passed'],
                  'TYPED_NUMERIC_TOLERANCE_CHANGED')
            checked[lane] = certificate(g,rhs,value['weights'],saved[lane],c['tolerance'])
            damaged = saved[lane]+g[0]
            try:
                certificate(g,rhs,value['weights'],damaged,c['tolerance'])
            except ValueError as error:
                check(str(error) == 'TYPED_NUMERIC_KKT','TYPED_NUMERIC_CONTROL_ERROR')
            else:
                raise ValueError('TYPED_NUMERIC_CONTROL_ACCEPTED')
            controls.append(dict(lane=lane,control='corrupt_displacement',rejected=True))
    check(len(checked) == 6,'TYPED_NUMERIC_COVERAGE')
    result = dict(schema='typed-numerical-readback-v1',report=evidence(a.root/'report.json'),
        displacements=evidence(a.root/'displacements.npz'),checked=checked,controls=controls,
        code=evidence(Path(__file__)),wall_seconds=time.perf_counter()-started,
        acceptance='author-checked',scope='linearized_constraints_not_actual_model_acceptance',
        serving=False,scientific_admission=False)
    write_committed(a.root/'readback.json',result)
    print(json.dumps(dict(pin=sha((a.root/'readback.json').read_bytes()),checked=checked)))


if __name__ == '__main__':
    main()
