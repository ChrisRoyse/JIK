"""Read every persisted numerical proposal back with independently blocked sums."""
import argparse
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_recognition import read,evidence
from claims.codata_source import check,sha
from claims.process_conditions import write_committed


def measure(g,rhs,weights,delta,tolerance):
    with mp.workdps(80):
        w = [mp.mpf(v) for v in weights]
        partials = [[] for _ in rhs]
        for begin in range(0,g.shape[1],256):
            subtotal = np.sum(g[:,begin:begin+256].astype(np.longdouble)*
                delta[begin:begin+256].astype(np.longdouble),axis=1,dtype=np.longdouble)
            for i,value in enumerate(subtotal):
                partials[i].append(mp.mpf(np.format_float_scientific(value,precision=25,unique=False)))
        residual = [mp.fsum(v)-mp.mpf(float(b)) for v,b in zip(partials,rhs)]
        stationarity = delta.astype(np.longdouble)-g.astype(np.longdouble).T@np.array(weights,dtype=np.longdouble)
        actual = dict(minimum_residual=float(min(residual)),minimum_multiplier=float(min(w)),
            complementarity=float(max(abs(a*b) for a,b in zip(w,residual))),
            stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
        actual['passed'] = bool(min(w) >= 0 and min(residual) >= -tolerance and
            max(abs(a*b) for a,b in zip(w,residual)) <= tolerance and
            np.max(np.abs(stationarity)) <= tolerance)
        return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    report = read(a.root/'report.json')
    source = Path(report['source'])
    check(evidence(source/'learning.json') == report['learning'] and
          evidence(source/'first-correction.npz') == report['source_arrays'] and
          evidence(a.root/'arrays.npz') == report['arrays'],'SUCCESSOR_NUMERIC_READBACK_BYTES')
    first = read(source/'first-correction.json')
    values,controls = {},[]
    with np.load(source/'first-correction.npz',allow_pickle=False) as original,\
         np.load(a.root/'arrays.npz',allow_pickle=False) as saved:
        for i,lane in enumerate(report['order']):
            entry = report['lanes'][lane]
            c = entry['result']
            prior = first['certificates'][lane]
            check(entry['original'] == prior and c['tolerance'] == prior['tolerance'] and
                  c['iteration_limit'] == prior['iteration_limit'] and
                  c['iterations'] <= prior['iteration_limit'],'SUCCESSOR_NUMERIC_READBACK_LIMITS')
            active = np.array(c['active'])
            g = original[f'gradients_{i}'][active]/np.array(c['norms'])[active,None]
            rhs = original[f'deficit_{i}'][active]/np.array(c['norms'])[active]
            delta = saved[f'delta_{i}']
            actual = measure(g,rhs,c['weights'],delta,c['tolerance'])
            check(actual['passed'] == (entry['physical'] is not None),'SUCCESSOR_NUMERIC_READBACK_VERDICT')
            if not entry['changed']:
                check(np.array_equal(delta,original[f'delta_{i}']),'SUCCESSOR_NUMERIC_READBACK_CONTROL_DELTA')
            broken = measure(g,rhs,c['weights'],delta+g[0],c['tolerance'])
            check(not broken['passed'],'SUCCESSOR_NUMERIC_READBACK_CONTROL_ACCEPTED')
            values[lane] = actual
            controls.append(dict(lane=lane,control='perturbed_displacement',rejected=True))
    check(all(v['passed'] for v in values.values()) == report['all_certified'],'SUCCESSOR_NUMERIC_READBACK_ALL')
    result = dict(schema='successor-numeric-readback-v1',report=evidence(a.root/'report.json'),
        arrays=report['arrays'],lanes=values,controls=controls,all_certified=report['all_certified'],
        model_executed=False,weights_changed=False,acceptance='author-checked',
        code=evidence(Path(__file__)),wall_seconds=time.perf_counter()-started)
    write_committed(a.root/'readback.json',result)
    print(json.dumps(dict(pin=sha((a.root/'readback.json').read_bytes()),lanes=values)))


if __name__ == '__main__':
    main()
