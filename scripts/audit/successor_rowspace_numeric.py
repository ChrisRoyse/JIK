"""Audit representable corrections on persisted typed gradients without model execution."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_recognition import read,evidence
from audit.proposal_learning_readback import fingerprint
from audit.typed_numeric_readback_v2 import certificate
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.rowspace_region_solve import refine


def binding(gradients,deficit,lane,learning):
    u = learning['losses'][0]['update']
    check(digest(fingerprint(torch.from_numpy(gradients))) == u['gradients'][lane],'SUCCESSOR_NUMERIC_GRADIENT')
    expected = np.array([r['floor'] for r in learning['binding']['successor_targets']['rows'][lane]])
    check(np.array_equal(expected-u['margins_before'][lane],deficit),'SUCCESSOR_NUMERIC_DEFICIT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(),'SUCCESSOR_NUMERIC_EXISTS')
    first,learning = read(a.root/'first-correction.json'),read(a.root/'learning.json')
    check(first['file'] == evidence(a.root/'first-correction.npz'),'SUCCESSOR_NUMERIC_SOURCE')
    u = learning['losses'][0]['update']
    check(first['certificates'] == u['certificates'],'SUCCESSOR_NUMERIC_CERTIFICATES')
    order = read(Path(learning['binding']['region_correction']['diagnostic']['root'])/'report.json')['instruments']
    reports,output,controls = {},{},[]
    source_loaded = time.perf_counter()
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'SUCCESSOR_NUMERIC_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('SUCCESSOR_NUMERIC_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    with np.load(a.root/'first-correction.npz',allow_pickle=False) as z:
        for i,lane in enumerate(order):
            gradients,deficit = z[f'gradients_{i}'],z[f'deficit_{i}']
            binding(gradients,deficit,lane,learning)
            original = u['certificates'][lane]
            if original['status'] == 'coordinate_parameter_certificate_failed':
                delta,result = refine(gradients,deficit,original)
            else:
                check(original['status'] == 'certified_minimum_norm','SUCCESSOR_NUMERIC_UNEXPECTED')
                delta,result = z[f'delta_{i}'],copy.deepcopy(original)
            output[f'delta_{i}'] = delta
            active = np.array(result['active'])
            normalized = gradients[active]/np.array(result['norms'])[active,None]
            rhs = deficit[active]/np.array(result['norms'])[active]
            physical,physical_rejection = None,None
            if result['actual_certified']:
                try:
                    physical = certificate(normalized,rhs,result['weights'],delta,result['tolerance'])
                except ValueError as error:
                    check('TYPED_NUMERIC_KKT' in str(error),'SUCCESSOR_NUMERIC_READER_ERROR')
                    physical_rejection = str(error)
                refuse('corrupt_displacement:'+lane,'TYPED_NUMERIC_KKT',lambda:certificate(
                    normalized,rhs,result['weights'],delta+normalized[0],result['tolerance']))
            reports[lane] = dict(original=original,result=result,physical=physical,
                physical_rejection=physical_rejection,changed=result != original)
            print(json.dumps(dict(lane=lane,status=result['status'],actual=result['actual'],
                iterations=result['iterations'],limit=result['iteration_limit'])),flush=True)
        lanes_finished = time.perf_counter()
        lane = order[0]
        g,d = z['gradients_0'].copy(),z['deficit_0'].copy()
        g[0,0] += 1
        refuse('changed_gradient','SUCCESSOR_NUMERIC_GRADIENT',lambda:binding(g,d,lane,learning))
        g,d = z['gradients_0'].copy(),z['deficit_0'].copy()
        d[0] += 1
        refuse('changed_deficit','SUCCESSOR_NUMERIC_DEFICIT',lambda:binding(g,d,lane,learning))
    a.output.mkdir()
    np.savez(a.output/'arrays.npz',**output)
    result = dict(schema='successor-rowspace-numeric-v1',source=str(a.root),learning=evidence(a.root/'learning.json'),
        source_arrays=first['file'],arrays=evidence(a.output/'arrays.npz'),order=order,lanes=reports,
        controls=controls,all_certified=all(r['physical'] is not None for r in reports.values()),
        model_executed=False,weights_changed=False,tolerances_unchanged=True,iteration_limits_unchanged=True,
        code={n:sha((Path(__file__).parents[1]/n).read_bytes()) for n in
              ('model/rowspace_region_solve.py','audit/successor_rowspace_numeric.py')},
        acceptance='author-checked',wall_seconds=time.perf_counter()-started,
        phases=dict(source=source_loaded-started,lanes=lanes_finished-source_loaded,
                    controls_and_arrays=time.perf_counter()-lanes_finished))
    write_committed(a.output/'report.json',result)
    print(json.dumps(dict(pin=sha((a.output/'report.json').read_bytes()),all_certified=result['all_certified'],
                         seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
