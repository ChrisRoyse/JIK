"""Read captured real branch arithmetic and scores independently on the CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_echo_readback import distance
from audit.proposal_recognition import read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.fresh_echo_population import FreshEchoPopulation


def arithmetic(arrays, index):
    h,parent,flat,separated,w0,w1,p = [arrays[f'{n}_{index}'] for n in
        ('h','parent_y','flat_y','separated_y','parent_w','trial_w','p')]
    h64,w064,w164 = h.astype(np.float64),w0.astype(np.float64),w1.astype(np.float64)
    gamma = h.shape[1]*np.finfo(np.float32).eps/(1-h.shape[1]*np.finfo(np.float32).eps)
    check(np.all(np.abs(h64@w064.T-parent) <= gamma*(np.abs(h64)@np.abs(w064).T)),
          'STABLE_READER_PARENT_ARITHMETIC')
    check(np.all(np.abs(h64@w164.T-flat) <= gamma*(np.abs(h64)@np.abs(w164).T)),
          'STABLE_READER_FLAT_ARITHMETIC')
    residual = (h64@p)@(w164-w064).T
    expected = parent+residual.astype(np.float32)
    check(np.array_equal(expected,separated),'STABLE_READER_SEPARATED_ARITHMETIC')
    return dict(rows=len(h),flat_equal=bool(np.array_equal(flat,parent)),
        separated_equal=bool(np.array_equal(separated,parent)),
        flat_drift=float(np.max(np.abs(flat.astype(np.float64)-parent))),
        separated_drift=float(np.max(np.abs(separated.astype(np.float64)-parent))),
        residual_max=float(np.max(np.abs(residual))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pin',required=True)
    a = parser.parse_args()
    raw = (a.root/'report.json').read_bytes()
    check(sha(raw) == a.pin,'STABLE_READER_REPORT')
    report = json.loads(raw)
    check(sha((a.root/'arrays.npz').read_bytes()) == report['arrays']['sha256'],'STABLE_READER_ARRAYS')
    previous = read(Path(report['source'])/'learning.json')
    check(sha((Path(report['source'])/'learning.json').read_bytes()) == report['source_learning']
          and digest(report['original_step']) == digest(previous['losses'][0]),'STABLE_READER_REPRODUCTION')
    check(report['flat_scores'] == report['original_step']['update']['trials'][0]['scores'],
          'STABLE_READER_SAME_TRIAL')
    parent = checkpoint(Path(report['parent']['root']),report['parent']['pin'])
    with np.load(a.root/'arrays.npz',allow_pickle=False) as source:
        arrays = {n:source[n].copy() for n in source.files}
    basis_root = Path(report['basis']['root'])
    check(sha((basis_root/'manifest.json').read_bytes()) == report['basis']['pin'],'STABLE_READER_BASIS')
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as source:
        basis_arrays = {n:source[n].copy() for n in source.files}
    branches = []
    for i,branch in enumerate(report['branches']):
        name = branch['name']
        check(np.array_equal(arrays[f'parent_w_{i}'],parent[0][name+'.weight'].numpy()) and
              np.array_equal(arrays[f'h_{i}'],basis_arrays[f'h_{i}']) and
              np.array_equal(arrays[f'parent_y_{i}'],basis_arrays[f'y_{i}']) and
              np.array_equal(arrays[f'p_{i}'],basis_arrays[f'p_{i}']),'STABLE_READER_REFERENCE')
        result = arithmetic(arrays,i)
        check(all(result[k] == branch[k] for k in ('rows','flat_equal','separated_equal')),
              'STABLE_READER_BRANCH_REPORT')
        branches.append(dict(name=name,**result))
    ceilings = previous['binding']['grounded_retention']['ceilings']
    comparisons = {}
    for mode in ('flat','separated'):
        evaluation = report[mode+'_evaluation']
        scores = report[mode+'_scores']
        checks = {(v['key'],v['metric'],v['instrument']):v for v in report[mode+'_checks']}
        rows = []
        for key,kinds in ceilings.items():
            item = evaluation[key]
            for kind,names in kinds.items():
                for name,ceiling in names.items():
                    if kind == 'native':
                        slot = item['assessment']['instruments'][name]
                        value = distance(np.array(slot['prediction_cosines']))
                    else:
                        prediction = item['termination']['prediction']
                        p = prediction['common'] if name == 'common' else prediction['instruments'][name]
                        truth = item['termination']['score']['outcome']
                        value = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
                    check(value == scores[key][kind][name] == checks[(key,kind,name)]['value']
                          and checks[(key,kind,name)]['retained'] == (value <= ceiling),
                          'STABLE_READER_RETAINED_SCORE')
                    rows.append(dict(key=key,kind=kind,instrument=name,value=value,ceiling=ceiling,retained=value<=ceiling))
        comparisons[mode] = rows
    config = read(a.root/'config.json')
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = FreshEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = previous['binding']['fresh_feedback_learning']['pairs']
    native = {}
    for name in previous['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as source:
            native[name] = {k:np.concatenate([source['vectors'][source['offsets'][r['unit']]:source['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    measured = {key:{mode:errors(report[mode+'_evaluation'][key],{n:v[pair['observation']] for n,v in native.items()})
                    for mode in ('flat','separated')} for key,pair in pairs.items()}
    damaged = {n:v.copy() for n,v in arrays.items()}
    damaged['separated_y_0'][0,0] += .1
    try:
        arithmetic(damaged,0)
    except ValueError as error:
        check('STABLE_READER_SEPARATED_ARITHMETIC' in str(error),'STABLE_READER_CONTROL_CODE')
    else:
        raise AssertionError('STABLE_READER_CONTROL_ACCEPTED')
    result = dict(schema='stable-response-readback-v1',acceptance='author-checked',report=a.pin,
        branches=branches,comparisons=comparisons,native=measured,
        controls=['altered_residual_output'],serving=False,scientific_admission=False)
    pin = write_committed(a.root/'numeric-readback.json',result)
    print(json.dumps(dict(pin=pin,flat_retained=sum(v['retained'] for v in comparisons['flat']),
        separated_retained=sum(v['retained'] for v in comparisons['separated']),branches=branches)))


if __name__ == '__main__':
    main()
