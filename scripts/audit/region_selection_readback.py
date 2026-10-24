"""Independently recover selection, native costs and objective terms from audit bytes."""
import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.proposal_echo_readback import decode, distance
from audit.proposal_learning_readback import checkpoint
from audit.proposal_recognition import evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.fresh_echo_population import FreshEchoPopulation
from claims.process_conditions import write_committed


def logsoftmax(values):
    values = values.astype(np.float64)
    centered = values-values.max()
    return centered-np.log(np.exp(centered).sum())


def native_error(prediction, target):
    p,t = prediction.astype(np.float64),target.astype(np.float64)
    p /= np.linalg.norm(p,axis=1,keepdims=True)
    t /= np.linalg.norm(t,axis=1,keepdims=True)
    return distance(np.clip(t@p.T,-1,1))


def target_check(actual, captured):
    check(np.array_equal(actual,captured),'REGION_READER_TARGET')


def selected_check(row, common):
    check(row['selected'] == int(common.argmax()) and row['selected_id'] == row['rows'][row['selected']],
          'REGION_READER_SELECTED')


def common_check(common, lanes):
    recomputed = sum(logsoftmax(v) for v in lanes)
    # CPU FP64 reconstruction of CUDA FP32 reductions; not an acceptance tolerance.
    check(np.allclose(common,recomputed,rtol=1e-6,atol=1e-6),'REGION_READER_COMMON')
    return float(np.max(np.abs(common-recomputed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pin',required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    raw = (a.root/'report.json').read_bytes()
    check(sha(raw) == a.pin,'REGION_READER_REPORT_PIN')
    report = read(a.root/'report.json')
    source = Path(report['source'])
    check(evidence(source/'learning.json') == report['source_learning'],'REGION_READER_SOURCE')
    learning = read(source/'learning.json')
    check(report['binding'] == digest(learning['binding']),'REGION_READER_BINDING')
    check(evidence(a.root/'arrays.npz') == report['arrays'],'REGION_READER_ARRAYS')
    with np.load(a.root/'arrays.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n] for n in archive.files}
    config = read(source/'config.json')
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = FreshEchoPopulation(inputs['population'],inputs['population_pin'])
    native = {}
    for name in report['instruments']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as slots:
            native[name] = {k:np.concatenate([slots['vectors'][slots['offsets'][r['unit']]:slots['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    results = {}
    for stage,state in report['states'].items():
        _,_,manifest = checkpoint(Path(state['checkpoint']),state['pin'])
        check(manifest['fingerprint'] == state['fingerprint'] and manifest['binding'] == learning['binding'],
              'REGION_READER_CHECKPOINT')
        results[stage] = {}
        for i,key in enumerate(report['examples']):
            row = state['examples'][key]
            prefix = stage+'_'+str(i)
            common,labels,allowed = [arrays[prefix+'_'+n] for n in ('common','labels','allowed')]
            check(bool(allowed.all()) and len(common) == len(row['rows']) == len(labels),'REGION_READER_MASK')
            selected_check(row,common)
            check(np.array_equal(labels,[v['outcome'] for v in row['bank']['rows']]),'REGION_READER_LABELS')
            prob = np.exp(logsoftmax(common))
            truth = np.array([float(j == row['outcome']) for j in range(3)])
            def brier(logits):
                p = np.exp(logsoftmax(logits))
                return float(sum((p[labels == j].sum()-truth[j])**2 for j in range(3)))
            roundoff = common_check(common,[arrays[prefix+'_'+str(j)+'_logits'] for j in range(len(report['instruments']))])
            common_brier = brier(common)
            check(abs(common_brier-row['common_brier']) < 1e-12,'REGION_READER_BRIER')
            shared = -logsoftmax(common)[row['target']]/len(report['instruments'])
            lanes = {}
            for j,name in enumerate(report['instruments']):
                tag = prefix+'_'+str(j)
                target = arrays[tag+'_target']
                target_check(native[name][row['input']['observation']],target)
                check(row['input'] == learning['binding']['fresh_feedback_learning']['pairs'][key],
                      'REGION_READER_INPUT_JOIN')
                costs,errors = [],[]
                for r in range(len(common)):
                    costs.append(native_error(arrays[tag+'_reference_'+str(r)],target))
                    errors.append(native_error(arrays[tag+'_placed_'+str(r)],target))
                check(np.allclose(costs,arrays[tag+'_costs'],rtol=1e-6,atol=1e-7),'REGION_READER_COSTS')
                evaluation = state['evaluation'][key]
                check(np.array_equal(arrays[tag+'_placed_'+str(row['selected'])],decode(evaluation['predictions'][name]))
                      and errors[row['selected']] == evaluation['assessment']['instruments'][name]['prediction_loss'],
                      'REGION_READER_ACTUAL_PLACEMENT')
                saved = row['instruments'][name]
                expected = float(prob@arrays[tag+'_costs'])
                lane_brier = brier(arrays[tag+'_logits'])
                teacher = errors[row['target']]
                check(abs(teacher-saved['placement']) < 1e-6 and abs(expected-saved['expected_native']) < 1e-6
                      and abs(lane_brier-saved['brier']) < 1e-12,'REGION_READER_OBJECTIVE_TERMS')
                lanes[name] = dict(actual=errors[row['selected']],teacher=teacher,expected_native=expected,
                    brier=lane_brier,all_native_errors=errors,all_reference_costs=costs,
                    exact_native_matches=[r for r in range(len(common)) if
                        np.array_equal(arrays[tag+'_reference_'+str(r)],target)])
            objective = shared+sum(v['teacher']+v['expected_native']+v['brier']+
                common_brier/len(lanes) for v in lanes.values())/len(lanes)
            check(abs(objective-row['objective']) < 2e-6,'REGION_READER_OBJECTIVE')
            results[stage][key] = dict(selected=row['selected'],target=row['target'],
                target_rank=int((common > common[row['target']]).sum())+1,
                target_margin=float(common[row['target']]-np.max(np.delete(common,row['target']))),
                probabilities=prob.tolist(),shared=shared,common_brier=common_brier,objective=objective,
                instruments=lanes,common_roundoff=roundoff)
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'REGION_READER_CONTROL_CODE')
        else:
            raise AssertionError('REGION_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    first = report['states']['start']['examples'][report['examples'][0]]
    target = arrays['start_0_0_target']
    bad = target.copy()
    bad.flat[0] += .1
    refuse('substituted_target','REGION_READER_TARGET',lambda:target_check(target,bad))
    bad_common = arrays['start_0_common'].copy()
    bad_common[0] += 1.
    refuse('changed_common_score','REGION_READER_COMMON',lambda:common_check(bad_common,
        [arrays['start_0_'+str(j)+'_logits'] for j in range(len(report['instruments']))]))
    wrong = copy.deepcopy(first)
    wrong['selected'] = (wrong['selected']+1)%len(wrong['rows'])
    refuse('mislabeled_selected_region','REGION_READER_SELECTED',lambda:selected_check(wrong,arrays['start_0_common']))
    result = dict(report=a.pin,source_learning=report['source_learning'],states=results,controls=controls,
        acceptance='author-checked',serving=False,scientific_admission=False,wall_seconds=time.perf_counter()-started)
    write_committed(a.root/'readback.json',result)
    print('readback '+sha((a.root/'readback.json').read_bytes()))
    for stage,examples in results.items():
        for key,row in examples.items():
            print(stage,key[:12],{n:row[n] for n in ('selected','target','target_rank','target_margin','objective','common_brier')})


if __name__ == '__main__':
    main()
