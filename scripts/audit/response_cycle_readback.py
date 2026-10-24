"""Read exact-failure margins, permitted derivatives and persisted cycle evidence."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.learning_evidence import read_learning
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_recognition import evidence, read
from audit.proposal_recognition_readback import softmax
from audit.response_learning_readback import updates
from claims.codata_source import check, sha
from claims.process_conditions import write_committed


def margins(saved, arrays, learning, basis, basis_arrays):
    pairs = learning['binding']['proposal_feedback']['pairs']
    train = learning['binding']['plan']['train']
    result, expected_arrays = {},set()
    for stage,diagnostic in saved['stages'].items():
        check(stage in ('before','after') and set(diagnostic['trace']) == set(train), 'CYCLE_READER_TRACE_ROSTER')
        layout = diagnostic['layout']
        check(len(layout) == len(basis['branches']), 'CYCLE_READER_LAYOUT')
        offset = 0
        for row,branch in zip(layout,basis['branches']):
            check(row['name'] == branch['parameter'] and row['shape'] == branch['shape'] and row['start'] == offset
                  and row['stop'] == offset+int(np.prod(row['shape'])), 'CYCLE_READER_LAYOUT')
            offset = row['stop']
        vectors = {}
        for label in ('objective',*pairs):
            prefix = stage+'/gradient/'+label+'/'
            expected_arrays.update((prefix+'raw',prefix+'projected'))
            raw, projected = arrays[prefix+'raw'],arrays[prefix+'projected']
            check(raw.shape == projected.shape == (offset,) and raw.dtype == projected.dtype == np.float64
                  and np.isfinite(raw).all() and np.isfinite(projected).all(), 'CYCLE_READER_GRADIENT_SHAPE')
            expected = np.concatenate([(raw[r['start']:r['stop']].reshape(r['shape'])@basis_arrays[f'p_{i}']).ravel()
                                       for i,r in enumerate(layout)])
            tolerance = 64*np.finfo(np.float64).eps*offset*max(1.,float(np.linalg.norm(raw)))
            check(np.max(np.abs(projected-expected)) <= tolerance, 'CYCLE_READER_PROJECTION')
            vectors[label] = raw,projected,tolerance
        raw_objective, objective, tolerance = vectors['objective']
        norm = np.linalg.norm(objective)
        check(abs(float(norm)-diagnostic['projected_objective_norm']) <= tolerance and norm > 0,
              'CYCLE_READER_OBJECTIVE_NORM')
        direction = -objective/norm
        expected_objective = learning['losses'][0]['update']['objective_before'] if stage == 'before' else learning['losses'][-1]['update']['objective_after']
        check(diagnostic['objective'] == expected_objective, 'CYCLE_READER_OBJECTIVE_VALUE')
        result[stage] = {}
        for key,row in diagnostic['trace'].items():
            prior = learning[stage][key]
            target = next(iter(learning['losses'][0]['details'][key].values()))['supervised_region']
            check(row['target'] == target and row['probabilities'] == prior['termination']['prediction'], 'CYCLE_READER_TARGET')
            selected = row['rows'].index(prior['region'])
            check(row['selected'] == selected, 'CYCLE_READER_SELECTED')
            prefix = stage+'/'+key+'/'
            common = arrays[prefix+'common'].astype(np.float64)
            expected_arrays.add(prefix+'common')
            probabilities = softmax(common)
            check(np.allclose(probabilities,row['probabilities']['common']['region_probabilities'],atol=1e-12,rtol=0),
                  'CYCLE_READER_PROBABILITIES')
            components = {n:np.zeros(len(common)) for n in ('full','base','native','fields')}
            lane_parts = {}
            for lane,values in row['lanes'].items():
                parts = {n:arrays[prefix+lane+'/'+n].astype(np.float64) for n in values}
                expected_arrays.update(prefix+lane+'/'+n for n in values)
                check(np.allclose(parts['base']+parts['native']+parts['fields'],parts['full'],atol=1e-12,rtol=0),
                      'CYCLE_READER_COMPONENT_SUM')
                for n,v in parts.items():
                    check(values[n] == float(v[target]-v[selected]), 'CYCLE_READER_SELECTED_MARGIN')
                    if n in components:
                        components[n] += v
                check(np.allclose(softmax(parts['full']),row['probabilities']['instruments'][lane]['region_probabilities'],
                                  atol=1e-12,rtol=0), 'CYCLE_READER_LANE_PROBABILITY')
                lane_parts[lane] = parts
            numerical = 32*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(common))))
            check(np.max(np.abs((common-common[selected])-(components['full']-components['full'][selected]))) <= numerical,
                  'CYCLE_READER_COMMON_COMPONENTS')
            if key not in pairs:
                continue
            m = diagnostic['margins'][key]
            rival_key, = [k for k in pairs if k != key]
            rival = pairs[rival_key]['region']
            check(m['rival_key'] == rival_key and m['rival'] == rival and m['target'] == target == pairs[key]['region'],
                  'CYCLE_READER_RIVAL')
            check(m['margin'] == float(np.float32(common[target])-np.float32(common[rival])), 'CYCLE_READER_MARGIN')
            raw, projected, tolerance = vectors[key]
            derivative = float(raw@direction)
            projected_norm = float(np.linalg.norm(projected))
            check(abs(derivative-m['objective_descent_derivative']) <= tolerance
                  and abs(projected_norm-m['projected_norm']) <= tolerance, 'CYCLE_READER_DERIVATIVE')
            check(projected_norm > 0 and abs(derivative/projected_norm-m['objective_descent_cosine']) <= tolerance,
                  'CYCLE_READER_GRADIENT_ANGLE')
            result[stage][key] = dict(target=target,rival=rival,selected=selected,margin=m['margin'],
                target_probability=float(probabilities[target]),rival_probability=float(probabilities[rival]),
                derivative=derivative,projected_margin_norm=projected_norm,
                objective_descent_cosine=derivative/projected_norm,
                components={n:float(v[target]-v[rival]) for n,v in components.items()},
                lanes={lane:{n:float(v[target]-v[rival]) for n,v in parts.items()} for lane,parts in lane_parts.items()})
    check(set(result) == {'before','after'} and set(arrays) == expected_arrays, 'CYCLE_READER_ARRAY_ROSTER')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    check(not a.output.exists(), 'CYCLE_READER_EXISTS')
    a.output.mkdir()
    config = read(a.root/'config.json')
    spec = config['spec']
    core,learning,basis,basis_arrays,native_arrays = read_learning(a.root,name='cycle-1',kind='response_cycle_candidate',
        parent=spec['continuations'][-1]['checkpoint'],basis_spec=spec['initial']['basis'],
        basis_parent=spec['initial']['continuation'])
    saved = read(a.root/'diagnostic.json')
    check(evidence(a.root/'diagnostic-arrays.npz') == saved['arrays'], 'CYCLE_READER_DIAGNOSTIC_BYTES')
    with np.load(a.root/'diagnostic-arrays.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n].copy() for n in archive.files}
    diagnosis = margins(saved,arrays,learning,basis,basis_arrays)
    controls = []
    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'CYCLE_READER_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('CYCLE_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name,code=code))
    key = next(iter(learning['binding']['proposal_feedback']['pairs']))
    bad = copy.deepcopy(saved)
    bad['stages']['after']['margins'][key]['rival'] = bad['stages']['after']['margins'][key]['target']
    refuse('swapped_rival','CYCLE_READER_RIVAL',lambda:margins(bad,arrays,learning,basis,basis_arrays))
    modified = dict(arrays)
    label = 'after/gradient/objective/projected'
    modified[label] = arrays[label].copy()
    modified[label][0] += .1
    refuse('changed_projection','CYCLE_READER_PROJECTION',lambda:margins(saved,modified,learning,basis,basis_arrays))
    bad_learning = copy.deepcopy(learning)
    retained_key = next(iter(basis['retained_keys']))
    bad_learning['losses'][0]['update']['after_scores'][retained_key]['termination']['common'] = 3.
    refuse('concealed_retention_failure','RESPONSE_READER_CEILING',lambda:updates(bad_learning))
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/cycle-1',broken)
    with (broken/'weights.pt').open('r+b') as handle:
        handle.seek(256); value=handle.read(1); handle.seek(256); handle.write(bytes([value[0]^1]))
    refuse('changed_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,learning['candidate_pin']))
    pair = learning['binding']['proposal_feedback']['pairs'][key]
    wrong = next(k for k in next(iter(native_arrays.values())) if k != pair['observation'])
    refuse('swapped_native_output','PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native_arrays.items()}))
    report = dict(core,schema='response-cycle-readback-v1',diagnosis=diagnosis,
        controls=read(a.root/'train-report.json')['controls']+controls,
        boundaries=['parent_and_immutable_basis','actual_margin_and_retained_update','cold_checkpoint_and_decision'])
    write_committed(a.output/'readback.json',report)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()),events=core['events'],
        accepted=core['accepted_updates'],controls=len(report['controls']),changed=len(core['changed']),
        objective_before=core['objective_before'],objective_after=core['objective_after'],
        margins={s:{k:v['margin'] for k,v in rows.items()} for s,rows in diagnosis.items()})))


if __name__ == '__main__':
    main()
