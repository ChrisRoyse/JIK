"""Independently recompute proposal score attribution from persisted arrays on CPU."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def check(ok, code):
    if not ok:
        raise ValueError('RECOGNITION_READER_'+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def evidence(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=sha(raw))


def softmax(values):
    values = np.exp(values-values.max())
    return values/values.sum()


def analyze(saved, arrays, prior):
    check(saved['train'] == prior['binding']['plan']['train'], 'TRAIN_ROSTER')
    result = {}
    expected_arrays = {'gradient/retention_example_cosines', 'gradient/example_gram'}
    for stage in ('before', 'after'):
        check(set(saved['stages'][stage]) == set(saved['train']), 'STAGE_ROSTER')
        result[stage] = {}
        for key, row in saved['stages'][stage].items():
            reference = prior[stage][key]
            target = next(iter(prior['losses'][0]['details'][key].values()))['supervised_region']
            check(row['target'] == target, 'TARGET')
            check(row['probabilities'] == reference['termination']['prediction'], 'PRIOR')
            selected = row['rows'].index(reference['region'])
            check(row['selected'] == selected, 'SELECTED')
            prefix = stage+'/'+key+'/'
            common = arrays[prefix+'common'].astype(np.float64)
            expected_arrays.add(prefix+'common')
            p = softmax(common)
            check(np.allclose(p, row['probabilities']['common']['region_probabilities'], atol=1e-12, rtol=0),
                  'COMMON_PROBABILITY')
            labels = np.array([r['outcome'] for r in row['probabilities']['bank']['rows']])
            nonzero = float(p[labels == 1].sum())
            components = {n:np.zeros(len(common)) for n in ('base', 'native', 'fields', 'full')}
            for lane, margins in row['lanes'].items():
                parts = {n:arrays[prefix+lane+'/'+n].astype(np.float64) for n in margins}
                expected_arrays.update(prefix+lane+'/'+n for n in margins)
                check(np.allclose(parts['base']+parts['native']+parts['fields'], parts['full'],
                                  atol=1e-12, rtol=0), 'COMPONENT_SUM')
                if 'independent_base' in parts:
                    tolerance = 8*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(parts['full']))))
                    check(np.max(np.abs(parts['base']-parts['independent_base'])) <= tolerance,
                          'INDEPENDENT_COMPONENT')
                for name, values in parts.items():
                    check(margins[name] == float(values[target]-values[selected]), 'MARGIN')
                    if name in components:
                        components[name] += values
                lane_p = softmax(parts['full'])
                check(np.allclose(lane_p, row['probabilities']['instruments'][lane]['region_probabilities'],
                                  atol=1e-12, rtol=0), 'LANE_PROBABILITY')
            # log-softmax normalizers are region-constant, so they cancel in margins.
            tolerance = 32*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(common))))
            check(np.max(np.abs((common-common[selected])-
                                (components['full']-components['full'][selected]))) <= tolerance,
                  'COMMON_COMPONENTS')
            margins = {n:float(v[target]-v[selected]) for n, v in components.items()}
            choices = {n:int(v.argmax()) for n, v in
                       dict(base=components['base'], native=components['base']+components['native'],
                            fields=components['full']).items()}
            result[stage][key] = dict(new=row['new'], target=target, selected=selected,
                nonzero_probability=nonzero, target_probability=float(p[target]),
                common_margin=float(common[target]-common[selected]), component_margins=margins,
                component_choices=choices, per_instrument_margins=row['lanes'])
    check(set(arrays) == expected_arrays, 'ARRAY_ROSTER')
    gradient = saved['gradient']
    projection = gradient['projection']
    gram, weights = np.array(projection['gram']), np.array(projection['weights'])
    check(projection['status'] == 'common_descent' and len(weights) == len(gradient['retained'])+1,
          'PROJECTION_SCOPE')
    energy = float(weights@gram@weights)
    check(abs(energy-projection['energy']) < 1e-12 and energy > 0, 'PROJECTION_ENERGY')
    objective_cosine = -float((gram@weights)[-1])/np.sqrt(energy)
    check(abs(objective_cosine-gradient['common_descent_objective_cosine']) < 1e-8, 'OBJECTIVE_DIRECTION')
    cross = arrays['gradient/retention_example_cosines']
    objectives = [-gradient['examples'][key]['objective_descent_cosine'] for key in saved['train']]
    cosines = -weights@np.vstack((cross, objectives))/np.sqrt(energy)
    for index, key in enumerate(saved['train']):
        check(abs(float(cosines[index])-gradient['examples'][key]['common_descent_cosine']) < 1e-8,
              'EXAMPLE_DIRECTION')
    check(all(r['retained'] and r['slack'] == r['ceiling']-r['value'] >= 0 for r in gradient['retained']),
          'RETENTION')
    return dict(stages=result, gradient=dict(objective_cosine=objective_cosine,
        examples=gradient['examples'], positive_slack=sum(r['slack'] > 0 for r in gradient['retained']),
        protected=len(gradient['retained']), projection_energy=energy,
        scope='local_linear_attribution_not_an_accepted_weight_update'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    saved = read(a.root/'diagnostic.json')
    seal_root = Path(saved['seal'])
    check(sha((seal_root/'seal.json').read_bytes()) == saved['pin'], 'SOURCE_SEAL')
    seal = read(seal_root/'seal.json')
    for name, item in {**{str(seal_root/n):v for n,v in seal['files'].items()}, **seal['external']}.items():
        check(evidence(Path(name)) == item, 'SOURCE_BYTES')
    prior = read(seal_root/'session/learning.json')
    check(evidence(a.root/'arrays.npz') == saved['arrays'], 'ARRAY_PIN')
    with np.load(a.root/'arrays.npz', allow_pickle=False) as archive:
        arrays = {n:archive[n].copy() for n in archive.files}
    result = analyze(saved, arrays, prior)
    controls = list(saved['controls'])

    def refuse(name, code, report, values):
        try:
            analyze(report, values, prior)
        except ValueError as error:
            check(str(error) == 'RECOGNITION_READER_'+code, 'CONTROL_CODE')
            controls.append(dict(name=name, code=code))
        else:
            raise AssertionError('RECOGNITION_READER_CONTROL_ACCEPTED')

    key = next(iter(saved['new']))
    lane = next(iter(saved['stages']['after'][key]['lanes']))
    bad = {n:v.copy() for n,v in arrays.items()}
    bad['after/'+key+'/'+lane+'/fields'][0] += .1
    refuse('changed_residual', 'COMPONENT_SUM', saved, bad)
    bad = copy.deepcopy(saved)
    bad['stages']['after'][key]['target'] = 0
    refuse('changed_target_index', 'TARGET', bad, arrays)
    bad = copy.deepcopy(saved)
    bad['stages']['after'][key]['probabilities']['common']['probabilities'][1] += .1
    refuse('changed_prediction', 'PRIOR', bad, arrays)
    for name, item in saved['journals'].items():
        check(evidence(a.root/name) == item, 'JOURNAL_BYTES')
    with sqlite3.connect((a.root/'experience.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        count, = db.execute('SELECT COUNT(*) FROM events').fetchone()
        head, = db.execute('SELECT hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
    check(count == saved['events'] and head == saved['head'], 'JOURNAL_STATE')
    for stage, snapshot in saved['checkpoints'].items():
        name = 'proposal-1-start' if stage == 'before' else 'proposal-1'
        path = Path(saved['source'])/'candidates'/name
        check(sha((path/'manifest.json').read_bytes()) == snapshot['pin'], 'CHECKPOINT_PIN')
        manifest = read(path/'manifest.json')
        check(manifest['fingerprint'] == snapshot['fingerprint'], 'CHECKPOINT_STATE')
        for file, item in manifest['files'].items():
            check(evidence(path/file) == item, 'CHECKPOINT_BYTES')
    report = dict(schema='proposal-recognition-readback-v1', acceptance='author-checked',
        diagnostic=evidence(a.root/'diagnostic.json'), arrays=saved['arrays'], events=count, head=head,
        controls=controls, boundaries=['source_to_preaction', 'checkpoint_to_components', 'diagnostic_without_update'],
        result=result, weights_optimizer_unchanged=saved['weights_optimizer_unchanged'],
        serving=False, scientific_admission=False)
    destination = a.root/'readback.json'
    with destination.open('x') as handle:
        json.dump(report, handle, sort_keys=True, separators=(',', ':'))
    print(json.dumps(dict(readback=sha(destination.read_bytes()), controls=len(controls),
        objective_descent_cosine=result['gradient']['objective_cosine'],
        new={k:v for k,v in result['stages']['after'].items() if v['new']})))


if __name__ == '__main__':
    main()
