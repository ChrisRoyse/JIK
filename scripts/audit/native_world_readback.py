"""Read native-world migration, Adam clocks, ledger and outcome errors independently."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.prospective_execution_readback import canonical, journal, sha, distance
from audit.rehearsal_readback import authenticated, same, tensors


def read(directory):
    m = json.loads(authenticated(directory/'manifest.json',(directory/'COMMITTED').read_text()))
    payload = {n:torch.load(io.BytesIO(authenticated(directory/n,e['sha256'])),map_location='cpu',weights_only=True)
               for n,e in m['files'].items()}
    return m,payload['weights.pt'],payload['training.pt']['optimizer']


def run(root):
    started = time.perf_counter()
    provenance = json.loads((root/'source.json').read_bytes())
    original = Path(provenance['root'])
    seal = json.loads(authenticated(original/'seal.json',provenance['seal']))
    for n,e in seal['files'].items():
        assert len(authenticated(original/n,e['sha256'])) == e['bytes']
    base = Path(provenance['upstream']['root'])
    base_seal = json.loads(authenticated(base/'seal.json',provenance['upstream']['seal']))
    source = json.loads(authenticated(base/'summary.json',base_seal['files']['summary.json']['sha256']))
    contract,old,old_payloads,old_head = journal(original/'experience.sqlite3')
    contract2,events,payloads,head = journal(root/'experience.sqlite3')
    assert contract2 == contract and events[:-1] == old and len(events) == len(old)+1
    assert all(payloads[k] == v for k,v in old_payloads.items())
    event = events[-1]
    assert event['kind'] == 'native_world_candidate' and event['result']['selected'] is False
    for n,pin in event['code'].items():
        authenticated(Path('scripts/model')/n,pin)
    prior,extra,core,old_opt = tensors(original/'candidates/rehearsal-2')
    retained = dict(extra,**{'core.'+n:v for n,v in core.items()})
    initial,w0,o0 = read(root/'native-1-start')
    middle,wm,om = read(root/'native-1-mid')
    final,w1,o1 = read(root/'candidates/native-1')
    assert sha((root/'candidates/native-1/manifest.json').read_bytes()) == event['pin']
    assert event['binding'] == sha(canonical(final['binding']))
    assert all(same(v,w0[n]) for n,v in retained.items())
    new = set(w0)-set(retained)
    assert new and all(n.startswith('query._native.') for n in new)
    assert all(torch.count_nonzero(w0[n]) == 0 for n in new if n.endswith('up.weight'))
    old_named = dict(zip(prior['names'],old_opt['param_groups'][0]['params']))
    initial_named = dict(zip(initial['names'],o0['param_groups'][0]['params']))
    assert all(same(old_opt['state'][i],o0['state'][initial_named[n]]) for n,i in old_named.items())
    assert all(initial_named[n] not in o0['state'] for n in new)
    changed = [n for n in w0 if not same(w0[n],w1[n])]
    assert changed and all(n.startswith(('query.','place.')) for n in changed)
    assert all(same(w0[n],w1[n]) for n in w0 if not n.startswith(('query.','place.')))
    clocks = {}
    for label,m,opt in [('initial',initial,o0),('mid',middle,om),('final',final,o1)]:
        by_name = dict(zip(m['names'],opt['param_groups'][0]['params']))
        values = {n:0 if i not in opt['state'] else int(opt['state'][i]['step']) for n,i in by_name.items()}
        assert all(v == (0 if n in new else 16)+m['step'] for n,v in values.items())
        clocks[label] = dict(legacy=sorted({v for n,v in values.items() if n not in new}),
                             native=sorted({v for n,v in values.items() if n in new}))
    trace = json.loads((root/'increment.json').read_bytes())
    payload = payloads[event['payload']]
    old_final = old_payloads[old[-1]['payload']]['candidate']
    assert payload['baseline'] == old_final == trace['before']
    assert payload['candidate'] == trace['after'] and event == trace['decision']
    assert trace['binding'] == final['binding'] and trace['exact_optimizer_recovery']
    binding = final['binding']
    forecasts = {sha(canonical(p['forecast'])):p['forecast'] for p in old_payloads.values() if 'forecast' in p}
    actual,targets = {},{}
    for e in source['config']['executions']:
        enrollment = json.loads(Path(e['enrollment']).read_bytes())
        capture,pop,native = map(Path,(e['observations'],enrollment['population'],enrollment['native']))
        manifest = json.loads(authenticated(capture/'manifest.json',enrollment['observation_pin']))
        population = json.loads(authenticated(pop/'report.json',enrollment['population_pin']))
        refs = [json.loads(v) for v in authenticated(pop/'references.jsonl',
                  population['files']['references.jsonl']['sha256']).splitlines()]
        report = json.loads(authenticated(native/'report.json',enrollment['native_pin']))
        arrays = {}
        for name,lane in report['lenses'].items():
            m = json.loads(authenticated(native/name/'report.json',lane['report_sha256']))
            with np.load(io.BytesIO(authenticated(native/name/'slots.npz',m['files']['slots.npz']['sha256'])),allow_pickle=False) as z:
                arrays[name] = z['vectors'].copy(),z['offsets'].copy()
        for observation,entry in manifest['observations'].items():
            execution = json.loads((capture/entry['execution']/'execution.json').read_bytes())
            actual[observation] = int(execution['result']['returncode'] != 0)
            units = [r['unit'] for r in refs if r['observation'] == observation and r['unit'] is not None]
            targets[observation] = {n:np.concatenate([v[o[u]:o[u+1]] for u in units]).astype(np.float64)
                                    for n,(v,o) in arrays.items()}
    phases = dict(source_ledger_and_tensor_readback=time.perf_counter()-started)
    tick = time.perf_counter()
    metrics,checks,mass_error = {},0,0.
    for role in ('baseline','candidate'):
        metrics[role] = {}
        for key,item in payload[role].items():
            observation = binding['evidence'][key]['execution']['observation']
            truth = actual[observation]
            p,g = item['termination']['prediction'],item['termination']['score']
            bank = p['bank']
            assert bank['pin'] == sha(canonical({k:v for k,v in bank.items() if k != 'pin'}))
            if key in event['evaluate']:
                assert bank == forecasts[key]['execution_anchor']['bank']
                assert all(r['observation'] != observation for r in bank['rows'])
            else:
                desc = binding['training_views'][key]['definition']
                historical = forecasts[key]['region_context']['definition']
                assert desc['history'] == historical['history'] and desc['private'] == historical['private']
                assert desc.get('observed_echoes',{}) == historical.get('observed_echoes',{})
            for row in bank['rows']:
                assert row['outcome'] == (2 if row['observation'] is None else actual[row['observation']])
            briers = {}
            for name,v in dict(p['instruments'],common=p['common']).items():
                probs = v['region_probabilities']
                sums = [sum(x for x,r in zip(probs,bank['rows']) if r['outcome'] == i) for i in range(3)]
                mass_error = max(mass_error,max(abs(x-y) for x,y in zip(sums,v['probabilities'])))
                assert abs(sum(probs)-1) < 1e-12 and mass_error < 1e-12
                brier = sum((x-float(i == truth))**2 for i,x in enumerate(v['probabilities']))
                assert brier == dict(g['instruments'],common=g['common'])[name]['brier']
                briers[name] = brier
            native_errors = {}
            for name,slot in item['predictions'].items():
                raw = base64.b64decode(slot['base64'],validate=True)
                assert sha(raw) == slot['sha256']
                prediction = np.frombuffer(raw,dtype='<f4').astype(np.float64).reshape(slot['shape'])
                error = distance(targets[observation][name],prediction)
                assert abs(error-item['assessment']['instruments'][name]['prediction_loss']) < 1e-12
                native_errors[name] = error
                checks += 1
            metrics[role][key] = dict(native=native_errors,termination=briers)
    retention = binding['grounded_retention']
    assert retention['keys'] == old[-1]['train'] and not set(retention['keys']) & set(event['evaluate'])
    before = {k:metrics['baseline'][k] for k in retention['keys']}
    assert before == retention['ceilings']
    order = [(k,kind,n) for k in retention['keys'] for kind in ('native','termination') for n in before[k][kind]]
    for loss in trace['losses']:
        update = loss['update']
        assert update['before_scores'] == before and update['protected'] == len(order)
        assert update['gradient_bytes'] == len(order)*update['parameter_coordinates']*8
        assert set(map(tuple,update['constraint_order'])) == set(order)
        for trial in update['trials']:
            kept = all(trial['scores'][k][kind][n] <= retention['ceilings'][k][kind][n] for k,kind,n in order)
            assert kept == trial['retained'] == trial['advances']
        if update['accepted']:
            assert update['after_scores'] == update['trials'][-1]['scores'] and update['trials'][-1]['advances']
        else:
            assert update['before'] == update['after'] and update['after_scores'] == before
        before = update['after_scores']
    assert before == {k:metrics['candidate'][k] for k in retention['keys']}
    assert all(before[k][kind][n] <= retention['ceilings'][k][kind][n] for k,kind,n in order)
    assert event['result']['metric_gate_passed'] is False
    phases['native_outcome_and_retention_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,acceptance='author-checked',head=head,source_head=old_head,
        checkpoint=event['pin'],changed=changed,new_branch_changed=[n for n in changed if n in new],
        exact_migration=True,legacy_moments_preserved=True,optimizer_clocks=clocks,native_score_checks=checks,
        metrics=metrics,maximum_mass_reconstruction_error=mass_error,
        retained_constraints=len(order),gradient_bytes=trace['losses'][0]['update']['gradient_bytes'],
        accepted_steps=sum(v['update']['accepted'] for v in trace['losses']),result=event['result'],
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,scientific_admission=False,serving=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    result = run(a.root)
    with a.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','result','changed','new_branch_changed')}))
