"""Independent journal, checkpoint tensor, native-error and termination readback."""
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


def authenticated(path,pin):
    raw = path.read_bytes()
    assert sha(raw) == pin,str(path)
    return raw


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def tensors(directory):
    m = json.loads(authenticated(directory/'manifest.json',(directory/'COMMITTED').read_text()))
    world = json.loads(authenticated(directory/'model/manifest.json',m['model']))
    extra = torch.load(io.BytesIO(authenticated(directory/'model/world.pt',world['payload_sha256'])),
                       map_location='cpu',weights_only=True)
    core = json.loads(authenticated(directory/'model/core/manifest.json',world['core']))
    weights = torch.load(io.BytesIO(authenticated(directory/'model/core/weights.pt',
        core['files']['weights.pt']['sha256'])),map_location='cpu',weights_only=True)
    training = torch.load(io.BytesIO(authenticated(directory/'training.pt',m['sha256'])),
                          map_location='cpu',weights_only=True)
    return m,extra,weights,training['optimizer']


def run(root):
    started = time.perf_counter()
    provenance = json.loads((root/'source.json').read_bytes())
    original = Path(provenance['root'])
    source_seal = json.loads(authenticated(original/'seal.json',provenance['seal']))
    for name,entry in source_seal['files'].items():
        assert len(authenticated(original/name,entry['sha256'])) == entry['bytes']
    source = json.loads(authenticated(original/'summary.json',provenance['summary']))
    contract,old,old_payloads,old_head = journal(Path(source['config']['output'])/'experience.sqlite3')
    contract2,events,payloads,head = journal(root/'experience.sqlite3')
    assert contract == contract2 and old_head == provenance['prior_head']
    assert events[:len(old)] == old and len(events) == len(old)+2
    assert all(payloads[k] == v for k,v in old_payloads.items())
    forecasts = {sha(canonical(p['forecast'])):p['forecast'] for p in old_payloads.values() if 'forecast' in p}
    actual,targets = {},{}
    for e in source['config']['executions']:
        enrollment = json.loads(Path(e['enrollment']).read_bytes())
        capture = Path(e['observations'])
        manifest = json.loads(authenticated(capture/'manifest.json',enrollment['observation_pin']))
        pop = Path(enrollment['population'])
        population = json.loads(authenticated(pop/'report.json',enrollment['population_pin']))
        refs = [json.loads(r) for r in authenticated(pop/'references.jsonl',
                population['files']['references.jsonl']['sha256']).splitlines()]
        native = Path(enrollment['native'])
        report = json.loads(authenticated(native/'report.json',enrollment['native_pin']))
        arrays = {}
        for name,lane in report['lenses'].items():
            m = json.loads(authenticated(native/name/'report.json',lane['report_sha256']))
            raw = authenticated(native/name/'slots.npz',m['files']['slots.npz']['sha256'])
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                arrays[name] = (z['vectors'].copy(),z['offsets'].copy())
        for observation,entry in manifest['observations'].items():
            execution = json.loads((capture/entry['execution']/'execution.json').read_bytes())
            actual[observation] = int(execution['result']['returncode'] != 0)
            rows = [r['unit'] for r in refs if r['observation'] == observation and r['unit'] is not None]
            targets[observation] = {n:np.concatenate([v[o[u]:o[u+1]] for u in rows]).astype(np.float64)
                                    for n,(v,o) in arrays.items()}
    phases = dict(authenticated_sources=time.perf_counter()-started)
    tick = time.perf_counter()
    results,previous,checks = [],None,0
    for number,event in enumerate(events[len(old):],1):
        assert event['kind'] == 'rehearsal_candidate' and event['result']['selected'] is False
        for name,pin in event['code'].items():
            authenticated(Path('scripts')/name,pin)
        folder = root/'candidates'/event['candidate']
        final,weights,core,opt = tensors(folder)
        initial,w0,c0,o0 = tensors(root/(event['candidate']+'-start'))
        assert sha((folder/'manifest.json').read_bytes()) == event['pin']
        assert final['binding']['head'] == event['prior_head']
        assert sha(canonical(final['binding'])) == event['binding']
        assert final['optimizer_steps'] == number*8 and initial['optimizer_steps'] == (number-1)*8
        assert same(core,c0)
        changed = [n for n in weights if not same(weights[n],w0[n])]
        assert changed and all(n.startswith(('query.','place.')) for n in changed)
        assert all(float(v['step']) == number*8 for v in opt['state'].values())
        if previous is not None:
            assert same(w0,previous[0]) and same(o0,previous[1]) and same(c0,previous[2])
        previous = weights,opt,core
        binding = final['binding']
        assert set(binding['training_views']) == set(event['train'])
        observations = {binding['evidence'][k]['execution']['observation'] for k in event['train']}
        for key,v in binding['training_views'].items():
            desc,original_view = v['definition'],forecasts[key]['region_context']['definition']
            assert desc['history'] == original_view['history']
            assert desc.get('observed_echoes',{}) == original_view.get('observed_echoes',{})
            assert desc['private'] == original_view['private']
            assert desc['rehearsal']['training_observations'] == sorted(observations)
            assert v['target_region'] in desc['shared'] and len(desc['shared']) <= 10
        payload = payloads[event['payload']]
        metrics = {}
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
                    assert g['scope'] == 'known_target_rehearsal_not_prospective'
                    assert all(r['observation'] is None or r['observation'] in observations for r in bank['rows'])
                for row in bank['rows']:
                    assert row['outcome'] == (2 if row['observation'] is None else actual[row['observation']])
                lanes = dict(p['instruments'],common=p['common'])
                scores = dict(g['instruments'],common=g['common'])
                briers = {}
                for name,v in lanes.items():
                    weights2 = v['region_probabilities']
                    sums = [sum(x for x,r in zip(weights2,bank['rows']) if r['outcome'] == i) for i in range(3)]
                    assert abs(sum(weights2)-1) < 1e-12
                    assert all(abs(x-y) < 1e-12 for x,y in zip(sums,v['probabilities']))
                    brier = sum((x-float(i == truth))**2 for i,x in enumerate(v['probabilities']))
                    assert brier == scores[name]['brier']
                    briers[name] = brier
                errors = {}
                for name,slot in item['predictions'].items():
                    raw = base64.b64decode(slot['base64'],validate=True)
                    assert sha(raw) == slot['sha256']
                    prediction = np.frombuffer(raw,dtype='<f4').astype(np.float64).reshape(slot['shape'])
                    error = distance(targets[observation][name],prediction)
                    assert abs(error-item['assessment']['instruments'][name]['prediction_loss']) < 1e-12
                    errors[name] = error
                    checks += 1
                metrics[role][key] = dict(native_errors=errors,briers=briers)
        retention = binding['grounded_retention']
        expected_keys = [] if number == 1 else events[len(old)+number-2]['train']
        assert retention['keys'] == expected_keys and not set(expected_keys) & set(event['evaluate'])
        if expected_keys:
            prior_metrics = results[-1]['metrics']['candidate']
            assert retention['ceilings'] == {k:dict(native=prior_metrics[k]['native_errors'],
                                                   termination=prior_metrics[k]['briers']) for k in expected_keys}
        trace = json.loads((root/(event['candidate']+'.json')).read_bytes())
        assert trace['binding'] == binding and trace['decision'] == event
        order = [(k,kind,n) for k in expected_keys for kind in ('native','termination')
                 for n in retention['ceilings'][k][kind]]
        def metric_rows(item):
            return {k:dict(native=item[k]['native_errors'],termination=item[k]['briers']) for k in expected_keys}
        before_step = metric_rows(metrics['baseline'])
        repaired = lambda values: all(values[k][kind][n] <= retention['ceilings'][k][kind][n]
                                     for k,kind,n in order)
        accepted_steps = 0
        for loss in trace['losses']:
            update = loss['update']
            assert update['before_scores'] == before_step and update['protected'] == len(order)
            assert {tuple(r) for r in update['constraint_order']} == set(order)
            if expected_keys:
                assert update['gradient_bytes'] == len(order)*update['parameter_coordinates']*8
                for trial in update['trials']:
                    values = trial['scores']
                    advances = all(values[k][kind][n] <= retention['ceilings'][k][kind][n] or
                        (before_step[k][kind][n] > retention['ceilings'][k][kind][n] and
                         values[k][kind][n] < before_step[k][kind][n]) for k,kind,n in order)
                    assert advances == trial['advances'] and repaired(values) == trial['retained']
                if update['accepted']:
                    assert update['trials'][-1]['advances']
                    assert update['after_scores'] == update['trials'][-1]['scores']
                    assert update['before'] != update['after']
                else:
                    assert update['before'] == update['after'] and update['scale'] == 0.
                    assert update['after_scores'] == before_step
            assert update['retained'] == repaired(update['after_scores'])
            before_step = update['after_scores']
            accepted_steps += int(update['accepted'])
        assert before_step == metric_rows(metrics['candidate'])
        gate = event['result']['grounded_training_retention']
        assert gate['retained'] == repaired(before_step) and len(gate['comparisons']) == len(order)
        for row in gate['comparisons']:
            k,kind,n = row['key'],row['metric'],row['instrument']
            assert row['ceiling'] == retention['ceilings'][k][kind][n]
            assert row['value'] == before_step[k][kind][n] and row['retained'] == (row['value'] <= row['ceiling'])
        if not gate['retained']:
            assert event['result']['metric_gate_passed'] is False
            assert 'GROUNDED_TRAINING_RETENTION_PENDING' in event['result']['reasons']
        results.append(dict(candidate=event['candidate'],pin=event['pin'],changed=len(changed),
            optimizer_steps=final['optimizer_steps'],metrics=metrics,result=event['result'],
            grounded_retained=gate['retained'],accepted_steps=accepted_steps,
            reused_costs={k:v['reused_costs'] for k,v in binding['training_views'].items()},
            new_costs={k:v['new_costs'] for k,v in binding['training_views'].items()}))
    _,branch,_,branch_head = journal(root/'withdrawal.sqlite3')
    assert branch[:-1] == events and branch[-1]['kind'] == 'invalidate'
    saved = json.loads((root/'withdrawal-snapshot.json').read_bytes())
    assert saved['head'] == branch_head
    assert all(saved['forecasts'][k]['invalidated'] for k in events[-1]['evaluate'])
    rejection_root = root/'persisted-rejection'
    rejection = json.loads((rejection_root/'report.json').read_bytes())
    _,rejected_events,_,rejected_head = journal(rejection_root/'experience.sqlite3')
    assert rejected_events == events[:-1] and rejected_head == rejection['head']
    assert sha((root/'experience.sqlite3').read_bytes()) == rejection['main_journal_unchanged']
    restored,rw,rc,ro = tensors(rejection_root/'restored')
    start,sw,sc,so = tensors(root/'rehearsal-2-start')
    assert restored['binding'] == start['binding'] and restored['optimizer_steps'] == start['optimizer_steps']
    assert same(rw,sw) and same(rc,sc) and same(ro,so)
    assert (rejection_root/'restored'/'COMMITTED').read_text() == rejection['restored_checkpoint']
    update = rejection['update']
    assert update['accepted'] is False and update['retained'] is False and update['scale'] == 0.
    assert update['before'] == update['after'] and update['before_scores'] == update['after_scores']
    assert len(update['trials']) == 4 and all(t['scores'] == update['before_scores'] and
        t['advances'] is False and t['retained'] is False for t in update['trials'])
    assert rejection['gate']['metric_gate_passed'] is False
    assert rejection['gate']['reasons'] == ['GROUNDED_TRAINING_RETENTION_PENDING']
    phases['checkpoint_and_metric_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,acceptance='author-checked',head=head,source_head=old_head,
        increments=results,native_score_checks=checks,withdrawal_head=branch_head,
        rejected_proposal_exact_restore=True,rejection_head=rejected_head,
        phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        scientific_admission=False,serving=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    report = run(a.root)
    with a.output.open('xb') as f:
        f.write(canonical(report))
    print(json.dumps(dict(complete=True,head=report['head'],native_score_checks=report['native_score_checks'])))
