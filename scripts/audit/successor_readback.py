"""Independent byte, native-window, retention and lineage audit of a successor."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_learning_readback import checkpoint, errors, fingerprint
from audit.proposal_recognition import read,evidence
from audit.successor_readback_math import retained,first_solve,floors
from audit.brier_region_bounds_readback import cache_readback
from claims.codata_source import check,sha
from claims.episodic_memory import canonical,digest,ZERO
from claims.next_echo_population import NextEchoPopulation
from claims.process_conditions import write_committed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(),'SUCCESSOR_READER_EXISTS')
    config,learning = read(a.root/'config.json'),read(a.root/'learning.json')
    binding,spec = read(a.root/'training-binding.json'),read(a.root/'learning-spec.json')
    baseline = read(a.root/'baseline.json')
    check(binding == learning['binding'] and evidence(a.root/'baseline.json')['sha256'] == spec['baseline']['pin'],
          'SUCCESSOR_READER_BINDING')
    checks = retained(learning)
    stages = [checkpoint(a.root/'candidates'/n,learning[p]) for n,p in
              (('successor-1-start','start_pin'),('successor-1-mid','mid_pin'),('successor-1','candidate_pin'))]
    start,mid,final = stages
    parent = config['spec']['checkpoint']
    old = checkpoint(Path(parent['root']),parent['pin'])
    check(start[2]['fingerprint'] == old[2]['fingerprint'] and
          all(torch.equal(v,start[0][n]) for n,v in old[0].items()),'SUCCESSOR_READER_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages),'SUCCESSOR_READER_CHECKPOINT_BINDING')
    check(all(start[2]['offsets'][n] == old[2]['offsets'][n]+old[2]['step'] for n in start[2]['names']),
          'SUCCESSOR_READER_CLOCK_CONTINUITY')
    changed = [n for n,v in start[0].items() if not torch.equal(v,final[0][n])]
    check(changed == learning['changed'] and set(changed) <= set(learning['eligible']),'SUCCESSOR_READER_PARAMETER_SCOPE')
    old_report = read(Path(spec['retained_report']['path']))
    check(evidence(Path(spec['retained_report']['path']))['sha256'] == spec['retained_report']['pin'],
          'SUCCESSOR_READER_PRIOR_REPORT')
    for key in old_report['after']:
        check(learning['before'][key] == old_report['after'][key],'SUCCESSOR_READER_OLD_BANK')
    basis_root = Path(spec['basis']['root'])
    basis = read(basis_root/'manifest.json')
    check(evidence(basis_root/'manifest.json')['sha256'] == spec['basis']['pin'] and
          basis['retained_keys'] == binding['grounded_retention']['keys'],'SUCCESSOR_READER_BASIS')
    check(evidence(basis_root/'arrays.npz') == basis['arrays'],'SUCCESSOR_READER_BASIS_ARRAYS')
    responses = []
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as z:
        for i,row in enumerate(basis['branches']):
            n = row['parameter']
            prefix = n.removesuffix('weight')
            h,p = z[f'h_{i}'].astype(np.float64),z[f'p_{i}']
            singular = np.linalg.svd(h,compute_uv=False)
            threshold = max(h.shape)*np.finfo(np.float64).eps*singular[0]
            check(row['rank'] == int(np.sum(singular > threshold)) and
                  np.max(np.abs(h@p)) <= 64*np.finfo(np.float64).eps*max(h.shape)*max(1.,np.linalg.norm(h,2)),
                  'SUCCESSOR_READER_PROJECTOR')
            check(all(torch.equal(s[0][prefix+suffix],old[0][prefix+suffix]) for s in stages
                      for suffix in ('anchor','projector')),'SUCCESSOR_READER_FROZEN_BUFFERS')
            residual = (h@old[0][prefix+'projector'].numpy())@(final[0][n].double().numpy()-old[0][n].double().numpy()).T
            responses.append(dict(name=n,rows=len(h),rank=row['rank'],free=h.shape[1]-row['rank'],
                residual_max=float(np.max(np.abs(residual))),runtime_buffers_unchanged=True))
    certificates,certificate = first_solve(a.root,learning,start)
    cache = cache_readback(a.root,learning)
    for cursor,stage in ((0,start),(len(learning['losses'])//2,mid)):
        expected = digest(dict(binding=digest(binding),rows=digest(binding['successor_targets']['rows']),
                               state=fingerprint(stage[0])))
        check(learning['losses'][cursor]['update']['cache_key'] == expected,'SUCCESSOR_READER_CACHE_CHECKPOINT')
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = NextEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = binding['successor_feedback_learning']['pairs']
    native = {}
    for name in learning['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as z:
            native[name] = {key:np.concatenate([z['vectors'][z['offsets'][r['unit']]:z['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == key and r['unit'] is not None])
                for key in population.observations.entries}
    errors_read = {key:{stage:errors(learning[stage][key],{n:v[pair['observation']] for n,v in native.items()})
                       for stage in ('before','after')} for key,pair in pairs.items()}
    head,count,decision = ZERO,0,None
    with sqlite3.connect((a.root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        all_events = list(db.execute('SELECT * FROM events ORDER BY seq'))
        for seq,previous,raw,pin in all_events:
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body) and
                  pin == digest(dict(seq=seq,previous=previous,body=body)),'SUCCESSOR_READER_JOURNAL')
            if body.get('payload'):
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'],'SUCCESSOR_READER_PAYLOAD')
            if body['kind'] == 'successor_candidate':
                decision = json.loads(payload)
                check(body == learning['decision'] and decision['after'] == learning['after'] and
                      decision['before'] == learning['before'],'SUCCESSOR_READER_DECISION')
            head,count = pin,count+1
    source = Path(config['source'])
    with sqlite3.connect((source/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        original = list(db.execute('SELECT * FROM events ORDER BY seq'))
        forecast_raw, = db.execute('SELECT data FROM payloads WHERE hash=?',(spec['experiment'],)).fetchone()
        source_snapshot = read(source/'snapshot-final.json')
        feedback_pin = source_snapshot['next_experiments'][spec['experiment']]['feedback']['payload']
        feedback_raw, = db.execute('SELECT data FROM payloads WHERE hash=?',(feedback_pin,)).fetchone()
    check(all_events[:-1] == original and sha(forecast_raw) == spec['experiment'] and
          sha(feedback_raw) == feedback_pin,'SUCCESSOR_READER_SOURCE_HISTORY')
    forecast,sealed = json.loads(forecast_raw),json.loads(feedback_raw)['scores']
    floors(learning,baseline,sealed,native)
    comparisons = {}
    for key,pair in pairs.items():
        entry = source_snapshot['native_feedback_contexts'][pair['context']]
        saved = forecast['predictions'][pair['condition']]
        check(entry['status'] == 'corrected_state' and entry['variant'] == 'corrected' and
              entry['observation'] == pair['observation'] and entry['receipt'] == pair['receipt'] and
              entry['native'] == pair['native'] and pair['prestate'] == forecast['state'] and
              pair['input'] == saved['values'] and pair['fields'] == saved['fields'] and
              pair['request'] == digest(saved['request']) and not pair['correction_state_used_as_input'] and
              not pair['heldout'],'SUCCESSOR_READER_PREACTION_JOIN')
        before,after = learning['before'][key]['termination'],learning['after'][key]['termination']
        truth = before['score']['outcome']
        predictions = dict(sealed_forecast=saved['models']['corrected']['termination'],
                           expanded_memory=before['prediction'],learned_weights=after['prediction'])
        stages_read = {stage:{n:dict(probabilities=p['probabilities'],
            brier=sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities'])))
            for n,p in dict(common=prediction['common'],**prediction['instruments']).items()}
            for stage,prediction in predictions.items()}
        check(before['prediction']['bank'] == after['prediction']['bank'],'SUCCESSOR_READER_COMPARISON_BANK')
        comparisons[key] = dict(condition=pair['condition'],observation=pair['observation'],outcome=truth,
                                scores=stages_read,heldout=False,source_forecast_unchanged=True)
    snapshot,cold = read(a.root/'snapshot-final.json'),read(a.root/'cold-report.json')
    check(snapshot['events'] == count == cold['events'] and snapshot['head'] == head == cold['head'] and
          digest(snapshot) == cold['snapshot'] and cold['journal'] == sha((a.root/'experience.sqlite3').read_bytes()),
          'SUCCESSOR_READER_COLD')
    check(learning['exact_optimizer_recovery'] and learning['rollback_behaviour_equal'] and
          learning['rollback_fingerprint'] == start[2]['fingerprint'] and decision is not None and
          not learning['decision']['result']['selected'],'SUCCESSOR_READER_ROLLBACK_ADMISSION')
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'SUCCESSOR_READER_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('SUCCESSOR_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    bad = copy.deepcopy(learning)
    key = next(iter(binding['grounded_retention']['ceilings']))
    bad['losses'][0]['update']['after_scores'][key]['termination']['common'] = 3.
    refuse('concealed_retention','TYPED_READER_RETENTION',lambda:retained(bad))
    bad = copy.deepcopy(learning)
    key = next(iter(pairs))
    bad['losses'][0]['update']['after_measurements']['brier'][key]['common'] = 3.
    refuse('concealed_new_brier_regression','TYPED_READER_NEW_RETENTION',lambda:retained(bad))
    bad = copy.deepcopy(learning)
    lane = next(iter(bad['binding']['successor_targets']['rows']))
    bad['binding']['successor_targets']['rows'][lane][0]['floor'] -= 1.
    refuse('weakened_confidence_floor','SUCCESSOR_READER_FLOOR',lambda:floors(bad,baseline,sealed,native))
    bad = copy.deepcopy(learning)
    bad['losses'][0]['update']['cache_key'] = '0'*64
    refuse('cache_substitution','BRIER_READER_CACHE_BINDING',lambda:cache_readback(a.root,bad))
    a.output.mkdir()
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/successor-1',broken)
    with (broken/'weights.pt').open('r+b') as f:
        f.seek(256)
        byte = f.read(1)
        f.seek(256)
        f.write(bytes([byte[0]^1]))
    refuse('corrupt_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,learning['candidate_pin']))
    key,pair = next(iter(pairs.items()))
    wrong = next(k for k in next(iter(native.values())) if k != pair['observation'])
    refuse('substituted_target','PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native.items()}))
    result = dict(schema='successor-readback-v1',learning=evidence(a.root/'learning.json'),
        candidate=learning['candidate_pin'],parent=parent,events=count,head=head,changed=changed,cache=cache,
        retained_checks=checks,retained_examples=len(binding['grounded_retention']['keys']),responses=responses,
        native=errors_read,comparison=comparisons,certificates=certificates,
        accepted_updates=sum(v['update']['accepted'] for v in learning['losses']),
        objective_before=learning['losses'][0]['update']['objective_before'],
        objective_after=learning['losses'][-1]['update']['objective_after'],
        controls=read(a.root/'train-report.json')['controls']+controls,exact_recovery=True,rollback=True,
        unchanged_source_history=True,acceptance='author-checked',serving=False,scientific_admission=False,
        useful_transfer_established=False,wall_seconds=time.perf_counter()-started)
    write_committed(a.output/'readback.json',result)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()),accepted=result['accepted_updates'],
        changed=len(changed),retained_checks=checks,objective_before=result['objective_before'],
        objective_after=result['objective_after'],native=errors_read)))


if __name__ == '__main__':
    main()
