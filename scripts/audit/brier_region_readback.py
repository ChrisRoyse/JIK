"""Read typed correction, strict retention and six first-step solves from persisted bytes."""
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
from audit.proposal_recognition import read, evidence
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest, ZERO
from claims.fresh_echo_population import FreshEchoPopulation
from claims.process_conditions import write_committed
from audit.brier_region_readback_math import retained, first_solve, floors, certificate
from audit.brier_region_bounds_readback import cache_readback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(),'TYPED_READER_EXISTS')
    config,learning = read(a.root/'config.json'),read(a.root/'learning.json')
    binding = read(a.root/'training-binding.json')
    check(binding == learning['binding'],'TYPED_READER_BINDING')
    audit = read(Path(config['spec']['correction']['root'])/'report.json')
    source_learning = read(Path(audit['source'])/'learning.json')
    check(learning['before'] == source_learning['before'] and binding['grounded_retention'] ==
          source_learning['binding']['grounded_retention'],'TYPED_READER_FIXED_BASELINE')
    checks = retained(learning)
    stages = [checkpoint(a.root/'candidates'/n,learning[p]) for n,p in
              (('brier-1-start','start_pin'),('brier-1-mid','mid_pin'),('brier-1','candidate_pin'))]
    start,mid,final = stages
    parent = config['spec']['checkpoint']
    old = checkpoint(Path(parent['root']),parent['pin'])
    check(all(torch.equal(v,start[0][n]) for n,v in old[0].items())
          and start[2]['fingerprint']['optimizer'] == old[2]['fingerprint']['optimizer'],
          'TYPED_READER_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages),'TYPED_READER_CHECKPOINT_BINDING')
    check(all(start[2]['offsets'][n] == old[2]['offsets'][n]+old[2]['step'] for n in start[2]['names']),
          'TYPED_READER_CLOCK_CONTINUITY')
    changed = [n for n,v in start[0].items() if not torch.equal(v,final[0][n])]
    check(changed == learning['changed'] and set(changed) <= set(learning['eligible']),'TYPED_READER_PARAMETER_SCOPE')
    for key in binding['grounded_retention']['keys']:
        check(learning['before'][key] == learning['after'][key],'TYPED_READER_OLD_BEHAVIOUR')
    basis_root = Path(config['spec']['basis']['root'])
    basis = read(basis_root/'manifest.json')
    check(sha((basis_root/'manifest.json').read_bytes()) == config['spec']['basis']['pin'],'TYPED_READER_BASIS')
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as z:
        basis_arrays = {n:z[n] for n in z.files}
    check(evidence(basis_root/'arrays.npz') == basis['arrays'],'TYPED_READER_BASIS_ARRAYS')
    responses = []
    for i,row in enumerate(basis['branches']):
        n = row['parameter']
        prefix = n.removesuffix('weight')
        h,p,y = [basis_arrays[v+'_'+str(i)] for v in ('h','p','y')]
        check(np.array_equal(start[0][prefix+'projector'].numpy(),p)
              and torch.equal(start[0][prefix+'anchor'],old[0][n])
              and all(torch.equal(s[0][prefix+suffix],start[0][prefix+suffix]) for s in stages
                      for suffix in ('projector','anchor')),'TYPED_READER_FROZEN_BUFFERS')
        residual = (h.astype(np.float64)@p)@(final[0][n].double().numpy()-old[0][n].double().numpy()).T
        check((y+residual.astype(np.float32)).tobytes() == y.tobytes(),'TYPED_READER_SEPARATED_ARITHMETIC')
        responses.append(dict(name=n,equal=True,residual_max=float(np.max(np.abs(residual)))))
    certificates, c = first_solve(a.root,learning,start)
    floors(learning,config)
    cache = cache_readback(a.root,learning)
    for cursor,stage in ((0,start),(len(learning['losses'])//2,mid)):
        expected = digest(dict(binding=digest(binding),rows=digest(binding['brier_region']['rows']),
                               state=fingerprint(stage[0])))
        check(learning['losses'][cursor]['update']['cache_key'] == expected,
              'BRIER_READER_CACHE_CHECKPOINT')
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = FreshEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = binding['fresh_feedback_learning']['pairs']
    native = {}
    for name in learning['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as z:
            native[name] = {k:np.concatenate([z['vectors'][z['offsets'][r['unit']]:z['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    errors_read = {key:{stage:errors(learning[stage][key],{n:v[pair['observation']] for n,v in native.items()})
                       for stage in ('before','after')} for key,pair in pairs.items()}
    head,count = ZERO,0
    with sqlite3.connect((a.root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        all_events = list(db.execute('SELECT * FROM events ORDER BY seq'))
        for seq,previous,raw,pin in all_events:
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body)
                  and pin == digest(dict(seq=seq,previous=previous,body=body)),'TYPED_READER_JOURNAL')
            if body.get('payload'):
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'],'TYPED_READER_PAYLOAD')
            if body['kind'] == 'brier_region_candidate':
                check(body == learning['decision'] and json.loads(payload)['after'] == learning['after'],
                      'TYPED_READER_DECISION')
            head,count = pin,count+1
    with sqlite3.connect((Path(config['source'])/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        original = list(db.execute('SELECT * FROM events ORDER BY seq'))
        forecast_raw, = db.execute('SELECT data FROM payloads WHERE hash=?',(config['spec']['experiment'],)).fetchone()
    check(all_events[:-1] == original,'TYPED_READER_SOURCE_HISTORY')
    check(sha(forecast_raw) == config['spec']['experiment'],'TYPED_READER_FORECAST_PIN')
    forecast = json.loads(forecast_raw)
    source_snapshot = read(Path(config['source'])/'snapshot-final.json')
    comparisons = {}
    for key,pair in pairs.items():
        entry = source_snapshot['native_feedback_contexts'][pair['context']]
        saved = forecast['predictions'][pair['condition']]
        check(entry['status'] == 'corrected_state' and entry['variant'] == 'corrected'
              and entry['observation'] == pair['observation'] and entry['receipt'] == pair['receipt']
              and entry['native'] == pair['native'] and pair['prestate'] == forecast['state']
              and pair['input'] == saved['values'] and pair['fields'] == saved['fields']
              and pair['request'] == digest(saved['request'])
              and not pair['correction_state_used_as_input'] and not pair['heldout'],
              'TYPED_READER_PREACTION_JOIN')
        before,after = learning['before'][key]['termination'],learning['after'][key]['termination']
        truth = before['score']['outcome']
        predictions = dict(sealed_forecast=saved['models']['corrected']['termination'],
                           expanded_memory=before['prediction'],learned_weights=after['prediction'])
        stages_read = {}
        for stage,prediction in predictions.items():
            stages_read[stage] = {n:dict(probabilities=p['probabilities'],
                brier=sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities'])))
                for n,p in dict(common=prediction['common'],**prediction['instruments']).items()}
        check(before['prediction']['bank'] == after['prediction']['bank'],'TYPED_READER_COMPARISON_BANK')
        comparisons[key] = dict(condition=pair['condition'],observation=pair['observation'],outcome=truth,
                                scores=stages_read,heldout=False,source_forecast_unchanged=True)
    snapshot,cold = read(a.root/'snapshot-final.json'),read(a.root/'cold-report.json')
    check(snapshot['events'] == count == cold['events'] and snapshot['head'] == head == cold['head']
          and digest(snapshot) == cold['snapshot'] and cold['journal'] == sha((a.root/'experience.sqlite3').read_bytes()),
          'TYPED_READER_COLD')
    check(learning['exact_optimizer_recovery'] and learning['rollback_behaviour_equal']
          and learning['rollback_fingerprint'] == start[2]['fingerprint'] and not learning['decision']['result']['selected'],
          'TYPED_READER_ROLLBACK_ADMISSION')
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'TYPED_READER_CONTROL_CODE')
        else:
            raise AssertionError('TYPED_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    bad = copy.deepcopy(learning)
    key = next(iter(binding['grounded_retention']['ceilings']))
    bad['losses'][0]['update']['after_scores'][key]['termination']['common'] = 3.
    refuse('concealed_retention','TYPED_READER_RETENTION',lambda:retained(bad))
    bad = copy.deepcopy(learning)
    new_key = next(iter(pairs))
    bad['losses'][0]['update']['after_measurements']['brier'][new_key]['common'] = 3.
    refuse('concealed_new_brier_regression','TYPED_READER_NEW_RETENTION',lambda:retained(bad))
    bad = copy.deepcopy(learning)
    lane = next(iter(bad['binding']['typed_correction']['rows']))
    odds = next(r for r in bad['binding']['typed_correction']['rows'][lane] if r['kind'] == 'odds')
    odds['floor'] += 1.
    refuse('altered_original_odds_floor','TYPED_READER_ODDS_FLOOR',lambda:floors(bad,config))
    bad = copy.deepcopy(learning)
    lane = next(iter(bad['binding']['brier_region']['rows']))
    bad['binding']['brier_region']['rows'][lane][0]['floor'] -= 1.
    refuse('weakened_confidence_floor','BRIER_READER_FLOOR',lambda:floors(bad,config))
    bad = copy.deepcopy(learning)
    bad['losses'][0]['update']['cache_key'] = '0'*64
    refuse('cache_key_substitution','BRIER_READER_CACHE_BINDING',lambda:cache_readback(a.root,bad))
    if c['status'] == 'certified_minimum_norm':
        bad_certificate = copy.deepcopy(c)
        bad_certificate['weights'][0] = -1.
        refuse('invalid_multiplier','PRECISE_READER_DUAL' if c['method'] != 'original_fp64' else 'REGION_READER_DUAL',lambda:certificate(bad_certificate))
    a.output.mkdir()
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/brier-1',broken)
    with (broken/'weights.pt').open('r+b') as f:
        f.seek(256)
        byte = f.read(1)
        f.seek(256)
        f.write(bytes([byte[0]^1]))
    refuse('corrupt_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,learning['candidate_pin']))
    key,pair = next(iter(pairs.items()))
    wrong = next(k for k in next(iter(native.values())) if k != pair['observation'])
    refuse('substituted_target','PROPOSAL_WEIGHT_NATIVE_SCORE',lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native.items()}))
    result = dict(schema='brier-region-readback-v1',learning=evidence(a.root/'learning.json'),
        candidate=learning['candidate_pin'],parent=parent,events=count,head=head,changed=changed,cache=cache,
        retained_checks=checks,retained_examples=len(binding['grounded_retention']['keys']),responses=responses,
        new_native_checks=sum(len(v) for v in learning['losses'][0]['update']['before_measurements']['native'].values())*len(learning['losses']),
        new_brier_checks=sum(len(v) for v in learning['losses'][0]['update']['before_measurements']['brier'].values())*len(learning['losses']),
        native=errors_read,comparison=comparisons,certificates=certificates,accepted_updates=sum(v['update']['accepted'] for v in learning['losses']),
        objective_before=learning['losses'][0]['update']['objective_before'],objective_after=learning['losses'][-1]['update']['objective_after'],
        original_objective_before=learning['losses'][0]['loss'],controls=read(a.root/'train-report.json')['controls']+controls,
        exact_recovery=True,rollback=True,unchanged_source_history=True,acceptance='author-checked',
        serving=False,scientific_admission=False,useful_transfer_established=False,wall_seconds=time.perf_counter()-started)
    write_committed(a.output/'readback.json',result)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()),accepted=result['accepted_updates'],
        objective_before=result['objective_before'],objective_after=result['objective_after'],native=errors_read)))


if __name__ == '__main__':
    main()
