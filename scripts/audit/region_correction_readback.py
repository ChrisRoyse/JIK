"""Read actual learning, strict retention and the first margin solve from persisted bytes."""
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


def retained(learning):
    ceilings = learning['binding']['grounded_retention']['ceilings']
    count = 0
    for row in learning['losses']:
        update = row['update']
        check(len(update['trials']) <= update['trial_limit'],'REGION_READER_TRIAL_CAP')
        for k,kinds in update['after_scores'].items():
            for kind,lanes in kinds.items():
                for n,value in lanes.items():
                    check(value <= ceilings[k][kind][n],'REGION_READER_RETENTION')
                    count += 1
        check(all(v <= update['before_native'][k][n] for k,lanes in update['after_native'].items()
                  for n,v in lanes.items()),'REGION_READER_NEW_RETENTION')
        before = sum(max(0.,r['cost']-v) for r,v in zip(update['rows'],update['margins_before']))
        after = sum(max(0.,r['cost']-v) for r,v in zip(update['rows'],update['margins_after']))
        # Reports use FP32 reductions; strict reported acceptance and actual native comparisons are separate.
        check(abs(before-update['objective_before']) < 1e-4 and abs(after-update['objective_after']) < 1e-4,
              'REGION_READER_MARGIN_SUM')
        check(update['objective_after'] < update['objective_before'] if update['accepted'] else
              update['objective_after'] == update['objective_before'],'REGION_READER_ADVANCE')
        if update['accepted']:
            chosen = update['trials'][update['selected_trial']]
            check(chosen['retained'] and chosen['new_native_retained'] and chosen['scores'] == update['after_scores']
                  and chosen['native'] == update['after_native'],'REGION_READER_SELECTED_TRIAL')
    return count


def certificate(c):
    if c['status'] != 'certified_minimum_norm':
        return dict(status=c['status'])
    gram,rhs,weights = [np.array(c[n]) for n in ('gram','rhs','weights')]
    residual = gram@weights-rhs
    tol = c['tolerance']
    check(np.min(weights) >= 0 and np.min(residual) >= -tol
          and np.max(np.abs(weights*residual)) <= tol and c['iterations'] <= c['iteration_limit'],
          'REGION_READER_DUAL')
    return dict(status=c['status'],minimum_residual=float(residual.min()),
                complementarity=float(np.max(np.abs(weights*residual))),tolerance=tol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(),'REGION_READER_EXISTS')
    config,learning = read(a.root/'config.json'),read(a.root/'learning.json')
    binding = read(a.root/'training-binding.json')
    check(binding == learning['binding'],'REGION_READER_BINDING')
    audit = read(Path(config['spec']['correction']['root'])/'report.json')
    source_learning = read(Path(audit['source'])/'learning.json')
    check(learning['before'] == source_learning['before'] and binding['grounded_retention'] ==
          source_learning['binding']['grounded_retention'],'REGION_READER_FIXED_BASELINE')
    checks = retained(learning)
    stages = [checkpoint(a.root/'candidates'/n,learning[p]) for n,p in
              (('region-1-start','start_pin'),('region-1-mid','mid_pin'),('region-1','candidate_pin'))]
    start,mid,final = stages
    parent = config['spec']['checkpoint']
    old = checkpoint(Path(parent['root']),parent['pin'])
    check(all(torch.equal(v,start[0][n]) for n,v in old[0].items())
          and start[2]['fingerprint']['optimizer'] == old[2]['fingerprint']['optimizer'],
          'REGION_READER_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages),'REGION_READER_CHECKPOINT_BINDING')
    check(all(start[2]['offsets'][n] == old[2]['offsets'][n]+old[2]['step'] for n in start[2]['names']),
          'REGION_READER_CLOCK_CONTINUITY')
    changed = [n for n,v in start[0].items() if not torch.equal(v,final[0][n])]
    check(changed == learning['changed'] and set(changed) <= set(learning['eligible']),'REGION_READER_PARAMETER_SCOPE')
    for key in binding['grounded_retention']['keys']:
        check(learning['before'][key] == learning['after'][key],'REGION_READER_OLD_BEHAVIOUR')
    basis_root = Path(config['spec']['basis']['root'])
    basis = read(basis_root/'manifest.json')
    check(sha((basis_root/'manifest.json').read_bytes()) == config['spec']['basis']['pin'],'REGION_READER_BASIS')
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as z:
        basis_arrays = {n:z[n] for n in z.files}
    check(evidence(basis_root/'arrays.npz') == basis['arrays'],'REGION_READER_BASIS_ARRAYS')
    responses = []
    for i,row in enumerate(basis['branches']):
        n = row['parameter']
        prefix = n.removesuffix('weight')
        h,p,y = [basis_arrays[v+'_'+str(i)] for v in ('h','p','y')]
        check(np.array_equal(start[0][prefix+'projector'].numpy(),p)
              and torch.equal(start[0][prefix+'anchor'],old[0][n])
              and all(torch.equal(s[0][prefix+suffix],start[0][prefix+suffix]) for s in stages
                      for suffix in ('projector','anchor')),'REGION_READER_FROZEN_BUFFERS')
        residual = (h.astype(np.float64)@p)@(final[0][n].double().numpy()-old[0][n].double().numpy()).T
        check((y+residual.astype(np.float32)).tobytes() == y.tobytes(),'REGION_READER_SEPARATED_ARITHMETIC')
        responses.append(dict(name=n,equal=True,residual_max=float(np.max(np.abs(residual)))))
    first = read(a.root/'first-correction.json')
    check(evidence(a.root/'first-correction.npz') == first['file'],'REGION_READER_FIRST_BYTES')
    with np.load(a.root/'first-correction.npz',allow_pickle=False) as z:
        arrays = {n:z[n] for n in z.files}
    update = learning['losses'][0]['update']
    g,deficit,delta = [arrays[n] for n in ('gradients','deficit','delta')]
    check(digest(fingerprint(torch.from_numpy(g))) == update['gradient'],'REGION_READER_GRADIENT_PIN')
    c = update['certificate']
    certificates = [certificate(v['update']['certificate']) for v in learning['losses']]
    if c['status'] == 'certified_minimum_norm':
        norms = np.linalg.norm(g,axis=1)
        normalized = g[np.array(c['active'])]/norms[np.array(c['active']),None]
        check(np.allclose(normalized@normalized.T,c['gram'],rtol=1e-12,atol=1e-12)
              and np.allclose(deficit[np.array(c['active'])]/norms[np.array(c['active'])],c['rhs'],rtol=1e-12,atol=1e-12)
              and np.allclose(normalized.T@np.array(c['weights']),delta,rtol=1e-11,atol=1e-11),
              'REGION_READER_LINEAR_SOLVE')
    offset = 0
    for i,n in enumerate(first['names']):
        before,after = arrays['before_'+str(i)],arrays['after_'+str(i)]
        part = delta[offset:offset+before.size].reshape(before.shape)
        check(np.array_equal(before,start[0][n].numpy()) and
              np.array_equal(after,(before.astype(np.float64)+update['scale']*part).astype(np.float32)),
              'REGION_READER_FIRST_STEP')
        offset += before.size
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
                  and pin == digest(dict(seq=seq,previous=previous,body=body)),'REGION_READER_JOURNAL')
            if body.get('payload'):
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'],'REGION_READER_PAYLOAD')
            if body['kind'] == 'native_region_correction_candidate':
                check(body == learning['decision'] and json.loads(payload)['after'] == learning['after'],
                      'REGION_READER_DECISION')
            head,count = pin,count+1
    with sqlite3.connect((Path(config['source'])/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        original = list(db.execute('SELECT * FROM events ORDER BY seq'))
        forecast_raw, = db.execute('SELECT data FROM payloads WHERE hash=?',(config['spec']['experiment'],)).fetchone()
    check(all_events[:-1] == original,'REGION_READER_SOURCE_HISTORY')
    check(sha(forecast_raw) == config['spec']['experiment'],'REGION_READER_FORECAST_PIN')
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
              'REGION_READER_PREACTION_JOIN')
        before,after = learning['before'][key]['termination'],learning['after'][key]['termination']
        truth = before['score']['outcome']
        predictions = dict(sealed_forecast=saved['models']['corrected']['termination'],
                           expanded_memory=before['prediction'],learned_weights=after['prediction'])
        stages_read = {}
        for stage,prediction in predictions.items():
            stages_read[stage] = {n:dict(probabilities=p['probabilities'],
                brier=sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities'])))
                for n,p in dict(common=prediction['common'],**prediction['instruments']).items()}
        check(before['prediction']['bank'] == after['prediction']['bank'],'REGION_READER_COMPARISON_BANK')
        comparisons[key] = dict(condition=pair['condition'],observation=pair['observation'],outcome=truth,
                                scores=stages_read,heldout=False,source_forecast_unchanged=True)
    snapshot,cold = read(a.root/'snapshot-final.json'),read(a.root/'cold-report.json')
    check(snapshot['events'] == count == cold['events'] and snapshot['head'] == head == cold['head']
          and digest(snapshot) == cold['snapshot'] and cold['journal'] == sha((a.root/'experience.sqlite3').read_bytes()),
          'REGION_READER_COLD')
    check(learning['exact_optimizer_recovery'] and learning['rollback_behaviour_equal']
          and learning['rollback_fingerprint'] == start[2]['fingerprint'] and not learning['decision']['result']['selected'],
          'REGION_READER_ROLLBACK_ADMISSION')
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'REGION_READER_CONTROL_CODE')
        else:
            raise AssertionError('REGION_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    bad = copy.deepcopy(learning)
    key = next(iter(binding['grounded_retention']['ceilings']))
    bad['losses'][0]['update']['after_scores'][key]['termination']['common'] = 3.
    refuse('concealed_retention','REGION_READER_RETENTION',lambda:retained(bad))
    if c['status'] == 'certified_minimum_norm':
        bad_certificate = copy.deepcopy(c)
        bad_certificate['weights'][0] = -1.
        refuse('invalid_multiplier','REGION_READER_DUAL',lambda:certificate(bad_certificate))
    a.output.mkdir()
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/region-1',broken)
    with (broken/'weights.pt').open('r+b') as f:
        f.seek(256)
        byte = f.read(1)
        f.seek(256)
        f.write(bytes([byte[0]^1]))
    refuse('corrupt_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,learning['candidate_pin']))
    key,pair = next(iter(pairs.items()))
    wrong = next(k for k in next(iter(native.values())) if k != pair['observation'])
    refuse('substituted_target','PROPOSAL_WEIGHT_NATIVE_SCORE',lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native.items()}))
    result = dict(schema='region-correction-readback-v1',learning=evidence(a.root/'learning.json'),
        candidate=learning['candidate_pin'],parent=parent,events=count,head=head,changed=changed,
        retained_checks=checks,retained_examples=len(binding['grounded_retention']['keys']),responses=responses,
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
