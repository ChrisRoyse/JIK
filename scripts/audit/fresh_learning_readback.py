"""Independent readback of newly observed feedback learning and retained responses."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
import sqlite3

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_echo_readback import distance
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_recognition import read
from audit.response_learning_readback import updates, basis_check
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from claims.fresh_echo_population import FreshEchoPopulation


def read_learning(root, *, name, kind, parent, basis_spec, basis_parent):
    root = Path(root)
    learning, config = read(root/'learning.json'),read(root/'config.json')
    binding = read(root/'training-binding.json')
    check(binding == learning['binding'], 'LEARNING_READER_BINDING')
    certificates = updates(learning)
    pins = [learning[k] for k in ('start_pin','mid_pin','candidate_pin')]
    stages = [checkpoint(root/'candidates'/n,p) for n,p in zip((name+'-start',name+'-mid',name),pins)]
    start, mid, final = stages
    previous = checkpoint(Path(parent['root']),parent['pin'])
    check(start[2]['fingerprint'] == previous[2]['fingerprint'], 'LEARNING_READER_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages), 'LEARNING_READER_CHECKPOINT_BINDINGS')
    check(all(start[2]['offsets'][n] == previous[2]['offsets'][n]+previous[2]['step'] for n in start[2]['names']),
          'LEARNING_READER_CLOCK_CONTINUITY')
    ceilings = binding['grounded_retention']['ceilings']
    prior_ceilings = previous[2]['binding']['grounded_retention']['ceilings']
    prior_report = read(Path(config['spec']['retained_report']['path']))
    check(sha(Path(config['spec']['retained_report']['path']).read_bytes()) ==
          config['spec']['retained_report']['pin'], 'FRESH_READER_PARENT_REPORT')
    retained_keys = previous[2]['binding']['plan']['train']
    check(set(ceilings) == set(retained_keys) and not set(retained_keys) & set(binding['plan']['evaluate']),
          'FRESH_READER_RETAINED_SPLIT')
    for key, kinds in ceilings.items():
        item = prior_report['after'][key]
        for group,names in kinds.items():
            for instrument,value in names.items():
                if group == 'native':
                    achieved = item['assessment']['instruments'][instrument]['prediction_loss']
                else:
                    prediction = item['termination']['prediction']
                    p = prediction['common'] if instrument == 'common' else prediction['instruments'][instrument]
                    truth = item['termination']['score']['outcome']
                    achieved = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
                old = prior_ceilings.get(key,{}).get(group,{}).get(instrument,achieved)
                check(value == min(old,achieved), 'FRESH_READER_CEILING_CHANGE')
    basis_root = Path(basis_spec['root'])
    basis = read(basis_root/'manifest.json')
    check(sha((basis_root/'manifest.json').read_bytes()) == basis_spec['pin'], 'LEARNING_READER_BASIS_PIN')
    raw = (basis_root/'arrays.npz').read_bytes()
    check(dict(bytes=len(raw),sha256=sha(raw)) == basis['arrays'], 'LEARNING_READER_BASIS_BYTES')
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n].copy() for n in archive.files}
    basis_report = basis_check(basis,arrays)
    original = checkpoint(Path(basis_parent['root']),basis_parent['pin'])
    basis_binding = copy.deepcopy(binding)
    basis_binding['fresh_feedback_learning'].pop('response_basis')
    check(basis['parent'] == basis_parent['pin'] and basis['parent_binding'] == digest(basis_binding)
          and basis['retained_keys'] == retained_keys, 'LEARNING_READER_BASIS_PARENT')
    eligible = {r['parameter'] for r in basis['branches']}
    changed = [n for n in start[0] if not torch.equal(start[0][n],final[0][n])]
    check(changed == learning['changed'] and set(changed) <= eligible == set(learning['eligible']), 'LEARNING_READER_SCOPE')
    drift = []
    for i,row in enumerate(basis['branches']):
        n = row['parameter']
        h = arrays[f'h_{i}'].astype(np.float64)
        old, new = original[0][n].double().numpy(),final[0][n].double().numpy()
        algebraic = h@(new-old).T
        gamma = h.shape[1]*np.finfo(np.float32).eps/(1-h.shape[1]*np.finfo(np.float32).eps)
        bound = float(np.max(np.abs(algebraic)+gamma*(np.abs(h)@(np.abs(old)+np.abs(new)).T)))
        actual = learning['losses'][-1]['update']['after_response_drift'][row['name']]
        check(actual['max_absolute'] <= bound and np.isfinite(bound), 'LEARNING_READER_ROUNDOFF')
        check(np.max(np.abs(h@old.T-arrays[f'y_{i}'])) <= np.max(gamma*(np.abs(h)@np.abs(old).T)),
              'LEARNING_READER_PARENT_RESPONSE')
        drift.append(dict(name=row['name'],actual=actual,weight_delta_response=float(np.max(np.abs(algebraic))),
                          floating_point_bound=bound))
    retained = []
    for key,kinds in binding['grounded_retention']['ceilings'].items():
        item = learning['after'][key]
        for instrument,ceiling in kinds['native'].items():
            saved = item['assessment']['instruments'][instrument]
            value = distance(np.array(saved['prediction_cosines']))
            check(value == saved['prediction_loss'] and value <= ceiling, 'LEARNING_READER_NATIVE_RETENTION')
            retained.append(dict(key=key,kind='native',name=instrument,value=value,ceiling=ceiling))
        for instrument,ceiling in kinds['termination'].items():
            prediction = item['termination']['prediction']
            p = prediction['common'] if instrument == 'common' else prediction['instruments'][instrument]
            truth = item['termination']['score']['outcome']
            value = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
            check(value <= ceiling, 'LEARNING_READER_OUTCOME_RETENTION')
            retained.append(dict(key=key,kind='termination',name=instrument,value=value,ceiling=ceiling))
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = FreshEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = binding['fresh_feedback_learning']['pairs']
    native_arrays = {}
    for instrument in learning['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/instrument/'slots.npz',allow_pickle=False) as data:
            native_arrays[instrument] = {k:np.concatenate([data['vectors'][data['offsets'][r['unit']]:data['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    native = {key:{stage:errors(learning[stage][key],{n:v[pair['observation']] for n,v in native_arrays.items()})
                  for stage in ('before','after')} for key,pair in pairs.items()}
    head, count, decision = ZERO,0,None
    with sqlite3.connect((root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous_pin,raw,pin in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous_pin == head and raw == canonical(body)
                  and pin == digest(dict(seq=seq,previous=previous_pin,body=body)), 'LEARNING_READER_CHAIN')
            if body.get('payload') is not None:
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'], 'LEARNING_READER_PAYLOAD')
            if body['kind'] == kind:
                check(decision is None and body == learning['decision'], 'LEARNING_READER_DECISION')
                decision = json.loads(payload)
                check(decision['before'] == learning['before'] and decision['after'] == learning['after'],
                      'LEARNING_READER_DECISION_SCORES')
            count, head = count+1,pin
    snapshot, cold = read(root/'snapshot-final.json'),read(root/'cold-report.json')
    check(count == snapshot['events'] and head == snapshot['head'] and cold['snapshot'] == digest(snapshot)
          and cold['journal'] == sha((root/'experience.sqlite3').read_bytes()), 'LEARNING_READER_COLD')
    check(decision is not None and not decision['result']['selected'], 'LEARNING_READER_NO_ADMISSION')
    report = dict(acceptance='author-checked',events=count,head=head,source_checkpoint=parent['pin'],candidate=pins[-1],
        basis=basis_report,response_drift=drift,certificates=certificates,retained=retained,native=native,changed=changed,
        accepted_updates=sum(r['accepted'] for r in certificates),objective_before=learning['losses'][0]['update']['objective_before'],
        objective_after=certificates[-1]['objective'],exact_optimizer_recovery=learning['exact_optimizer_recovery'],
        rollback=learning['rollback_behaviour_equal'],serving=False,scientific_admission=False,useful_transfer_established=False)
    return report,learning,basis,arrays,native_arrays


def trial_diagnostics(learning):
    ceilings = learning['binding']['grounded_retention']['ceilings']
    rows = []
    for step,loss in enumerate(learning['losses'],1):
        update = loss['update']
        for index,trial in enumerate(update['trials']):
            failures = [dict(key=k,kind=kind,instrument=n,excess=value-ceilings[k][kind][n])
                for k,kinds in trial['scores'].items() for kind,names in kinds.items()
                for n,value in names.items() if value > ceilings[k][kind][n]]
            rows.append(dict(step=step,trial=index,scale=trial['scale'],failures=failures,
                objective_before=update['objective_before'],objective=trial['objective'],
                objective_decreased=trial['objective_decreased'],retained=trial['retained'],
                algebraic_response=update['projection']['algebraic_response'],
                actual_response_drift=trial['response_drift']))
    return dict(rows=rows,cause='unresolved',
        interpretation='Small response drift and score excess can motivate a numerical-method audit; no tolerance is granted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    check(not a.output.exists(),'FRESH_READER_EXISTS')
    config = read(a.root/'config.json')
    spec = config['spec']
    report,learning,basis,arrays,native = read_learning(a.root,name='fresh-1',
        kind='fresh_learning_candidate',parent=spec['checkpoint'],basis_spec=spec['basis'],
        basis_parent=spec['checkpoint'])
    pairs = learning['binding']['fresh_feedback_learning']['pairs']
    source = Path(config['source'])
    source_snapshot = read(source/'snapshot-final.json')
    snapshot = read(a.root/'snapshot-final.json')
    check({k:v for k,v in snapshot.items() if k not in ('head','events','fresh_learning_candidates')} ==
          {k:v for k,v in source_snapshot.items() if k not in ('head','events')},'FRESH_READER_STATE_ISOLATION')
    with sqlite3.connect((source/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        raw, = db.execute('SELECT data FROM payloads WHERE hash=?',(spec['experiment'],)).fetchone()
        check(sha(raw) == spec['experiment'],'FRESH_READER_FORECAST_PAYLOAD')
        forecast = json.loads(raw)
    comparison = {}
    for key,pair in pairs.items():
        entry = source_snapshot['native_feedback_contexts'][pair['context']]
        saved = forecast['predictions'][pair['condition']]
        check(entry['status'] == 'corrected_state' and entry['variant'] == 'corrected'
              and entry['observation'] == pair['observation'] and entry['receipt'] == pair['receipt']
              and entry['native'] == pair['native'] and pair['prestate'] == forecast['state']
              and pair['input'] == saved['values'] and pair['fields'] == saved['fields']
              and pair['request'] == digest(saved['request'])
              and not pair['correction_state_used_as_input'] and not pair['heldout'],
              'FRESH_READER_SUPERVISION_JOIN')
        before = learning['before'][key]['termination']
        after = learning['after'][key]['termination']
        truth = before['score']['outcome']
        original = saved['models']['corrected']['termination']
        stages = dict(sealed_forecast=original,expanded_memory=before['prediction'],
                      learned_weights=after['prediction'])
        scores = {}
        for stage,prediction in stages.items():
            scores[stage] = {}
            for name,p in dict(common=prediction['common'],**prediction['instruments']).items():
                scores[stage][name] = dict(probabilities=p['probabilities'],
                    brier=sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities'])))
        check(before['prediction']['bank'] == after['prediction']['bank'], 'FRESH_READER_COMPARISON_BANK')
        check(len(original['bank']['rows']) == learning['binding']['fresh_feedback_learning']['regions']['first']
              and len(before['prediction']['bank']['rows']) == len(original['bank']['rows'])+len(pairs),
              'FRESH_READER_REGION_EXTENSION')
        comparison[key] = dict(condition=pair['condition'],observation=pair['observation'],outcome=truth,
            scores=scores,heldout=False,source_forecast_unchanged=True)
    controls = []

    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'FRESH_READER_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('FRESH_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name,code=code))

    damaged = {n:v.copy() for n,v in arrays.items()}
    damaged['p_0'][0,0] += .1
    refuse('changed_projector','RESPONSE_READER_PROJECTOR',lambda:basis_check(basis,damaged))
    bad = copy.deepcopy(learning)
    bad['losses'][0]['update']['after_scores'][basis['retained_keys'][0]]['termination']['common'] = 3.
    refuse('concealed_retention_failure','RESPONSE_READER_CEILING',lambda:updates(bad))
    a.output.mkdir()
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/fresh-1',broken)
    with (broken/'weights.pt').open('r+b') as handle:
        handle.seek(256)
        value = handle.read(1)
        handle.seek(256)
        handle.write(bytes([value[0]^1]))
    refuse('changed_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,learning['candidate_pin']))
    key,pair = next(iter(pairs.items()))
    wrong = next(k for k in next(iter(native.values())) if k != pair['observation'])
    refuse('swapped_native_output','PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native.items()}))
    report.update(schema='fresh-learning-readback-v1',comparison=comparison,
        trial_diagnostics=trial_diagnostics(learning),
        controls=read(a.root/'train-report.json')['controls']+controls,
        boundaries=['observed_outputs_to_preaction_supervision','weights_and_all_prior_retention',
                    'checkpoint_optimizer_rollback_and_cold_decision'],source_state_unchanged=True)
    write_committed(a.output/'readback.json',report)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()),events=report['events'],
        controls=len(report['controls']),changed=len(report['changed']),retained=len(report['retained']),
        accepted_updates=report['accepted_updates'],objective_before=report['objective_before'],
        objective_after=report['objective_after'])))


if __name__ == '__main__':
    main()
