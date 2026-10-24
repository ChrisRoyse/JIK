"""Independent CPU readback of response preservation, checkpoints and real outcomes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_echo_readback import distance
from audit.proposal_recognition import read
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from claims.proposal_echo_population import ProposalEchoPopulation


def basis_check(manifest, arrays):
    result = []
    for i,row in enumerate(manifest['branches']):
        h,y,p,s = [arrays[f'{n}_{i}'] for n in ('h','y','p','s')]
        width = row['shape'][1]
        check(h.shape == (row['rows'],width) and y.shape == (row['rows'],row['shape'][0]),
              'RESPONSE_READER_ROWS')
        _, singular, vh = np.linalg.svd(h.astype(np.float64),full_matrices=False)
        threshold = max(h.shape)*np.finfo(np.float64).eps*singular[0]
        rank = int(np.sum(singular > threshold))
        expected = np.eye(width)-vh[:rank].T@vh[:rank]
        tolerance = 64*np.finfo(np.float64).eps*max(h.shape)*max(1.,float(singular[0]))
        check(rank == row['rank'] and np.allclose(s,singular,rtol=0,atol=tolerance), 'RESPONSE_READER_RANK')
        symmetry = float(np.max(np.abs(p-p.T)))
        idempotence = float(np.max(np.abs(p@p-p)))
        annihilation = float(np.max(np.abs(h.astype(np.float64)@p)))
        check(max(symmetry,idempotence,annihilation,float(np.max(np.abs(p-expected)))) <= tolerance,
              'RESPONSE_READER_PROJECTOR')
        check([c['start'] for c in row['calls']] == [0]+[c['stop'] for c in row['calls'][:-1]]
              and row['calls'][-1]['stop'] == len(h)
              and set(c['key'] for c in row['calls']) == set(manifest['retained_keys']), 'RESPONSE_READER_COVERAGE')
        result.append(dict(name=row['name'],rows=len(h),rank=rank,free=width-rank,
            symmetry=symmetry,idempotence=idempotence,annihilation=annihilation,tolerance=tolerance))
    return result


def updates(learning):
    ceilings = learning['binding']['grounded_retention']['ceilings']
    order = {(k,t,n) for k,v in ceilings.items() for t,names in v.items() for n in names}
    previous = None
    output = []
    def kept(scores):
        check(set(scores) == set(ceilings) and all(set(scores[k]) == set(v) for k,v in ceilings.items())
              and all(set(scores[k][t]) == set(names) for k,v in ceilings.items() for t,names in v.items()),
              'RESPONSE_READER_SCORE_ROSTER')
        return all(scores[k][t][n] <= ceiling for k,v in ceilings.items() for t,names in v.items()
                   for n,ceiling in names.items())
    for row in learning['losses']:
        u, p = row['update'],row['update']['projection']
        check(set(map(tuple,u['constraint_order'])) == order and kept(u['before_scores']) and kept(u['after_scores']),
              'RESPONSE_READER_CEILING')
        if previous:
            check(previous['after_scores'] == u['before_scores'] and previous['objective_after'] == u['objective_before']
                  and previous['after'] == u['before'], 'RESPONSE_READER_STEP_CHAIN')
        check(0 <= u['scale'] <= 1 and len(u['trials']) <= u['trial_limit'] == 4,
              'RESPONSE_READER_BUDGET')
        if p['status'] == 'projected_objective':
            tolerance = 128*np.finfo(np.float64).eps*max(1.,p['adam_norm'])
            check(abs(p['norm']-p['adam_norm']) <= tolerance and p['objective_direction'] < 0,
                  'RESPONSE_READER_NORM')
        feasible = []
        for i,t in enumerate(u['trials']):
            check(t['retained'] == kept(t['scores']) and t['objective_decreased'] == (t['objective'] < u['objective_before']),
                  'RESPONSE_READER_TRIAL')
            if t['retained'] and t['objective_decreased']:
                feasible.append(i)
        best = min(feasible,key=lambda i:u['trials'][i]['objective']) if feasible else None
        check(u['selected_trial'] == best and u['accepted'] == (best is not None), 'RESPONSE_READER_SELECTION')
        if best is not None:
            t = u['trials'][best]
            check(u['after_scores'] == t['scores'] and u['objective_after'] == t['objective']
                  and u['after_response_drift'] == t['response_drift'] and u['scale'] == t['scale'], 'RESPONSE_READER_ACCEPTANCE')
        else:
            check(u['before'] == u['after'] and u['before_scores'] == u['after_scores']
                  and u['objective_before'] == u['objective_after'] and u['scale'] == 0, 'RESPONSE_READER_RESTORE')
        output.append(dict(accepted=u['accepted'],scale=u['scale'],objective=u['objective_after'],trials=len(u['trials'])))
        previous = u
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    check(not a.output.exists(), 'RESPONSE_READER_EXISTS')
    a.output.mkdir()
    learning, config = read(a.root/'learning.json'),read(a.root/'config.json')
    binding = read(a.root/'training-binding.json')
    check(binding == learning['binding'], 'RESPONSE_READER_BINDING')
    certs = updates(learning)
    pins = [learning[k] for k in ('start_pin','mid_pin','candidate_pin')]
    stages = [checkpoint(a.root/'candidates'/n,p) for n,p in zip(('response-1-start','response-1-mid','response-1'),pins)]
    start, mid, final = stages
    parent = checkpoint(Path(config['spec']['continuation']['root']),config['spec']['continuation']['pin'])
    check(start[2]['fingerprint'] == parent[2]['fingerprint'], 'RESPONSE_READER_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages), 'RESPONSE_READER_CHECKPOINT_BINDING')
    check(binding['grounded_retention']['ceilings'] == parent[2]['binding']['grounded_retention']['ceilings'],
          'RESPONSE_READER_CEILING_CHANGE')
    basis = read(a.root/'basis/manifest.json')
    check(sha((a.root/'basis/manifest.json').read_bytes()) == config['spec']['basis']['pin'], 'RESPONSE_READER_BASIS_PIN')
    check(sha((a.root/'basis/arrays.npz').read_bytes()) == basis['arrays']['sha256'], 'RESPONSE_READER_ARRAY_BYTES')
    with np.load(a.root/'basis/arrays.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n].copy() for n in archive.files}
    basis_report = basis_check(basis,arrays)
    eligible = {r['parameter'] for r in basis['branches']}
    changed = [n for n in start[0] if not torch.equal(start[0][n],final[0][n])]
    check(changed == learning['changed'] and set(changed) <= eligible == set(learning['eligible']), 'RESPONSE_READER_SCOPE')
    drift = []
    for i,row in enumerate(basis['branches']):
        n = row['parameter']
        h = arrays[f'h_{i}'].astype(np.float64)
        old, new = start[0][n].double().numpy(),final[0][n].double().numpy()
        algebraic = h@(new-old).T
        gamma = h.shape[1]*np.finfo(np.float32).eps/(1-h.shape[1]*np.finfo(np.float32).eps)
        rounding = gamma*(np.abs(h)@(np.abs(old)+np.abs(new)).T)
        actual = learning['losses'][-1]['update']['after_response_drift'][row['name']]
        bound = float(np.max(np.abs(algebraic)+rounding))
        check(actual['max_absolute'] <= bound and np.isfinite(bound), 'RESPONSE_READER_ROUNDOFF')
        check(np.max(np.abs(h@old.T-arrays[f'y_{i}'])) <= np.max(gamma*(np.abs(h)@np.abs(old).T)),
              'RESPONSE_READER_PARENT_RESPONSE')
        drift.append(dict(name=row['name'],actual=actual,weight_delta_response=float(np.max(np.abs(algebraic))),
                          floating_point_bound=bound))
    retained = []
    for key,kinds in binding['grounded_retention']['ceilings'].items():
        item = learning['after'][key]
        for name,ceiling in kinds['native'].items():
            saved = item['assessment']['instruments'][name]
            value = distance(np.array(saved['prediction_cosines']))
            check(value == saved['prediction_loss'] and value <= ceiling, 'RESPONSE_READER_NATIVE_RETENTION')
            retained.append(dict(key=key,kind='native',name=name,value=value,ceiling=ceiling))
        for name,ceiling in kinds['termination'].items():
            prediction = item['termination']['prediction']
            p = prediction['common'] if name == 'common' else prediction['instruments'][name]
            truth = item['termination']['score']['outcome']
            value = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
            check(value <= ceiling, 'RESPONSE_READER_OUTCOME_RETENTION')
            retained.append(dict(key=key,kind='termination',name=name,value=value,ceiling=ceiling))
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = ProposalEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = binding['proposal_feedback']['pairs']
    native_arrays = {}
    for name in learning['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as data:
            native_arrays[name] = {k:np.concatenate([data['vectors'][data['offsets'][r['unit']]:data['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    native = {key:{stage:errors(learning[stage][key],{n:v[pair['observation']] for n,v in native_arrays.items()})
                  for stage in ('before','after')} for key,pair in pairs.items()}
    head, count, decision = ZERO,0,None
    with sqlite3.connect((a.root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous,raw,pin in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body)
                  and pin == digest(dict(seq=seq,previous=previous,body=body)), 'RESPONSE_READER_CHAIN')
            if body.get('payload') is not None:
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'], 'RESPONSE_READER_PAYLOAD')
            if body['kind'] == 'response_learning_candidate':
                check(decision is None and body == learning['decision'], 'RESPONSE_READER_DECISION')
                decision = json.loads(payload)
                check(decision['before'] == learning['before'] and decision['after'] == learning['after'],
                      'RESPONSE_READER_DECISION_SCORES')
            count, head = count+1,pin
    snapshot, cold = read(a.root/'snapshot-final.json'),read(a.root/'cold-report.json')
    check(count == snapshot['events'] and head == snapshot['head'] and cold['snapshot'] == digest(snapshot)
          and cold['journal'] == sha((a.root/'experience.sqlite3').read_bytes()), 'RESPONSE_READER_COLD')
    check(decision is not None and not decision['result']['selected'], 'RESPONSE_READER_NO_ADMISSION')
    controls = []
    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'RESPONSE_READER_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('RESPONSE_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name,code=code))
    damaged = {n:v.copy() for n,v in arrays.items()}
    damaged['p_0'][0,0] += .1
    refuse('changed_projector','RESPONSE_READER_PROJECTOR',lambda:basis_check(basis,damaged))
    bad = copy.deepcopy(learning)
    bad['losses'][0]['update']['after_scores'][next(iter(basis['retained_keys']))]['termination']['common'] = 3.
    refuse('concealed_retention_failure','RESPONSE_READER_CEILING',lambda:updates(bad))
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/response-1',broken)
    with (broken/'weights.pt').open('r+b') as handle:
        handle.seek(256); value=handle.read(1); handle.seek(256); handle.write(bytes([value[0]^1]))
    refuse('changed_checkpoint','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,pins[-1]))
    key, pair = next(iter(pairs.items()))
    wrong = next(k for k in population.observations.entries if k != pair['observation'])
    refuse('swapped_native_output','PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key],{n:v[wrong] for n,v in native_arrays.items()}))
    report = dict(schema='response-learning-readback-v1',acceptance='author-checked',events=count,head=head,
        source_checkpoint=config['spec']['continuation']['pin'],candidate=pins[-1],basis=basis_report,response_drift=drift,
        controls=read(a.root/'train-report.json')['controls']+controls,certificates=certs,retained=retained,native=native,
        changed=changed,accepted_updates=sum(r['accepted'] for r in certs),
        objective_before=learning['losses'][0]['update']['objective_before'],objective_after=certs[-1]['objective'],
        exact_optimizer_recovery=learning['exact_optimizer_recovery'],rollback=learning['rollback_behaviour_equal'],
        boundaries=['parent_and_actual_activation_basis','scoped_updates_and_fixed_ceilings','cold_decision_and_rollback'],
        serving=False,scientific_admission=False,useful_transfer_established=False)
    write_committed(a.output/'readback.json',report)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()),events=count,controls=len(report['controls']),
        accepted_updates=report['accepted_updates'],changed=len(changed),retained=len(retained),
        objective_before=report['objective_before'],objective_after=report['objective_after'])))


if __name__ == '__main__':
    main()
