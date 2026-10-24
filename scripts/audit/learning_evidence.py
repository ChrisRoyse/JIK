"""Shared CPU checks for bounded learning artifacts, independent of training code."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from audit.proposal_echo_readback import distance
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_recognition import read
from audit.response_learning_readback import updates, basis_check
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.proposal_echo_population import ProposalEchoPopulation


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
    check(binding['grounded_retention']['ceilings'] == previous[2]['binding']['grounded_retention']['ceilings'],
          'LEARNING_READER_CEILING_CHANGE')
    basis_root = Path(basis_spec['root'])
    basis = read(basis_root/'manifest.json')
    check(sha((basis_root/'manifest.json').read_bytes()) == basis_spec['pin'], 'LEARNING_READER_BASIS_PIN')
    raw = (basis_root/'arrays.npz').read_bytes()
    check(dict(bytes=len(raw),sha256=sha(raw)) == basis['arrays'], 'LEARNING_READER_BASIS_BYTES')
    with np.load(basis_root/'arrays.npz',allow_pickle=False) as archive:
        arrays = {n:archive[n].copy() for n in archive.files}
    basis_report = basis_check(basis,arrays)
    original = checkpoint(Path(basis_parent['root']),basis_parent['pin'])
    check(basis['parent'] == basis_parent['pin'] and basis['parent_binding'] == digest(original[2]['binding']),
          'LEARNING_READER_BASIS_PARENT')
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
    population = ProposalEchoPopulation(inputs['population'],inputs['population_pin'])
    pairs = binding['proposal_feedback']['pairs']
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
