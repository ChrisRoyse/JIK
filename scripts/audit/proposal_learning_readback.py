"""Read persisted weight changes, native target errors and journal decisions independently."""
import argparse
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.proposal_echo_readback import decode, distance
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from claims.proposal_echo_population import ProposalEchoPopulation


def fingerprint(value):
    if isinstance(value,torch.Tensor):
        a = value.detach().cpu().numpy()
        return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha(a.tobytes()))
    if isinstance(value,dict):
        return {str(k):fingerprint(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [fingerprint(v) for v in value]
    return value


def checkpoint(root, pin):
    raw = (root/'manifest.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(),'PROPOSAL_WEIGHT_PIN')
    m = json.loads(raw)
    payload = {}
    for name,item in m['files'].items():
        content = (root/name).read_bytes()
        check(len(content) == item['bytes'] and sha(content) == item['sha256'],'PROPOSAL_WEIGHT_BYTES')
        payload[name] = torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
    weights, optimizer = payload['weights.pt'],payload['training.pt']['optimizer']
    check(digest(fingerprint(weights)) == m['fingerprint']['model']
          and digest(fingerprint(optimizer)) == m['fingerprint']['optimizer'],'PROPOSAL_WEIGHT_FINGERPRINT')
    group, = optimizer['param_groups']
    check(len(group['params']) == len(m['names']),'PROPOSAL_WEIGHT_OPTIMIZER_ORDER')
    for name,index in zip(m['names'],group['params']):
        check(float(optimizer['state'][index]['step']) == m['offsets'][name]+m['step'],
              'PROPOSAL_WEIGHT_CLOCK')
    check(not m['serving'] and not m['scientific_admission'],'PROPOSAL_WEIGHT_SCOPE')
    return weights,optimizer,m


def errors(item, actual):
    result = {}
    for name,values in actual.items():
        predicted = decode(item['predictions'][name]).astype(np.float64)
        values = values.astype(np.float64)
        predicted /= np.linalg.norm(predicted,axis=1,keepdims=True)
        values /= np.linalg.norm(values,axis=1,keepdims=True)
        cosines = np.clip(values@predicted.T,-1,1)
        loss = distance(cosines)
        saved = item['assessment']['instruments'][name]
        check(np.array_equal(cosines,np.asarray(saved['prediction_cosines'])) and loss == saved['prediction_loss'],
              'PROPOSAL_WEIGHT_NATIVE_SCORE')
        result[name] = loss
    return result


def readback(root, output):
    check(not output.exists(),'PROPOSAL_WEIGHT_READBACK_EXISTS')
    output.mkdir()
    config = json.loads((root/'config.json').read_bytes())
    learning = json.loads((root/'learning.json').read_bytes())
    binding = json.loads((root/'training-binding.json').read_bytes())
    check(binding == learning['binding'],'PROPOSAL_WEIGHT_BINDING')
    paths = ('proposal-1-start','proposal-1-mid','proposal-1')
    pins = [learning[k] for k in ('start_pin','mid_pin','candidate_pin')]
    checkpoints = [checkpoint(root/'candidates'/name,pin) for name,pin in zip(paths,pins)]
    start,mid,final = checkpoints
    parent = checkpoint(Path(config['spec']['checkpoint']['root']),config['spec']['checkpoint']['pin'])
    check(start[2]['fingerprint'] == parent[2]['fingerprint'],'PROPOSAL_WEIGHT_CONTINUATION')
    check(all(v[2]['binding'] == binding for v in checkpoints),'PROPOSAL_WEIGHT_BOUND_CHECKPOINTS')
    changed = [n for n in start[0] if not torch.equal(start[0][n],final[0][n])]
    check(changed == learning['changed'] and set(changed) <= set(final[2]['names']),
          'PROPOSAL_WEIGHT_FROZEN_DRIFT')
    retained = []
    for key,ceilings in binding['grounded_retention']['ceilings'].items():
        scored = learning['after'][key]
        for name,value in ceilings['native'].items():
            actual = scored['assessment']['instruments'][name]['prediction_loss']
            check(actual <= value,'PROPOSAL_WEIGHT_NATIVE_RETENTION')
            retained.append(dict(key=key,kind='native',name=name,ceiling=value,value=actual))
        for name,value in ceilings['termination'].items():
            pred = scored['termination']['prediction']
            p = pred['common'] if name == 'common' else pred['instruments'][name]
            truth = scored['termination']['score']['outcome']
            actual = sum((x-float(i == truth))**2 for i,x in enumerate(p['probabilities']))
            check(actual <= value,'PROPOSAL_WEIGHT_OUTCOME_RETENTION')
            retained.append(dict(key=key,kind='termination',name=name,ceiling=value,value=actual))
    echo_config = json.loads((Path(config['source'])/'config.json').read_bytes())
    inputs = echo_config['inputs']
    population = ProposalEchoPopulation(inputs['population'],inputs['population_pin'])
    arrays = {}
    for name in learning['before'][next(iter(binding['proposal_feedback']['pairs']))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz',allow_pickle=False) as data:
            arrays[name] = {k:np.concatenate([data['vectors'][data['offsets'][r['unit']]:data['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    native = {}
    for key,pair in binding['proposal_feedback']['pairs'].items():
        actual = {n:a[pair['observation']] for n,a in arrays.items()}
        native[key] = {stage:errors(learning[stage][key],actual) for stage in ('before','after')}
    head,count,decision = ZERO,0,None
    with sqlite3.connect((root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous,raw,pin in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body)
                  and pin == digest(dict(seq=seq,previous=previous,body=body)),'PROPOSAL_WEIGHT_CHAIN')
            if body.get('payload') is not None:
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(sha(payload) == body['payload'],'PROPOSAL_WEIGHT_PAYLOAD')
            if body['kind'] == 'proposal_learning_candidate':
                check(decision is None and body == learning['decision'],'PROPOSAL_WEIGHT_DECISION')
                decision = json.loads(payload)
                check(decision['before'] == learning['before'] and decision['after'] == learning['after'],
                      'PROPOSAL_WEIGHT_DECISION_SCORES')
            count,head = count+1,pin
    saved = json.loads((root/'snapshot-final.json').read_bytes())
    cold = json.loads((root/'cold-report.json').read_bytes())
    check(decision is not None and count == saved['events'] and head == saved['head']
          and cold['snapshot'] == digest(saved),'PROPOSAL_WEIGHT_COLD_REPLAY')
    check(not decision['result']['selected'] and not decision['result']['useful_capability_established'],
          'PROPOSAL_WEIGHT_CAPABILITY_SCOPE')
    controls = []

    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'PROPOSAL_WEIGHT_CONTROL_CODE')
        else:
            raise AssertionError('PROPOSAL_WEIGHT_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))

    refuse('wrong_checkpoint_pin','PROPOSAL_WEIGHT_PIN',lambda:checkpoint(root/'candidates/proposal-1','0'*64))
    broken = output/'corrupt-checkpoint'
    shutil.copytree(root/'candidates/proposal-1',broken)
    path = broken/'weights.pt'
    with path.open('r+b') as f:
        f.seek(256); value=f.read(1); f.seek(256); f.write(bytes([value[0]^1]))
    refuse('changed_checkpoint_bytes','PROPOSAL_WEIGHT_BYTES',lambda:checkpoint(broken,pins[-1]))
    key,pair = next(iter(binding['proposal_feedback']['pairs'].items()))
    wrong = next(k for k in population.observations.entries if k != pair['observation'])
    refuse('swapped_output_windows','PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key],{n:a[wrong] for n,a in arrays.items()}))
    training = json.loads((root/'train-report.json').read_bytes())
    report = dict(schema='proposal-learning-readback-v1',acceptance='author-checked',head=head,events=count,
        controls=training['controls']+controls,changed_parameters=changed,retention=retained,native=native,
        accepted_updates=sum(r['update']['accepted'] for r in learning['losses']),
        objective_before=learning['losses'][0]['update']['objective_before'],
        objective_after=learning['losses'][-1]['update']['objective_after'],
        exact_optimizer_recovery=learning['exact_optimizer_recovery'],rollback=learning['rollback_behaviour_equal'],
        source_checkpoint=config['spec']['checkpoint']['pin'],candidate=pins[-1],
        boundaries=['captured_output_to_training_region','weights_and_retention','journal_restart_and_rollback'],
        useful_transfer_established=False,scientific_admission=False,serving=False)
    write_committed(output/'readback.json',report)
    return dict(pin=sha((output/'readback.json').read_bytes()),events=count,controls=len(report['controls']),
                accepted_updates=report['accepted_updates'],changed=len(changed),retained=len(retained))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(readback(args.root,args.output)))
