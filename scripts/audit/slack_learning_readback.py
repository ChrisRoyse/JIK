"""Read real slack-update certificates, weights, native errors and decisions on CPU."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.proposal_learning_readback import checkpoint, errors
from audit.proposal_echo_readback import distance
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from claims.proposal_echo_population import ProposalEchoPopulation


def read(path):
    return json.loads(path.read_bytes())


def certificates(learning):
    ceilings = learning['binding']['grounded_retention']['ceilings']
    names = list(next(iter(learning['before'].values()))['predictions'])
    order = [(k, kind, name) for k in learning['binding']['grounded_retention']['keys']
             for kind in ('native', 'termination') for name in (names if kind == 'native' else names+['common'])]
    result = []
    for row in learning['losses']:
        update, p = row['update'], row['update']['projection']
        by_slack = {(k, kind, n):ceilings[k][kind][n]-update['before_scores'][k][kind][n]
                    for k, kind, n in order}
        actual_order = update['constraint_order']
        check(len(actual_order) == len(order) and set(map(tuple, actual_order)) == set(order),
              'SLACK_READBACK_CONSTRAINT_ORDER')
        slack = np.array([by_slack[tuple(k)] for k in actual_order])
        check(np.array_equal(slack, np.array(p['slack'])) and np.min(slack) >= 0, 'SLACK_READBACK_SLACK')
        norms, active = np.array(p['norms']), np.array(p['active'], dtype=bool)
        check(np.array_equal(active, norms > 0), 'SLACK_READBACK_ACTIVE')
        bounds = .5*slack[active]/norms[active]
        check(np.allclose(bounds, p['bounds'], atol=1e-14, rtol=1e-13), 'SLACK_READBACK_BOUND')
        gram, rhs, raw = np.array(p['gram']), np.array(p['rhs']), np.array(p['raw_direction'])
        check(gram.shape == (int(active.sum()),)*2 and np.allclose(gram, gram.T, atol=1e-12, rtol=0)
              and np.allclose(np.diag(gram), 1., atol=1e-12, rtol=0)
              and np.linalg.eigvalsh(gram).min() >= -1e-10, 'SLACK_READBACK_GRAM')
        check(np.allclose(raw-bounds, rhs, atol=1e-14, rtol=1e-13), 'SLACK_READBACK_RHS')
        check(p['iterations'] <= p['iterations_limit'] == 3*len(rhs), 'SLACK_READBACK_SOLVER_BUDGET')
        if p['status'] == 'certified_slack_projection':
            weights = np.array(p['weights'])
            residual = gram@weights-rhs
            tol = p['tolerance']
            check(np.min(weights) >= 0 and np.min(residual) >= -tol
                  and np.max(np.abs(weights*residual)) <= tol, 'SLACK_READBACK_DUAL')
            check(np.allclose(residual, p['residual'], atol=tol, rtol=0), 'SLACK_READBACK_RESIDUAL')
            direction = np.array(p['direction'])
            check(np.allclose(raw-gram@weights, direction, atol=4*tol, rtol=0)
                  and np.all(direction <= bounds+tol) and p['norm'] <= p['proposal_norm']+tol,
                  'SLACK_READBACK_PRIMAL')
        else:
            check(not update['accepted'], 'SLACK_READBACK_UNCERTIFIED_ACCEPTANCE')
        trials = update['trials']
        check(len(trials) <= update['trial_limit'] == 4 and update['constraints'] == len(order),
              'SLACK_READBACK_TRIAL_BUDGET')
        for trial in trials:
            kept = all(trial['scores'][k][kind][n] <= ceilings[k][kind][n] for k, kind, n in order)
            check(kept == trial['retained'] and trial['objective_decreased'] ==
                  (trial['objective'] < update['objective_before']), 'SLACK_READBACK_TRIAL')
        if update['accepted']:
            chosen = trials[update['selected_trial']]
            feasible = [t['objective'] for t in trials if t['retained'] and t['objective_decreased']]
            check(chosen['retained'] and chosen['objective'] == min(feasible) == update['objective_after']
                  and chosen['scores'] == update['after_scores'], 'SLACK_READBACK_ACTUAL_ACCEPTANCE')
        else:
            check(update['before'] == update['after'] and update['before_scores'] == update['after_scores']
                  and update['objective_before'] == update['objective_after'], 'SLACK_READBACK_REJECT_RESTORE')
        result.append(dict(status=p['status'], accepted=update['accepted'], iterations=p['iterations'],
            trials=len(trials), scale=update['scale'], objective=update['objective_after']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    check(not a.output.exists(), 'SLACK_READBACK_EXISTS')
    a.output.mkdir()
    learning, config = read(a.root/'learning.json'), read(a.root/'config.json')
    binding = read(a.root/'training-binding.json')
    check(binding == learning['binding'], 'SLACK_READBACK_BINDING')
    certs = certificates(learning)
    pins = [learning[k] for k in ('start_pin', 'mid_pin', 'candidate_pin')]
    stages = [checkpoint(a.root/'candidates'/n, p) for n, p in zip(('slack-1-start', 'slack-1-mid', 'slack-1'), pins)]
    start, mid, final = stages
    parent = checkpoint(Path(config['spec']['continuation']['root']), config['spec']['continuation']['pin'])
    check(start[2]['fingerprint'] == parent[2]['fingerprint'], 'SLACK_READBACK_CONTINUATION')
    check(all(s[2]['binding'] == binding for s in stages), 'SLACK_READBACK_CHECKPOINT_BINDINGS')
    check(binding['grounded_retention']['ceilings'] == parent[2]['binding']['grounded_retention']['ceilings'],
          'SLACK_READBACK_CEILING_CHANGE')
    changed = [n for n in start[0] if not torch.equal(start[0][n], final[0][n])]
    check(changed == learning['changed'] and set(changed) <= set(final[2]['names']), 'SLACK_READBACK_FROZEN_DRIFT')
    retained = []
    for key, kinds in binding['grounded_retention']['ceilings'].items():
        item = learning['after'][key]
        for name, ceiling in kinds['native'].items():
            saved = item['assessment']['instruments'][name]
            value = distance(np.array(saved['prediction_cosines']))
            check(value == saved['prediction_loss'] and value <= ceiling, 'SLACK_READBACK_NATIVE_RETENTION')
            retained.append(dict(key=key, kind='native', name=name, value=value, ceiling=ceiling))
        for name, ceiling in kinds['termination'].items():
            prediction = item['termination']['prediction']
            p = prediction['common'] if name == 'common' else prediction['instruments'][name]
            truth = item['termination']['score']['outcome']
            value = sum((v-float(i == truth))**2 for i,v in enumerate(p['probabilities']))
            check(value <= ceiling, 'SLACK_READBACK_OUTCOME_RETENTION')
            retained.append(dict(key=key, kind='termination', name=name, value=value, ceiling=ceiling))
    inputs = read(Path(config['source'])/'config.json')['inputs']
    population = ProposalEchoPopulation(inputs['population'], inputs['population_pin'])
    pairs = binding['proposal_feedback']['pairs']
    arrays = {}
    for name in learning['before'][next(iter(pairs))]['predictions']:
        with np.load(Path(inputs['native'])/name/'slots.npz', allow_pickle=False) as data:
            arrays[name] = {k:np.concatenate([data['vectors'][data['offsets'][r['unit']]:data['offsets'][r['unit']+1]]
                for r in population.refs if r['observation'] == k and r['unit'] is not None])
                for k in population.observations.entries}
    native = {key:{stage:errors(learning[stage][key], {n:a[pair['observation']] for n,a in arrays.items()})
                  for stage in ('before', 'after')} for key,pair in pairs.items()}
    head, count, decision = ZERO, 0, None
    with sqlite3.connect((a.root/'experience.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        for seq, previous, raw, pin in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous == head and raw == canonical(body)
                  and pin == digest(dict(seq=seq, previous=previous, body=body)), 'SLACK_READBACK_CHAIN')
            if body.get('payload') is not None:
                payload, = db.execute('SELECT data FROM payloads WHERE hash=?', (body['payload'],)).fetchone()
                check(sha(payload) == body['payload'], 'SLACK_READBACK_PAYLOAD')
            if body['kind'] == 'slack_learning_candidate':
                check(decision is None and body == learning['decision'], 'SLACK_READBACK_DECISION')
                decision = json.loads(payload)
                check(decision['before'] == learning['before'] and decision['after'] == learning['after'],
                      'SLACK_READBACK_DECISION_SCORES')
            count, head = count+1, pin
    snapshot, cold = read(a.root/'snapshot-final.json'), read(a.root/'cold-report.json')
    check(count == snapshot['events'] and head == snapshot['head'] and cold['snapshot'] == digest(snapshot)
          and cold['journal'] == sha((a.root/'experience.sqlite3').read_bytes()), 'SLACK_READBACK_COLD')
    check(decision is not None and not decision['result']['selected'], 'SLACK_READBACK_NO_ADMISSION')
    controls = []

    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'SLACK_READBACK_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('SLACK_READBACK_CONTROL_ACCEPTED')
        controls.append(dict(name=name, code=code))

    certified = next((i for i,r in enumerate(learning['losses'])
                      if r['update']['projection']['status'] == 'certified_slack_projection'), None)
    if certified is not None:
        bad = copy.deepcopy(learning)
        bad['losses'][certified]['update']['projection']['weights'][0] = -1.
        refuse('negative_dual_weight', 'SLACK_READBACK_DUAL', lambda:certificates(bad))
    bad = copy.deepcopy(learning)
    bad['losses'][0]['update']['projection']['slack'][0] += .1
    refuse('invented_slack', 'SLACK_READBACK_SLACK', lambda:certificates(bad))
    broken = a.output/'corrupt-checkpoint'
    shutil.copytree(a.root/'candidates/slack-1', broken)
    with (broken/'weights.pt').open('r+b') as handle:
        handle.seek(256); value=handle.read(1); handle.seek(256); handle.write(bytes([value[0]^1]))
    refuse('changed_checkpoint', 'PROPOSAL_WEIGHT_BYTES', lambda:checkpoint(broken, pins[-1]))
    key, pair = next(iter(pairs.items()))
    wrong = next(k for k in population.observations.entries if k != pair['observation'])
    refuse('swapped_native_output', 'PROPOSAL_WEIGHT_NATIVE_SCORE',
           lambda:errors(learning['after'][key], {n:a[wrong] for n,a in arrays.items()}))
    report = dict(schema='slack-learning-readback-v1', acceptance='author-checked', events=count, head=head,
        source_checkpoint=config['spec']['continuation']['pin'], candidate=pins[-1],
        controls=read(a.root/'train-report.json')['controls']+controls, certificates=certs,
        retained=retained, native=native, changed=changed, accepted_updates=sum(r['accepted'] for r in certs),
        objective_before=learning['losses'][0]['update']['objective_before'], objective_after=certs[-1]['objective'],
        exact_optimizer_recovery=learning['exact_optimizer_recovery'], rollback=learning['rollback_behaviour_equal'],
        boundaries=['parent_and_fixed_ceilings', 'actual_updates_and_certificates', 'cold_decision_and_rollback'],
        serving=False, scientific_admission=False, useful_transfer_established=False)
    write_committed(a.output/'readback.json', report)
    print(json.dumps(dict(pin=sha((a.output/'readback.json').read_bytes()), events=count, controls=len(report['controls']),
        accepted_updates=report['accepted_updates'], changed=len(changed), retained=len(retained),
        objective_before=report['objective_before'], objective_after=report['objective_after'])))


if __name__ == '__main__':
    main()
