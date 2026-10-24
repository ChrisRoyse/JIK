"""Independent persisted-weight, SQL, native-distance and outcome-score recount."""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.field_echo_readback import decode, distance, files, write
from audit.prospective_execution_readback import journal
from claims.codata_source import check, sha
from claims.episodic_memory import digest


def tensors(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        check(np.isfinite(array).all(), 'CONTROLLED_READBACK_FINITE')
        return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=sha(array.tobytes()))
    if isinstance(value, dict):
        return {str(k):tensors(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensors(v) for v in value]
    return value


def checkpoint(path):
    raw = (path/'manifest.json').read_bytes()
    m = json.loads(raw)
    check(sha(raw) == (path/'COMMITTED').read_text() and not m['serving'] and not m['scientific_admission'],
          'CONTROLLED_READBACK_CHECKPOINT')
    files(path, m['files'])
    weights = torch.load(io.BytesIO((path/'weights.pt').read_bytes()), weights_only=True, map_location='cpu')
    training = torch.load(io.BytesIO((path/'training.pt').read_bytes()), weights_only=True, map_location='cpu')
    optimizer = training['optimizer']
    check(dict(model=digest(tensors(weights)), optimizer=digest(tensors(optimizer))) == m['fingerprint'],
          'CONTROLLED_READBACK_FINGERPRINT')
    frozen = {n:v for n,v in weights.items() if not n.startswith(('query.','place.'))}
    check(digest(tensors(frozen)) == m['frozen'], 'CONTROLLED_READBACK_FROZEN')
    group, = optimizer['param_groups']
    check(len(group['params']) == len(m['names']), 'CONTROLLED_READBACK_OPTIMIZER_ROSTER')
    clocks = {}
    for name, index in zip(m['names'], group['params']):
        clock = m['offsets'][name]+m['step']
        check(float(optimizer['state'][index]['step']) == clock, 'CONTROLLED_READBACK_CLOCK')
        clocks[name] = clock
    return m, weights, clocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root, start = a.root, time.perf_counter()
    source = json.loads((root/'source.json').read_bytes())
    prior = Path(source['root'])
    check(sha((prior/'seal.json').read_bytes()) == source['seal'] and
          sha((prior/'experience.sqlite3').read_bytes()) == source['journal'], 'CONTROLLED_READBACK_SOURCE')
    old_contract, old_events, old_payloads, _ = journal(prior/'experience.sqlite3')
    contract, events, payloads, head = journal(root/'experience.sqlite3')
    check(contract == old_contract and events[:-1] == old_events, 'CONTROLLED_READBACK_PREFIX')
    decision = events[-1]
    check(decision['kind'] == 'controlled_feedback_candidate', 'CONTROLLED_READBACK_DECISION')
    value = payloads[decision['payload']]
    run = json.loads((root/'learning.json').read_bytes())
    check(decision == run['decision'] and value['baseline'] == run['before'] and value['candidate'] == run['after'],
          'CONTROLLED_READBACK_RESULTS')
    initial, initial_weights, initial_clocks = checkpoint(root/'candidates/controlled-1-start')
    middle, _, _ = checkpoint(root/'candidates/controlled-1-mid')
    final, final_weights, final_clocks = checkpoint(root/'candidates/controlled-1')
    check(initial['step'] == 0 and middle['step'] == 4 and final['step'] == 8 and
          final['binding'] == run['binding'] and digest(final['binding']) == decision['binding'],
          'CONTROLLED_READBACK_BINDING')
    changed = [n for n in initial_weights if not torch.equal(initial_weights[n], final_weights[n])]
    check(set(changed) == set(run['changed']) and all(n.startswith(('query.','place.')) for n in changed),
          'CONTROLLED_READBACK_CHANGED')
    check(initial['frozen'] == middle['frozen'] == final['frozen'] and
          all(final_clocks[n] == initial_clocks[n]+8 for n in initial_clocks), 'CONTROLLED_READBACK_CONTINUITY')
    check(run['rollback_fingerprint'] == initial['fingerprint'] and run['rollback_behaviour_equal']
          and run['exact_optimizer_recovery'], 'CONTROLLED_READBACK_RECOVERY')
    binding = run['binding']
    protected = binding['grounded_retention']['ceilings']
    retention = []
    for key, categories in protected.items():
        result = run['after'][key]
        actual = dict(native={n:r['prediction_loss'] for n,r in result['assessment']['instruments'].items()},
            termination={n:r['brier'] for n,r in dict(result['termination']['score']['instruments'],
                                                     common=result['termination']['score']['common']).items()})
        for kind, lanes in categories.items():
            for name, ceiling in lanes.items():
                retention.append(dict(example=key, metric=kind, instrument=name,
                                      ceiling=ceiling, value=actual[kind][name], retained=actual[kind][name] <= ceiling))
    check(len(retention) == 26 and all(r['retained'] for r in retention), 'CONTROLLED_READBACK_RETENTION')
    accepted = sum(r['update']['accepted'] for r in run['losses'])
    check(len(run['losses']) == 8 and all(r['update']['retained'] for r in run['losses']),
          'CONTROLLED_READBACK_STEP_RETENTION')
    echo_snapshot = json.loads((prior/'snapshot-final.json').read_bytes())
    pairs = binding['controlled_feedback']['pairs']
    measured, native_checks, outcome_checks = {}, 0, 0
    for key, pair in pairs.items():
        entry = echo_snapshot['native_feedback_contexts'][pair['context']]
        check(digest(entry) == pair['entry'] and entry['status'] == 'corrected_state' and
              pair['prestate'] == entry['fork']['prestate'] and pair['correction_state_used_as_input'] is False,
              'CONTROLLED_READBACK_CAUSAL_JOIN')
        spec = next(e['spec'] for e in old_events if e['kind'] == 'field_echo_enrollment' and
                    e['spec']['experiment'] == entry['experiment'])
        population = Path(spec['population'])
        refs = [json.loads(r) for r in (population/'references.jsonl').read_bytes().splitlines()]
        units = [r['unit'] for r in refs if r['observation'] == pair['observation'] and r['unit'] is not None]
        feedback = next(e for e in old_events if e['kind'] == 'field_experiment_feedback' and e['forecast'] == entry['experiment'])
        manifest_raw = (Path(feedback['root'])/'manifest.json').read_bytes()
        check(sha(manifest_raw) == feedback['pin'], 'CONTROLLED_READBACK_OBSERVATION_PIN')
        manifest = json.loads(manifest_raw)
        ob = manifest['observations'][pair['observation']]
        check(ob['condition'] == entry['condition'], 'CONTROLLED_READBACK_CONDITION')
        result_raw = (Path(feedback['root'])/ob['execution']/'execution.json').read_bytes()
        check(sha(result_raw) == manifest['files'][ob['execution']+'/execution.json']['sha256'],
              'CONTROLLED_READBACK_EXECUTION_BYTES')
        outcome = int(json.loads(result_raw)['result']['returncode'] != 0)
        row = dict(condition=entry['condition'], outcome=outcome, native={}, termination={})
        for phase in ('before','after'):
            item = run[phase][key]
            for name, packed in item['predictions'].items():
                lane = Path(spec['native'])/name
                lane_report = json.loads((lane/'report.json').read_bytes())
                files(lane, lane_report['files'])
                with np.load(lane/'slots.npz', allow_pickle=False) as z:
                    targets = np.concatenate([z['vectors'][z['offsets'][u]:z['offsets'][u+1]] for u in units]).astype(np.float64)
                predicted = decode(packed).astype(np.float64)
                predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
                targets /= np.linalg.norm(targets, axis=1, keepdims=True)
                loss = distance(np.clip(targets@predicted.T, -1, 1))
                check(abs(loss-item['assessment']['instruments'][name]['prediction_loss']) < 1e-12,
                      'CONTROLLED_READBACK_NATIVE_SCORE')
                row['native'].setdefault(name,{})[phase] = loss
                native_checks += 1
            prediction, score = item['termination']['prediction'], item['termination']['score']
            for name, values in dict(prediction['instruments'], common=prediction['common']).items():
                probabilities = values['probabilities']
                brier = sum((v-float(i == outcome))**2 for i,v in enumerate(probabilities))
                declared = score['common'] if name == 'common' else score['instruments'][name]
                check(abs(brier-declared['brier']) < 1e-12, 'CONTROLLED_READBACK_OUTCOME_SCORE')
                row['termination'].setdefault(name,{})[phase] = brier
                outcome_checks += 1
        measured[key] = row
    cold = json.loads((root/'cold-report.json').read_bytes())
    snapshot = json.loads((root/'snapshot.json').read_bytes())
    check(cold['head'] == head == snapshot['head'] and cold['snapshot'] == digest(snapshot),
          'CONTROLLED_READBACK_COLD')
    check(not decision['result']['selected'] and not decision['result']['useful_capability_established'] and
          snapshot['active_model'] == echo_snapshot['active_model'], 'CONTROLLED_READBACK_QUARANTINE')
    gains = {kind:sum(v['after'] < v['before'] for row in measured.values() for v in row[kind].values())
             for kind in ('native','termination')}
    report = dict(complete=True, acceptance='author-checked', head=head,
        journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        candidate_pin=decision['pin'], changed_parameters=changed, accepted_updates=accepted,
        retained_checks=retention, native_score_checks=native_checks, outcome_score_checks=outcome_checks,
        controlled_measurements=measured, improved_checks=gains,
        optimizer_clock_range=[min(final_clocks.values()),max(final_clocks.values())],
        frozen_unchanged=True, cold_replay_equal=True, active_checkpoint_unchanged=True,
        metric_gate_passed=decision['result']['metric_gate_passed'], reasons=decision['result']['reasons'],
        useful_heldout_capability=False, serving=False, scientific_admission=False,
        wall_seconds=time.perf_counter()-start)
    write(root/'independent-readback.json', report)
    print(json.dumps({k:report[k] for k in ('complete','head','candidate_pin','accepted_updates','improved_checks','metric_gate_passed','reasons')}))


if __name__ == '__main__':
    main()
