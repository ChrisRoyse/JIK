"""Independent persisted-weight, SQL, native-distance and outcome-score recount."""
import argparse
import io
import json
import math
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
        check(np.isfinite(array).all(), 'CORRECTION_READBACK_FINITE')
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
          'CORRECTION_READBACK_CHECKPOINT')
    files(path, m['files'])
    weights = torch.load(io.BytesIO((path/'weights.pt').read_bytes()), weights_only=True, map_location='cpu')
    training = torch.load(io.BytesIO((path/'training.pt').read_bytes()), weights_only=True, map_location='cpu')
    optimizer = training['optimizer']
    check(dict(model=digest(tensors(weights)), optimizer=digest(tensors(optimizer))) == m['fingerprint'],
          'CORRECTION_READBACK_FINGERPRINT')
    frozen = {n:v for n,v in weights.items() if not n.startswith(('query.','place.'))}
    check(digest(tensors(frozen)) == m['frozen'], 'CORRECTION_READBACK_FROZEN')
    group, = optimizer['param_groups']
    check(len(group['params']) == len(m['names']), 'CORRECTION_READBACK_OPTIMIZER_ROSTER')
    clocks = {}
    for name, index in zip(m['names'], group['params']):
        clock = m['offsets'][name]+m['step']
        check(float(optimizer['state'][index]['step']) == clock, 'CORRECTION_READBACK_CLOCK')
        clocks[name] = clock
    return m, weights, clocks


def common_certificate(projection):
    gram, weights = (np.asarray(projection[k],dtype=np.float64) for k in ('gram','weights'))
    products = gram@weights
    energy = float(weights@products)
    tolerance = 64*np.finfo(np.float64).eps*max(1.,float(np.linalg.norm(gram,np.inf)))
    check(projection['tolerance'] == tolerance and 0 < projection['iterations'] <= 3*len(weights) and
          np.all(weights >= 0) and abs(weights.sum()-1.) <= tolerance and
          np.min(products-energy) >= -tolerance and np.max(np.abs(weights*(products-energy))) <= tolerance and
          energy > 0 and np.min(products) > 0 and abs(energy-projection['energy']) <= tolerance and
          np.max(np.abs(products-np.asarray(projection['products']))) <= tolerance and
          np.max(np.asarray(projection['projected_dot'])) < 0 and projection['scale'] > 0 and
          np.max(np.abs(np.asarray(projection['projected_dot'])+projection['scale']*products)) <=
          tolerance*max(1.,projection['scale']), 'CORRECTION_READBACK_COMMON_DESCENT_KKT')


def trial_selection(update, retention):
    ceilings = retention['ceilings']
    names = sorted(ceilings[retention['keys'][0]]['native'])
    order = [(k,kind,n) for k in retention['keys']
             for kind,lanes in (('native',names),('termination',names+['common'])) for n in lanes]
    trials, feasible, lower, upper = update['trials'], [], None, None
    check(update['trial_limit'] == 4 and len(trials) <= 4, 'CORRECTION_READBACK_TRIAL_BUDGET')
    for index, trial in enumerate(trials):
        scale, value = trial['scale'], trial['objective']
        check(0 < scale <= 1. and (scale == 1. if index == 0 else scale == trials[index-1]['next_scale']),
              'CORRECTION_READBACK_TRIAL_SCALE')
        kept = all(trial['scores'][k][kind][n] <= ceilings[k][kind][n] for k,kind,n in order)
        gain = value < update['objective_before']
        check(kept == trial['retained'] and gain == trial['objective_decreased'], 'CORRECTION_READBACK_TRIAL_VERDICT')
        if kept and gain:
            feasible.append(index)
            lower = scale
        else:
            upper = scale
        if 'next_scale' not in trial:
            check(index == len(trials)-1,'CORRECTION_READBACK_MISSING_NEXT_SCALE')
            continue
        if trial['next_rule'] == 'geometric_bracket':
            check(lower is not None and upper is not None and 0 < lower < upper <= 1. and
                  trial['lower'] == lower and trial['upper'] == upper and
                  trial['next_scale'] == math.sqrt(lower*upper),'CORRECTION_READBACK_BRACKET')
        else:
            check(trial['next_rule'] == 'curvature_contraction' and lower is None,
                  'CORRECTION_READBACK_CONTRACTION_RULE')
            proposals, roots = [], []
            if not kept:
                for (k,kind,n),slope in zip(order,update['projected_direction']):
                    start, end = update['before_scores'][k][kind][n], trial['scores'][k][kind][n]
                    ceiling = ceilings[k][kind][n]
                    if end <= ceiling:
                        continue
                    curvature = (end-start-scale*slope)/(scale*scale)
                    check(start <= ceiling and curvature > 0,'CORRECTION_READBACK_CONTRACTION_BOUNDARY')
                    roots.append((-slope+math.sqrt(slope*slope+4*curvature*(ceiling-start)))/(2*curvature))
                proposals.append(.5*min(scale,*roots))
            if not gain:
                slope = update['projected_direction'][-1]
                curvature = (value-update['objective_before']-scale*slope)/(scale*scale)
                check(slope < 0 and curvature > 0,'CORRECTION_READBACK_OBJECTIVE_CONTRACTION')
                proposals.append(.5*min(scale,-slope/curvature))
            check(proposals and trial['next_scale'] == min(proposals),'CORRECTION_READBACK_CONTRACTION_SCALE')
    best = min(feasible,key=lambda i:trials[i]['objective']) if feasible else None
    check(update['accepted'] == bool(feasible) and update['selected_trial'] == best,
          'CORRECTION_READBACK_BEST_TRIAL')
    if best is not None:
        check(update['scale'] == trials[best]['scale'] and update['after_scores'] == trials[best]['scores'] and
              update['objective_after'] == trials[best]['objective'],'CORRECTION_READBACK_BEST_RESTORE')
    else:
        check(update['scale'] == 0 and update['after_scores'] == update['before_scores'],
              'CORRECTION_READBACK_NO_FEASIBLE_RESTORE')
    return dict(trials=len(trials),selected=best,restored_earlier=best is not None and best < len(trials)-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root, start = a.root, time.perf_counter()
    source = json.loads((root/'source.json').read_bytes())
    prior = Path(source['root'])
    check(sha((prior/'seal.json').read_bytes()) == source['seal'] and
          sha((prior/'experience.sqlite3').read_bytes()) == source['journal'], 'CORRECTION_READBACK_SOURCE')
    old_contract, old_events, old_payloads, _ = journal(prior/'experience.sqlite3')
    contract, events, payloads, head = journal(root/'experience.sqlite3')
    check(contract == old_contract and events[:-1] == old_events, 'CORRECTION_READBACK_PREFIX')
    decision = events[-1]
    check(decision['kind'] == 'corrective_feedback_candidate', 'CORRECTION_READBACK_DECISION')
    value = payloads[decision['payload']]
    run = json.loads((root/'learning.json').read_bytes())
    check(decision == run['decision'] and value['baseline'] == run['before'] and value['candidate'] == run['after'],
          'CORRECTION_READBACK_RESULTS')
    initial, initial_weights, initial_clocks = checkpoint(root/'candidates/corrective-1-start')
    middle, _, _ = checkpoint(root/'candidates/corrective-1-mid')
    final, final_weights, final_clocks = checkpoint(root/'candidates/corrective-1')
    check(initial['step'] == 0 and middle['step'] == 4 and final['step'] == 8 and
          final['binding'] == run['binding'] and digest(final['binding']) == decision['binding'],
          'CORRECTION_READBACK_BINDING')
    changed = [n for n in initial_weights if not torch.equal(initial_weights[n], final_weights[n])]
    check(set(changed) == set(run['changed']) and all(n.startswith(('query.','place.')) for n in changed),
          'CORRECTION_READBACK_CHANGED')
    check(initial['frozen'] == middle['frozen'] == final['frozen'] and
          all(final_clocks[n] == initial_clocks[n]+8 for n in initial_clocks), 'CORRECTION_READBACK_CONTINUITY')
    check(run['rollback_fingerprint'] == initial['fingerprint'] and run['rollback_behaviour_equal']
          and run['exact_optimizer_recovery'], 'CORRECTION_READBACK_RECOVERY')
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
    check(len(retention) == 52 and all(r['retained'] for r in retention), 'CORRECTION_READBACK_RETENTION')
    accepted = sum(r['update']['accepted'] for r in run['losses'])
    check(len(run['losses']) == 8 and all(r['update']['retained'] for r in run['losses']),
          'CORRECTION_READBACK_STEP_RETENTION')
    certificates, searches = [], []
    for row in run['losses']:
        update = row['update']
        check(update['constraints'] == 52 and update['objective_before'] == row['loss'],
              'CORRECTION_READBACK_OBJECTIVE_BINDING')
        check(update['objective_after'] < update['objective_before'] if update['accepted'] else
              update['objective_after'] == update['objective_before'], 'CORRECTION_READBACK_OBJECTIVE_DESCENT')
        check(all(update['after_scores'][key][kind][name] <= ceiling
                  for key,groups in protected.items() for kind,lanes in groups.items()
                  for name,ceiling in lanes.items()), 'CORRECTION_READBACK_STEP_CEILINGS')
        if update.get('search_rule') == 'bounded_geometric_feasibility_bracket_v1':
            searches.append(trial_selection(update,binding['grounded_retention']))
        projection = update['projection']
        if projection['status'] == 'common_descent':
            common_certificate(projection)
            certificates.append(projection)
        if projection['status'] == 'projected':
            gram, dot, dual = (np.asarray(projection[k],dtype=np.float64) for k in ('gram','dot','dual'))
            after = dot-gram@dual
            tolerance = projection['tolerance']
            check(np.all(dual >= 0) and np.max(after) <= tolerance and
                  np.max(np.abs(dual*after)) <= tolerance and
                  np.max(np.abs(after-np.asarray(projection['projected_dot']))) <= tolerance,
                  'CORRECTION_READBACK_PROJECTION_KKT')
    certificate_controls = []
    if certificates:
        for key in ('weights','projected_dot'):
            altered = json.loads(json.dumps(certificates[0]))
            altered[key][0] += 1.
            try:
                common_certificate(altered)
            except ValueError as error:
                check('CORRECTION_READBACK_COMMON_DESCENT_KKT' in str(error), 'CORRECTION_READBACK_CONTROL_CODE')
            else:
                raise AssertionError('CORRECTION_READBACK_ACCEPTED_ALTERED_CERTIFICATE')
            certificate_controls.append(dict(altered=key,code='CORRECTION_READBACK_COMMON_DESCENT_KKT'))
    search_controls = []
    if searches:
        altered = json.loads(json.dumps(run['losses'][0]['update']))
        altered['selected_trial'] = None if altered['selected_trial'] is not None else 0
        try:
            trial_selection(altered,binding['grounded_retention'])
        except ValueError as error:
            check('CORRECTION_READBACK_BEST_TRIAL' in str(error),'CORRECTION_READBACK_SEARCH_CONTROL_CODE')
        else:
            raise AssertionError('CORRECTION_READBACK_ACCEPTED_ALTERED_SELECTION')
        search_controls.append(dict(altered='selected_trial',code='CORRECTION_READBACK_BEST_TRIAL'))
    echo_snapshot = json.loads((prior/'snapshot-final.json').read_bytes())
    pairs = binding['controlled_feedback']['pairs']
    measured, native_checks, outcome_checks = {}, 0, 0
    for key, pair in pairs.items():
        entry = echo_snapshot['native_feedback_contexts'][pair['context']]
        check(digest(entry) == pair['entry'] and entry['status'] == 'corrected_state' and
              pair['prestate'] == entry['fork']['prestate'] and pair['correction_state_used_as_input'] is False,
              'CORRECTION_READBACK_CAUSAL_JOIN')
        spec = next(e['spec'] for e in old_events if e['kind'] == 'field_echo_enrollment' and
                    e['spec']['experiment'] == entry['experiment'])
        population = Path(spec['population'])
        refs = [json.loads(r) for r in (population/'references.jsonl').read_bytes().splitlines()]
        units = [r['unit'] for r in refs if r['observation'] == pair['observation'] and r['unit'] is not None]
        feedback = next(e for e in old_events if e['kind'] == 'field_experiment_feedback' and e['forecast'] == entry['experiment'])
        manifest_raw = (Path(feedback['root'])/'manifest.json').read_bytes()
        check(sha(manifest_raw) == feedback['pin'], 'CORRECTION_READBACK_OBSERVATION_PIN')
        manifest = json.loads(manifest_raw)
        ob = manifest['observations'][pair['observation']]
        check(ob['condition'] == entry['condition'], 'CORRECTION_READBACK_CONDITION')
        result_raw = (Path(feedback['root'])/ob['execution']/'execution.json').read_bytes()
        check(sha(result_raw) == manifest['files'][ob['execution']+'/execution.json']['sha256'],
              'CORRECTION_READBACK_EXECUTION_BYTES')
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
                      'CORRECTION_READBACK_NATIVE_SCORE')
                row['native'].setdefault(name,{})[phase] = loss
                native_checks += 1
            prediction, score = item['termination']['prediction'], item['termination']['score']
            for name, values in dict(prediction['instruments'], common=prediction['common']).items():
                probabilities = values['probabilities']
                brier = sum((v-float(i == outcome))**2 for i,v in enumerate(probabilities))
                declared = score['common'] if name == 'common' else score['instruments'][name]
                check(abs(brier-declared['brier']) < 1e-12, 'CORRECTION_READBACK_OUTCOME_SCORE')
                row['termination'].setdefault(name,{})[phase] = brier
                outcome_checks += 1
        measured[key] = row
    cold = json.loads((root/'cold-report.json').read_bytes())
    snapshot = json.loads((root/'snapshot.json').read_bytes())
    check(cold['head'] == head == snapshot['head'] and cold['snapshot'] == digest(snapshot),
          'CORRECTION_READBACK_COLD')
    check(not decision['result']['selected'] and not decision['result']['useful_capability_established'] and
          snapshot['active_model'] == echo_snapshot['active_model'], 'CORRECTION_READBACK_QUARANTINE')
    gains = {kind:sum(v['after'] < v['before'] for row in measured.values() for v in row[kind].values())
             for kind in ('native','termination')}
    report = dict(complete=True, acceptance='author-checked', head=head,
        journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        candidate_pin=decision['pin'], changed_parameters=changed, accepted_updates=accepted,
        retained_checks=retention, native_score_checks=native_checks, outcome_score_checks=outcome_checks,
        controlled_measurements=measured, improved_checks=gains,
        common_descent_certificates=len(certificates),certificate_controls=certificate_controls,
        trial_searches=searches,search_controls=search_controls,
        optimizer_clock_range=[min(final_clocks.values()),max(final_clocks.values())],
        frozen_unchanged=True, cold_replay_equal=True, active_checkpoint_unchanged=True,
        metric_gate_passed=decision['result']['metric_gate_passed'], reasons=decision['result']['reasons'],
        useful_heldout_capability=False, serving=False, scientific_admission=False,
        wall_seconds=time.perf_counter()-start)
    write(root/'independent-readback.json', report)
    print(json.dumps({k:report[k] for k in ('complete','head','candidate_pin','accepted_updates','improved_checks','metric_gate_passed','reasons')}))


if __name__ == '__main__':
    main()
