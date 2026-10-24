"""Independent SQL/NumPy readback of captured bytes, native lanes and feedback states.

No model forward or producer's observation/measurement adapters are imported.
"""
import argparse
import base64
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.prospective_execution_readback import journal
from claims.codata_source import check, sha
from claims.episodic_memory import digest


def write(path, value):
    with path.open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def decode(value):
    raw = base64.b64decode(value['base64'], validate=True)
    check(sha(raw) == value['sha256'] and value['dtype'] == 'float32-le', 'ECHO_READBACK_TENSOR')
    array = np.frombuffer(raw, dtype='<f4').reshape(value['shape'])
    check(np.isfinite(array).all(), 'ECHO_READBACK_FINITE')
    return array


def files(root, entries):
    for name, info in entries.items():
        raw = (root/name).read_bytes()
        check(len(raw) == info['bytes'] and sha(raw) == info['sha256'], 'ECHO_READBACK_BYTES')


def load_native(root, population_pin):
    report = json.loads((root/'report.json').read_bytes())
    check(report['complete'] and not report['errors'] and report['inputs'] == 3 and
          report['calibration_report_sha256'] == population_pin, 'ECHO_READBACK_NATIVE_REPORT')
    arrays, instruments = {}, {}
    for name, ref in report['lenses'].items():
        raw = (root/name/'report.json').read_bytes()
        check(sha(raw) == ref['report_sha256'], 'ECHO_READBACK_LANE_PIN')
        lane = json.loads(raw)
        files(root/name, lane['files'])
        with np.load(root/name/'slots.npz', allow_pickle=False) as z:
            x, ptr = z['vectors'].copy(), z['offsets'].copy()
        check(len(ptr) == 4 and ptr[0] == 0 and ptr[-1] == len(x) and np.all(np.diff(ptr) > 0)
              and np.isfinite(x).all() and np.max(np.abs(np.linalg.norm(x, axis=1)-1)) < 1e-4,
              'ECHO_READBACK_NATIVE_ARRAYS')
        arrays[name] = (x, ptr)
        definition = dict(schema='loop-instrument-v1', name=name, channel='language', layout='multi',
            dims=x.shape[1], frozen=True, weights_sha256=lane['contract']['weights_sha256'],
            training_lineage={'status':'upstream_not_independently_attested'},
            measurement_contract=lane['contract'], lens_id=lane['lens_id'])
        instruments[name] = digest(definition)
    return arrays, instruments


def distance(cosines):
    n, m = cosines.shape
    cost = np.zeros((n+1, m+1), dtype=np.float64)
    cost[:,0], cost[0,:] = np.arange(n+1), np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost[i,j] = min(cost[i-1,j]+1, cost[i,j-1]+1, cost[i-1,j-1]+1-cosines[i-1,j-1])
    return float(cost[-1,-1]/max(n,m))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root, start = a.root, time.perf_counter()
    source = json.loads((root/'source.json').read_bytes())
    prior = Path(source['root'])
    check(sha((prior/'seal.json').read_bytes()) == source['seal'] and
          sha((prior/'experience.sqlite3').read_bytes()) == source['journal'], 'ECHO_READBACK_PARENT')
    old_contract, old_events, _, _ = journal(prior/'experience.sqlite3')
    contract, events, payloads, head = journal(root/'experience.sqlite3')
    check(contract == old_contract and events[:len(old_events)] == old_events, 'ECHO_READBACK_PREFIX')
    new = events[len(old_events):]
    check(len(new) == 5 and new[0]['kind'] == 'field_echo_enrollment' and
          all(e['kind'] == 'field_echo_transition' for e in new[1:]), 'ECHO_READBACK_EVENT_ROSTER')
    enrollment = payloads[new[0]['payload']]
    contexts = copy.deepcopy(enrollment['contexts'])
    spec = new[0]['spec']
    native, population = Path(spec['native']), Path(spec['population'])
    population_raw = (population/'report.json').read_bytes()
    check(sha(population_raw) == spec['population_pin'] == (population/'COMMITTED').read_text(),
          'ECHO_READBACK_POPULATION_PIN')
    pop = json.loads(population_raw)
    files(population, pop['files'])
    refs = [json.loads(r) for r in (population/'references.jsonl').read_bytes().splitlines()]
    rows = pq.read_table(population/'process/chunks/atoms.parquet').to_pylist()
    check(len(rows) == 3 and len(refs) == 4 and sum(r['unit'] is not None for r in refs) == 2,
          'ECHO_READBACK_STREAMS')
    check(sha((native/'report.json').read_bytes()) == spec['native_pin'], 'ECHO_READBACK_NATIVE_PIN')
    arrays, instruments = load_native(native, spec['population_pin'])
    check(len(arrays) == enrollment['instrument_count'] == 6, 'ECHO_READBACK_ROSTER')
    execution = prior/'execution'
    manifest_raw = (execution/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    check(sha(manifest_raw) == enrollment['native']['observations'] == (execution/'COMMITTED').read_text(),
          'ECHO_READBACK_EXECUTION_PIN')
    files(execution, manifest['files'])
    for ref in refs:
        observed = manifest['observations'][ref['observation']]
        result = json.loads((execution/observed['execution']/'execution.json').read_bytes())['result']
        raw = base64.b64decode(result[ref['stream']]['base64'], validate=True)
        check(len(raw) == ref['bytes'] and sha(raw) == ref['text_sha256'], 'ECHO_READBACK_CAPTURE')
        check(raw == b'' if ref['unit'] is None else rows[ref['unit']]['text'].encode() == raw,
              'ECHO_READBACK_CAPTURE_TEXT')
    forecast = json.loads((prior/'forecast/forecast.json').read_bytes())
    _, _, forecast_payloads, _ = journal(prior/'forecast/journal.sqlite3')
    predictions = forecast_payloads[forecast['forecast']['payload']]['predictions']
    observed_arrays = {}
    for context, entry in contexts.items():
        ob = entry['observation']
        check(entry['condition'] == manifest['observations'][ob]['condition'] and
              context == 'controlled:'+digest([spec['experiment'], entry['condition']]),
              'ECHO_READBACK_CONTEXT_JOIN')
        units = [r['unit'] for r in refs if r['observation'] == ob and r['unit'] is not None]
        observed_arrays[context] = {n:np.concatenate([x[ptr[u]:ptr[u+1]] for u in units])
                                    for n,(x,ptr) in arrays.items()}
        check(entry['state']['sequence'] % 5 == 3 and entry['status'] == 'awaiting_echo',
              'ECHO_READBACK_START_PHASE')
        decode(entry['state']['hidden'])
    checks, deltas = 0, []
    for event in new[1:]:
        r, value = event['request'], payloads[event['payload']]
        entry = contexts[r['context']]
        check(entry['observation'] == r['observation'] and r['origin'] == 'execution_receipt' and
              r['same_occurrence_reflection'] == (r['phase'] == 4) and value['before'] == entry['state']
              and r['before'] == digest(value['before']) and r['native'] == entry['native'],
              'ECHO_READBACK_TRANSITION_JOIN')
        before, after = decode(value['before']['hidden']), decode(value['after']['hidden'])
        check(value['before']['sequence'] % 5 == r['phase'] and
              value['after']['sequence'] == value['before']['sequence']+1 and
              value['after']['last_event'] == digest(r) and
              value['before']['standard'] == value['after']['standard'], 'ECHO_READBACK_PHASE')
        for i, (name, observed) in enumerate(observed_arrays[r['context']].items()):
            actual_hash = dict(shape=list(observed.shape), dtype=observed.dtype.str,
                               sha256=sha(observed.tobytes()))
            check(actual_hash == value['observed'][name], 'ECHO_READBACK_OBSERVED_WINDOWS')
            prediction = predictions[entry['condition']]['constellation'][name]
            check(prediction['instrument'] == instruments[name], 'ECHO_READBACK_INSTRUMENT')
            pred = decode(prediction).astype(np.float64)
            target = observed.astype(np.float64)
            pred /= np.linalg.norm(pred, axis=1, keepdims=True)
            target /= np.linalg.norm(target, axis=1, keepdims=True)
            cosines = np.clip(target@pred.T, -1, 1)
            err = value['prediction_error'][name]
            check(np.array_equal(cosines, np.asarray(err['cosines'])) and
                  abs(distance(cosines)-err['ordered_distance']) < 1e-12, 'ECHO_READBACK_DISCREPANCY')
            delta = float(np.linalg.norm(after[i].astype(np.float64)-before[i].astype(np.float64)))
            check(delta > 0 and abs(delta-value['state_delta_by_instrument'][name]) < 1e-5,
                  'ECHO_READBACK_DELTA')
            checks += 1
        deltas.append(dict(context=r['context'], phase=r['phase'], values=value['state_delta_by_instrument']))
        entry.update(state=value['after'], status='awaiting_correction' if r['phase'] == 3 else 'corrected_state',
                     last_transition=event['payload'])
    final = json.loads((root/'snapshot-final.json').read_bytes())
    cold = json.loads((root/'cold-report.json').read_bytes())
    check(final['head'] == head == cold['head'] and digest(final) == cold['snapshot'] and
          final['native_feedback_contexts'] == contexts, 'ECHO_READBACK_FINAL')
    original = json.loads((prior/'snapshot-final.json').read_bytes())
    excluded = {'head','events','native_feedback_contexts','native_feedback_enrollments'}
    check({k:v for k,v in final.items() if k not in excluded} ==
          {k:v for k,v in original.items() if k not in excluded}, 'ECHO_READBACK_ORIGINAL_STATE')
    broken = root/'negative-native'
    shutil.copytree(native, broken)
    name = next(iter(arrays))
    with (broken/name/'slots.npz').open('ab') as f:
        f.write(b'corrupt-negative-control')
    try:
        load_native(broken, spec['population_pin'])
    except ValueError as error:
        check('ECHO_READBACK_BYTES' in str(error), 'ECHO_READBACK_NEGATIVE_CODE')
    else:
        raise AssertionError('ECHO_READBACK_CORRUPTION_ACCEPTED')
    report = dict(complete=True, acceptance='author-checked', head=head,
        journal_sha256=sha((root/'experience.sqlite3').read_bytes()), contexts=len(contexts),
        transitions=len(new)-1, independent_lane_transition_checks=checks, captured_bytes=sum(r['bytes'] for r in refs),
        native_windows_including_probe={n:len(x) for n,(x,_) in arrays.items()}, deltas=deltas,
        corrupted_native_refused=True, original_state_unchanged=True, cold_replay_equal=True,
        wall_seconds=time.perf_counter()-start, learned_understanding=False, new_weight_update=False,
        serving=False, scientific_admission=False)
    write(root/'independent-readback.json', report)
    print(json.dumps({k:report[k] for k in ('complete','head','contexts','transitions','independent_lane_transition_checks')}))


if __name__ == '__main__':
    main()
