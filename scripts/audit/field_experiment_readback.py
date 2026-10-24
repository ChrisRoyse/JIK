"""Independent SQL/byte/window/clock/score readback of controlled field executions."""
import argparse
import base64
from datetime import datetime
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.prospective_execution_readback import canonical, journal, sha


def read(path, pin):
    raw = path.read_bytes()
    assert sha(raw) == pin, str(path)
    return raw


def moment(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-12, (a,b)


def native(root, pin):
    report = json.loads(read(root/'report.json', pin))
    result = {}
    for name, row in report['lenses'].items():
        lane = json.loads(read(root/name/'report.json', row['report_sha256']))
        with np.load(io.BytesIO(read(root/name/'slots.npz', lane['files']['slots.npz']['sha256'])),
                     allow_pickle=False) as z:
            result[name] = z['vectors'].copy(), z['offsets'].copy()
    return result


def check_fields(payload, source):
    f3 = json.loads((source/'source.json').read_bytes())
    config = json.loads((Path(f3['upstream']['root'])/'summary.json').read_bytes())['config']
    action_arrays = native(Path(config['native']), config['native_pin'])
    pop = Path(config['population'])
    report = json.loads(read(pop/'report.json', config['population_pin']))
    spans = [json.loads(v) for v in read(pop/'reading_spans.jsonl',
        report['files']['reading_spans.jsonl']['sha256']).splitlines()]
    cache, result = {}, {}
    action = payload['historical_request']['reference']
    for condition, value in payload['predictions'].items():
        desc = value['fields']
        measured = desc['measured_fields']
        pin = measured['native']
        if pin not in cache:
            provider, = [v for v in config['providers'] if v['condition_native_pin'] == pin]
            arrays = native(Path(provider['condition_native']), pin)
            pop = Path(provider['condition_population'])
            report = json.loads(read(pop/'report.json', provider['condition_population_pin']))
            rows = json.loads(read(pop/'fields.json', report['files']['fields.json']['sha256']))
            cache[pin] = arrays, rows
        arrays, rows = cache[pin]
        selected = [r for r in rows if r['condition'] == condition]
        assert desc['historical_request'] == sha(canonical(payload['historical_request']))
        assert desc['fresh_conditions'] == dict(condition=condition, pin=payload['conditions']['pin'])
        assert desc['numerical_identifiers'] is False and desc['scope'] == 'planned_resource_intervention'
        fingerprints, counts = {}, {}
        for name, (vectors, ptr) in action_arrays.items():
            pieces = [vectors[ptr[r['atom']]:ptr[r['atom']+1]] for r in spans if r['unit'] == action['unit']]
            av = np.concatenate(pieces)
            cursor = len(av)
            expected = [dict(role='action', start=0, end=cursor, text_sha256=action['text_sha256'])]
            pieces = [av]
            values, offsets = arrays[name]
            role_counts = dict(action=len(av), program=0, wheel_filename=0)
            for row in selected:
                part = values[offsets[row['unit']]:offsets[row['unit']+1]]
                pieces.append(part)
                expected.append(dict(role=row['field'], start=cursor, end=cursor+len(part), text_sha256=row['text_sha256']))
                role_counts[row['field']] += len(part)
                cursor += len(part)
            assert expected == desc['fields'][name]
            joined = np.concatenate(pieces).astype(np.float32)[None]
            fingerprints[name] = dict(shape=list(joined.shape), dtype=joined.dtype.str, sha256=sha(joined.tobytes()))
            counts[name] = role_counts
        assert sha(canonical(fingerprints)) == value['values']
        result[condition] = counts
    return result


def score(prediction, grade, outcome, observation, committed):
    bank = prediction['bank']
    assert bank['pin'] == sha(canonical({k:v for k,v in bank.items() if k != 'pin'}))
    assert prediction['serving'] is False and prediction['scientific_admission'] is False and prediction['calibrated'] is False
    assert all(r['observation'] != observation for r in bank['rows'])
    seen = [r for r in bank['rows'] if r['observation'] is not None]
    assert all(moment(r['finished']) < committed for r in seen)
    latest = max(seen, key=lambda r:r['unit']) if seen else None
    prior = latest['outcome'] if latest is not None else 2
    for name, value in {**prediction['instruments'], 'common':prediction['common']}.items():
        p, regions = value['probabilities'], value['region_probabilities']
        assert len(p) == 3 and len(regions) == len(bank['rows'])
        assert all(math.isfinite(v) and 0 <= v <= 1+1e-12 for v in p+regions)
        close(sum(p), 1.)
        close(sum(regions), 1.)
        expected = [sum(v for v,r in zip(regions,bank['rows']) if r['outcome'] == k) for k in range(3)]
        for a,b in zip(p,expected):
            close(a,b)
        actual = grade['common'] if name == 'common' else grade['instruments'][name]
        close(actual['brier'], sum((v-float(k == outcome))**2 for k,v in enumerate(p)))
        close(actual['uniform_brier'], .5)
        close(actual['prior_observed_brier'], sum((float(k == prior)-float(k == outcome))**2 for k in range(3)))
        close(actual['unsupported_mass'], p[2])
    assert grade['prediction'] == sha(canonical(prediction)) and grade['outcome'] == outcome
    return len(prediction['instruments'])+1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.root
    source_meta = json.loads((root/'source.json').read_bytes())
    source = Path(source_meta['root'])
    source_seal = json.loads(read(source/'seal.json', source_meta['seal']))
    for name, entry in source_seal['files'].items():
        assert len(read(source/name, entry['sha256'])) == entry['bytes']
    assert sha((source/'experience.sqlite3').read_bytes()) == source_meta['journal']
    seal_pin = (root/'forecast/COMMITTED').read_text()
    seal = json.loads(read(root/'forecast/forecast.json', seal_pin))
    read(root/'forecast/journal.sqlite3', seal['journal'])
    c0,e0,p0,h0 = journal(source/'experience.sqlite3')
    cf,ef,pf,hf = journal(root/'forecast/journal.sqlite3')
    c,e,payloads,head = journal(root/'experience.sqlite3')
    assert c0 == cf == c and ef[:-1] == e0 and e[:-1] == ef
    assert len(ef) == seal['events'] and hf == seal['head'] and ef[-1] == seal['forecast']
    forecast = pf[ef[-1]['payload']]
    assert forecast['candidate']['pin'] == json.loads((source/'increment.json').read_bytes())['decision']['pin']
    feedback = payloads[e[-1]['payload']]
    assert e[-1]['kind'] == 'field_experiment_feedback' and e[-1]['prior_head'] == hf
    assert e[-1]['forecast'] == ef[-1]['payload'] and e[-1]['forecast_pin'] == seal_pin
    assert feedback['new_weight_update'] is False and feedback['why_status'] == 'unresolved'
    manifest_pin = (root/'execution/COMMITTED').read_text()
    manifest = json.loads(read(root/'execution/manifest.json', manifest_pin))
    assert e[-1]['pin'] == manifest_pin and manifest['controlled_forecast'] == seal_pin
    for name, entry in manifest['files'].items():
        assert len(read(root/'execution'/name, entry['sha256'])) == entry['bytes']
    committed = moment(seal['committed_at'])
    input_commit = json.loads((root/'execution/condition-commitment.json').read_bytes())
    assert sha(canonical(input_commit)) == forecast['conditions']['pin'] == manifest['condition_commitment']
    public = json.loads(read(root/'execution/condition-inputs.json', input_commit['inputs']))['inputs']
    assert moment(input_commit['committed_at']) < committed
    assert set(public) == set(forecast['predictions'])
    window_counts = check_fields(forecast, source)
    phases = dict(source_journals_and_native_readback=time.perf_counter()-started)
    tick = time.perf_counter()
    rows, checks = [], 0
    for observation, entry in manifest['observations'].items():
        assert sha(canonical(entry)) == observation and entry['experiment'] == seal_pin
        condition = entry['condition']
        path = root/'execution'/entry['execution']
        execution = json.loads((path/'execution.json').read_bytes())
        before, after = (json.loads((path/(n+'.json')).read_bytes()) for n in ('before','after'))
        actual = execution['result']
        assert before['Id'] == after['Id'] == entry['occurrence'] == execution['container_id']
        assert execution['status'] == 'observed' and execution['exitcode'] == 0 and not execution['oom_killed']
        assert not after['State']['Running'] and after['State']['ExitCode'] == 0
        assert moment(after['State']['StartedAt']) > committed
        assert moment(after['State']['FinishedAt']) > moment(after['State']['StartedAt'])
        for state in (before, after):
            h = state['HostConfig']
            assert state['Image'] == public[condition]['image'] and state['Config']['User'] == '65534:65534'
            assert h['NetworkMode'] == 'none' and h['ReadonlyRootfs'] and not h['Privileged'] and h['CapDrop'] == ['ALL']
            assert h['Memory'] == h['MemorySwap'] == 512*1024*1024 and h['NanoCpus'] == 1_000_000_000 and h['PidsLimit'] == 64
            assert all(not v['RW'] for v in state['Mounts'])
        assert actual['program_sha256'] == public[condition]['program_sha256']
        assert actual['wheels'] == public[condition]['wheels'] and actual['harness_sha256'] == public[condition]['harness_sha256']
        captured = {}
        for name in ('stdout', 'stderr'):
            value = actual[name]
            raw = base64.b64decode(value['base64'], validate=True)
            assert len(raw) == value['bytes'] and sha(raw) == value['sha256']
            captured[name] = raw.decode()
        markers = [v.split('=',1)[1] for v in (path/'stdout.bin').read_text().splitlines() if v.startswith('MNLM_EXECUTION_RESULT=')]
        assert len(markers) == 1 and json.loads(markers[0]) == actual
        grade = feedback['scores'][condition]
        assert grade['observation'] == observation
        close(grade['seconds_before_execution'], (moment(after['State']['StartedAt'])-committed).total_seconds())
        for name in ('candidate', 'field_disabled'):
            checks += score(forecast['predictions'][condition][name], grade[name], int(actual['returncode'] != 0), observation, committed)
        rows.append(dict(condition=condition, resources=list(public[condition]['wheels']), returncode=actual['returncode'],
            captured=captured, seconds_before_execution=grade['seconds_before_execution'],
            candidate=forecast['predictions'][condition]['candidate']['common']['probabilities'],
            field_disabled=forecast['predictions'][condition]['field_disabled']['common']['probabilities'],
            candidate_brier=grade['candidate']['common']['brier'], field_disabled_brier=grade['field_disabled']['common']['brier']))
    snap = json.loads((root/'snapshot-final.json').read_bytes())
    old = json.loads((source/'snapshot.json').read_bytes())
    assert snap['head'] == head and snap['active_model'] == old['active_model'] and snap['states'] == old['states']
    new = set(snap['memory']['grounded_world_claims'])-set(old['memory']['grounded_world_claims'])
    assert new == set(feedback['episodes']) and len(new) == len(rows) == 2
    episodes = [v['event']['episode'] for v in feedback['memory_records'] if v['event']['op'] == 'intake']
    for ep in episodes:
        assert ep['id'] == sha(canonical({k:v for k,v in ep.items() if k != 'id'}))
        if ep['id'] in new:
            assert ep['claim_kind'] == 'event_occurrence' and ep['why'] is None and ep['why_status'] == 'unresolved'
            assert ep['native_measured'] is False and snap['memory']['states'][ep['id']]['grounded_scope'] == 'observed_process_in_pinned_runtime'
    cold = json.loads((root/'cold-report.json').read_bytes())
    assert cold['snapshot'] == sha(canonical(snap)) and cold['head'] == head and cold['complete']
    phases['execution_scores_and_memory'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    result = dict(schema='field-experiment-independent-readback-v1', complete=True, acceptance='author-checked',
        head=head, forecast_pin=seal_pin, observation_pin=manifest_pin, source_unchanged=True,
        outcome_score_checks=checks, new_grounded_occurrences=len(new), measured=rows, window_counts=window_counts,
        scientific_admission=False, serving=False, new_weight_update=False, phases=phases,
        wall_seconds=wall, phase_coverage=sum(phases.values())/wall)
    with (root/'independent-readback.json').open('xb') as stream:
        stream.write(canonical(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
