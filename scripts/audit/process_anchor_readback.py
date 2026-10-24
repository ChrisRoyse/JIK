"""Read SQLite, capture bytes and scalar forecasts independently of model/oracle code."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.prospective_execution_readback import canonical, journal, sha


def run(config):
    started = time.perf_counter()
    c = json.loads(config.read_bytes())
    root = Path(c['output'])
    _, events, payloads, head = journal(root/'experience.sqlite3')
    actual, files = {}, 0
    for e in c['executions']:
        directory = Path(e['observations'])
        raw = (directory/'manifest.json').read_bytes()
        assert sha(raw) == (directory/'COMMITTED').read_text()
        manifest = json.loads(raw)
        for name, receipt in manifest['files'].items():
            blob = (directory/name).read_bytes()
            assert len(blob) == receipt['bytes'] and sha(blob) == receipt['sha256']
            files += 1
        seal = json.loads((directory/'forecast/forecast.json').read_bytes())
        for key, entry in manifest['observations'].items():
            assert key == sha(canonical(entry)) and key not in actual
            execution = json.loads((directory/entry['execution']/'execution.json').read_bytes())
            after = json.loads((directory/entry['execution']/'after.json').read_bytes())
            assert execution['container_id'] == after['Id'] and not after['State']['Running']
            began = datetime.fromisoformat(after['State']['StartedAt'].replace('Z','+00:00'))
            sealed = datetime.fromisoformat(seal['sealed_at'])
            assert began > sealed
            result = execution['result']
            assert type(result['returncode']) is int
            outer = (directory/entry['execution']/'stdout.bin').read_text()
            rows = [json.loads(r.split('=',1)[1]) for r in outer.splitlines()
                    if r.startswith('MNLM_EXECUTION_RESULT=')]
            assert rows == [result]
            actual[key] = dict(returncode=result['returncode'],outcome=int(result['returncode'] != 0),
                observation_set=sha(raw),sealed=sealed,started=began,seal=seal,
                finished=datetime.fromisoformat(after['State']['FinishedAt'].replace('Z','+00:00')))
    phases = dict(source_authentication=time.perf_counter()-started)
    tick = time.perf_counter()
    predictions, executions, published, scored, enrollment_order = {}, {}, {}, [], {}
    for sequence,event in enumerate(events):
        if event['kind'] == 'execution_enrollment':
            for forecast, observation in event['links'].items():
                assert forecast in predictions and observation not in enrollment_order
                enrollment_order[observation] = sequence
                assert actual[observation]['seal']['forecasts'][forecast]['execution_anchor'] == predictions[forecast]
        if event['kind'] == 'grow':
            for unit, support in event['supports'].items():
                attachment = executions[support['forecast']]
                observation = attachment['observation']
                assert enrollment_order[observation] < sequence and int(unit) not in published
                published[int(unit)] = attachment
        payload = payloads.get(event.get('payload'),{})
        if 'forecast' in payload:
            record = payload['forecast']
            key = sha(canonical(record))
            p = record['execution_anchor']
            assert p['serving'] is False and p['calibrated'] is False and p['scientific_admission'] is False
            bank = p['bank']
            assert bank['pin'] == sha(canonical({k:v for k,v in bank.items() if k != 'pin'}))
            assert bank['classes'] == ['returncode_zero','returncode_nonzero','unobserved_region']
            assert [r['unit'] for r in bank['rows']] == record['region_context']['definition']['units']
            for row in bank['rows']:
                earlier = published.get(row['unit'])
                if earlier is None:
                    assert row == dict(unit=row['unit'],outcome=2,observation=None)
                else:
                    a = actual[earlier['observation']]
                    assert row['observation'] == earlier['observation']
                    assert row['native'] == earlier['native']
                    assert row['returncode'] == a['returncode'] and row['outcome'] == a['outcome']
                    assert row['observation_set'] == a['observation_set']
            assert set(p['instruments']) == set(record['predictions'])
            for v in (*p['instruments'].values(),p['common']):
                weights = v['region_probabilities']
                assert len(weights) == len(bank['rows'])
                assert all(math.isfinite(x) and x >= 0 for x in weights)
                assert abs(sum(weights)-1) < 1e-12
                sums = [sum(x for r,x in zip(bank['rows'],weights) if r['outcome'] == i) for i in range(3)]
                assert all(abs(x-y) < 1e-12 for x,y in zip(v['probabilities'],sums))
            predictions[key] = p
        if 'execution' in payload:
            attachment = payload['execution']
            observation = attachment['observation']
            matches = [k for k,p in predictions.items() if k in actual[observation]['seal']['forecasts']]
            assert len(matches) == 1
            key = matches[0]
            executions[key] = attachment
            p, a, grade = predictions[key], actual[observation], payload['execution_anchor_score']
            assert grade['prediction'] == sha(canonical(p)) and grade['observation'] == observation
            assert grade['returncode'] == a['returncode'] and grade['outcome'] == a['outcome']
            prior = [r for r in p['bank']['rows'] if r['observation'] is not None]
            assert all(actual[r['observation']]['finished'] < a['sealed'] for r in prior)
            assert observation not in [r['observation'] for r in prior]
            latest = max(prior,key=lambda r:r['unit']) if prior else None
            last_class = latest['outcome'] if latest else 2
            assert grade['prior_observation'] == (latest['observation'] if latest else None)
            lanes = {**p['instruments'],'common':p['common']}
            scores = {**grade['instruments'],'common':grade['common']}
            for name,v in lanes.items():
                probabilities = v['probabilities']
                brier = sum((x-float(i == a['outcome']))**2 for i,x in enumerate(probabilities))
                s = scores[name]
                assert s['brier'] == brier and s['uniform_brier'] == .5
                assert s['prior_observed_brier'] == (0. if last_class == a['outcome'] else 2.)
                assert s['unsupported_mass'] == probabilities[2]
                assert s['zero_returncode_mass_interval'] == [probabilities[0],min(1.,probabilities[0]+probabilities[2])]
            scored.append(dict(forecast=key,observation=observation,returncode=a['returncode'],
                prior_outcomes=len(prior),heads=len(lanes),seconds_before_execution=(a['started']-a['sealed']).total_seconds(),
                probabilities={n:v['probabilities'] for n,v in lanes.items()},scores=scores))
    assert len(scored) == len(actual) == 3 and len(published) == 2
    phases['forecast_publication_and_outcome_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    return dict(complete=True,acceptance='author-checked',head=head,source_files=files,
                forecasts=len(predictions),published_actual_regions=len(published),scored=scored,
                phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
                scientific_admission=False,task_correctness_established=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    value = run(a.config)
    with a.output.open('xb') as f:
        f.write(canonical(value))
    print(json.dumps(dict(complete=True,forecasts=value['forecasts'],source_files=value['source_files'])))
