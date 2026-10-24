"""Read persisted forecast/process/journal boundaries with falsifying controls."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims import proposal_forecast
from claims.proposal_observation import ProposalObservations
from claims.process_conditions import write_committed


def readback(root, output):
    check(not output.exists(),'PROPOSAL_READBACK_EXISTS')
    output.mkdir()
    pin = (root/'forecast/COMMITTED').read_text()
    seal, forecasts = proposal_forecast.read(root/'forecast',pin)
    op = (root/'execution/COMMITTED').read_text()
    observed = ProposalObservations(root/'execution',op,root/'forecast',pin)
    feedback = json.loads((root/'feedback.json').read_bytes())
    snapshot = json.loads((root/'snapshot-final.json').read_bytes())
    config = json.loads((root/'config.json').read_bytes())
    before = json.loads((Path(config['source'])/'snapshot-final.json').read_bytes())
    controls = []

    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'PROPOSAL_READBACK_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('PROPOSAL_READBACK_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))

    refuse('wrong_forecast_pin','PROPOSAL_FORECAST_PIN',lambda:proposal_forecast.read(root/'forecast','0'*64))
    refuse('wrong_observation_pin','PROPOSAL_OBSERVATION_PIN',
           lambda:ProposalObservations(root/'execution','0'*64,root/'forecast',pin))
    results = {}
    for ob,entry in observed.entries.items():
        condition = entry['condition']
        actual = observed.read(ob)
        prediction = forecasts['predictions'][condition]
        score = feedback['scores'][condition]
        lead = proposal_forecast.match(seal,forecasts,condition,actual)
        check(score['observation'] == ob and score['seconds_before_execution'] == lead,
              'PROPOSAL_READBACK_FORECAST_JOIN')
        truth = [float(i == int(actual['result']['returncode'] != 0)) for i in range(3)]
        models = {}
        for name,value in prediction['models'].items():
            p = value['termination']
            check(all(r['observation'] != ob for r in p['bank']['rows']),'PROPOSAL_READBACK_TARGET_LEAK')
            checked = {}
            for lane,v in {**p['instruments'],'common':p['common']}.items():
                expected = sum((a-b)**2 for a,b in zip(v['probabilities'],truth))
                saved = score['models'][name]['common' if lane == 'common' else 'instruments']
                saved = saved if lane == 'common' else saved[lane]
                check(expected == saved['brier'] and v['probabilities'][2] == saved['unsupported_mass'],
                      'PROPOSAL_READBACK_SCORE')
                checked[lane] = dict(probabilities=v['probabilities'],brier=expected)
            models[name] = checked
        episode = observed.episode(ob,evidence=(observed.condition_inputs().episode(condition)['id'],))
        check(episode['id'] in feedback['episodes'],'PROPOSAL_READBACK_EPISODE')
        results[entry['name']] = dict(observation=ob,returncode=actual['result']['returncode'],
            stdout=actual['captured']['stdout'],stderr=actual['captured']['stderr'],
            lead_seconds=lead,models=models,episode=episode['id'])
        if not any(c['name'] == 'late_forecast' for c in controls):
            late = dict(seal,committed_at=actual['finished'])
            refuse('late_forecast','PROPOSAL_FORECAST_LATE',lambda:proposal_forecast.match(late,forecasts,condition,actual))
            changed = copy.deepcopy(actual)
            changed['conditions']['commitment'] = '0'*64
            refuse('different_conditions','PROPOSAL_FORECAST_CONDITION',
                   lambda:proposal_forecast.match(seal,forecasts,condition,changed))
            future = copy.deepcopy(forecasts)
            row = next(r for r in future['predictions'][condition]['models']['before']['termination']['bank']['rows']
                       if r['observation'] is not None)
            row['finished'] = actual['finished']
            refuse('future_anchor','PROPOSAL_FORECAST_FUTURE_BANK',
                   lambda:proposal_forecast.match(seal,future,condition,actual))
    corrupt = output/'corrupt-capture'
    shutil.copytree(root/'execution',corrupt)
    path = corrupt/'with_dependency/execution/stdout.bin'
    path.write_bytes(path.read_bytes()+b'changed')
    refuse('changed_persisted_capture','PROPOSAL_OBSERVATION_BYTES',
           lambda:ProposalObservations(corrupt,op,root/'forecast',pin))
    head, count, last = ZERO, 0, None
    with sqlite3.connect((root/'experience.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        for seq,previous,raw,actual in db.execute('SELECT * FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == count and previous == head and canonical(body) == raw
                  and digest(dict(seq=seq,previous=previous,body=body)) == actual,'PROPOSAL_READBACK_CHAIN')
            if seq == before['events']-1:
                check(actual == before['head'],'PROPOSAL_READBACK_PREFIX')
            count, head, last = count+1,actual,body
        check(head == snapshot['head'] and count == snapshot['events'] == before['events']+2
              and last['kind'] == 'proposal_feedback','PROPOSAL_READBACK_JOURNAL')
        payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(last['payload'],)).fetchone()
        check(sha(payload) == last['payload'] and json.loads(payload) == feedback,'PROPOSAL_READBACK_FEEDBACK')
    cold = json.loads((root/'cold-stage.json').read_bytes())
    check(cold['cold_replay'] and cold['snapshot'] == digest(snapshot),'PROPOSAL_READBACK_RESTART')
    report = dict(schema='proposal-prospective-readback-v1',forecast=pin,observations=op,
        before=dict(head=before['head'],events=before['events']),after=dict(head=head,events=count),
        controls=controls,results=results,cold_snapshot=cold['snapshot'],
        boundaries=['frozen_forecast_before_occurrence','committed_inputs_to_captured_process','occurrence_to_journal_and_restart'],
        author_checked=True,scientific_admission=False,serving=False,exposure_closure_complete=False,
        native_output_echo_measured=False,new_weight_update=False)
    report_pin = write_committed(output/'readback.json',report)
    return dict(pin=report_pin,controls=len(controls),events=count,
        results={k:dict(returncode=v['returncode'],common={n:r['common'] for n,r in v['models'].items()})
                 for k,v in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    print(json.dumps(readback(a.root,a.output)))
