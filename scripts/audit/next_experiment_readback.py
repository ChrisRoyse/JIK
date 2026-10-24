"""Read persisted forecast/process/journal boundaries with falsifying controls."""
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
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims import next_forecast
from claims.next_observation import NextObservations
from claims.process_conditions import write_committed
from model.loop_training_checkpoint import tensor_fingerprint
from model.process_anchor import validate as validate_prediction


def readback(root, output):
    check(not output.exists(),'PROPOSAL_READBACK_EXISTS')
    original = {p.relative_to(root).as_posix():sha(p.read_bytes())
                for p in root.rglob('*') if p.is_file()}
    output.mkdir()
    pin = (root/'forecast/COMMITTED').read_text()
    seal, forecasts = next_forecast.read(root/'forecast',pin)
    op = (root/'execution/COMMITTED').read_text()
    observed = NextObservations(root/'execution',op,root/'forecast',pin)
    feedback = json.loads((root/'feedback.json').read_bytes())
    snapshot = json.loads((root/'snapshot-final.json').read_bytes())
    config = json.loads((root/'config.json').read_bytes())
    before = json.loads((Path(config['source'])/'snapshot-final.json').read_bytes())
    check(forecasts['same_expanded_reference_bank'] and forecasts['fixed_historical_context']
          and not forecasts['new_weight_update'], 'NEXT_READBACK_COMPARISON_SCOPE')
    for field in ('standard','states','active_model','world','world_truth_established','serving'):
        check(snapshot[field] == before[field],'NEXT_READBACK_RETAINED_STATE:'+field)
    for name, checkpoint in forecasts['checkpoints'].items():
        spec = config['experiment'][name if name == 'before' else 'corrected']
        folder = Path(spec['root'])
        raw = (folder/'manifest.json').read_bytes()
        manifest = json.loads(raw)
        check(sha(raw) == spec['pin'] == checkpoint['pin'], 'NEXT_READBACK_CHECKPOINT')
        for filename, item in manifest['files'].items():
            raw = (folder/filename).read_bytes()
            check(item == dict(bytes=len(raw),sha256=sha(raw)), 'NEXT_READBACK_WEIGHT_BYTES')
        weights = torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
        training = torch.load(folder/'training.pt',weights_only=True,map_location='cpu')
        if name == 'before':
            # Independently reconstruct only the representation change: the
            # baseline parameters and Adam state remain the original parent's.
            basis = config['experiment']['learning']['basis']
            path = Path(basis['root'])
            raw = (path/'manifest.json').read_bytes()
            check(sha(raw) == basis['pin'],'NEXT_READBACK_BASIS')
            bm = json.loads(raw)
            raw = (path/'arrays.npz').read_bytes()
            check(dict(bytes=len(raw),sha256=sha(raw)) == bm['arrays'],'NEXT_READBACK_BASIS_BYTES')
            with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
                for i,row in enumerate(bm['branches']):
                    weights[row['name']+'.anchor'] = weights[row['parameter']].clone()
                    weights[row['name']+'.projector'] = torch.from_numpy(arrays[f'p_{i}'].copy())
        check(checkpoint['fingerprint'] == dict(model=digest(tensor_fingerprint(weights)),
            optimizer=digest(tensor_fingerprint(training['optimizer']))),'NEXT_READBACK_CHECKPOINT_STATE')
    controls = []

    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'PROPOSAL_READBACK_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('PROPOSAL_READBACK_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))

    refuse('wrong_forecast_pin','PROPOSAL_FORECAST_PIN',lambda:next_forecast.read(root/'forecast','0'*64))
    refuse('wrong_observation_pin','PROPOSAL_OBSERVATION_PIN',
           lambda:NextObservations(root/'execution','0'*64,root/'forecast',pin))
    results = {}
    for ob,entry in observed.entries.items():
        condition = entry['condition']
        actual = observed.read(ob)
        prediction = forecasts['predictions'][condition]
        check(prediction['models']['before']['termination']['bank'] ==
              prediction['models']['corrected']['termination']['bank'], 'NEXT_READBACK_BANK')
        score = feedback['scores'][condition]
        lead = next_forecast.match(seal,forecasts,condition,actual)
        check(score['observation'] == ob and score['seconds_before_execution'] == lead,
              'PROPOSAL_READBACK_FORECAST_JOIN')
        truth = [float(i == int(actual['result']['returncode'] != 0)) for i in range(3)]
        models = {}
        for name,value in prediction['models'].items():
            p = value['termination']
            validate_prediction(p)
            check(len(p['bank']['rows']) == 17,'NEXT_READBACK_REGION_COUNT')
            check(all(r['observation'] != ob for r in p['bank']['rows']),'PROPOSAL_READBACK_TARGET_LEAK')
            prior_row = max((r for r in p['bank']['rows'] if r['observation'] is not None),key=lambda r:r['unit'])
            prior_brier = sum((float(i == prior_row['outcome'])-v)**2 for i,v in enumerate(truth))
            checked = {}
            for lane,v in {**p['instruments'],'common':p['common']}.items():
                expected = sum((a-b)**2 for a,b in zip(v['probabilities'],truth))
                saved = score['models'][name]['common' if lane == 'common' else 'instruments']
                saved = saved if lane == 'common' else saved[lane]
                check(expected == saved['brier'] and v['probabilities'][2] == saved['unsupported_mass']
                      and saved['uniform_brier'] == .5 and saved['prior_observed_brier'] == prior_brier,
                      'PROPOSAL_READBACK_SCORE')
                checked[lane] = dict(probabilities=v['probabilities'],brier=expected,
                                     uniform_brier=.5,prior_observed_brier=prior_brier)
            models[name] = checked
        episode = observed.episode(ob,evidence=(observed.condition_inputs().episode(condition)['id'],))
        check(episode['id'] in feedback['episodes'],'PROPOSAL_READBACK_EPISODE')
        results[entry['name']] = dict(observation=ob,returncode=actual['result']['returncode'],
            stdout=actual['captured']['stdout'],stderr=actual['captured']['stderr'],
            lead_seconds=lead,models=models,episode=episode['id'])
        if not any(c['name'] == 'late_forecast' for c in controls):
            late = dict(seal,committed_at=actual['finished'])
            refuse('late_forecast','PROPOSAL_FORECAST_LATE',lambda:next_forecast.match(late,forecasts,condition,actual))
            changed = copy.deepcopy(actual)
            changed['conditions']['commitment'] = '0'*64
            refuse('different_conditions','PROPOSAL_FORECAST_CONDITION',
                   lambda:next_forecast.match(seal,forecasts,condition,changed))
            other = next(c for c in forecasts['predictions'] if c != condition)
            refuse('swapped_observation','PROPOSAL_FORECAST_CONDITION',
                   lambda:next_forecast.match(seal,forecasts,other,actual))
            future = copy.deepcopy(forecasts)
            row = next(r for r in future['predictions'][condition]['models']['before']['termination']['bank']['rows']
                       if r['observation'] is not None)
            row['finished'] = actual['finished']
            refuse('future_anchor','PROPOSAL_FORECAST_FUTURE_BANK',
                   lambda:next_forecast.match(seal,future,condition,actual))
            wrong = copy.deepcopy(prediction['models']['before']['termination'])
            wrong['bank']['rows'][0]['unit'] += 1
            refuse('changed_bank','PROCESS_ANCHOR_BANK_PIN',lambda:validate_prediction(wrong))
    corrupt = output/'corrupt-capture'
    shutil.copytree(root/'execution',corrupt)
    shutil.copyfile(root/'forecast-restart.json',output/'forecast-restart.json')
    path = corrupt/'with_dependency/execution/stdout.bin'
    path.write_bytes(path.read_bytes()+b'changed')
    refuse('changed_persisted_capture','PROPOSAL_OBSERVATION_BYTES',
           lambda:NextObservations(corrupt,op,root/'forecast',pin))
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
              and last['kind'] == 'next_feedback','PROPOSAL_READBACK_JOURNAL')
        payload, = db.execute('SELECT data FROM payloads WHERE hash=?',(last['payload'],)).fetchone()
        check(sha(payload) == last['payload'] and json.loads(payload) == feedback,'PROPOSAL_READBACK_FEEDBACK')
    cold = json.loads((root/'cold-stage.json').read_bytes())
    check(cold['cold_replay'] and cold['snapshot'] == digest(snapshot),'PROPOSAL_READBACK_RESTART')
    restart = json.loads((root/'forecast-restart.json').read_bytes())
    check(restart['cold_replay'] and not restart['executed'] and restart['head'] == seal['head']
          and restart['forecast'] == pin and restart['snapshot'] == digest(json.loads(
              (root/'snapshot-forecast.json').read_bytes())), 'NEXT_READBACK_PROSPECTIVE_RESTART')
    check(snapshot['next_experiments'][seal['forecast']['payload']]['status'] == 'observed'
          and feedback['why_status'] == 'unresolved' and not feedback['task_correctness_established']
          and not feedback['new_weight_update'], 'NEXT_READBACK_EPISTEMIC_SCOPE')
    stage_controls = {s:json.loads((root/(s+'-stage.json')).read_bytes())['controls']
                      for s in ('forecast','intake')}
    check(len(stage_controls['forecast']) == 2 and len(stage_controls['intake']) == 1,
          'NEXT_READBACK_STAGE_CONTROLS')
    check({p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()} == original,
          'NEXT_READBACK_SOURCE_MUTATION')
    report = dict(schema='next-prospective-readback-v1',forecast=pin,observations=op,
        before=dict(head=before['head'],events=before['events']),after=dict(head=head,events=count),
        controls=controls,stage_controls=stage_controls,results=results,cold_snapshot=cold['snapshot'],
        source_bytes_unchanged=True,checkpoints=forecasts['checkpoints'],forecast_restart=restart,
        boundaries=['frozen_forecast_before_occurrence','committed_inputs_to_captured_process','occurrence_to_journal_and_restart'],
        author_checked=True,scientific_admission=False,serving=False,exposure_closure_complete=False,
        native_output_echo_measured=False,new_weight_update=False,
        useful_transfer_established=False,
        baseline_scope='process_termination_only_must_beat_prior_not_just_parent')
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
