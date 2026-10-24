"""Seal, cold-replay, execute and intake prospective transfer after typed correction."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.attempt_open import open_attempt, prepare, recipe
from audit.proposal_recognition import read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims import attempt_forecast
from lens.lenscommon import run_cli, take_lease
from oracle.attempt_experiment import run as execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('forecast','restart','execute','intake','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,source = Path(config['root']),Path(config['source'])
    prior,previous = recipe(config)
    controls = []
    if a.stage == 'forecast':
        prepare(config,prior,previous,root)
        write_committed(root/'config.json',config)
        for field,code in (('corrected','ATTEMPT_CANDIDATE_CHECKPOINT'),('native','ATTEMPT_NATIVE_PIN'),
                           ('input_readback','ATTEMPT_INPUT_READBACK')):
            bad = copy.deepcopy(config)
            target = bad if field == 'input_readback' else bad['experiment']
            target[field]['native_pin' if field == 'native' else 'pin'] = '0'*64
            try:
                recipe(bad)
            except ValueError as error:
                check(code in str(error),'ATTEMPT_CONTROL_CODE:'+str(error))
            else:
                raise AssertionError('ATTEMPT_CONTROL_ACCEPTED:'+field)
            controls.append(dict(name='wrong_'+field,code=code))
    else:
        check(read(root/'config.json') == config,'ATTEMPT_CONFIG_CHANGED')
    if a.stage == 'execute':
        pin = (root/'forecast/COMMITTED').read_text()
        observed = execute(root/'execution',root/'forecast',pin)
        value = dict(observation_pin=observed,wall_seconds=time.perf_counter()-started)
        write_committed(root/'execute-stage.json',value)
        print(json.dumps(value),flush=True)
        return
    if a.stage in ('forecast','restart'):
        check(not (root/'execution').exists(),'ATTEMPT_FORECAST_LATE')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','attempt-transfer-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = read(source/'snapshot-final.json' if a.stage == 'forecast' else
                 root/('snapshot-final.json' if a.stage == 'cold' else 'snapshot-forecast.json'))
    session,sessions = open_attempt(config,prior,previous,root,saved)
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    if a.stage == 'forecast':
        body = session.forecast_attempt(dict(config['experiment'],execution=str(root/'execution')))
        snapshot = session.snapshot()
        try:
            session.forecast_attempt(dict(config['experiment'],execution=str(root/'execution')))
        except ValueError as error:
            check('ATTEMPT_FORECAST_PREFIX' in str(error),'ATTEMPT_DUPLICATE_FORECAST_CODE')
        else:
            raise AssertionError('ATTEMPT_DUPLICATE_FORECAST_ACCEPTED')
        check(session.snapshot() == snapshot,'ATTEMPT_DUPLICATE_FORECAST_MUTATION')
        controls.append(dict(name='duplicate_forecast',code='ATTEMPT_FORECAST_PREFIX'))
        pin = attempt_forecast.publish(session,root/'forecast',body)
        write_committed(root/'snapshot-forecast.json',session.snapshot())
        result = dict(forecast=pin,body=body)
    elif a.stage == 'intake':
        pin = (root/'forecast/COMMITTED').read_text()
        before = attempt_forecast.read(root/'forecast',pin)
        observed_pin = (root/'execution/COMMITTED').read_text()
        body = session.observe_attempt(root/'execution',observed_pin,root/'forecast',pin)
        check(attempt_forecast.read(root/'forecast',pin) == before,'ATTEMPT_FORECAST_REWRITTEN')
        snapshot = session.snapshot()
        try:
            session.observe_attempt(root/'execution',observed_pin,root/'forecast',pin)
        except ValueError as error:
            check('ATTEMPT_NOT_PENDING' in str(error),'ATTEMPT_DUPLICATE_CODE')
        else:
            raise AssertionError('ATTEMPT_DUPLICATE_ACCEPTED')
        check(session.snapshot() == snapshot,'ATTEMPT_DUPLICATE_MUTATION')
        controls.append(dict(name='duplicate_observation',code='ATTEMPT_NOT_PENDING'))
        write_committed(root/'snapshot-final.json',snapshot)
        write_committed(root/'feedback.json',session.journal.payload(body['payload']))
        result = dict(body=body,forecast_unchanged=True)
    else:
        check(session.snapshot() == saved,'ATTEMPT_COLD_REPLAY')
        pin = (root/'forecast/COMMITTED').read_text()
        attempt_forecast.read(root/'forecast',pin)
        result = dict(forecast=pin,snapshot=digest(saved),cold_replay=True,executed=a.stage != 'restart')
    result.update(stage=a.stage,head=session.head,events=session.journal.count,controls=controls)
    for item in sessions:
        item.close()
    finished = time.perf_counter()
    result.update(phases=dict(authentication_replay=replayed-started,action_close=finished-replayed),
        wall_seconds=finished-started,journal=sha((root/'experience.sqlite3').read_bytes()),
        code=sha(Path(__file__).read_bytes()),serving=False,scientific_admission=False,acceptance='author-checked')
    write_committed(root/('forecast-restart.json' if a.stage == 'restart' else a.stage+'-stage.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k != 'body'}),flush=True)


if __name__ == '__main__':
    run_cli(main)
