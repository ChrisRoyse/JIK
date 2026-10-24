"""Connect a fresh sealed forecast, actual execution, episodic intake and restart."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims import fresh_forecast
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.fresh_experiment import FreshLoop
from oracle.fresh_experiment import run as execute


def validate_recipe(config, seal, previous):
    spec = config['experiment']
    check(spec['learning'] == previous['spec'] and config['source'] == previous['source']
          and config['prospective'] == previous['prospective'] and config['echo'] == previous['echo'],
          'FRESH_SOURCE_RECIPE')
    check(spec['before'] == spec['learning']['initial']['previous']['proposal']['checkpoint'],
          'FRESH_BASELINE_CHECKPOINT')
    root = Path(spec['corrected']['root'])
    check(root == Path(previous['root'])/'candidates/cycle-1'
          and evidence(root/'manifest.json') == seal['files']['session/candidates/cycle-1/manifest.json']
          and sha((root/'manifest.json').read_bytes()) == spec['corrected']['pin'],
          'FRESH_CANDIDATE_CHECKPOINT')
    native = spec['native']
    check(sha((Path(native['native'])/'report.json').read_bytes()) == native['native_pin'],
          'FRESH_NATIVE_PIN')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('forecast','execute','intake','cold'),required=True)
    a = p.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,source = Path(config['root']),Path(config['source'])
    seal = authenticate(Path(config['proof']['root']),config['proof']['pin'])
    previous = read(Path(config['proof']['root'])/'session/config.json')
    validate_recipe(config,seal,previous)
    echo = authenticate(Path(config['echo']['root']),config['echo']['pin'])
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(evidence(source/name) == echo['files']['session/'+name], 'FRESH_PREFIX_BYTES')
    prospective = read(Path(config['prospective'])/'config.json')
    ancestor = Path(prospective['source'])
    controls = []
    if a.stage == 'forecast':
        check(not root.exists(),'FRESH_EXPERIMENT_EXISTS')
        root.mkdir()
        (root/'base').mkdir()
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        shutil.copyfile(ancestor/'experience.sqlite3',root/'base/experience.sqlite3')
        write_committed(root/'config.json',config)
        for field,code in (('corrected','FRESH_CANDIDATE_CHECKPOINT'),('native','FRESH_NATIVE_PIN')):
            bad = copy.deepcopy(config)
            bad['experiment'][field]['native_pin' if field == 'native' else 'pin'] = '0'*64
            try:
                validate_recipe(bad,seal,previous)
            except ValueError as error:
                check(code in str(error), 'FRESH_CONTROL_CODE')
            else:
                raise AssertionError('FRESH_CONTROL_ACCEPTED:'+field)
            controls.append(dict(name='wrong_'+field,code=code))
        check(evidence(root/'experience.sqlite3') == evidence(source/'experience.sqlite3'),
              'FRESH_CONTROL_MUTATION')
    else:
        check(read(root/'config.json') == config,'FRESH_EXPERIMENT_CONFIG')
    if a.stage == 'execute':
        pin = (root/'forecast/COMMITTED').read_text()
        output = execute(root/'execution',root/'forecast',pin)
        write_committed(root/'execution-stage.json',dict(pin=output,wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(stage=a.stage,observation_pin=output)),flush=True)
        return
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','fresh-proposal-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original,_ = open_session(ancestor,prospective['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    saved = read(source/'snapshot-final.json' if a.stage == 'forecast' else
                 root/('snapshot-forecast.json' if a.stage == 'intake' else 'snapshot-final.json'))
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
                             loop_type=partial(FreshLoop,learning_base=base))
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    if a.stage == 'forecast':
        spec = dict(config['experiment'],execution=str(root/'execution'))
        body = session.forecast_fresh(spec)
        pin = fresh_forecast.publish(session,root/'forecast',body)
        write_committed(root/'snapshot-forecast.json',session.snapshot())
        result = dict(forecast_pin=pin,body=body)
    elif a.stage == 'intake':
        pin = (root/'forecast/COMMITTED').read_text()
        before = fresh_forecast.read(root/'forecast',pin)
        observed_pin = (root/'execution/COMMITTED').read_text()
        body = session.observe_fresh(root/'execution',observed_pin,root/'forecast',pin)
        check(fresh_forecast.read(root/'forecast',pin) == before,'FRESH_FORECAST_REWRITTEN')
        snapshot = digest(session.snapshot())
        try:
            session.observe_fresh(root/'execution',observed_pin,root/'forecast',pin)
        except ValueError as error:
            check('PROPOSAL_NOT_PENDING' in str(error),'FRESH_DUPLICATE_CODE')
        else:
            raise AssertionError('FRESH_DUPLICATE_ACCEPTED')
        check(digest(session.snapshot()) == snapshot,'FRESH_DUPLICATE_MUTATION')
        controls.append(dict(name='duplicate_observation',code='PROPOSAL_NOT_PENDING'))
        write_committed(root/'snapshot-final.json',session.snapshot())
        write_committed(root/'feedback.json',session.journal.payload(body['payload']))
        result = dict(body=body,forecast_unchanged=True)
    else:
        check(session.snapshot() == saved,'FRESH_COLD_REPLAY')
        result = dict(snapshot=digest(saved),cold_replay=True)
    result.update(stage=a.stage,head=session.head,events=session.journal.count,controls=controls)
    session.close()
    original.close()
    finished = time.perf_counter()
    result.update(phases=dict(authentication_replay=replayed-started,action_close=finished-replayed),
                  wall_seconds=finished-started,journal=sha((root/'experience.sqlite3').read_bytes()))
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k != 'body'}),flush=True)


if __name__ == '__main__':
    run_cli(main)
