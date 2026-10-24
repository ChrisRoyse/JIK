"""Real forecast, occurrence, intake and cold replay stages for a selected proposal."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.corrective_open import open_session
from audit.fsv_field_world import authenticate
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims import proposal_forecast
from lens.lenscommon import run_cli, take_lease
from model.proposal_experiment import ProposalLoop
from oracle.proposal_experiment import run as execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('forecast','execute','intake','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    root, source = Path(config['root']), Path(config['source'])
    if a.stage == 'forecast':
        check(not root.exists(),'PROPOSAL_EXPERIMENT_EXISTS')
        authenticate(source,config['source_pin'])
        root.mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        write_committed(root/'config.json',config)
    else:
        check(config == json.loads((root/'config.json').read_bytes()),'PROPOSAL_EXPERIMENT_CONFIG')
    if a.stage == 'execute':
        pin = (root/'forecast/COMMITTED').read_text()
        output = execute(root/'execution',root/'forecast',pin)
        write_committed(root/'execution-stage.json',dict(pin=output,wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(stage=a.stage,observation_pin=output)),flush=True)
        return
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','proposal-'+a.stage)
    with (root/('gpu-'+a.stage+'.txt')).open('x') as stream:
        stream.write(gpu)
    configure()
    saved_path = (source/'snapshot-final.json' if a.stage == 'forecast' else
                  root/('snapshot-forecast.json' if a.stage == 'intake' else 'snapshot-final.json'))
    saved = json.loads(saved_path.read_bytes())
    session, _ = open_session(source,config['source_pin'],root,saved,loop_type=ProposalLoop)
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    if a.stage == 'forecast':
        spec = dict(config['experiment'],execution=str(root/'execution'))
        body = session.forecast_proposal(spec)
        pin = proposal_forecast.publish(session,root/'forecast',body)
        write_committed(root/'snapshot-forecast.json',session.snapshot())
        result = dict(forecast_pin=pin,body=body)
    elif a.stage == 'intake':
        pin = (root/'forecast/COMMITTED').read_text()
        before = proposal_forecast.read(root/'forecast',pin)
        body = session.observe_proposal(root/'execution',(root/'execution/COMMITTED').read_text(),
                                        root/'forecast',pin)
        check(proposal_forecast.read(root/'forecast',pin) == before,'PROPOSAL_FORECAST_REWRITTEN')
        write_committed(root/'snapshot-final.json',session.snapshot())
        write_committed(root/'feedback.json',session.journal.payload(body['payload']))
        result = dict(body=body,forecast_unchanged=True)
    else:
        check(session.snapshot() == saved,'PROPOSAL_COLD_REPLAY')
        result = dict(snapshot=digest(saved),cold_replay=True)
    finished = time.perf_counter()
    result.update(stage=a.stage,head=session.head,events=session.journal.count,
        phases=dict(authentication_replay=replayed-started,action=finished-replayed),wall_seconds=finished-started)
    write_committed(root/(a.stage+'-stage.json'),result)
    session.journal.db.close()
    print(json.dumps({k:v for k,v in result.items() if k != 'body'}),flush=True)


if __name__ == '__main__':
    run_cli(main)
