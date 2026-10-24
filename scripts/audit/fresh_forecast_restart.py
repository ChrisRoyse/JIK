"""Restart a forecast-only journal before any independent execution is allowed."""
import argparse
from functools import partial
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims import fresh_forecast
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.fresh_experiment import FreshLoop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root = Path(config['root'])
    check(config == read(root/'config.json') and not (root/'execution').exists(),
          'FRESH_RESTART_BEFORE_EXECUTE')
    pin = (root/'forecast/COMMITTED').read_text()
    before = fresh_forecast.read(root/'forecast',pin)
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','fresh-forecast-restart')
    (root/'gpu-forecast-restart.txt').open('x').write(gpu)
    configure()
    previous = read(Path(config['prospective'])/'config.json')
    ancestor = Path(previous['source'])
    original,_ = open_session(ancestor,previous['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,previous['experiment']['correction'])
    saved = read(root/'snapshot-forecast.json')
    session,_ = open_session(ancestor,previous['source_pin'],root,saved,
                             loop_type=partial(FreshLoop,learning_base=base))
    check(session.snapshot() == saved and fresh_forecast.read(root/'forecast',pin) == before,
          'FRESH_RESTART_FORECAST_CHANGED')
    session.close()
    original.close()
    result = dict(forecast=pin,snapshot=digest(saved),events=saved['events'],head=saved['head'],
        forecast_unchanged=True,cold_replay=True,executed=False,wall_seconds=time.perf_counter()-started,
        code=sha(Path(__file__).read_bytes()),acceptance='author-checked')
    report = write_committed(root/'forecast-restart.json',result)
    print(report,flush=True)


if __name__ == '__main__':
    run_cli(main)
