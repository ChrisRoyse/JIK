"""Prepare real feedback, learn with retention, resume, roll back and cold replay."""
import argparse
import copy
import shutil
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read,evidence
from audit.successor_box_open import authenticate,prepare,open_successor
from claims.codata_source import check,sha
from claims.episodic_memory import digest,canonical
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli,take_lease
from model.successor_box_data import BoxData as SuccessorData
from model import successor_checkpoint,successor_training,response_basis,rehearsal_retention
from model import box_correction as correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,source = Path(config['root']),Path(config['source'])
    previous = authenticate(config)
    if a.stage == 'train':
        prepare(config,previous,root)
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'SUCCESSOR_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','successor-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = read((root if a.stage == 'cold' else source)/'snapshot-final.json')
    session,sessions = open_successor(config,previous,root,saved)
    replayed = time.perf_counter()
    print(canonical(dict(stage=a.stage,replayed_events=session.journal.count)).decode(),flush=True)
    controls = []
    def refuse(name,code,fn):
        before = digest(session.snapshot())
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'SUCCESSOR_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('SUCCESSOR_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == before,'SUCCESSOR_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))
    if a.stage == 'train':
        spec = config['spec']
        data = SuccessorData(session,session.transfer_data,spec)
        write_committed(root/'learning-spec.json',spec)
        write_committed(root/'training-binding.json',data.binding)
        shutil.copyfile(spec['baseline']['path'],root/'baseline.json')
        wrong = copy.deepcopy(spec)
        wrong['checkpoint']['pin'] = '0'*64
        refuse('wrong_parent','SUCCESSOR_PARENT_PIN',lambda:SuccessorData(session,session.transfer_data,wrong))
        wrong = copy.deepcopy(spec)
        wrong['contexts'] = list(reversed(wrong['contexts']))
        refuse('context_order','SUCCESSOR_CONTEXT_ROSTER',lambda:SuccessorData(session,session.transfer_data,wrong))
        wrong = copy.deepcopy(spec)
        wrong['baseline']['pin'] = '0'*64
        refuse('wrong_baseline','SUCCESSOR_BASELINE_PIN',lambda:SuccessorData(session,session.transfer_data,wrong))
        wrong = copy.deepcopy(spec)
        wrong['numerical']['pin'] = '0'*64
        refuse('wrong_numerical_pin','BOX_DATA_PIN',lambda:SuccessorData(session,session.transfer_data,wrong))
        def record(matrices,deficits,deltas,before,params,partition):
            path = root/'first-correction.npz'
            if path.exists():
                return
            arrays = {}
            for i,(lane,names) in enumerate(partition.items()):
                arrays[f'gradients_{i}'] = matrices[lane].cpu().numpy()
                arrays[f'deficit_{i}'] = deficits[lane].cpu().numpy()
                arrays[f'delta_{i}'] = (np.zeros(matrices[lane].shape[1]) if deltas[lane] is None
                                      else deltas[lane].cpu().numpy())
                for j,n in enumerate(names):
                    arrays[f'before_{i}_{j}'] = before[n].cpu().numpy()
                    arrays[f'after_{i}_{j}'] = params[n].detach().cpu().numpy()
            np.savez(path,**arrays)
        runs = []
        def factory(session,_base,spec):
            item = SuccessorData(session,session.transfer_data,spec)
            item.record_typed_correction = record
            runs.append(item)
            return item
        result = successor_training.increment(session,root/'candidates',name='successor-1',spec=spec,
            data_factory=factory,update=correction.step,review=lambda s,n,p,c:s.consider_successor(n,p,c),
            eligible=lambda d,m:d.response_basis.names)
        write_committed(root/'learning.json',result)
        write_committed(root/'cache-trace.json',runs[0].cache_trace)
        write_committed(root/'first-correction.json',dict(file=evidence(root/'first-correction.npz'),
            partition=result['losses'][0]['update']['partition'],certificates=result['losses'][0]['update']['certificates']))
        refuse('duplicate_candidate','SUCCESSOR_DUPLICATE',lambda:session.consider_successor(
            'successor-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    else:
        check(session.snapshot() == saved,'SUCCESSOR_COLD')
    final = session.snapshot()
    for item in sessions:
        item.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),
        journal=sha((root/'experience.sqlite3').read_bytes()),controls=controls,
        phases=dict(authentication_replay=replayed-started,action_and_close=finished-replayed),
        wall_seconds=finished-started,serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(canonical(report).decode(),flush=True)


if __name__ == '__main__':
    run_cli(main)
