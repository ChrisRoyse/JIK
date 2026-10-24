"""Execute bounded typed correction, exact optimizer recovery, rollback and cold replay."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fresh_learning_open import prepare, open_fresh
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read, evidence
from claims.codata_source import check, sha
from claims.episodic_memory import digest, canonical
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.typed_correction_session import TypedData, TypedLoop
from model import typed_correction, stable_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    check('basis' in config['spec'],'TYPED_RUN_BASIS_REQUIRED')
    root = Path(config['root'])
    resolved = prepare(config,root) if a.stage == 'train' else None
    if a.stage == 'train':
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'TYPED_RUN_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','typed-correction-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = read(root/'snapshot-final.json') if a.stage == 'cold' else None
    session,base,sessions = open_fresh(config,root,saved=saved,loop_type=TypedLoop,sources_value=resolved)
    replayed = time.perf_counter()
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error),'TYPED_RUN_CONTROL_CODE:'+str(error))
        else:
            raise AssertionError('TYPED_RUN_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    if a.stage == 'train':
        data = TypedData(session,base,config['spec'])
        write_committed(root/'training-binding.json',data.binding)
        wrong = copy.deepcopy(config['spec'])
        wrong['typed']['pin'] = '0'*64
        refuse('wrong_source','TYPED_CORRECTION_SOURCE_PIN',lambda:TypedData(session,base,wrong))
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
        def factory(session,base,spec):
            item = TypedData(session,base,spec)
            item.record_typed_correction = record
            return item
        result = stable_training.increment(session,root/'candidates',name='typed-1',spec=config['spec'],
            data_factory=factory,update=typed_correction.step,review=lambda s,n,p,c:s.consider_typed(n,p,c),
            eligible=lambda d,m:d.response_basis.names)
        write_committed(root/'learning.json',result)
        write_committed(root/'first-correction.json',dict(file=evidence(root/'first-correction.npz'),
            partition=result['losses'][0]['update']['partition'],
            certificates=result['losses'][0]['update']['certificates']))
        snapshot = session.snapshot()
        refuse('duplicate_candidate','TYPED_CORRECTION_DUPLICATE',lambda:session.consider_typed(
            'typed-1',result['candidate_pin'],config['spec']))
        check(session.snapshot() == snapshot,'TYPED_RUN_CONTROL_STATE')
        write_committed(root/'snapshot-final.json',snapshot)
    else:
        snapshot = session.snapshot()
        check(snapshot == saved,'TYPED_RUN_COLD')
    for item in sessions:
        item.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=snapshot['events'],head=snapshot['head'],snapshot=digest(snapshot),
        journal=sha((root/'experience.sqlite3').read_bytes()),controls=controls,
        phases=dict(authentication_replay=replayed-started,learning_controls_close=finished-replayed),
        wall_seconds=finished-started,serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(canonical(report).decode(),flush=True)


if __name__ == '__main__':
    run_cli(main)
