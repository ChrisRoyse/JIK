"""Real proposal correction, frozen-response basis, bounded learning and cold replay."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import authenticate, evidence, read
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.slack_learning import SlackData
from model.response_learning import ResponseData, ResponseLearningLoop, increment
from model import field_checkpoint, response_basis
from senses.contract import SenseError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    proof = Path(config['proof'])
    seal = authenticate(proof,config['proof_pin'])
    source = Path(config['source'])
    previous = read(proof/'session/config.json')
    check(config['spec']['previous'] == previous['spec'], 'RESPONSE_PREVIOUS_SPEC')
    echo = Path(config['echo_proof'])
    check(sha((echo/'seal.json').read_bytes()) == config['echo_pin'], 'RESPONSE_ECHO_PIN')
    echo_seal = read(echo/'seal.json')
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(evidence(source/name) == echo_seal['files']['session/'+name], 'RESPONSE_PREFIX_BYTES')
    evidence_path = Path(config['spec']['evidence']['path'])
    check(evidence(evidence_path) == seal['files']['session/learning.json'], 'RESPONSE_LEARNING_SOURCE')
    root = Path(config['root'])
    prospective = read(Path(config['prospective'])/'config.json')
    ancestor = Path(prospective['source'])
    before = read(source/'snapshot-final.json')
    if a.stage == 'train':
        check(not root.exists() and 'basis' not in config['spec'], 'RESPONSE_ROOT_EXISTS')
        root.mkdir()
        (root/'base').mkdir()
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        shutil.copyfile(ancestor/'experience.sqlite3',root/'base/experience.sqlite3')
        write_committed(root/'config-request.json',config)
    else:
        check(read(root/'config.json') == config, 'RESPONSE_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','response-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original, _ = open_session(ancestor,prospective['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    saved = before if a.stage == 'train' else read(root/'snapshot-final.json')
    session, _ = open_session(ancestor,prospective['source_pin'],root,saved,
                             loop_type=partial(ResponseLearningLoop,learning_base=base))
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls = []

    def refuse(name, code, fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error), 'RESPONSE_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('RESPONSE_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot, 'RESPONSE_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    spec = config['spec']
    basis_seconds = 0.
    if a.stage == 'train':
        tick = time.perf_counter()
        old_data = SlackData(session,base,spec['previous'])
        parent, _, _ = field_checkpoint.load(Path(spec['continuation']['root']),old_data,
                                              expected=spec['continuation']['pin'])
        check(old_data.evaluate(parent) == read(evidence_path)['after'], 'RESPONSE_PARENT_BEHAVIOUR')
        spec['basis'] = response_basis.build(root/'basis',old_data,parent,spec['continuation']['pin'])
        write_committed(root/'config.json',config)
        basis_seconds = time.perf_counter()-tick
        bad = copy.deepcopy(spec)
        bad['continuation']['pin'] = '0'*64
        refuse('wrong_parent','RESPONSE_PARENT_PIN',lambda:ResponseData(session,base,bad))
        data = ResponseData(session,base,spec)
        basis = data.response_basis
        arrays = {n:v.copy() for n,v in basis.arrays.items()}
        arrays['h_0'] = arrays['h_0'][1:]
        refuse('missing_activation_row','RESPONSE_BASIS_ROWS',lambda:response_basis.validate(basis.manifest,arrays))
        arrays = {n:v.copy() for n,v in basis.arrays.items()}
        arrays['p_0'][0,0] += .1
        refuse('corrupt_projector','RESPONSE_BASIS_PROJECTOR',lambda:response_basis.validate(basis.manifest,arrays))
        name, parameter = next((n,p) for n,p in parent.named_parameters() if n not in basis.names)
        old = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.add_(.01)
            refuse('noneligible_weight_change','RESPONSE_NONELIGIBLE_DRIFT',lambda:basis.measure(data,parent))
        finally:
            with torch.no_grad():
                parameter.copy_(old)
        data.fresh()
        basis.measure(data,parent)
        refuse('wrong_checkpoint_pin','NATIVE_WORLD_CHECKPOINT_PIN',lambda:field_checkpoint.load(
            Path(spec['continuation']['root']),data,expected='0'*64))
        write_committed(root/'training-binding.json',data.binding)
        result = increment(session,root/'candidates',name='response-1',spec=spec)
        write_committed(root/'learning.json',result)
        refuse('duplicate_decision','RESPONSE_DUPLICATE_DECISION',lambda:session.consider_response_learning(
            'response-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check({k:v for k,v in final.items() if k not in ('head','events','response_learning_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head','events')}, 'RESPONSE_OTHER_STATE_CHANGED')
    session.close()
    original.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),
        controls=controls,phases=dict(authentication_replay=replayed-started,basis=basis_seconds,
            learning_controls_close=finished-replayed-basis_seconds),wall_seconds=finished-started,
        journal=sha((root/'experience.sqlite3').read_bytes()),serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run_cli(main)
