"""Learn captured fresh consequences with retained prior responses and cold replay."""
import argparse
import copy
from dataclasses import replace
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
from model.fresh_learning import FreshData
from model.fresh_learning_session import FreshLearningLoop
from model.response_cycles import CycleData, CycleLoop
from model import bounded_training, field_checkpoint, response_basis, response_update
from senses.contract import SenseError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root,source = Path(config['root']),Path(config['source'])
    seal = authenticate(Path(config['proof']),config['proof_pin'])
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(evidence(source/name) == seal['files']['session/'+name],'FRESH_LEARNING_SOURCE_BYTES')
    echo_config = read(source/'config.json')
    upstream = read(Path(echo_config['source'])/'config.json')
    prospective = read(Path(upstream['prospective'])/'config.json')
    ancestor,parent_source = Path(prospective['source']),Path(upstream['source'])
    check(config['spec']['checkpoint'] == upstream['experiment']['corrected'],
          'FRESH_LEARNING_SOURCE_CHECKPOINT')
    before = read(source/'snapshot-final.json')
    if a.stage == 'train':
        check(not root.exists() and 'basis' not in config['spec'],'FRESH_LEARNING_ROOT_EXISTS')
        root.mkdir()
        for name,path in (('base',ancestor),('parent',parent_source)):
            (root/name).mkdir()
            (root/name/'candidates').mkdir()
            shutil.copyfile(path/'experience.sqlite3',root/name/'experience.sqlite3')
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        write_committed(root/'config-request.json',config)
    else:
        check(read(root/'config.json') == config,'FRESH_LEARNING_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','fresh-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original,_ = open_session(ancestor,prospective['source_pin'],root/'base',read(ancestor/'snapshot-final.json'))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    parent_session,_ = open_session(ancestor,prospective['source_pin'],root/'parent',
        read(parent_source/'snapshot-final.json'),loop_type=partial(CycleLoop,learning_base=base))
    parent_data = CycleData(parent_session,base,upstream['experiment']['learning'])
    saved = before if a.stage == 'train' else read(root/'snapshot-final.json')
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(FreshLearningLoop,learning_base=base,increment_base=parent_data))
    replayed = time.perf_counter()
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    controls = []

    def refuse(name,code,fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'FRESH_LEARNING_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('FRESH_LEARNING_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot,'FRESH_LEARNING_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    spec = config['spec']
    basis_seconds = 0.
    if a.stage == 'train':
        tick = time.perf_counter()
        parent,_,_ = field_checkpoint.load(Path(spec['checkpoint']['root']),parent_data,
                                           expected=spec['checkpoint']['pin'])
        prior_report = read(Path(spec['retained_report']['path']))
        check(parent_data.evaluate(parent) == prior_report['after'],'FRESH_LEARNING_PARENT_BEHAVIOUR')
        data = FreshData(session,base,spec)
        spec['basis'] = response_basis.build(root/'basis',data,parent,spec['checkpoint']['pin'])
        write_committed(root/'config.json',config)
        basis_seconds = time.perf_counter()-tick
        for field,code in (('checkpoint','PROPOSAL_LEARNING_PARENT_PIN'),
                           ('retained_report','PROPOSAL_LEARNING_RETAINED_BYTES')):
            bad = copy.deepcopy(spec)
            bad[field]['pin'] = '0'*64
            refuse('wrong_'+field,code,lambda bad=bad:FreshData(session,base,bad))
        bad = copy.deepcopy(spec)
        bad['contexts'] = list(reversed(bad['contexts']))
        refuse('swapped_context_order','PROPOSAL_LEARNING_CONTEXT_ROSTER',lambda:FreshData(session,base,bad))
        context = spec['contexts'][0]
        entry = session.echo_contexts[context]
        old_status = entry['status']
        try:
            entry['status'] = 'awaiting_correction'
            refuse('incomplete_echo','PROPOSAL_LEARNING_FEEDBACK_INCOMPLETE',lambda:FreshData(session,base,spec))
        finally:
            entry['status'] = old_status
        bad = copy.deepcopy(spec)
        bad['basis']['pin'] = '0'*64
        refuse('wrong_basis','RESPONSE_BASIS_PIN',lambda:FreshData(session,base,bad))
        data = FreshData(session,base,spec)
        key = next(iter(data.new))
        e = data.examples[key]
        old_state = e['state']
        try:
            e['state'] = replace(old_state,sequence=old_state.sequence+3)
            refuse('correction_as_preaction','PROPOSAL_LEARNING_PRESTATE',lambda:data.forward(parent,key))
        finally:
            e['state'] = old_state
        basis = data.response_basis
        arrays = {n:v.copy() for n,v in basis.arrays.items()}
        arrays['p_0'][0,0] += .1
        refuse('corrupt_projector','RESPONSE_BASIS_PROJECTOR',lambda:response_basis.validate(basis.manifest,arrays))
        _,parameter = next((n,p) for n,p in parent.named_parameters() if n not in basis.names)
        old = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.add_(.01)
            refuse('noneligible_weight_change','RESPONSE_NONELIGIBLE_DRIFT',lambda:basis.measure(data,parent))
        finally:
            with torch.no_grad():
                parameter.copy_(old)
        basis.measure(data,parent)
        refuse('wrong_checkpoint_pin','NATIVE_WORLD_CHECKPOINT_PIN',lambda:field_checkpoint.load(
            Path(spec['checkpoint']['root']),data,expected='0'*64))
        write_committed(root/'training-binding.json',data.binding)
        result = bounded_training.increment(session,root/'candidates',name='fresh-1',spec=spec,
            data_factory=FreshData,update=response_update.step,
            review=lambda s,n,p,r:s.consider_fresh_learning(n,p,r),
            eligible=lambda d,_model:d.response_basis.names)
        write_committed(root/'learning.json',result)
        refuse('duplicate_decision','CANDIDATE_REVIEW_DUPLICATE',lambda:session.consider_fresh_learning(
            'fresh-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check({k:v for k,v in final.items() if k not in ('head','events','fresh_learning_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head','events')},'FRESH_LEARNING_OTHER_STATE_CHANGED')
    session.close()
    parent_session.close()
    original.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),controls=controls,
        phases=dict(authentication_replay=replayed-started,basis=basis_seconds,
            learning_controls_close=finished-replayed-basis_seconds),wall_seconds=finished-started,
        journal=sha((root/'experience.sqlite3').read_bytes()),serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run_cli(main)
