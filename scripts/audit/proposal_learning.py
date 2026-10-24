"""Run a real proposal-feedback weight correction and replay its persisted decision."""
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
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.corrective_learning import CorrectiveData
from model.proposal_learning import ProposalData
from model.proposal_learning_session import ProposalLearningLoop
from model.proposal_learning_update import increment
from model import field_checkpoint
from senses.contract import SenseError


def authenticate(config):
    root = Path(config['proof'])
    raw = (root/'seal.json').read_bytes()
    check(sha(raw) == config['proof_pin'],'PROPOSAL_LEARNING_SOURCE_SEAL')
    seal = json.loads(raw)
    for name,item in {**{str(root/n):v for n,v in seal['files'].items()},**seal['external']}.items():
        path = Path(name)
        check(path.stat().st_size == item['bytes'] and sha(path.read_bytes()) == item['sha256'],
              'PROPOSAL_LEARNING_SOURCE_BYTES')
    source = Path(config['source'])
    for name in ('experience.sqlite3','snapshot-final.json','config.json'):
        check(sha((source/name).read_bytes()) == seal['files']['session/'+name]['sha256'],
              'PROPOSAL_LEARNING_PREFIX_BYTES')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    authenticate(config)
    root,source = Path(config['root']),Path(config['source'])
    prospective = json.loads((Path(config['prospective'])/'config.json').read_bytes())
    ancestor = Path(prospective['source'])
    before = json.loads((source/'snapshot-final.json').read_bytes())
    if a.stage == 'train':
        check(not root.exists(),'PROPOSAL_LEARNING_ROOT_EXISTS')
        root.mkdir()
        (root/'base').mkdir()
        (root/'candidates').mkdir()
        shutil.copyfile(source/'experience.sqlite3',root/'experience.sqlite3')
        shutil.copyfile(ancestor/'experience.sqlite3',root/'base/experience.sqlite3')
        write_committed(root/'config.json',config)
    else:
        check(json.loads((root/'config.json').read_bytes()) == config,'PROPOSAL_LEARNING_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28','proposal-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    original,_ = open_session(ancestor,prospective['source_pin'],root/'base',
                              json.loads((ancestor/'snapshot-final.json').read_bytes()))
    base = CorrectiveData(original,prospective['experiment']['correction'])
    saved = before if a.stage == 'train' else json.loads((root/'snapshot-final.json').read_bytes())
    session,_ = open_session(ancestor,prospective['source_pin'],root,saved,
        loop_type=partial(ProposalLearningLoop,learning_base=base))
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)
    replayed = time.perf_counter()
    controls = []

    def refuse(name,code,fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'PROPOSAL_LEARNING_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('PROPOSAL_LEARNING_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot,'PROPOSAL_LEARNING_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    spec = config['spec']
    if a.stage == 'train':
        bad = copy.deepcopy(spec)
        bad['checkpoint']['pin'] = '0'*64
        refuse('wrong_parent_pin','PROPOSAL_LEARNING_PARENT_PIN',lambda:ProposalData(session,base,bad))
        context = spec['contexts'][0]
        entry = session.echo_contexts[context]
        old = copy.deepcopy(entry)
        try:
            entry['status'] = 'awaiting_echo'
            refuse('incomplete_feedback','PROPOSAL_LEARNING_FEEDBACK_INCOMPLETE',lambda:ProposalData(session,base,spec))
        finally:
            entry.clear(); entry.update(old)
        data = ProposalData(session,base,spec)
        model,_,_ = field_checkpoint.start(data)
        key = next(iter(data.new))
        e = data.examples[key]
        state = e['state']
        try:
            e['state'] = session.echo_runtime[data.new[key]['context']]['state']
            refuse('correction_state_as_input','PROPOSAL_LEARNING_PRESTATE',lambda:data.forward(model,key))
        finally:
            e['state'] = state
        slot = next(iter(e['native'].values()))
        value = slot.values.clone()
        try:
            slot.values[0,0,0] += .001
            refuse('changed_sealed_input','PROPOSAL_LEARNING_INPUT',lambda:data.forward(model,key))
        finally:
            slot.values.copy_(value)
        target = next(iter(e['target'].values()))
        value = target.values.clone()
        try:
            target.values[0,0,0] += .001
            refuse('changed_target','PROPOSAL_LEARNING_TARGET_CHANGED',lambda:data.forward(model,key))
        finally:
            target.values.copy_(value)
        write_committed(root/'training-binding.json',data.binding)
        result = increment(session,root/'candidates',name='proposal-1',spec=spec)
        write_committed(root/'learning.json',result)
        refuse('duplicate_decision','PROPOSAL_LEARNING_DUPLICATE_DECISION',
               lambda:session.consider_proposal_learning('proposal-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    check({k:v for k,v in final.items() if k not in ('head','events','proposal_learning_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head','events')},'PROPOSAL_LEARNING_OTHER_STATE_CHANGED')
    session.close()
    original.close()
    done = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),
        controls=controls,phases=dict(authentication_replay=replayed-started,learning_and_close=done-replayed),
        wall_seconds=done-started,journal=sha((root/'experience.sqlite3').read_bytes()),
        serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run_cli(main)
