"""Execute, recover and cold-replay the bounded separated-response learning path."""
import argparse
import copy
import io
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.fresh_learning_open import prepare, open_fresh
from audit.fsv_identity_loop import configure
from audit.proposal_recognition import read
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from lens.lenscommon import run_cli, take_lease
from model.stable_learning import StableData, StableLoop
from model import field_checkpoint, response_basis, response_update, stable_checkpoint, stable_training
from senses.contract import SenseError


def damaged_checkpoint(source, target, change):
    target.mkdir()
    manifest = read(source/'manifest.json')
    weights = torch.load(source/'weights.pt',map_location='cpu',weights_only=True)
    change(weights)
    stream = io.BytesIO()
    torch.save(weights,stream)
    raw = stream.getvalue()
    (target/'weights.pt').write_bytes(raw)
    (target/'training.pt').write_bytes((source/'training.pt').read_bytes())
    manifest['files']['weights.pt'] = dict(bytes=len(raw),sha256=sha(raw))
    raw = canonical(manifest)
    (target/'manifest.json').write_bytes(raw)
    pin = sha(raw)
    (target/'COMMITTED').write_text(pin)
    return pin


def validate_config(config):
    check('basis' in config['spec'],
          'STABLE_RUN_BASIS_REQUIRED: use the completed source run config with its pinned response basis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('train','cold'),required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    validate_config(config)
    root = Path(config['root'])
    resolved = prepare(config,root) if a.stage == 'train' else None
    if a.stage == 'train':
        write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json') == config,'STABLE_RUN_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','stable-learning-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    saved = None if a.stage == 'train' else read(root/'snapshot-final.json')
    session,base,sessions = open_fresh(config,root,saved=saved,loop_type=StableLoop,sources_value=resolved)
    replayed = time.perf_counter()
    print(canonical(dict(stage=a.stage,replayed_events=session.journal.count)).decode(),flush=True)
    controls = []

    def refuse(name,code,fn):
        snapshot = digest(session.snapshot())
        try:
            fn()
        except (ValueError,SenseError) as error:
            check(code in str(error),'STABLE_RUN_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('STABLE_RUN_CONTROL_ACCEPTED:'+name)
        check(digest(session.snapshot()) == snapshot,'STABLE_RUN_CONTROL_MUTATION')
        controls.append(dict(name=name,code=code))

    spec = config['spec']
    if a.stage == 'train':
        missing_basis = copy.deepcopy(config)
        missing_basis['spec'].pop('basis')
        refuse('missing_launch_basis','STABLE_RUN_BASIS_REQUIRED',lambda:validate_config(missing_basis))
        for field,pin_name,code in (('stable','pin','STABLE_LEARNING_DIAGNOSTIC_PIN'),
                                   ('stable','readback_pin','STABLE_LEARNING_NUMERIC_PIN'),
                                   ('checkpoint','pin','PROPOSAL_LEARNING_PARENT_PIN'),
                                   ('basis','pin','RESPONSE_BASIS_PIN')):
            bad = copy.deepcopy(spec)
            bad[field][pin_name] = '0'*64
            refuse('wrong_'+field+'_'+pin_name,code,lambda bad=bad:StableData(session,base,bad))
        data = StableData(session,base,spec)
        parent,optimizer,_ = stable_checkpoint.start(data)
        plain,_,_ = field_checkpoint.start(data)
        check(data.evaluate(parent) == data.evaluate(plain),'STABLE_RUN_INITIAL_BEHAVIOUR')
        for field in ('anchor','projector'):
            module = next(iter(response_basis.modules(parent).values()))
            value = getattr(module,field)
            old = value.clone()
            with torch.no_grad():
                value.add_(.01)
            refuse('changed_'+field,'STABLE_RESPONSE_FROZEN_BUFFER',lambda:stable_checkpoint.validate(parent,optimizer,data,0))
            with torch.no_grad():
                value.copy_(old)
            module.seal()
            stable_checkpoint.validate(parent,optimizer,data,0)
        _,parameter = next((n,p) for n,p in parent.named_parameters() if n not in data.response_basis.names)
        old = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(.01)
        refuse('noneligible_parameter','RESPONSE_NONELIGIBLE_DRIFT',lambda:data.response_basis.measure(data,parent))
        with torch.no_grad():
            parameter.copy_(old)
        control_root = root/'checkpoint-controls'
        control_root.mkdir()
        intact = control_root/'intact'
        intact_pin = stable_checkpoint.save(intact,parent,optimizer,data,step=0,parent=data.parent_pin)
        refuse('legacy_loader','NATIVE_WORLD_CHECKPOINT_CONTRACT',
               lambda:field_checkpoint.load(intact,data,expected=intact_pin))
        anchor = next(n for n in parent.state_dict() if n.endswith('.anchor'))
        missing = control_root/'missing-anchor'
        missing_pin = damaged_checkpoint(intact,missing,lambda weights:weights.pop(anchor))
        refuse('missing_anchor','STABLE_CHECKPOINT_STATE',
               lambda:stable_checkpoint.load(missing,data,expected=missing_pin))
        changed = control_root/'changed-anchor'
        changed_pin = damaged_checkpoint(intact,changed,lambda weights:weights[anchor].add_(.01))
        refuse('persisted_changed_anchor','STABLE_CHECKPOINT_BUFFERS',
               lambda:stable_checkpoint.load(changed,data,expected=changed_pin))
        corrupt = control_root/'corrupt-bytes'
        shutil.copytree(intact,corrupt)
        with (corrupt/'weights.pt').open('r+b') as handle:
            handle.seek(256)
            value = handle.read(1)
            handle.seek(256)
            handle.write(bytes([value[0]^1]))
        refuse('corrupt_checkpoint','STABLE_CHECKPOINT_BYTES',
               lambda:stable_checkpoint.load(corrupt,data,expected=intact_pin))
        write_committed(root/'training-binding.json',data.binding)
        result = stable_training.increment(session,root/'candidates',name='stable-1',spec=spec,
            data_factory=StableData,update=response_update.step,
            review=lambda s,n,p,r:s.consider_stable(n,p,r),eligible=lambda d,_m:d.response_basis.names)
        write_committed(root/'learning.json',result)
        refuse('duplicate_decision','STABLE_LEARNING_DUPLICATE',lambda:session.consider_stable('stable-1',result['candidate_pin'],spec))
        write_committed(root/'snapshot-final.json',session.snapshot())
    final = session.snapshot()
    before = read(Path(config['source'])/'snapshot-final.json')
    check({k:v for k,v in final.items() if k not in ('head','events','stable_response_candidates')} ==
          {k:v for k,v in before.items() if k not in ('head','events')},'STABLE_RUN_STATE_CHANGED')
    for s in sessions:
        s.close()
    finished = time.perf_counter()
    report = dict(stage=a.stage,events=final['events'],head=final['head'],snapshot=digest(final),controls=controls,
        phases=dict(authentication_replay=replayed-started,learning_controls_close=finished-replayed),
        wall_seconds=finished-started,journal=sha((root/'experience.sqlite3').read_bytes()),
        serving=False,scientific_admission=False)
    write_committed(root/(a.stage+'-report.json'),report)
    print(canonical(report).decode(),flush=True)


if __name__ == '__main__':
    run_cli(main)
