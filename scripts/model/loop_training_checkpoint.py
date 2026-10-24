"""Recover the exact intrinsic-loop AdamW step, bound to checked experience."""
import io
import json
import os
from pathlib import Path

import torch

from model import loop_checkpoint
from model.loop_learning import check
from production.publication import sync_dir
from senses.contract import canonical,digest


def sha(raw):return loop_checkpoint.sha(raw)


def optimizer_for(model,plan):
    return torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'],foreach=False,fused=False)


def code_pins():
    root=Path(__file__).parent
    return {n:sha((root/n).read_bytes()) for n in ('loop_training_checkpoint.py','loop_learning.py','loop_session.py')}


def tensor_fingerprint(value):
    if isinstance(value,torch.Tensor):
        a=value.detach().cpu().contiguous().numpy()
        return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha(a.tobytes()))
    if isinstance(value,dict):return {str(k):tensor_fingerprint(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_fingerprint(x) for x in value]
    return value


def fingerprint(model,optimizer):
    return dict(model=digest(tensor_fingerprint(model.state_dict())),optimizer=digest(tensor_fingerprint(optimizer.state_dict())))


def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True


def validate_optimizer(model,optimizer,plan,step):
    check(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups)==1,'OPTIMIZER_RECIPE')
    group=optimizer.param_groups[0];params=list(model.parameters())
    check(len(group['params'])==len(params) and all(a is b for a,b in zip(group['params'],params)),'OPTIMIZER_ORDER')
    expected=dict(lr=plan['learning_rate'],weight_decay=plan['weight_decay'],betas=(.9,.999),eps=1e-8,
                  amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False)
    check(all(group[k]==v for k,v in expected.items()),'OPTIMIZER_RECIPE')
    for p,state in optimizer.state.items():
        check(set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==step,'OPTIMIZER_CURSOR')
        check(all(state[k].shape==p.shape and state[k].dtype==p.dtype and state[k].device==p.device
                  for k in ('exp_avg','exp_avg_sq')) and finite(state),'OPTIMIZER_STATE')
    check((step==0)==(not optimizer.state),'OPTIMIZER_CURSOR')


def put(path,raw):
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def save(root,model,optimizer,data,*,step,parent=None):
    root=Path(root);data.fresh(disk=True);model.check_reference()
    check(not root.exists(),'TRAINING_SNAPSHOT_EXISTS')
    check(type(step) is int and 0<=step<=128 and type(optimizer) is torch.optim.AdamW,'TRAINING_RECIPE')
    validate_optimizer(model,optimizer,data.plan,step)
    check(finite(optimizer.state_dict()),'NONFINITE_OPTIMIZER')
    root.mkdir(parents=True)
    states=tuple(model.start(c) for c in sorted(data.session.states))
    model_pin=loop_checkpoint.save(model,states,root/'model')
    payload=dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.initial.device))
    buf=io.BytesIO();torch.save(payload,buf);raw=buf.getvalue()
    model_bytes=sum(p.stat().st_size for p in (root/'model').iterdir() if p.is_file())
    check(len(raw)+model_bytes<=model.config.checkpoint_bytes,'TRAINING_SNAPSHOT_BUDGET')
    manifest=dict(schema='intrinsic-loop-learning-v1',model=model_pin,standard=model.standard_id,source_id=model.source_id,
                  binding=data.binding,memory_head=data.head,memory_contract=data.contract,plan=data.plan,step=step,parent=parent,
                  names=[n for n,_ in model.named_parameters()],optimizer='AdamW',optimizer_bytes=len(raw),optimizer_sha256=sha(raw),
                  code_sha256=code_pins(),fingerprint=fingerprint(model,optimizer),torch_version=str(torch.__version__),
                  production_admission=False,behavioural_compliance_established=False,world_truth_established=False)
    put(root/'training.pt',raw);m=canonical(manifest);put(root/'manifest.json',m);put(root/'COMMITTED',sha(m).encode())
    sync_dir(root);sync_dir(root.parent)
    return sha(m)


def load(root,data,*,expected,device='cuda'):
    root=Path(root);data.fresh(disk=True)
    check((root/'COMMITTED').is_file() and (root/'manifest.json').is_file(),'TRAINING_SNAPSHOT_INCOMPLETE')
    check((root/'manifest.json').stat().st_size<=1<<20,'TRAINING_SNAPSHOT_BUDGET')
    raw=(root/'manifest.json').read_bytes();m=json.loads(raw)
    check(sha(raw)==expected==(root/'COMMITTED').read_text() and canonical(m)==raw,'TRAINING_SNAPSHOT_PIN')
    check(m['schema']=='intrinsic-loop-learning-v1' and m['binding']==data.binding and m['plan']==data.plan
          and m['memory_head']==data.head and m['memory_contract']==data.contract
          and m['standard']==data.model.standard_id and m['source_id']==data.model.source_id
          and m['code_sha256']==code_pins() and m['torch_version']==str(torch.__version__),'TRAINING_SNAPSHOT_CONTRACT')
    check(m['production_admission'] is False and m['behavioural_compliance_established'] is False and m['world_truth_established'] is False,'TRAINING_SNAPSHOT_SCOPE')
    model,_,_=loop_checkpoint.load(root/'model',expected_checkpoint=m['model'],expected_standard=m['standard'],device=device)
    check([n for n,_ in model.named_parameters()]==m['names'] and m['optimizer']=='AdamW','OPTIMIZER_ORDER')
    model_bytes=sum(p.stat().st_size for p in (root/'model').iterdir() if p.is_file())
    check(m['optimizer_bytes']+model_bytes<=model.config.checkpoint_bytes
          and (root/'training.pt').stat().st_size==m['optimizer_bytes'],'TRAINING_SNAPSHOT_BUDGET')
    raw=(root/'training.pt').read_bytes();check(sha(raw)==m['optimizer_sha256'],'OPTIMIZER_PIN')
    p=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    check(set(p)=={'optimizer','cpu_rng','cuda_rng'} and finite(p['optimizer']),'OPTIMIZER_SCHEMA')
    opt=optimizer_for(model,m['plan']);opt.load_state_dict(p['optimizer'])
    validate_optimizer(model,opt,m['plan'],m['step'])
    check(fingerprint(model,opt)==m['fingerprint'],'OPTIMIZER_READBACK')
    check(type(m['step']) is int and 0<=m['step']<=128,'TRAINING_CURSOR')
    torch.set_rng_state(p['cpu_rng']);torch.cuda.set_rng_state(p['cuda_rng'],model.initial.device)
    return model,opt,m
