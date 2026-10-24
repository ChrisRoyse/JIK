"""Durable, source-bound complete-native training steps for a derived codebook."""
import io
import os

import torch

from model.loop_training_checkpoint import tensor_fingerprint, validate_optimizer
from model.native_rq_data import matched_donors
from senses.contract import digest
from text.constellation_corpus import file_sha, need


def fingerprint(value):
    return digest(tensor_fingerprint(value))


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu(v) for v in value)
    return value


def branch_pins(values,roster):
    branches=['encoder.','books.','to_latent.']+['projectors.'+n+'.' for n in roster]+['decoders.'+n+'.' for n in roster]
    return {b:fingerprint({n:v for n,v in values.items() if n.startswith(b)}) for b in branches}


def save(path,model,optimizer,binding,step,rows,ledger):
    state=dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),binding=binding,
               step=step,rows=rows,ledger=ledger)
    buffer=io.BytesIO();torch.save(state,buffer)
    with path.open('xb') as f:f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    actual=torch.load(path,map_location='cpu',weights_only=True)
    need(fingerprint(actual)==fingerprint(state),'POPULATION_CHECKPOINT_READBACK')
    return file_sha(path)


def load(path,pin,model,optimizer,binding,step,rows,plan):
    need(file_sha(path)==pin,'POPULATION_CHECKPOINT_PIN')
    state=torch.load(path,map_location='cpu',weights_only=True)
    need(state['binding']==binding and state['step']==step and state['rows']==rows,
         'POPULATION_CHECKPOINT_BINDING')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    model.set_step(step);model.check_statistics();validate_optimizer(model,optimizer,plan,step)
    return state


def step(model,optimizer,frames,ids,cases,number,plan):
    need(type(number) is int and number>=0 and 0<len(ids)<=128 and len(frames)==len(cases)==len(ids),
         'POPULATION_STEP_SCOPE')
    device=model.encoder.query.device
    model.set_step(number);model.check_statistics();before=cpu(model.state_dict())
    donors,eligible=matched_donors(frames,ids,cases,model.roster);native_before=fingerprint(frames)
    native=tuple({n:x.to(device) for n,x in frame.items()} for frame in frames)
    optimizer.zero_grad(set_to_none=True);result=model.training_loss(native,donors.to(device),eligible.to(device))
    need(bool(torch.isfinite(result['loss'])),'POPULATION_NONFINITE_LOSS');result['loss'].backward()
    gradients=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()),'POPULATION_GRADIENT')
    optimizer.step();model.check_statistics();after=cpu(model.state_dict());moments=cpu(optimizer.state_dict())
    need(all(bool(torch.isfinite(x).all()) for x in after.values()),'POPULATION_PARAMETER')
    # CPU copies avoid one accelerator synchronization per moment/parameter check.
    for state in moments['state'].values():
        need(set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==number+1
             and all(bool(torch.isfinite(x).all()) for x in state.values()),'POPULATION_OPTIMIZER_STATE')
    need(len(moments['state'])==len(list(model.parameters())),'POPULATION_OPTIMIZER_COVERAGE')
    need(native_before==fingerprint(frames)==fingerprint(cpu(native)),'POPULATION_NATIVE_CHANGED')
    return dict(step=number+1,ids=ids,donors=donors.tolist(),eligible=eligible.tolist(),native=native_before,
        before=fingerprint(before),after=fingerprint(after),gradient=fingerprint(gradients),optimizer=fingerprint(moments),
        branch_before=branch_pins(before,model.roster),branch_after=branch_pins(after,model.roster),
        branch_gradient=branch_pins(gradients,model.roster),
        **{n:float(result[n].detach()) for n in ['loss','reconstruction','contrast','quantizer_loss']})
