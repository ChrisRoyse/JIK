"""Whole-byte embedding masking for an explicit TRAIN-only decoder experiment."""
import hashlib
import struct
import time

import torch

from model import mixed_surface_batches as mixed
from model.native_population_training import cpu,fingerprint
from model.surface_device_runtime import batches,sync
from model import surface_inversion_training as inversion
from senses.contract import canonical
from text.constellation_corpus import need

SEED='decoder-prefix-v1-20261025'


def eligible(batch):
    lengths=torch.tensor([r['prefix_length'] for r in batch.records],device=batch.prefix.device)
    return (torch.arange(batch.prefix.shape[1],device=batch.prefix.device)[None]<lengths[:,None]) & (batch.prefix<256)


def mask_for(batch,step,rate):
    need(type(step) is int and step>=0 and type(rate) is float and rate in (0.,.4),'PREFIX_MASK_POLICY')
    result=torch.zeros(batch.prefix.shape,dtype=torch.bool)
    threshold=int(rate*(1<<64))
    for row,r in enumerate(batch.records):
        raw=canonical(dict(seed=SEED,step=step,occurrence=r['occurrence'],offset=r['offset']))
        # One small address-bound hash stream per row. Global RNG is untouched.
        values=struct.unpack('<'+'Q'*batch.prefix.shape[1],hashlib.shake_256(raw).digest(8*batch.prefix.shape[1]))
        result[row]=torch.tensor([v<threshold for v in values],dtype=torch.bool)
    return result.to(batch.prefix.device) & eligible(batch)


def forward(model,batch,mask):
    need(model.training and torch.is_grad_enabled(),'PREFIX_MASK_TRAIN_ONLY')
    need(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and mask.shape==batch.prefix.shape
         and mask.device==batch.prefix.device and not mask.requires_grad,'PREFIX_MASK_LAYOUT')
    need(not bool((mask & ~eligible(batch)).any()),'PREFIX_MASK_NONBYTE_OR_PADDING')
    need(not model.tokens._forward_hooks,'PREFIX_MASK_EXISTING_HOOK')
    calls=[]
    def apply(module,inputs,output):
        need(len(inputs)==1 and inputs[0] is batch.prefix and output.shape[:2]==mask.shape,'PREFIX_MASK_CALL')
        calls.append(1)
        return output.masked_fill(mask[:,:,None],0.)
    # The zero-mask arm executes the original forward arithmetic.
    handle=model.tokens.register_forward_hook(apply) if bool(mask.any()) else None
    try:result=model(batch.prefix,batch.native,offsets=batch.offsets)
    finally:
        if handle is not None:handle.remove()
    need(len(calls)==(1 if handle is not None else 0) and not model.tokens._forward_hooks,'PREFIX_MASK_LIFETIME')
    return result


def evaluate(model,examples,*,step,rate):
    need(examples and all(e.side=='train' for e in examples),'PREFIX_MASK_NONTRAIN')
    device=next(model.parameters()).device; began=time.perf_counter(); positions=0;objective=0.;traces=[]
    scores=[dict(losses=torch.full((len(e.raw)+1,),float('nan')),
                 argmax=torch.full((len(e.raw)+1,),-1,dtype=torch.int64)) for e in examples]
    for host in batches(examples,model.config,model.specs,'gathered',128):
        b=host.to(device);mask=mask_for(b,step,rate)
        logits=forward(model,b,mask)['logits'];loss,token=mixed.loss(logits,b);loss.backward();sync(device)
        token=token.detach().cpu();predicted=logits.detach().argmax(-1).cpu()
        traces.append(dict(records=b.records,mask=mask.cpu(),eligible=eligible(b).cpu()))
        objective+=float(loss.detach())
        for row,r in enumerate(b.records):
            i=r['atom'];a,z=r['scored'];left=a-r['offset'];right=z-r['offset']
            need(bool(torch.isnan(scores[i]['losses'][a:z]).all()),'PREFIX_MASK_DUPLICATE_POSITION')
            scores[i]['losses'][a:z]=token[row,left:right];scores[i]['argmax'][a:z]=predicted[row,left:right];positions+=z-a
    need(positions==sum(len(e.raw)+1 for e in examples)
         and all(bool(v['losses'].isfinite().all()) and bool((v['argmax']>=0).all()) for v in scores),'PREFIX_MASK_COVERAGE')
    return scores,dict(objective=objective,positions=positions,mask_pin=fingerprint(traces),
        masked=sum(int(t['mask'].sum()) for t in traces),eligible=sum(int(t['eligible'].sum()) for t in traces),
        seconds=time.perf_counter()-began),traces


def update(model,optimizer,examples,step,*,rate):
    need(0<len(examples)<=128 and all(e.side=='train' for e in examples),'PREFIX_MASK_NONTRAIN')
    before,_=inversion.check_state(model,optimizer,step)
    native=fingerprint([{n:x.values for n,x in e.native.items()} for e in examples])
    model.train();optimizer.zero_grad(set_to_none=True)
    scores,profile,traces=evaluate(model,examples,step=step,rate=rate)
    gradients=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()),'PREFIX_MASK_GRADIENT')
    branches={}
    for s in model.specs:
        for prefix in ['projectors.'+s.name+'.','blocks.0.branches.'+s.name+'.']:
            values={n:v for n,v in gradients.items() if n.startswith(prefix)}
            need(values and any(bool(v.abs().any()) for v in values.values()),'PREFIX_MASK_DISCONNECTED')
            branches[prefix]=fingerprint(values)
    optimizer.step();after,moments=inversion.check_state(model,optimizer,step+1)
    need(native==fingerprint([{n:x.values for n,x in e.native.items()} for e in examples]),'PREFIX_MASK_NATIVE_CHANGED')
    return dict(step=step+1,before=fingerprint(before),after=fingerprint(after),optimizer=fingerprint(moments),
        gradient=fingerprint(gradients),branch_gradients=branches,scores=fingerprint(scores),native=native,
        positions=profile['positions'],objective=profile['objective']),profile,dict(scores=scores,traces=traces)
