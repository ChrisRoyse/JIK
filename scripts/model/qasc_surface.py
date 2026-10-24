"""Complete-byte decoding on fixed private-target and frozen predictor conditions."""
import math
import numpy as np
import torch
from model.qasc_context import LANES,check
from model.strict_surface import losses_encoded,render_complete

CONFIG=dict(width=256,layers=4,heads=8,steps=600,batch=32,seed=25,lr=.0003,weight_decay=.05,
            betas=[.9,.95],clip_gradient_norm=1.,capacity=512,gaussian_norm_sigma=.3,precision='bf16-fp32-loss')
ARMS=('true','gaussian','predicted')
ROUTES=('true','predicted','other-true','other-predicted')

def conditions(ctx,folder):
    target=torch.as_tensor(ctx['target_index'],device='cuda');truth=[];predicted=[]
    with np.load(folder/'codebook.npz',allow_pickle=False) as b, np.load(folder/'seed-25/region/outputs.npz',allow_pickle=False) as p:
        for k,name in enumerate(LANES):
            y=torch.as_tensor(ctx['truth'][k],device='cuda');mean=torch.as_tensor(b[name+'_mean'],device='cuda')
            truth.append(torch.nn.functional.normalize(y-mean,dim=1)[target]);predicted.append(torch.as_tensor(p[f'slots_{k}'],device='cuda'))
    gen=torch.Generator(device='cuda').manual_seed(25)
    gaussian=[torch.nn.functional.normalize(x+(.3/math.sqrt(768))*torch.randn(x.shape,device='cuda',generator=gen),dim=1) for x in truth]
    return dict(true=truth,predicted=predicted,gaussian=gaussian)

def plan(rows):
    train=np.array([i for i,r in enumerate(rows) if r['partition']=='train'],np.int64)
    heldout=np.array([i for i,r in enumerate(rows) if r['partition']!='train'],np.int64)
    selected={s:np.array([i for i,r in enumerate(rows) if r['partition']==s][:128],np.int64) for s in ('calibration','evaluation')}
    check(all(len(v)==128 for v in selected.values()),'SURFACE_SELECTION')
    return np.random.default_rng(25).choice(train,(600,32),replace=True),heldout,selected

def summarize(records):
    result={}
    for split in ('calibration','evaluation'):
        result[split]={}
        for route in ROUTES:
            rr=[r for r in records if r['partition']==split and r['route']==route]
            result[split][route]=dict(rows=len(rr),bytes=sum(r['bytes'] for r in rr),
                bits_per_byte=sum(r['byte_nll_bits'] for r in rr)/sum(r['bytes'] for r in rr),
                mean_eos_bits=sum(r['eos_nll_bits'] for r in rr)/len(rr))
    return result

@torch.inference_mode()
def evaluate(model,values,packed,lengths,indices,rows,donors):
    model.eval();records=[]
    for route in ROUTES:
        arrays=values[route.removeprefix('other-')];other=route.startswith('other-')
        for start in range(0,len(indices),32):
            ix=indices[start:start+32];source=donors[ix] if other else ix;slots=[x[source] for x in arrays];length=int(lengths[ix].max())
            with torch.autocast('cuda',dtype=torch.bfloat16):_,bb,eb,counts=losses_encoded(model,slots,packed[0][ix,:length],packed[1][ix,:length])
            for i,s,b,e,c in zip(ix,source,bb.tolist(),eb.tolist(),counts.tolist()):
                records.append(dict(row=int(i),conditioning_row=int(s),partition=rows[i]['partition'],route=route,byte_nll_bits=b,eos_nll_bits=e,bytes=c))
    return records

@torch.inference_mode()
def render(model,values,selected,rows,texts,arm,policy):
    model.eval();records=[]
    for split,indices in selected.items():
        for route in ('true','predicted'):
            for start in range(0,len(indices),32):
                ix=indices[start:start+32];slots=[x[ix] for x in values[route]]
                with torch.autocast('cuda',dtype=torch.bfloat16):out=render_complete(model,slots,capacity=512)
                for i,value in zip(ix,out):
                    records.append(dict(row=int(i),case=rows[i]['case'],group=rows[i]['group'],policy=policy,arm=arm,
                        partition=split,route=route,exact_provider_target=value['text']==texts[i],asserted=False,**value))
    return records
