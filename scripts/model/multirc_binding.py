"""Typed native question/answer binding to complete source-window sets.

Only learned latent tokens interact across instruments. Missing observations use
explicit learned absence tokens, and labels/identifiers are outside forward.
"""
import numpy as np
import torch
from torch import nn
from model.multirc_context import check

CONFIG=dict(width=96,heads=4,steps=2000,batch=64,seeds=[25,26,27],learning_rate=.0005,
            weight_decay=.01,max_gradient_norm=1.,precision='fp32',dropout=0.,bootstrap_draws=999)
ARMS=('joint','no-context','no-question','no-answer','sentences')
FIELDS={'question','answer','context','mask','levels','geometry'}

class Store:
    def __init__(self,ctx,device='cuda'):
        self.device=device
        self.native=[torch.as_tensor(v,device=device) for v in ctx['native']]
        self.means=[torch.as_tensor(v,device=device) for v in ctx['means']]
        self.maps=[{n:torch.as_tensor(v,device=device) for n,v in m.items()} for m in ctx['maps']]
        self.docs=torch.tensor([r['document'] for r in ctx['rows']],device=device)
        self.available=np.array([r['available'] for r in ctx['rows']],bool)
    def batch(self,ids,intervention=None,donors=None):
        check(self.available[ids].all(),'BATCH_UNAVAILABLE')
        ii=torch.as_tensor(ids,device=self.device);dd=self.docs[ii]
        qq=ii
        if intervention in ('question-swap','passage-swap'):
            check(donors is not None and self.available[donors].all(),'DONOR_UNAVAILABLE')
            alt=torch.as_tensor(donors,device=self.device)
            if intervention=='question-swap':qq=alt
            else:dd=self.docs[alt]
        elif intervention not in (None,'remove-context'):raise ValueError('MNLM_MULTIRC_BINDING_INTERVENTION')
        result=[]
        for values,mean,m in zip(self.native,self.means,self.maps):
            result.append(dict(question=values[m['question'][qq]]-mean,answer=values[m['answer'][ii]]-mean,
                               context=values[m['indices'][dd]]-mean,mask=m['mask'][dd],
                               levels=m['levels'][dd],geometry=m['geometry'][dd]))
        return result

class Binding(nn.Module):
    def __init__(self):
        super().__init__();w=CONFIG['width']
        self.adapters=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,w)) for _ in range(6)])
        self.pair=nn.ModuleList([nn.MultiheadAttention(w,4,dropout=0.,batch_first=True) for _ in range(6)])
        self.read=nn.ModuleList([nn.MultiheadAttention(w,4,dropout=0.,batch_first=True) for _ in range(6)])
        self.relate=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,4*w),nn.GELU(),nn.Linear(4*w,w)) for _ in range(6)])
        self.geometry=nn.ModuleList([nn.Linear(3,w,bias=False) for _ in range(6)])
        self.level=nn.ModuleList([nn.Embedding(4,w) for _ in range(6)])
        self.role=nn.Parameter(torch.randn(6,3,w)*.02)
        self.absent=nn.Parameter(torch.randn(6,3,w)*.02)
        self.sense=nn.Parameter(torch.randn(6,w)*.02)
        self.task=nn.Parameter(torch.randn(1,1,w)*.02)
        self.binding=nn.MultiheadAttention(w,4,dropout=0.,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2))
    def forward(self,inputs,arm):
        check(arm in ARMS and type(inputs)==list and len(inputs)==6,'FORWARD_SCOPE')
        check(all(type(m)==dict and set(m)==FIELDS for m in inputs),'FORWARD_FIELDS')
        b=len(inputs[0]['question']);tokens=[]
        for k,m in enumerate(inputs):
            n=m['context'].shape[1]
            check(m['question'].shape==m['answer'].shape==(b,768) and m['context'].shape==(b,n,768)
                  and m['mask'].shape==m['levels'].shape==(b,n) and m['geometry'].shape==(b,n,3),'FORWARD_SHAPE')
            q=self.absent[k,0].expand(b,-1) if arm=='no-question' else self.adapters[k](m['question'])
            a=self.absent[k,1].expand(b,-1) if arm=='no-answer' else self.adapters[k](m['answer'])
            pair=torch.stack((q+self.role[k,0],a+self.role[k,1]),1)
            h=pair+self.pair[k](pair,pair,pair,need_weights=False)[0]
            # All retained context windows remain separate attention positions.
            if arm=='no-context':
                context=self.absent[k,2].expand(b,1,-1)+self.role[k,2];mask=None
            else:
                context=self.adapters[k](m['context'])+self.geometry[k](m['geometry'])+self.level[k](m['levels'])+self.role[k,2]
                mask=~(m['mask'] & (m['levels']==1)) if arm=='sentences' else ~m['mask']
            h=h+self.read[k](h,context,context,key_padding_mask=mask,need_weights=False)[0]
            tokens.append(h+self.relate[k](h)+self.sense[k])
        bound=torch.cat(tokens,1);query=self.task.expand(b,-1,-1)
        return self.head(query+self.binding(query,bound,bound,need_weights=False)[0])[:,0]

def schedule(train,seed):
    rng=np.random.default_rng(seed)
    return np.asarray([rng.choice(train,CONFIG['batch'],replace=False) for _ in range(CONFIG['steps'])])

def donors(rows):
    out=np.full(len(rows),-1,np.int64)
    for partition in ('train','calibration','evaluation'):
        groups={}
        for i,r in enumerate(rows):
            if r['available'] and r['partition']==partition:groups.setdefault(r['group'],[]).append(i)
        keys=sorted(groups);check(len(keys)>1,'DONOR_GROUPS')
        for j,key in enumerate(keys):out[groups[key]]=groups[keys[(j+1)%len(keys)]][0]
    return out

@torch.inference_mode()
def infer(model,store,arm,ids,intervention=None,donor=None):
    result=[]
    for start in range(0,len(ids),CONFIG['batch']):
        ii=ids[start:start+CONFIG['batch']]
        batch=store.batch(ii,intervention,None if donor is None else donor[ii])
        value=model(batch,'no-context' if intervention=='remove-context' else arm)
        result.append(value.cpu().numpy())
    return np.concatenate(result)

@torch.inference_mode()
def controls(model,store,ids):
    inputs=store.batch(ids[:64]);out=model(inputs,'joint')
    bad=[{**m,'isAnswer':0} for m in inputs]
    try:model(bad,'joint')
    except ValueError as exc:check('FORWARD_FIELDS' in str(exc),'PRIVATE_ERROR')
    else:raise ValueError('MNLM_MULTIRC_BINDING_PRIVATE_ACCEPTED')
    for arm,field in [('no-question','question'),('no-answer','answer'),('no-context','context')]:
        changed=[{**m,field:m[field]+17} for m in inputs]
        check(torch.equal(model(inputs,arm),model(changed,arm)),'ABSENT_INPUT_LEAKAGE')
    padded=[{**m,'context':torch.where(m['mask'][...,None],m['context'],m['context']+17)} for m in inputs]
    check(torch.equal(out,model(padded,'joint')),'PAD_LEAKAGE')
    changes={}
    for field in ('question','answer','context'):
        changes[field]=float((out-model([{**m,field:-m[field]} for m in inputs],'joint')).abs().max())
        check(changes[field]>0,'IGNORED_INPUT')
    return dict(private_fields_refused=True,absent_inputs_bitwise_irrelevant=True,padding_bitwise_irrelevant=True,
                input_changes=changes,source_correctness_is_world_truth=False)
