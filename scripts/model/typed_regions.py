"""Separate native-sense codebooks and a from-scratch typed region predictor."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.qasc_context import check

CONFIG=dict(width=96,layers=2,heads=4,regions=64,lloyd_updates=25,steps=600,batch=32,
            seeds=[25,26,27],learning_rate=.001,weight_decay=.01,temperature=.1,precision='fp32')
FIELDS=('question','fact1','fact2')

def normalize(x):
    check(bool(torch.isfinite(x).all()) and bool((x.norm(dim=-1)>1e-8).all()),'REGION_ZERO_OR_NONFINITE')
    return F.normalize(x,dim=-1)

class Store:
    def __init__(self,ctx,device='cuda'):
        self.native=[torch.as_tensor(x,device=device) for x in ctx['inputs']]
        self.truth=[torch.as_tensor(x,device=device) for x in ctx['truth']]
        self.maps=torch.as_tensor(ctx['maps'],device=device);self.target_index=torch.as_tensor(ctx['target_index'],device=device)
        self.device=device
    def prepare(self,train):
        unique=torch.unique(self.target_index[train],sorted=True)
        self.means=[x[unique].mean(0) for x in self.truth]
        self.x=[normalize(x-m) for x,m in zip(self.native,self.means)]
        self.y=[normalize(x-m) for x,m in zip(self.truth,self.means)]
        return unique
    def batch(self,ids,scope='both-facts',donor=None):
        idx=self.maps[ids].clone()
        if donor is not None:idx[:,1:]=self.maps[donor][:,1:]
        v={field:[x[idx[:,j]] for x in self.x] for j,field in enumerate(FIELDS)}
        for field in FIELDS:
            if ((scope=='facts-only' and field=='question') or (scope=='question-only' and field!='question')
                    or (scope=='fact1-only' and field=='fact2') or (scope=='fact2-only' and field=='fact1')):
                v[field]=[torch.zeros_like(x) for x in v[field]]
        return v

@torch.inference_mode()
def build_books(store,unique):
    books=[];histories=[];rng=np.random.default_rng(25)
    initial=rng.choice(len(unique),CONFIG['regions'],replace=False)
    for y in store.y:
        values=y[unique];centers=values[initial].clone();history=[]
        for _ in range(CONFIG['lloyd_updates']):
            codes=(values@centers.T).argmax(1);membership=F.one_hot(codes,CONFIG['regions']).float()
            counts=membership.sum(0);sums=membership.T@values;occupied=counts>0
            centers[occupied]=normalize(sums[occupied]);history.append(counts.cpu().numpy())
        books.append(centers);histories.append(np.array(history))
    codes=torch.stack([(y@b.T).argmax(1) for y,b in zip(store.y,books)],1)
    return books,codes,histories,initial

class Predictor(nn.Module):
    def __init__(self,books):
        super().__init__();w=CONFIG['width'];self.K=6
        self.adapters=nn.ModuleList([nn.Sequential(nn.LayerNorm(768),nn.Linear(768,w)) for _ in range(6)])
        self.sense=nn.Parameter(torch.randn(6,w)*.02);self.role=nn.Parameter(torch.randn(3,w)*.02)
        self.query=nn.Parameter(torch.randn(6,w)*.02)
        block=nn.TransformerEncoderLayer(w,CONFIG['heads'],4*w,dropout=0,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,CONFIG['layers'],enable_nested_tensor=False)
        self.norm=nn.LayerNorm(w);self.codes=nn.ModuleList([nn.Linear(w,64) for _ in range(6)])
        self.residual=nn.ModuleList([nn.Linear(w,768) for _ in range(6)])
        for k,b in enumerate(books):self.register_buffer('book_'+str(k),b.clone())
    def forward(self,inputs,arm='region'):
        check(type(inputs)==dict and set(inputs)==set(FIELDS),'REGION_FORWARD_FIELDS')
        check(arm in ('region','direct'),'REGION_ARM')
        for field in FIELDS:
            check(type(inputs[field])==list and len(inputs[field])==6,'REGION_FORWARD_SENSES')
            check(all(x.ndim==2 and x.shape==inputs['question'][0].shape and x.shape[1]==768 for x in inputs[field]),'REGION_FORWARD_SHAPE')
        tokens=torch.stack([self.adapters[k](inputs[f][k])+self.sense[k]+self.role[j]
                            for j,f in enumerate(FIELDS) for k in range(6)],1)
        # Concatenation is over typed token positions; native vectors never concatenate.
        h=self.norm(self.encoder(torch.cat((tokens,self.query[None].expand(len(tokens),-1,-1)),1))[:,-6:])
        logits=[self.codes[k](h[:,k]) for k in range(6)];slots=[]
        for k in range(6):
            delta=self.residual[k](h[:,k])
            if arm=='region':delta=getattr(self,'book_'+str(k))[logits[k].argmax(1)]+.5*delta
            slots.append(normalize(delta))
        return slots,logits

def objective(slots,logits,truth,codes,target_ids):
    same=target_ids[:,None]==target_ids[None,:];same.fill_diagonal_(False)
    ce=[];cos=[];contrast=[];labels=torch.arange(len(target_ids),device=target_ids.device)
    for k,(x,y) in enumerate(zip(slots,truth)):
        ce.append(F.cross_entropy(logits[k],codes[:,k]));cos.append((1-(x*y).sum(1)).mean())
        scores=(x@y.T/CONFIG['temperature']).masked_fill(same,float('-inf'))
        contrast.append(F.cross_entropy(scores,labels))
    terms=torch.stack([torch.stack(v).mean() for v in (ce,cos,contrast)])
    return terms.sum(),terms

def schedule(rows,seed):
    train=np.array([i for i,r in enumerate(rows) if r['partition']=='train'],np.int64)
    return np.random.default_rng(seed).choice(train,(CONFIG['steps'],CONFIG['batch']),replace=True)

@torch.inference_mode()
def infer(model,store,arm,scope='both-facts',donors=None):
    values=[[] for _ in range(6)];logits=[[] for _ in range(6)]
    for start in range(0,len(store.maps),32):
        ids=np.arange(start,min(start+32,len(store.maps)))
        x,l=model(store.batch(ids,scope,None if donors is None else donors[ids]),arm)
        for k in range(6):values[k].append(x[k].cpu().numpy());logits[k].append(l[k].cpu().numpy())
    return [np.vstack(v) for v in values],[np.vstack(v) for v in logits]

@torch.inference_mode()
def controls(model,store,arm):
    inputs=store.batch(np.arange(32));base,_=model(inputs,arm)
    try:model({**inputs,'target':torch.zeros(32,768,device=store.device)},arm)
    except ValueError as exc:check('FORWARD_FIELDS' in str(exc),'REGION_TARGET_ERROR')
    else:raise ValueError('MNLM_QASC_REGION_TARGET_ACCEPTED')
    changes={}
    for field in FIELDS:
        changed={**inputs,field:[-x for x in inputs[field]]};out,_=model(changed,arm)
        changes[field]=max(float((x-y).abs().max()) for x,y in zip(base,out))
        check(changes[field]>0,'REGION_IGNORED_INPUT')
    return dict(private_field_refused=True,input_changes=changes,teacher_forced_region=False,unsnapped=True)
