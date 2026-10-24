"""Provisional source-anchor readout from separate, complete native window memories.

This research head does not admit senses or authorize production assertions.
The forward signature cannot receive entity IDs, source text or answer labels.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.anchor_population_readback import arrays,lines

CONFIG=dict(width=64,heads=4,questions=3,classes=6,batch=32,steps=200,
            learning_rate=0.001,weight_decay=0.01,seeds=[25,26,27],precision="fp32")


def check(ok,code):
    if not ok:raise ValueError("MNLM_ANCHOR_READOUT_"+code)


def array_sha(value):
    h=hashlib.sha256();h.update(str(value.dtype).encode());h.update(str(value.shape).encode())
    h.update(memoryview(np.ascontiguousarray(value)).cast('B'));return h.hexdigest()


def bind_native(x,ptr,rows,maximum):
    """No semantic pooling: copy every native window to a padded per-lens memory."""
    out=np.zeros((len(rows),maximum,x.shape[1]),np.float32);mask=np.zeros((len(rows),maximum),bool)
    for i,row in enumerate(rows):
        cursor=0
        for c in row["chunks"]:
            part=x[ptr[c]:ptr[c+1]];end=cursor+len(part)
            check(end<=maximum,"WINDOW_OVERFLOW");out[i,cursor:end]=part;mask[i,cursor:end]=True;cursor=end
        check(cursor>0,"EMPTY_MEMORY")
    return out,mask


def prepare(root):
    report=json.loads((root/"report.json").read_bytes());rows=lines(root/"rows.jsonl")
    memories=[];masks=[];pins=[]
    for lane in report["lanes"]:
        x,ptr=arrays(root,lane);memory,mask=bind_native(x,ptr,rows,lane["max_document_windows"])
        check(int(mask.sum())==lane["windows"],"NATIVE_CONSERVATION")
        memories.append(torch.from_numpy(memory));masks.append(torch.from_numpy(mask))
        pins.append(dict(name=lane["name"],memory_sha256=array_sha(memory),mask_sha256=array_sha(mask),
                         shape=list(memory.shape),valid_windows=int(mask.sum())))
    question=np.asarray([r["question"] for r in rows],np.int64);target=np.zeros((len(rows),6),bool)
    for i,r in enumerate(rows):target[i,r["positive_classes"]]=True
    train=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"],np.int64)
    return rows,memories,masks,torch.from_numpy(question),torch.from_numpy(target),train,pins


class AnchorReadout(nn.Module):
    def __init__(self,dimensions):
        super().__init__();w=CONFIG["width"];h=CONFIG["heads"]
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(d,w),nn.LayerNorm(w)) for d in dimensions])
        self.window_attention=nn.ModuleList([nn.MultiheadAttention(w,h,batch_first=True,dropout=0.0) for _ in dimensions])
        self.question=nn.Embedding(3,w);self.instrument=nn.Parameter(torch.randn(len(dimensions),w)*0.02)
        self.binding=nn.MultiheadAttention(w,h,batch_first=True,dropout=0.0)
        self.norm=nn.LayerNorm(w);self.heads=nn.ModuleList([nn.Linear(w,6) for _ in range(3)])

    def forward(self,memories,masks,questions,active_lanes):
        check(len(memories)==len(masks)==len(self.projections)==10 and active_lanes in (6,10),"MODEL_INPUT")
        query=self.question(questions).unsqueeze(1);typed=[]
        for k in range(active_lanes):
            native=self.projections[k](memories[k])
            reading,_=self.window_attention[k](query,native,native,key_padding_mask=~masks[k],need_weights=False)
            typed.append(reading+self.instrument[k])
        # Stack typed learned branches, never concatenate/average native coordinates.
        memory=torch.cat(typed,dim=1)
        bound,_=self.binding(query,memory,memory,need_weights=False)
        hidden=self.norm((query+bound).squeeze(1))
        logits=torch.stack([head(hidden) for head in self.heads],dim=1)
        return logits[torch.arange(len(questions),device=questions.device),questions]


def set_loss(logits,positive):
    check(bool(positive.any(1).all()),"EMPTY_TARGET")
    return -torch.logsumexp(logits.log_softmax(1).masked_fill(~positive,-torch.inf),1).mean()


def batch(memories,masks,questions,index):
    return [m[index] for m in memories],[m[index] for m in masks],questions[index]


def schedule(train,seed):
    rng=np.random.default_rng(seed)
    return np.asarray([rng.choice(train,CONFIG["batch"],replace=False) for _ in range(CONFIG["steps"])])


def target_permutation(rows,train,seed):
    """Only training targets are permuted, separately within each fixed question."""
    rng=np.random.default_rng(seed);source=np.arange(len(rows),dtype=np.int64)
    for q in range(3):
        ids=np.asarray([i for i in train if rows[i]["question"]==q]);source[ids]=rng.permutation(ids)
    return source


def score(probabilities,rows):
    out={}
    for q in range(3):
        out[str(q)]={}
        for split in ("train","calibration","evaluation"):
            ids=[i for i,r in enumerate(rows) if r["question"]==q and r["split"]==split]
            p=probabilities[ids].astype(np.float64);truth=[rows[i]["positive_classes"] for i in ids]
            supported=np.asarray([p[j,classes].sum() for j,classes in enumerate(truth)])
            correct=sum(int(int(p[j].argmax()) in classes) for j,classes in enumerate(truth))
            out[str(q)][split]=dict(rows=len(ids),correct=correct,accuracy=correct/len(ids),
                mean_supported_probability=float(supported.mean()),nll_nats=float(-np.log(supported).mean()))
    return out


def prior(rows,train):
    result=np.empty((len(rows),6),np.float64)
    for q in range(3):
        counts=np.zeros(6,np.float64)
        for i in train:
            r=rows[i]
            if r["question"]==q:counts[r["positive_classes"]]+=1/len(r["positive_classes"])
        counts/=counts.sum()
        for i,r in enumerate(rows):
            if r["question"]==q:result[i]=counts
    return result
