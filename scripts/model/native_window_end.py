"""Predict a positive native-window count by position-wise continuation and stop."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Linear
from text.constellation_corpus import need


def physical_limits(target,output_bytes):
    """Limits come from existing allocations, never observed target lengths."""
    c=target.config;dims=target.dimensions;lanes=len(dims)
    tokens=min(c['max_native_tokens'],math.isqrt(c['max_attention_elements']//c['heads']))
    limits={n:min(tokens-(2*lanes-1),
        (c['max_native_elements']-(sum(dims.values())-d))//d,
        (output_bytes//4-(sum(dims.values())-d))//d) for n,d in dims.items()}
    need(all(v>0 for v in limits.values()),'WINDOW_END_PHYSICAL_ALLOWANCE');return limits


class WindowEnd(nn.Module):
    """P(W=w)=P(continue at 1..w-1)*P(stop at w), with shared parameters."""
    TILE=32

    def __init__(self,width,frequency,limit):
        super().__init__();need(type(limit) is int and limit>0,'WINDOW_END_LIMIT')
        self.width,self.limit=width,limit
        self.register_buffer('frequency',frequency.detach().clone())
        self.context=Linear(width,width);self.position=Linear(width,width);self.out=Linear(width,1)
        # A declared geometric initial count law, independent of TRAIN/QUERY lengths.
        with torch.no_grad():self.out.weight.zero_();self.out.bias.zero_()

    def logits(self,hidden,positions):
        need(hidden.dtype==torch.float32 and hidden.ndim==2 and hidden.shape[1]==self.width
             and 0<len(hidden)<=64 and bool(torch.isfinite(hidden).all()),'WINDOW_END_HIDDEN')
        need(positions.dtype==torch.int64 and positions.ndim==1 and positions.device==hidden.device
             and 0<len(positions)<=self.TILE and bool(((positions>=1)&(positions<=self.limit)).all()),
             'WINDOW_END_POSITION')
        phase=(positions-1)[:,None]*self.frequency[None]
        position=torch.stack((phase.sin(),phase.cos()),-1).flatten(1)
        x=self.context(hidden)[:,None,:]+self.position(position)[None]
        result=self.out(F.gelu(x)).squeeze(-1)
        need(bool(torch.isfinite(result).all()),'WINDOW_END_NONFINITE');return result

    def loss(self,hidden,counts):
        need(counts.dtype==torch.int64 and counts.shape==(len(hidden),) and counts.device==hidden.device
             and bool(((counts>=1)&(counts<=self.limit)).all()),'WINDOW_END_TARGET_PHYSICAL_SCOPE')
        last=int(counts.max());parts=[];traces=[]
        for first in range(1,last+1,self.TILE):
            positions=torch.arange(first,min(first+self.TILE,last+1),device=hidden.device)
            logits=self.logits(hidden,positions).double();live=positions[None]<=counts[:,None]
            stop=positions[None]==counts[:,None]
            terms=torch.where(stop,F.softplus(-logits),F.softplus(logits))
            parts.append((terms*live).sum(1));traces.append(logits)
        values=torch.stack(parts).sum(0)
        return dict(loss=values.mean(),nll=values,logits=torch.cat(traces,1))

    def forecast(self,hidden,*,generator=None):
        counts=torch.zeros(len(hidden),dtype=torch.int64,device=hidden.device);traces=[]
        survival=torch.zeros(len(hidden),dtype=torch.float64,device=hidden.device)
        best=torch.full_like(survival,-torch.inf)
        for first in range(1,self.limit+1,self.TILE):
            positions=torch.arange(first,min(first+self.TILE,self.limit+1),device=hidden.device)
            logits=self.logits(hidden,positions);traces.append(logits)
            if generator is None:
                continuation=F.logsigmoid(-logits.double());prefix=continuation.cumsum(1)-continuation
                probability=survival[:,None]+prefix+F.logsigmoid(logits.double())
                value,index=probability.max(1);counts=torch.where(value>best,index+first,counts)
                best=torch.maximum(best,value);survival=survival+continuation.sum(1)
                # Remaining tail mass bounds every later count probability. This
                # certifies the global mode without a TRAIN-derived search limit.
                complete=bool((survival<=best).all())
            else:
                stop=torch.rand(logits.shape,device=hidden.device,generator=generator)<logits.sigmoid()
                present=stop.any(1);selected=stop.to(torch.int64).argmax(1)+first
                counts=torch.where((counts==0)&present,selected,counts);complete=bool((counts>0).all())
            if complete:return dict(counts=counts,logits=torch.cat(traces,1))
        raise ValueError('WINDOW_END_PHYSICAL_MEMORY_EXHAUSTED_NO_TRUNCATION')
