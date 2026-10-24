"""Bounded batches of complete causal windows from separately typed atoms."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from model.native import NativeInput
from model.stream_surface import schedule
from senses.contract import digest
from text.constellation_corpus import need


@dataclass(frozen=True)
class MixedSurfaceBatch:
    prefix: torch.Tensor
    labels: torch.Tensor
    offsets: torch.Tensor
    denominators: torch.Tensor
    native: dict
    records: tuple
    atoms: int

    def to(self, device):
        return MixedSurfaceBatch(self.prefix.to(device), self.labels.to(device), self.offsets.to(device),
            self.denominators.to(device), {n:x.to(device) for n,x in self.native.items()}, self.records, self.atoms)


def validate(examples, config, specs, rows):
    need(type(rows) is int and 0 < rows <= 512, 'MIXED_SURFACE_ROW_LIMIT')
    need(isinstance(examples, (tuple,list)) and 0 < len(examples) <= 128, 'MIXED_SURFACE_ATOM_TILE')
    names={s.name for s in specs};total=0
    for e in examples:
        need(isinstance(e.raw,bytes) and e.raw and set(e.native)==names, 'MIXED_SURFACE_SOURCE')
        observed=None
        for spec in specs:
            x=e.native[spec.name];x.check(spec,config)
            need(spec.layout=='multi' and len(x.rows)==1 and not bool(x.absent.any())
                 and not bool(x.padding.any()), 'MIXED_SURFACE_COMPLETE_NATIVE')
            need(observed is None or observed==x.rows, 'MIXED_SURFACE_SOURCE_ROWS')
            observed=x.rows;total+=x.values.numel()
    need(total<=config.max_native_elements, 'MIXED_SURFACE_SOURCE_MEMORY')


def fits(records, examples, config, specs):
    b=len(records);width=max(len(item['prefix']) for _,item in records)
    lengths={s.name:max(examples[i].native[s.name].values.shape[1] for i,_ in records) for s in specs}
    return layout_fits(b,width,lengths,config,specs)


def layout_fits(b,width,lengths,config,specs):
    return (sum(b*lengths[s.name]*s.dims for s in specs)<=config.max_native_elements
            and b*config.heads*width*width<=config.max_attention_elements
            and all(b*config.heads*width*n<=config.max_attention_elements for n in lengths.values()))


def assemble(records, examples, config, specs):
    need(records and fits(records,examples,config,specs), 'MIXED_SURFACE_PHYSICAL_MEMORY')
    b=len(records);width=max(len(item['prefix']) for _,item in records)
    prefix=torch.zeros((b,width),dtype=torch.int64);labels=torch.full_like(prefix,-100)
    offsets=torch.empty(b,dtype=torch.int64);denominators=torch.empty(b,dtype=torch.float32)
    meta=[];ids=[]
    for row,(i,item) in enumerate(records):
        e=examples[i];length=len(item['prefix'])
        prefix[row,:length]=torch.tensor(item['prefix'],dtype=torch.int64)
        labels[row,:length]=torch.tensor(item['labels'],dtype=torch.int64)
        offsets[row]=item['offset'];denominators[row]=len(e.raw)+1
        record=dict(atom=i,occurrence=e.occurrence,uid=e.uid,offset=item['offset'],scored=item['scored'],prefix_length=length)
        meta.append(record);ids.append(digest(dict(source=next(iter(e.native.values())).rows,**record)))
    native={}
    for spec in specs:
        n=max(examples[i].native[spec.name].values.shape[1] for i,_ in records)
        values=torch.zeros((b,n,spec.dims),dtype=torch.float32);padding=torch.ones((b,n),dtype=torch.bool)
        for row,(i,_) in enumerate(records):
            value=examples[i].native[spec.name].values
            need(value.device.type=='cpu','MIXED_SURFACE_HOST_INPUT_REQUIRED')
            length=value.shape[1];values[row,:length]=value[0];padding[row,:length]=False
        native[spec.name]=NativeInput(spec.identity,tuple(ids),values,padding,torch.zeros(b,dtype=torch.bool))
        native[spec.name].check(spec,config)
    return MixedSurfaceBatch(prefix,labels,offsets,denominators,native,tuple(meta),len(examples))


def batches(examples, config, specs, *, rows=128):
    """Stream schedules; never materialize the corpus or truncate an observation."""
    validate(examples,config,specs,rows);pending=[];width=0;lengths={s.name:0 for s in specs}
    for i,e in enumerate(examples):
        native_lengths={s.name:e.native[s.name].values.shape[1] for s in specs}
        for item in schedule(e.raw,width=config.max_prefix):
            new_width=max(width,len(item['prefix']))
            new_lengths={n:max(lengths[n],v) for n,v in native_lengths.items()}
            if pending and (len(pending)==rows or not layout_fits(len(pending)+1,new_width,new_lengths,config,specs)):
                yield assemble(pending,examples,config,specs);pending=[]
                new_width=len(item['prefix']);new_lengths=native_lengths.copy()
            need(layout_fits(len(pending)+1,new_width,new_lengths,config,specs),'MIXED_SURFACE_SINGLE_ROW_MEMORY')
            pending.append((i,item))
            width=new_width;lengths=new_lengths
    if pending:yield assemble(pending,examples,config,specs)


def loss(logits, batch):
    """Mean of complete per-atom losses, including every original byte and EOS."""
    need(logits.shape[:2]==batch.labels.shape and logits.device==batch.labels.device,
         'MIXED_SURFACE_LOGIT_SHAPE')
    token_loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.labels.reshape(-1),
                               ignore_index=-100,reduction='none').reshape(batch.labels.shape)
    objective=(token_loss.sum(1)/batch.denominators).sum()/batch.atoms
    need(bool(torch.isfinite(objective)),'MIXED_SURFACE_LOSS_FINITE')
    return objective,token_loss
