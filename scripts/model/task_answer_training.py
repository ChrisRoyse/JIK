"""Durable task labels and complete-answer checkpoint continuation."""
import json
from pathlib import Path
from types import MappingProxyType

import torch

from model.canonical_answer_inputs import AnswerLabelCache,CanonicalAnswerInputs,report
from model.native_open_window_answer import OpenPositionRQ,answer_model
from model.native_population_training import cpu,fingerprint
from model.native_placement_checked import optimizer_for,validate_optimizer
from text.capability_answer_source import stamp
from text.constellation_corpus import canonical,digest,file_sha,need


def json_value(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


class PublishedLabels(AnswerLabelCache):
    """Read the already cold-recounted labels; never re-encode per pass."""
    def __init__(self,store,root,publication,target):
        self.store=store;self.path=Path(root)/'labels.pt';ref=publication['files']['labels.pt']
        need(file_sha(self.path)==ref['sha256'] and self.path.stat().st_size==ref['bytes'],'TASK_TRAIN_LABEL_BYTES')
        value=torch.load(self.path,map_location='cpu',weights_only=True)
        self.ids=tuple(value['ids']);self.codes,self.counts=value['codes'],value['counts']
        self.binding,self.pin,self.seal=value['binding'],value['pin'],value['seal']
        need(self.ids==tuple(r['id'] for r in store.plan.records)
             and self.codes.dtype==self.counts.dtype==torch.int64
             and self.codes.shape==(len(self.ids),len(target.sizes))
             and self.counts.shape==(len(self.ids),len(store.roster)), 'TASK_TRAIN_LABEL_LAYOUT')
        need(self.binding['store']==store.pin and self.binding['codebook']==fingerprint(target.state_dict())
             and self.pin==publication['label_pin']==fingerprint((self.binding,self.ids,self.codes,self.counts))
             and self.seal==fingerprint((self.codes,self.counts)), 'TASK_TRAIN_LABEL_BINDING')
        need(all(bool(((self.codes[:,i]>=0)&(self.codes[:,i]<size)).all()) for i,size in enumerate(target.sizes))
             and self.counts.tolist()==[[store.plan.tokens[n]['windows'][i] for n in store.roster]
                                       for i in range(len(self.ids))], 'TASK_TRAIN_LABEL_VALUES')
        self.index=MappingProxyType({key:i for i,key in enumerate(self.ids)})
        self.versions=(self.codes._version,self.counts._version);self.binding_seal=digest(canonical(self.binding))
        self.row_seals=tuple(fingerprint((self.codes[i],self.counts[i])) for i in range(len(self.ids)))
        self.stamp=stamp(self.path);self.fresh()

    def fresh(self):
        self.store.fresh()
        need(stamp(self.path)==self.stamp and digest(canonical(self.binding))==self.binding_seal
             and (self.codes._version,self.counts._version)==self.versions,'TASK_TRAIN_STALE_LABELS')


def task_inputs(schedule,root,pin,cold,cold_pin,target):
    first,second=report(root,pin),report(cold,cold_pin)
    need(first['complete'] and first['mode']=='bind' and first['experimental_items_ready']
         and second['complete'] and second['source']==dict(path=str(root),pin=pin)
         and second['all_native_inputs_checked'] and second['all_learner_items_checked']
         and second['code']==first['code'],'TASK_TRAIN_CONSUMER_READBACK')
    base=Path(__file__).resolve().parents[1]
    for name,h in first['code'].items():need(file_sha(base/name)==h,'TASK_TRAIN_CONSUMER_CODE')
    for name,ref in first['files'].items():need(file_sha(Path(root)/name)==ref['sha256'],'TASK_TRAIN_CONSUMER_PAYLOAD')
    refs=first['native'];a,b=refs['native'],refs['cold']
    store=CanonicalAnswerInputs(schedule,a['path'],a['pin'],b['path'],b['pin'])
    labels=PublishedLabels(store,root,first,target)
    receipts={r['task']:r for r in json.loads((Path(root)/'items.json').read_bytes())}
    def item(key):
        row,request=store.item(key,labels,fitting=schedule.rows[key]['split']=='train');ref=receipts[key]
        need(row['binding']==ref['binding'] and row['native_pin']==ref['native'] and row['label_pin']==ref['labels']
             and request.pin==ref['request'],'TASK_TRAIN_ITEM_READBACK')
        return row,request
    return dict(store=store,labels=labels,item=item,binding=dict(consumer=pin,cold=cold_pin,store=store.pin,labels=labels.pin),
                close=labels.fresh)


def extension(bundle):
    model=OpenPositionRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    model.load_state_dict(bundle['target'],strict=True);model.eval().requires_grad_(False)
    base=Path(__file__).resolve().parents[1]
    pin=digest(canonical(dict(parent=bundle['target_ref'],runtime=file_sha(base/'model/native_open_window_answer.py'),
        physical_policy=file_sha(base/'model/native_window_end.py'),rule='preserve_TRAIN_position_prefix_then_TRAIN_mean')))
    return model,pin


def make(bundle,target,pin,architecture,device):
    model=answer_model(target,bundle,architecture,target_pin=pin).to(device)
    return model,optimizer_for(model,dict(bundle['policy'],device=device))


def checkpoint(model,optimizer,binding,ledger):
    validate_optimizer(model,optimizer,binding['policy'],len(ledger))
    return cpu(dict(schema='complete-task-answer-checkpoint-v1',binding=binding,cursor=len(ledger),
        ledger=ledger,model=model.state_dict(),optimizer=optimizer.state_dict()))


def restore(value,model,optimizer,binding):
    need(set(value)=={'schema','binding','cursor','ledger','model','optimizer'}
         and value['schema']=='complete-task-answer-checkpoint-v1' and value['binding']==binding,
         'TASK_TRAIN_CHECKPOINT_BINDING')
    step=value['cursor'];ledger=value['ledger']
    need(type(step) is int and 0<=step<=binding['updates'] and len(ledger)==step
         and [r['step'] for r in ledger]==list(range(1,step+1)),'TASK_TRAIN_CHECKPOINT_CURSOR')
    need(step==0 or (fingerprint(value['model'])==ledger[-1]['update']['after']
         and fingerprint(value['optimizer'])==ledger[-1]['update']['optimizer']),'TASK_TRAIN_CHECKPOINT_STATE')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    validate_optimizer(model,optimizer,binding['policy'],step)
    return ledger
