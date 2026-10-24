"""Complete byte traces and per-instrument discrimination; never admission bits."""
from collections import defaultdict
import math

import numpy as np
import torch
from torch.nn import functional as F

from claims.capability_population import need
from model.stream_surface import batches


def trace(decoder, raw, native):
    c=decoder.config;elements=sum(x.values.numel() for x in native.values())
    widest=max(x.values.shape[1] for x in native.values())
    size=min(8,c.max_native_elements//elements,
             c.max_attention_elements//(c.heads*c.max_prefix*max(c.max_prefix,widest)))
    need(size>0,'RESCORE_PHYSICAL_BATCH');pieces=[];canonical=0.;cursor=0
    for prefix,labels,offsets,slots,items in batches(raw,native,c,batch_size=size):
        logits=decoder(prefix,slots,offsets=offsets)['logits']
        flat=F.cross_entropy(logits.reshape(-1,c.vocabulary),labels.reshape(-1),
                             ignore_index=-100,reduction='none').reshape_as(labels)
        canonical+=float(F.cross_entropy(logits.reshape(-1,c.vocabulary),labels.reshape(-1),
                            ignore_index=-100,reduction='sum')/(len(raw)+1))
        for index,item in enumerate(items):
            lo,hi=item['scored'];need(lo==cursor,'RESCORE_BYTE_ORDER')
            values=flat[index,labels[index]!=-100].cpu().numpy().copy()
            need(len(values)==hi-lo and np.isfinite(values).all(),'RESCORE_BYTE_COVERAGE')
            pieces.append(values);cursor=hi
    need(cursor==len(raw)+1,'RESCORE_BYTE_COMPLETE')
    return np.concatenate(pieces),dict(canonical_loss=canonical,physical_batch=size,positions=cursor)


def coverage(a,b):
    """Symmetric MaxSim; metric only, with bounded native matrix tiles."""
    a=F.normalize(a,dim=-1);b=F.normalize(b,dim=-1)
    left=torch.full((len(a),),-torch.inf,device=a.device)
    right=torch.full((len(b),),-torch.inf,device=b.device)
    for i in range(0,len(a),512):
        for j in range(0,len(b),512):
            m=a[i:i+512]@b[j:j+512].T
            left[i:i+512]=torch.maximum(left[i:i+512],m.max(1).values)
            right[j:j+512]=torch.maximum(right[j:j+512],m.max(0).values)
    return float((left.mean()+right.mean())/2)


def array(slot):return slot.values[0,~slot.padding[0]]


def discriminate(pred,target,alternatives):
    result={}
    for name,slot in pred.items():
        value=coverage(array(slot),array(target[name]))
        scores=[coverage(array(slot),array(other[name])) for other in alternatives]
        result[name]=dict(target=value,distractors=scores,wins=sum(value>x for x in scores),
                         ties=sum(value==x for x in scores),losses=sum(value<x for x in scores))
    return result


def difference(correct,shuffled):
    need(correct.shape==shuffled.shape and correct.ndim==1 and len(correct)>1,'RESCORE_DELTA_SHAPE')
    values=shuffled.astype(np.float64)-correct.astype(np.float64)
    return dict(positions=len(values),bytes=len(values)-1,
        mean_byte_delta=math.fsum(values[:-1])/max(1,len(values)-1),eos_delta=float(values[-1]),
        mean_position_delta=math.fsum(values)/len(values),positive=int((values>0).sum()),
        negative=int((values<0).sum()),ties=int((values==0).sum()))


def summarize(rows):
    cells=defaultdict(list)
    for r in rows:
        cells[(r['arm'],r['operation'],'all')].append(r)
        cells[(r['arm'],r['operation'],r['group'])].append(r)
    result=[]
    for (arm,operation,group),values in sorted(cells.items()):
        cell=dict(arm=arm,operation=operation,group=group,tasks=len(values),
                  groups=len({r['group'] for r in values}),lanes={},decoder={})
        for name in values[0]['lanes']:
            lanes=[r['lanes'][name] for r in values];wins=sum(x['wins'] for x in lanes)
            ties=sum(x['ties'] for x in lanes);losses=sum(x['losses'] for x in lanes)
            nonidentical=[(x['target'],s) for r,x in zip(values,lanes)
                          for s,same in zip(x['distractors'],r['identical_surfaces']) if not same]
            cell['lanes'][name]=dict(wins=wins,ties=ties,losses=losses,pairs=wins+ties+losses,
                strict_accuracy=wins/(wins+ties+losses),tie_half_accuracy=(wins+.5*ties)/(wins+ties+losses),
                nonidentical_pairs=len(nonidentical),nonidentical_wins=sum(a>b for a,b in nonidentical),
                nonidentical_ties=sum(a==b for a,b in nonidentical))
        for kind in ('predicted','oracle'):
            ds=[r['deltas'][kind] for r in values];total=sum(x['bytes'] for x in ds)
            cell['decoder'][kind]=dict(mean_task_delta=math.fsum(x['mean_byte_delta'] for x in ds)/len(ds),
                mean_byte_delta=math.fsum(x['mean_byte_delta']*x['bytes'] for x in ds)/total,bytes=total,
                better=sum(x['mean_byte_delta']>0 for x in ds),worse=sum(x['mean_byte_delta']<0 for x in ds),
                ties=sum(x['mean_byte_delta']==0 for x in ds))
        result.append(cell)
    return result
