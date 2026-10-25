"""Independent source, ledger, checkpoint and complete-byte audit operations."""
from dataclasses import replace
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F
from model import gathered_surface_batches as gathered
from model import protected_corpus_lineage as lineage
from model import protected_corpus_training as training
from model import protected_surface_state as state
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha,need


def census(source,ids,plan):
    data=source['data'];offsets=source['surfaces'].offsets;cursor=byte_count=prefix_rows=0
    for d,(lo,hi) in enumerate(data.documents):
        if data.doc_sides[d]!=0:continue
        lo,hi=int(lo),int(hi)
        need(np.array_equal(ids[cursor:cursor+hi-lo],np.arange(lo,hi,dtype=np.int64)),
             'PROTECTED_READBACK_TRAIN_COVERAGE')
        u=data.mapping[lo:hi];lengths=offsets[u+1]-offsets[u]
        cursor+=hi-lo;byte_count+=int(lengths.sum());prefix_rows+=int(((lengths+16)//16).sum())
    result=dict(atoms=cursor,bytes=byte_count,prefix_rows=prefix_rows)
    need(cursor==len(ids) and result==plan['train'],'PROTECTED_READBACK_CENSUS')
    return result


def smallest_documents(source,plan):
    """Choose by complete source size only, before any new forward or score."""
    data=source['data'];offsets=source['surfaces'].offsets;result={}
    for side,code in [('development',1),('query',2)]:
        candidates=[]
        for d in plan['evaluation'][side]['documents']:
            lo,hi=map(int,data.documents[d]);u=data.mapping[lo:hi];lengths=offsets[u+1]-offsets[u]
            need(data.doc_sides[d]==code and hi>lo,'PROTECTED_READBACK_DOCUMENT_SIDE')
            candidates.append((int(lengths.sum())+hi-lo,d))
        _,document=min(candidates);lo,hi=map(int,data.documents[document]);windows={};raw=positions=0
        for start in range(lo,hi,128):
            for e in source['adapter'].fetch(list(range(start,min(start+128,hi))),side=side):
                raw+=len(e.raw);positions+=len(e.raw)+1
                for n,v in e.native.items():windows[n]=windows.get(n,0)+len(v.values[0])
        result[side]=dict(document=document,component=int(data.components[document]),atoms=hi-lo,
                          bytes=raw,positions=positions,windows=windows)
        need(positions==min(candidates)[0],'PROTECTED_READBACK_DOCUMENT_BYTES')
    return result


def row(record,*,ids,cursor,step,previous,before,branches,source):
    v=record['row'];selected=ids[cursor:cursor+128].tolist()
    need(selected and v['step']==step+1 and v['occurrences']==selected and v['cursor']==cursor+len(selected)
         and v['previous']==previous and digest(v)==record['pin'] and v['before']==before,
         'PROTECTED_READBACK_LEDGER')
    raw=source['surfaces'].occurrences(source['data'],selected,side='train',fit=True)
    uids=source['data'].mapping[selected]
    need(v['input']==digest([dict(occurrence=i,uid=int(u),raw=b.hex()) for i,u,b in zip(selected,uids,raw)])
         and v['positions']==sum(len(x)+1 for x in raw),'PROTECTED_READBACK_INPUT')
    need(set(v['branches'])==branches and all(type(x['nonzero']) is bool and x['nonzero']==(step>0)
         and re.fullmatch('[0-9a-f]{64}',x['pin']) for x in v['branches'].values()),
         'PROTECTED_READBACK_BRANCHES')
    need(all(isinstance(v[k],str) and re.fullmatch('[0-9a-f]{64}',v[k])
             for k in ('native','gradient','after','optimizer','scores')) and math.isfinite(v['objective']),
         'PROTECTED_READBACK_HASHES')
    return v['cursor'],v['step'],record['pin'],v['after']


def checkpoint(ref,record,binding,model):
    v=record['row']
    need(ref['cursor']==v['cursor'] and ref['step']==v['step'] and ref['ledger']==record['pin']
         and ref['after']==v['after'] and ref['optimizer']==v['optimizer'],'PROTECTED_READBACK_CHECKPOINT_LEDGER')
    bound=training.progress(binding,ref['cursor'],ref['step'],ref['ledger'],ref['size'])
    payload,m=state.read(ref['root'],ref['pin'],binding=bound,parent=model.parent_pin)
    need(ref['state']==m['state'] and ref['step']==m['step'] and fingerprint(payload)==ref['state']
         and fingerprint(payload['correction'])==v['after'] and fingerprint(payload['optimizer'])==v['optimizer'],
         'PROTECTED_READBACK_CHECKPOINT_STATE')
    expected=model.correction.state_dict();actual=payload['correction']
    need(actual.keys()==expected.keys() and all(actual[n].shape==expected[n].shape and actual[n].dtype==expected[n].dtype
         for n in expected),'PROTECTED_READBACK_PARAMETER_SHAPES')
    moments=payload['optimizer'];parameters=list(model.correction.parameters())
    indices=[i for group in moments['param_groups'] for i in group['params']]
    need(len(indices)==len(parameters) and len(set(indices))==len(indices) and set(moments['state'])==set(indices),
         'PROTECTED_READBACK_MOMENT_COVERAGE')
    for i,p in zip(indices,parameters):
        value=moments['state'][i]
        need(set(value)=={'step','exp_avg','exp_avg_sq'} and value['step'].numel()==1 and float(value['step'])==ref['step']
             and all(value[k].shape==p.shape and value[k].dtype==p.dtype for k in ('exp_avg','exp_avg_sq')),
             'PROTECTED_READBACK_MOMENT_STATE')
    return dict(step=ref['step'],pin=ref['pin'],state=ref['state'],shards=len(m['shards']),
                bytes=sum(x['bytes'] for x in m['shards']))


def ledger(report,source,ids,model):
    b=report['binding'];size=b['size'];selected=ids[:size];last=report['checkpoints'][-1]
    need(size==(len(ids) if report['full_training'] else 384),'PROTECTED_READBACK_SCOPE')
    lineage.origin(last,b);segments=[];tip=last
    while True:
        path=Path(tip['origin']['path']);o=json.loads(path.read_bytes());segments.append((tip,o,path.parent))
        if o['resume'] is None:break
        tip=o['resume']['reference']
    segments.reverse();cursor=step=0;previous=digest(b);before=fingerprint(model.correction.state_dict());checks=[]
    branches={prefix for s in model.specs for prefix in ['projectors.'+s.name+'.',
        *[f'blocks.{i}.branches.{s.name}.' for i in range(model.config.layers)]]}
    current_refs=[];native_replayed=0
    for tip,o,root in segments:
        need(o['start']==dict(cursor=cursor,step=step,ledger=previous) and o['initial_correction']==before,
             'PROTECTED_READBACK_ANCESTRY')
        need(json.loads((root/'binding.json').read_bytes())==b,'PROTECTED_READBACK_SEGMENT_BINDING')
        refs=[]
        with (root/'steps.jsonl').open() as f:
            for line in f:
                record=json.loads(line)
                cursor,step,previous,before=row(record,ids=selected,cursor=cursor,step=step,previous=previous,
                                               before=before,branches=branches,source=source)
                if not report['full_training']:
                    examples=source['adapter'].fetch(record['row']['occurrences'],side='train',fit=True)
                    need(fingerprint([{n:x.values for n,x in e.native.items()} for e in examples])==record['row']['native'],
                         'PROTECTED_READBACK_NATIVE');native_replayed+=len(examples)
                required=step%256==0 or step==tip['step'] or (not report['full_training'] and step>=2)
                if required:
                    refpath=root/f'checkpoint-{step:06d}.json'
                    need(refpath.is_file(),'PROTECTED_READBACK_CHECKPOINT_MISSING');ref=json.loads(refpath.read_bytes())
                    need(Path(ref['root']).resolve()==(root/f'checkpoint-{step:06d}').resolve()
                         and ref['origin']==tip['origin'] and ref['size']==size,'PROTECTED_READBACK_CHECKPOINT_PATH')
                    checks.append(checkpoint(ref,record,b,model));refs.append(ref)
                if step==tip['step']:
                    need(ref==tip,'PROTECTED_READBACK_TIP')
                    if root==Path(last['root']).parent:need(not f.read().strip(),'PROTECTED_READBACK_LEDGER_TAIL')
                    break
        need(step==tip['step'] and cursor==tip['cursor'] and previous==tip['ledger'],'PROTECTED_READBACK_SEGMENT_END')
        current_refs=refs
    need(current_refs==report['checkpoints'] and cursor==report['cursor']==size and step==report['steps']
         and previous==report['ledger'] and before==last['after'],'PROTECTED_READBACK_COMPLETE')
    return dict(segments=len(segments),rows=step,source_occurrences=cursor,checkpoints=checks,
                native_occurrences_replayed=native_replayed,full_gradient_replay=False)


def complete_scores(scores,examples):
    need(len(scores)==len(examples) and all(set(x)=={'losses','argmax'} and x['losses'].shape==x['argmax'].shape==(len(e.raw)+1,)
         and bool(x['losses'].isfinite().all()) and bool(((x['argmax']>=0)&(x['argmax']<258)).all())
         for x,e in zip(scores,examples)),'PROTECTED_READBACK_MISSING_BYTE')


def score(model,examples):
    """Independent loss/scatter readout, preserving the declared gathered-128 geometry."""
    result=[dict(losses=torch.full((len(e.raw)+1,),float('nan')),
                 argmax=torch.full((len(e.raw)+1,),-1,dtype=torch.int64)) for e in examples]
    model.eval();device=next(model.parameters()).device
    with torch.no_grad():
        for host in gathered.batches(examples,model.config,model.specs,rows=128):
            b=host.to(device);logits=model(b.prefix,b.native,offsets=b.offsets)['logits']
            losses=F.cross_entropy(logits.reshape(-1,258),b.labels.reshape(-1),ignore_index=-100,
                                   reduction='none').reshape(b.labels.shape).cpu()
            predicted=logits.argmax(-1).cpu()
            for j,r in enumerate(host.records):
                lo,hi=r['scored'];left=lo-r['offset'];right=hi-r['offset'];item=result[r['atom']]
                need(bool(item['losses'][lo:hi].isnan().all()),'PROTECTED_READBACK_DUPLICATE_BYTE')
                item['losses'][lo:hi]=losses[j,left:right];item['argmax'][lo:hi]=predicted[j,left:right]
            del logits,losses,predicted,b,host
    complete_scores(result,examples);return result


def donors(source,ids,train):
    data=source['data'];result=[]
    for i in ids:
        c=data.components[np.searchsorted(data.documents[:,1],i,side='right')]
        start=int(digest(dict(seed='inversion-donor-v1',occurrence=i))[:16],16)%len(train)
        for offset in range(len(train)):
            d=int(train[(start+offset)%len(train)]);dc=data.components[np.searchsorted(data.documents[:,1],d,side='right')]
            if dc!=c and data.mapping[d]!=data.mapping[i]:result.append(d);break
        else:raise ValueError('PROTECTED_READBACK_NO_DONOR')
    return result


def metadata(values,examples,donor_ids,source):
    expected=[dict(occurrence=e.occurrence,uid=e.uid,donor=d,bytes=len(e.raw),
                   train_present=bool(source['continuation'].presence[e.uid]&1)) for e,d in zip(examples,donor_ids)]
    need(values==expected,'PROTECTED_READBACK_EVALUATION_METADATA')


def evaluation(refs,source,model,groups,train):
    need([(r['side'],r['document']) for r in refs]==[(s,d) for s,ds in groups.items() for d in ds],
         'PROTECTED_READBACK_EVALUATION_POPULATION')
    result={}
    for ref in refs:
        path=Path(ref['path']);need(file_sha(path)==ref['pin'],'PROTECTED_READBACK_EVALUATION_PIN')
        saved=torch.load(path,map_location='cpu',weights_only=True);lo,hi=map(int,source['data'].documents[ref['document']])
        need(len(saved['scores'])==len(saved['metadata'])==hi-lo==ref['atoms']
             and ref['component']==int(source['data'].components[ref['document']]),'PROTECTED_READBACK_EVALUATION_RANGE')
        positions=matches=exact=0;correct_sum=shuffled_sum=0.
        for start in range(lo,hi,128):
            ids=list(range(start,min(start+128,hi)));ds=donors(source,ids,train)
            examples=source['adapter'].fetch(ids,side=ref['side']);other=source['adapter'].fetch(ds,side='train')
            offset=start-lo;count=len(ids);metadata(saved['metadata'][offset:offset+count],examples,ds,source)
            correct=score(model,examples);shuffled=score(model,[replace(e,native=d.native) for e,d in zip(examples,other)])
            actual=[dict(correct=x,shuffled=y) for x,y in zip(correct,shuffled)]
            need(fingerprint(actual)==fingerprint(saved['scores'][offset:offset+count]),'PROTECTED_READBACK_EVERY_BYTE')
            for e,x,y in zip(examples,correct,shuffled):
                positions+=len(e.raw)+1;labels=torch.tensor([*e.raw,257]);matches+=int((labels==x['argmax']).sum())
                exact+=int(torch.equal(labels,x['argmax']))
                correct_sum+=float(x['losses'].double().sum());shuffled_sum+=float(y['losses'].double().sum())
        need(all(ref[k]==v for k,v in dict(positions=positions,byte_matches=matches,exact_atoms=exact,
             correct_sum=correct_sum,shuffled_sum=shuffled_sum,shuffled_minus_correct=(shuffled_sum-correct_sum)/positions).items()),
             'PROTECTED_READBACK_DENOMINATOR')
        t=result.setdefault(ref['side'],dict(documents=0,atoms=0,positions=0,correct_sum=0.,shuffled_sum=0.,
                                            byte_matches=0,exact_atoms=0))
        for k,v in dict(documents=1,atoms=hi-lo,positions=positions,correct_sum=correct_sum,shuffled_sum=shuffled_sum,
                        byte_matches=matches,exact_atoms=exact).items():t[k]+=v
    return result
