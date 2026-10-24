"""Independent recount of task joins, native bytes, code assignments and capacity."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.canonical_answer_consumer import ROOT,prepare_source,code,reject
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest,describe
from model.native_target_cache import FrozenLabels
from text.constellation_corpus import canonical,digest,file_sha,need


def recount(saved,schedule,target):
    ids={v['id']:i for i,v in enumerate(schedule.plan.records)}
    caps={n:len(getattr(target,'position_'+str(i))) for i,n in enumerate(target.roster)}
    need(saved['count_head_capacity']==caps and saved['windows_truncated']==0,'ANSWER_READBACK_CAPACITY')
    need([r['task'] for r in saved['rows']]==sorted(schedule.rows),'ANSWER_READBACK_TASK_COVERAGE')
    totals={side:dict(tasks=0,context_fields=0,answer_atoms=0,end_labels=0,decisions=0,unsupported_tasks=0,
        instruments={n:dict(max_answer_windows=0,unsupported_atoms=0,unsupported_tasks=0) for n in caps})
        for side in ('train','query')}
    for row in saved['rows']:
        key=row['task'];original=schedule.targets.rows[key];request=schedule.targets.request(key)
        answer=schedule.targets.target(key,fitting=original['split']=='train')
        contexts=[digest(f['text'].encode()) for f in request['context']]
        targets=[a['measurement'] for a in answer['atoms']];c=len(contexts);a=len(targets)
        need(row['side']==original['split'] and row['family']==original['family']
             and row['group']==original['split_group'] and row['operation']==request['operation']
             and row['contexts']==contexts and row['answers']==targets and row['ending']==answer['end'],
             'ANSWER_READBACK_SOURCE_JOIN')
        decisions=[dict(position=c-1+i,answer_index=i if i<a else None,end=i==a) for i in range(a+1)]
        need(row['decisions']==decisions and row['request']==NativeRequest(describe(schedule,key)).pin,
             'ANSWER_READBACK_SUPERVISION')
        total=totals[row['side']];total['tasks']+=1;total['context_fields']+=c;total['answer_atoms']+=a
        total['end_labels']+=1;total['decisions']+=a+1;bad=False
        for n in caps:
            counts=[schedule.plan.tokens[n]['windows'][ids[i]] for i in targets]
            excess=[j for j,v in enumerate(counts) if v>caps[n]]
            need(row['counts'][n]==counts and row['unsupported'][n]==excess,'ANSWER_READBACK_NATIVE_COUNTS')
            t=total['instruments'][n];t['max_answer_windows']=max(t['max_answer_windows'],*counts)
            t['unsupported_atoms']+=len(excess);t['unsupported_tasks']+=bool(excess);bad|=bool(excess)
        total['unsupported_tasks']+=bad
    need(totals==saved['summary'],'ANSWER_READBACK_DENOMINATORS')
    need(saved['source_native_windows']=={n:schedule.plan.tokens[n]['total_windows'] for n in caps},
         'ANSWER_READBACK_WINDOW_DENOMINATOR')
    return totals


def frame_check(frame,expected):
    need(set(frame)==set(expected),'ANSWER_READBACK_MISSING_INSTRUMENT')
    for n in expected:
        need(frame[n].shape==expected[n].shape and np.array_equal(frame[n],expected[n]),'ANSWER_READBACK_WINDOW_BYTES')


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        need(file_sha(a.input/'report.json')==a.pin,'ANSWER_READBACK_REPORT_PIN')
        report=json.loads((a.input/'report.json').read_bytes())
        need(report['complete'] and report['schema']=='canonical-answer-consumer-v1','ANSWER_READBACK_SCOPE')
        for n,r in report['files'].items():
            need(file_sha(a.input/n)==r['sha256'] and (a.input/n).stat().st_size==r['bytes'],'ANSWER_READBACK_FILE')
        schedule,source,bundle,target=prepare_source(a.root)
        need(code(source)==report['code'],'ANSWER_READBACK_CODE')
        saved=json.loads((a.input/'census.json').read_bytes());summary=recount(saved,schedule,target)
        phases['source_and_schedule_recount']=time.perf_counter()-tick;tick=time.perf_counter()
        controls=[];bad=deepcopy(saved);bad['rows'][0]['answers'].pop()
        controls.append(reject('ANSWER_READBACK_SOURCE_JOIN',lambda:recount(bad,schedule,target)))
        bad=deepcopy(saved);bad['summary']['query']['tasks']-=1
        controls.append(reject('ANSWER_READBACK_DENOMINATORS',lambda:recount(bad,schedule,target)))
        # Real canonical reused input, never an invented task reading.
        original={}
        for n in target.roster:
            with np.load(schedule.plan.root/n/'reused.npz',allow_pickle=False) as z:
                original[n]=z['vectors'][z['offsets'][0]:z['offsets'][1]].copy()
        frame_check(original,original)
        broken=dict(original);del broken[target.roster[0]]
        controls.append(reject('ANSWER_READBACK_MISSING_INSTRUMENT',lambda:frame_check(broken,original)))
        broken=dict(original);broken[target.roster[0]]=broken[target.roster[0]][:-1]
        controls.append(reject('ANSWER_READBACK_WINDOW_BYTES',lambda:frame_check(broken,original)))
        checked=dict(native_inputs=0,native_windows=0,labels=0,learner_items=0)
        if report['mode']=='bind':
            refs=report['native'];root=Path(refs['native']['path']);cold=Path(refs['cold']['path'])
            authorities=[]
            for label,path in [('native',root),('cold',cold)]:
                need(file_sha(path/'report.json')==refs[label]['pin'],'ANSWER_READBACK_NATIVE_PIN')
                authorities.append(json.loads((path/'report.json').read_bytes()))
            values={};offsets={}
            for n in target.roster:
                pairs=[]
                for path,r in zip((root,cold),authorities):
                    lane=path/n;need(file_sha(lane/'report.json')==r['instruments'][n]['pin'],'ANSWER_READBACK_LANE_PIN')
                    lane_report=json.loads((lane/'report.json').read_bytes())
                    need(file_sha(lane/'reference.npz')==lane_report['files']['reference.npz']['sha256'],
                         'ANSWER_READBACK_ARRAY_PIN')
                    with np.load(lane/'reference.npz',allow_pickle=False) as z:pairs.append((z['vectors'],z['bounds']))
                x,b=pairs[0];y,d=pairs[1]
                expected=np.r_[0,np.cumsum(schedule.plan.tokens[n]['windows'])]
                need(np.array_equal(b,d) and np.array_equal(b,expected) and np.array_equal(x,y),
                     'ANSWER_READBACK_COLD_VALUES')
                values[n],offsets[n]=x,b;checked['native_windows']+=len(x)
            ids=tuple(v['id'] for v in schedule.plan.records);index={v:i for i,v in enumerate(ids)}
            def frame(identity):
                i=index[identity]
                return {n:torch.from_numpy(v[offsets[n][i]:offsets[n][i+1]].copy()) for n,v in values.items()}
            payload=torch.load(a.input/'labels.pt',map_location='cpu',weights_only=True)
            need(tuple(payload['ids'])==ids and fingerprint((payload['codes'],payload['counts']))==payload['seal'],
                 'ANSWER_READBACK_LABEL_PAYLOAD')
            labels=FrozenLabels(target)
            with torch.no_grad():
                for i,identity in enumerate(ids):
                    f=frame(identity);residual=labels.runtime.latent((f,)).numpy().copy();codes=[]
                    for book in labels.runtime.books:
                        b=book.numpy();j=int(np.square(residual[:,None]-b[None]).sum(-1).argmin())
                        codes.append(j);residual-=b[j]
                    need(payload['codes'][i].tolist()==codes and payload['counts'][i].tolist()==[len(f[n]) for n in target.roster],
                         'ANSWER_READBACK_LABEL_ASSIGNMENT')
                    checked['labels']+=1;checked['native_inputs']+=1
            need(payload['binding']==dict(store=digest(canonical(report['native_binding'])),codebook=labels.state,
                 runtime=labels.runtime_state,runtime_policy='canonical-singleton-fp64-with-independent-numpy-argmin')
                 and fingerprint((payload['binding'],ids,payload['codes'],payload['counts']))==payload['pin']==report['label_pin'],
                 'ANSWER_READBACK_LABEL_BINDING')
            records=json.loads((a.input/'items.json').read_bytes())
            need([v['task'] for v in records]==sorted(schedule.rows),'ANSWER_READBACK_ITEM_COVERAGE')
            for receipt in records:
                key=receipt['task'];r=schedule.rows[key];public=schedule.request(key);private=r['private'];targets=private['targets']
                context=tuple(frame(f['input']) for f in public['fields']);answers=tuple(frame(t) for t in targets)
                ix=[index[t] for t in targets];codes=payload['codes'][ix]
                counts={n:payload['counts'][ix,j] for j,n in enumerate(target.roster)};req=NativeRequest(describe(schedule,key))
                provenance=dict(store=payload['binding']['store'],labels=payload['pin'],request=req.pin,observed=len(context),
                    component=r['group'],family=r['family'],operation=public['operation'],position_kind=public['position_kind'],
                    targets=list(targets),end=private['end'],scope='complete_recorded_task_answer')
                expected=dict(task=key,side=r['split'],binding=fingerprint((key,r['split'],provenance)),
                    native=fingerprint((context,answers)),labels=fingerprint((codes,counts)),request=req.pin,
                    observed=len(context),answers=len(answers))
                need(receipt==expected,'ANSWER_READBACK_LEARNER_ITEM');checked['learner_items']+=1
            labels.fresh()
        phases['controls_native_labels_and_items']=time.perf_counter()-tick;tick=time.perf_counter()
        write_committed(a.output/'controls.json',controls)
        for key in schedule.rows:schedule.fresh(key)
        phases['persist_and_final_authentication']=time.perf_counter()-tick
        elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'ANSWER_READBACK_PHASE_COVERAGE')
        out=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),mode=report['mode'],summary=summary,checked=checked,
            all_native_inputs_checked=report['mode']=='bind' and checked['native_inputs']==report['unique_inputs'],
            all_learner_items_checked=checked['learner_items']==489,controls=controls,code=report['code'],
            acceptance='author-checked',scientific_admission=False,full_answer_capability_demonstrated=False,
            phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        pin=write_committed(a.output/'report.json',out);print(json.dumps(dict(pin=pin,mode=report['mode'],checked=checked,seconds=elapsed)))
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=repr(exc),phases=phases));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
