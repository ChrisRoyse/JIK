"""Independent persisted task-transfer census, cold trajectories and score accounting."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.task_answer_transfer import read,rejected,checked_request
from claims.process_conditions import write_committed as write
from model import task_answer_transfer as transfer
from model.task_answer_public_pack import TaskPublicPack
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.constellation_corpus import canonical,digest,file_sha,need


def census(pack,plan):
    queries=[r for r in pack.cases if r['split']=='query'];trains=[r for r in pack.cases if r['split']=='train']
    need([r['task'] for r in plan['rows']]==[r['task'] for r in queries],'TASK_TRANSFER_READBACK_QUERY_UNION')
    def signature(r):
        d=r['descriptor'];windows=[]
        for f in r['fields']:
            j=pack.index[f['input']];windows.append([int(pack.offsets[n][j+1])-int(pack.offsets[n][j]) for n in pack.roster])
        return dict(family=r['family'],operation=d['operation'],position_kind=d['position_kind'],
            fields=[{k:v for k,v in f.items() if k!='clock'} for f in d['fields']],windows=windows)
    train_signatures={r['task']:signature(r) for r in trains};matched=0
    for ordinal,(row,query) in enumerate(zip(plan['rows'],queries,strict=True)):
        s=signature(query);candidates=[]
        for train in trains:
            if train['group']==query['group'] or signature_inputs(train)==signature_inputs(query):continue
            if train_signatures[train['task']]==s:candidates.append(train)
        candidates.sort(key=lambda t:digest(canonical(['task-transfer-donor-v1',query['task'],t['task']])))
        donor=dict(task=candidates[0]['task'],public=candidates[0]['pin']) if candidates else None
        need(row['ordinal']==ordinal and row['public']==query['pin'] and row['group']==query['group']
             and row['structure']==digest(canonical(s)) and row['donor']==donor
             and row['donor_candidates']==len(candidates),'TASK_TRANSFER_READBACK_DONOR')
        matched+=donor is not None
    return dict(queries=len(queries),train=len(trains),matched=matched,unavailable=len(queries)-matched,
                query_union_exact=True,train_only_different_group=True,hash_selection_exact=True)


def signature_inputs(row):return tuple(f['input'] for f in row['fields'])


def metric(events,future,roster,refusal):
    # Recount directly from persisted emitted events and original reference arrays.
    atoms=[e for e in events if e['kind']=='constellation'];ended=bool(events and events[-1]['kind']=='end')
    count=len(future);per={n:[-1.]*count for n in roster}
    for j,event in enumerate(atoms[:count]):
        for n in roster:
            x=np.asarray(event['slots'][n],dtype=np.float64);y=np.asarray(future[j][n],dtype=np.float64)
            x=x/np.linalg.norm(x,axis=1,keepdims=True);y=y/np.linalg.norm(y,axis=1,keepdims=True)
            matrix=np.clip(x@y.T,-1.,1.)
            per[n][j]=float(min(np.mean(np.max(matrix,axis=0)),np.mean(np.max(matrix,axis=1))))
    extra=max(0,len(atoms)-count)
    return dict(expected_atoms=count,generated_atoms=len(atoms),ended=ended,ended_at_reference_boundary=ended and len(atoms)==count,
        censored=not ended,matched_positions=min(len(atoms),count),missing=max(0,count-len(atoms)),extra=extra,
        per_slot=per,full_output_mean={n:(sum(per[n])-extra)/(count+extra) for n in roster},missing_similarity=-1.,extra_similarity=-1.,
        refusal=refusal,reference_scored_after_generation=True,semantic_validity_established=False)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);m=checked_request(a)
    left=read(a.generated,a.generated_pin);right=read(a.cold,a.cold_pin);score=read(a.scored,a.scored_pin)
    need(left['complete'] and right['complete'] and left['request']==right['request']==score['request']==a.pin
         and score['generation']==a.generated_pin,'TASK_TRANSFER_READBACK_BINDING')
    l=read(left['run']['path'],left['run']['pin']);r=read(right['run']['path'],right['run']['pin'])
    need(l['binding']==r['binding'] and l['binding']['ordinals']==m['ordinals']
         and [v['ordinal'] for v in l['results']]==[v['ordinal'] for v in r['results']]==m['ordinals'],
         'TASK_TRANSFER_READBACK_COHORT')
    pack=TaskPublicPack(m['pack']['path'],m['pack']['pin'])
    try:
        counted=census(pack,m['plan']);phases=dict(authenticate_census=time.perf_counter()-tick);tick=time.perf_counter()
        from audit.task_answer_public_pack import source
        schedule,store,binding=source();need(binding==pack.report['binding']==score['private_binding'],'TASK_TRANSFER_READBACK_SOURCE')
        phases['private_source']=time.perf_counter()-tick;tick=time.perf_counter()
        events=trajectories=ends=refusals=0;cases=[];first=None
        lr=Path(left['run']['path']).parent;rr=Path(right['run']['path']).parent
        for x,y,s in zip(l['results'],r['results'],score['cases'],strict=True):
            need(x['ordinal']==y['ordinal']==s['ordinal'],'TASK_TRANSFER_READBACK_ORDINAL')
            row=m['plan']['rows'][x['ordinal']];_,_,public,unavailable=transfer.inputs(pack,row)
            kw=dict(row=row,plan_pin=m['plan_pin'],identities=l['binding']['checkpoints'],expected=l['binding']['expected'],
                public=public,unavailable=unavailable,target_state=l['binding']['frozen'],caches=l['binding']['cache_bytes'],
                authority=dict(m['authority'],plan=m['plan_pin'],task=row['task'],public=row['public'],ordinal=row['ordinal']))
            p=lr/x['path'];q=rr/y['path']
            c=transfer.check_case(p,x['pin'],**kw);d=transfer.check_case(q,y['pin'],**kw)
            scored=read(a.scored.parent/s['path'],s['pin']);private=schedule.supervision(row['task'],fitting=False)
            # The reference reads occur after both generations and their complete grids have sealed.
            future=tuple(store.frame(i) for i in private['targets'])
            need(scored['task']==row['task'] and scored['reference_atoms']==len(future),'TASK_TRANSFER_READBACK_REFERENCE')
            lookup={(v['architecture'],v['proposal'],v['mode']):v for v in scored['arms']}
            for u,v in zip(c['traces'],d['traces'],strict=True):
                need(all(u[k]==v[k] for k in ('architecture','proposal','mode')),'TASK_TRANSFER_READBACK_ARM')
                aa=list(replay(p/u['path'],u['pin']));bb=list(replay(q/v['path'],v['pin']))
                need(fingerprint(aa)==fingerprint(bb),'TASK_TRANSFER_COLD_EVENTS')
                tr=read(p/u['path']/'report.json',u['pin'])
                cold_trace=read(q/v['path']/'report.json',v['pin'])
                need(all(tr[k]==cold_trace[k] for k in ('status','refusal','generated_atoms','events','model_ended')),
                     'TASK_TRANSFER_COLD_TERMINAL')
                key=u['architecture'],u['proposal'],u['mode'];measured=metric(aa,future,pack.roster,tr['refusal'])
                need(lookup[key]['metrics']==measured,'TASK_TRANSFER_READBACK_METRIC')
                events+=len(aa);trajectories+=1;ends+=measured['ended'];refusals+=measured['censored']
            for pair in scored['paired']:
                key=pair['architecture'],pair['proposal'];correct=lookup[*key,'correct']['metrics'];control=lookup.get((*key,pair['control']))
                expected=None if control is None else {n:correct['full_output_mean'][n]-control['metrics']['full_output_mean'][n] for n in pack.roster}
                need(pair['available']==(control is not None) and pair['per_slot']==expected,'TASK_TRANSFER_READBACK_PAIR')
            cases.append(dict(ordinal=row['ordinal'],family=row['family'],traces=len(c['traces']),reference_atoms=len(future),unavailable=unavailable))
            if first is None:first=(p,c,kw)
        phases['cold_and_score_recount']=time.perf_counter()-tick;tick=time.perf_counter()
        controls={};bad=deepcopy(m['plan']);bad['rows'].pop()
        controls['omitted_query']=rejected(lambda:census(pack,bad))
        row=deepcopy(next(v for v in m['plan']['rows'] if v['donor'] is not None));other=next(v for v in pack.cases if v['split']=='query')
        row['donor']=dict(task=other['task'],public=other['pin'])
        controls['query_donor']=rejected(lambda:transfer.inputs(pack,row))
        p,c,kw=first
        for fault in ('missing_proposal','wrong_public','wrong_target','wrong_cache'):
            dest=a.output/fault;dest.mkdir();value=deepcopy(c)
            for ref in c['traces']:(dest/ref['path']).symlink_to(os.path.relpath(p/ref['path'],dest),target_is_directory=True)
            if fault=='missing_proposal':value['traces'].pop()
            elif fault=='wrong_public':value['public']['correct']['prefix']='0'*64
            elif fault=='wrong_target':value['target_state']='0'*64
            else:value['cache_bytes']['shared']+=1
            pin=write(dest/'report.json',value)
            controls[fault]=rejected(lambda:transfer.check_case(dest,pin,**kw))
        src=p/c['traces'][0]['path'];dest=a.output/'altered_event';dest.mkdir()
        for name in ('report.json','request.json','index.jsonl'):(dest/name).symlink_to(os.path.relpath(src/name,dest))
        shutil.copyfile(src/'events.bin',dest/'events.bin')
        with (dest/'events.bin').open('r+b') as f:
            byte=f.read(1);f.seek(0);f.write(bytes([(byte[0] if byte else 0)^1]));f.flush();os.fsync(f.fileno())
        controls['altered_event']=rejected(lambda:list(replay(dest,c['traces'][0]['pin'])))
        write(a.output/'controls.json',controls);store.fresh();pack.close();pack=None
        phases['controls']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_TRANSFER_READBACK_PHASES')
        value=dict(complete=True,request=a.pin,producer=a.generated_pin,cold=a.cold_pin,scored=a.scored_pin,
            census=counted,cases=cases,trajectories=trajectories,events=events,ends=ends,physical_refusals=refusals,
            all_cold_event_states_exact=True,all_reference_metrics_exact=True,controls=controls,phases=phases,seconds=seconds,
            model_loads=0,encoder_calls=0,optimizer_updates=0,gpu_used=False,acceptance='author-checked',
            full_population_transfer_measured=False,capability_demonstrated=False,architecture_winner=False,
            readback_code=file_sha(Path(__file__)))
        pin=write(a.output/'report.json',value);print(json.dumps(dict(complete=True,pin=pin,cases=len(cases),trajectories=trajectories,
            events=events,ends=ends,refusals=refusals,controls=len(controls),seconds=seconds)),flush=True)
    finally:
        if pack is not None:pack.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','generated','cold','scored','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pin','generated-pin','cold-pin','scored-pin'):p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
