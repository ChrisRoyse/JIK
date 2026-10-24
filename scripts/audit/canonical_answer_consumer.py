"""Publish complete task joins, output-capacity limits and canonical cached labels."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''  # The declared canonical label runtime is CPU fp64.
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_grouped_placement_training as training
from claims.process_conditions import write_committed
from model.answer_training_schedule import AnswerSchedule,decisions
from model.canonical_answer_inputs import CanonicalAnswerInputs,AnswerLabelCache
from model.native_conditioned_answer import check
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest,describe
from model.native_rq_packed import PackedNativeRQ
from model.native_target_cache import FrozenLabels
from text.constellation_corpus import canonical,digest,file_sha,need

ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
PLAN=dict(path=str(ROOT/'answer-native-plan-v1-20261023'),
          pin='22993ed7704f1d31ca48df404c0af14acff1d466624b412a9c55be2132cf1098')
NEW=('model/canonical_answer_inputs.py','audit/canonical_answer_consumer.py',
     'audit/canonical_answer_consumer_readback.py','audit/canonical_answer_consumer_queue.py')


def prepare_source(root):
    schedule=AnswerSchedule(PLAN['path'],PLAN['pin']);source,bundle=training.source_bundle(root)
    target=PackedNativeRQ(bundle['dimensions'],bundle['config'],bundle['sizes'],'no_encoder_quantizer',bundle['statistics'])
    target.load_state_dict(bundle['target'],strict=True);target.eval().requires_grad_(False)
    need(tuple(bundle['dimensions'])==tuple(schedule.plan.report['roster']), 'ANSWER_CONSUMER_ROSTER')
    return schedule,source,bundle,target


def code(source):
    base=Path(__file__).resolve().parents[1]
    names=set(source['code'])|set(NEW)|{'model/answer_training_schedule.py','model/native_conditioned_answer.py',
        'model/native_request_input.py','model/native_requested_answer.py','model/native_request_stream.py',
        'model/native_target_cache.py','model/capability_answer_targets.py','text/answer_native_plan.py',
        'text/answer_native_measure.py'}
    return {n:file_sha(base/n) for n in sorted(names)}


def census(schedule,target):
    plan=schedule.plan;ids={r['id']:i for i,r in enumerate(plan.records)};roster=tuple(target.roster)
    caps={n:len(getattr(target,'position_'+str(i))) for i,n in enumerate(roster)}
    rows=[]
    for key,row in sorted(schedule.rows.items()):
        public=schedule.request(key);private=schedule.supervision(key,fitting=row['split']=='train')
        inputs=[f['input'] for f in public['fields']];targets=private['targets']
        counts={n:[plan.tokens[n]['windows'][ids[i]] for i in targets] for n in roster}
        excessive={n:[j for j,v in enumerate(counts[n]) if v>caps[n]] for n in roster}
        rows.append(dict(task=key,side=row['split'],family=row['family'],group=row['group'],
            operation=public['operation'],contexts=inputs,answers=list(targets),counts=counts,
            ending=private['end'],decisions=private['decisions'],
            request=digest(canonical(describe(schedule,key))),unsupported=excessive))
    summary={}
    for side in ('train','query'):
        subset=[r for r in rows if r['side']==side]
        summary[side]=dict(tasks=len(subset),context_fields=sum(len(r['contexts']) for r in subset),
            answer_atoms=sum(len(r['answers']) for r in subset),end_labels=len(subset),
            decisions=sum(len(r['decisions']) for r in subset),
            unsupported_tasks=sum(any(r['unsupported'].values()) for r in subset),
            instruments={n:dict(max_answer_windows=max(v for r in subset for v in r['counts'][n]),
                unsupported_atoms=sum(len(r['unsupported'][n]) for r in subset),
                unsupported_tasks=sum(bool(r['unsupported'][n]) for r in subset)) for n in roster})
    return dict(rows=rows,summary=summary,count_head_capacity=caps,
        source_native_windows={n:plan.tokens[n]['total_windows'] for n in roster},
        target_codebook='existing_legal_frozen_transfer_baseline',windows_truncated=0)


def reject(reason,fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError) as exc:
        need(reason in str(exc),'ANSWER_CONSUMER_WRONG_REFUSAL');return dict(expected=reason,observed=str(exc))
    raise ValueError('ANSWER_CONSUMER_NEGATIVE_ACCEPTED:'+reason)


def source_controls(schedule):
    train=next(k for k,r in schedule.rows.items() if r['split']=='train')
    query=next(k for k,r in schedule.rows.items() if r['split']=='query')
    out=[reject('ANSWER_SCHEDULE_FIT_SPLIT',lambda:schedule.supervision(query,fitting=True)),
         reject('ANSWER_SCHEDULE_FIT_SPLIT',lambda:schedule.supervision(train,fitting=False))]
    bad=describe(schedule,train);bad['answer_length']=len(schedule.rows[train]['private']['targets'])
    out.append(reject('REQUEST_INPUT_SCHEMA',lambda:NativeRequest(bad)))
    original=deepcopy(schedule.rows[train]);schedule.rows[train]['private']['targets'].pop()
    try:out.append(reject('ANSWER_SCHEDULE_CHANGED',lambda:schedule.request(train)))
    finally:schedule.rows[train]=original
    schedule.fresh(train);return out


def native_controls(store,labels):
    key=next(k for k,r in store.schedule.rows.items() if r['split']=='query')
    before=store.stats['frame_reads']
    out=[reject('ANSWER_SCHEDULE_FIT_SPLIT',lambda:store.item(key,labels,fitting=True))]
    need(store.stats['frame_reads']==before,'ANSWER_CONSUMER_PRIVATE_READ_BEFORE_SPLIT')
    identity=next(iter(store.index));name=store.roster[0]
    direct=store.frame(identity);original=fingerprint(direct);direct[name].zero_()
    need(fingerprint(store.frame(identity))==original,'ANSWER_CONSUMER_MUTABLE_ALIAS')
    out.append(dict(expected='RETURNED_FRAME_OWNED',observed='caller mutation leaves resident native bytes exact'))
    a=store.arrays[name];saved=a[0,0].copy();a.flags.writeable=True;a[0,0]+=1
    try:out.append(reject('TASK_NATIVE_CACHE_CHANGED',lambda:store.frame(identity)))
    finally:a[0,0]=saved;a.flags.writeable=False
    value=labels.codes[0,0].item();labels.codes[0,0]+=1
    try:out.append(reject('TASK_NATIVE_LABEL_CACHE_CHANGED',lambda:labels.get([identity])))
    finally:labels.codes[0,0]=value;labels.versions=(labels.codes._version,labels.counts._version)
    need(fingerprint((labels.codes,labels.counts))==labels.seal,'ANSWER_CONSUMER_CONTROL_RESTORE')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu0=time.process_time();phases={}
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        schedule,source,bundle,target=prepare_source(a.root);pins=code(source)
        (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter()
        counted=census(schedule,target);controls=source_controls(schedule)
        write_committed(a.output/'census.json',counted)
        refs={};cache=None;items=[];store=None
        phases['complete_schedule_and_capacity']=time.perf_counter()-tick;tick=time.perf_counter()
        if a.mode=='bind':
            need(all(v is not None for v in (a.native,a.native_pin,a.cold,a.cold_pin)), 'ANSWER_CONSUMER_NATIVE_REQUIRED')
            store=CanonicalAnswerInputs(schedule,a.native,a.native_pin,a.cold,a.cold_pin)
            refs=dict(native=dict(path=str(a.native),pin=a.native_pin),cold=dict(path=str(a.cold),pin=a.cold_pin))
            phases['resident_native_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
            cache=AnswerLabelCache(store,FrozenLabels(target))
            training.runtime.persist(a.output/'labels.pt',dict(ids=cache.ids,codes=cache.codes,counts=cache.counts,
                binding=cache.binding,pin=cache.pin,seal=cache.seal))
            phases['canonical_unique_labels']=time.perf_counter()-tick;tick=time.perf_counter()
            for key,row in sorted(schedule.rows.items()):
                item,request=store.item(key,cache,fitting=row['split']=='train')
                if row['split']=='train':need(check(item)==row['private']['decisions'],'ANSWER_CONSUMER_TRAIN_SCHEDULE')
                else:
                    need(decisions(len(item['context']),len(item['answers']))==row['private']['decisions'],
                         'ANSWER_CONSUMER_QUERY_SCHEDULE')
                need(request.pin==item['provenance']['request'],'ANSWER_CONSUMER_REQUEST')
                items.append(dict(task=key,side=row['split'],binding=item['binding'],native=item['native_pin'],
                    labels=item['label_pin'],request=request.pin,observed=len(item['context']),answers=len(item['answers'])))
            controls+=native_controls(store,cache);store.fresh();cache.fresh()
            phases['learner_items_and_controls']=time.perf_counter()-tick;tick=time.perf_counter()
            write_committed(a.output/'items.json',items)
        else:
            # Exercise the production consumer's missing/pin refusal; no stand-in native arrays.
            missing=a.output/'unpublished-native'
            controls.append(reject('TASK_NATIVE_REPORT_PIN',lambda:CanonicalAnswerInputs(schedule,missing,'0'*64,missing,'1'*64)))
        write_committed(a.output/'controls.json',controls)
        for key in schedule.rows:schedule.fresh(key)
        need(code(source)==pins,'ANSWER_CONSUMER_CODE_CHANGED')
        phases['publication_and_final_authentication']=time.perf_counter()-tick
        elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'ANSWER_CONSUMER_PHASE_COVERAGE')
        names=['census.json','controls.json','nvidia-before.txt']+(['labels.pt','items.json'] if cache else [])
        result=dict(schema='canonical-answer-consumer-v1',mode=a.mode,complete=True,plan=PLAN,
            target=dict(source=training.SOURCE[1],target=bundle['target_ref']),code=pins,
            native=refs,summary=counted['summary'],count_head_capacity=counted['count_head_capacity'],
            unique_inputs=len(schedule.plan.records),native_binding=None if store is None else store.binding,
            label_pin=None if cache is None else cache.pin,stats=None if store is None else store.stats,
            experimental_items_ready=cache is not None,training_updates=0,scientific_admission=False,
            full_answer_capability_demonstrated=False,acceptance='author-checked',phases=phases,seconds=elapsed,
            cpu_seconds=time.process_time()-cpu0,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            files={n:dict(sha256=file_sha(a.output/n),bytes=(a.output/n).stat().st_size) for n in names})
        pin=write_committed(a.output/'report.json',result)
        print(json.dumps(dict(pin=pin,mode=a.mode,summary=result['summary'],seconds=elapsed)),flush=True)
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=repr(exc),phases=phases));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['plan','bind'],required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    for n in ('native','cold'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    run(p.parse_args())
