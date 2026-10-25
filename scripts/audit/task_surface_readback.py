"""Recount persisted task scores from original bytes and replay decoder forwards."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import task_surface_scoring as producer
from claims.process_conditions import write_committed as write
from model.answer_surface_input import predicted_atom
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from model.stream_surface import batches,EOS
from model.task_surface_scoring import score_trace,paired,native_pin
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import digest,file_sha,need


def recount(saved,raw):
    need(saved['raw_sha256']==digest(raw) and saved['raw_bytes']==len(raw)
         and saved['logits'].shape==(len(raw)+1,258) and bool(saved['logits'].isfinite().all()),
         'TASK_SURFACE_READBACK_POSITIONS')
    targets=torch.tensor([*raw,EOS],dtype=torch.int64)
    need(torch.equal(saved['targets'],targets),'TASK_SURFACE_READBACK_TARGET')
    scores=saved['logits'].double();losses=F.cross_entropy(scores,targets,reduction='none')
    need(torch.equal(losses,saved['losses']) and torch.equal(scores.argmax(-1)==targets,saved['correct']),
         'TASK_SURFACE_READBACK_LOSS')
    return dict(positions=len(raw)+1,correct=int(saved['correct'].sum()),nll=float(losses.sum()))


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False)
    r=producer.contract.read(a.input/'report.json',a.pin)
    need(r['complete'] and r['teacher_forced'] and not r['free_generation_tested'] and not r['gpu_used'],
         'TASK_SURFACE_READBACK_CPU_SCOPE')
    for n,h in r['code'].items():
        if (producer.BASE/n).resolve()!=Path(__file__).resolve():
            need(file_sha(producer.BASE/n)==h,'TASK_SURFACE_READBACK_CODE')
    m,generated,cases,binding=producer.authenticate(Path(r['request']['path']),r['request']['pin'],
        Path(r['generated']['path']),r['generated']['pin'],[0,1,3])
    schedule=producer.AnswerSchedule(binding['plan']['path'],binding['plan']['pin'])
    texts={x['id']:x['text'] for x in schedule.plan.records}
    args=[]
    for path,pin in producer.DECODER:args.extend((producer.P/path,pin))
    decoder=TrainedSurfaceDecoder(*args,device='cpu',cpu_diagnostic=True)
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    trajectories=events=atoms=positions=correct=0;negative_sample=None;rows=[]
    try:
        need(decoder.receipt==r['decoder'] and len(cases)==len(r['results'])==3,'TASK_SURFACE_READBACK_DECODER')
        for (path,case),ref in zip(cases,r['results'],strict=True):
            root=a.input/ref['path'];report=producer.contract.read(root/'report.json',ref['pin'])
            need(report['case']==case['case'] and report['unavailable']==case['unavailable']
                 and len(report['arms'])==len(case['traces']),'TASK_SURFACE_READBACK_CASE')
            references=producer.references(schedule,case['case']['task'],texts)
            for arm,trace_ref in zip(report['arms'],case['traces'],strict=True):
                need(all(arm[k]==trace_ref[k] for k in ('architecture','mode','proposal')),
                     'TASK_SURFACE_READBACK_GRID')
                stored=producer.contract.read(root/arm['path']/'report.json',arm['pin'])
                need(all(arm[k]==v for k,v in stored.items()) and arm['trace']==trace_ref['pin'],
                     'TASK_SURFACE_READBACK_REPORT')
                n=0;seen=0;matched=0;nlls=[];hits=0;ended=False
                for event in replay(path/trace_ref['path'],trace_ref['pin']):
                    events+=1;seen+=1
                    if event['kind']=='end':ended=True;continue
                    native=predicted_atom(event,decoder.specs,decoder.config,region_pin=arm['region_pin'],
                        region_sizes=arm['region_sizes'],trace_pin=trace_ref['pin'],position=n).slots
                    if n<len(references):
                        raw=references[n]['raw'];item=arm['atoms'][n];p=root/arm['path']/item['path']
                        need(file_sha(p)==item['pin'],'TASK_SURFACE_READBACK_BYTES')
                        saved=torch.load(p,map_location='cpu',weights_only=True);count=recount(saved,raw)
                        need(saved['native']==native_pin(native)==item['native'] and item['reference']==references[n]['id']
                             and all(item[k]==v for k,v in count.items()),'TASK_SURFACE_READBACK_NATIVE')
                        coverage=[]
                        with torch.no_grad():
                            for prefix,labels,offsets,slots,items in batches(raw,native,decoder.config):
                                logits=decoder.model(prefix,slots,offsets=offsets)['logits']
                                for i,s in enumerate(items):
                                    lo,hi=s['scored'];left=lo-s['offset']
                                    need(torch.equal(logits[i,left:left+hi-lo],saved['logits'][lo:hi]),
                                         'TASK_SURFACE_READBACK_FORWARD')
                                    coverage.extend(range(lo,hi))
                        need(coverage==list(range(len(raw)+1)),'TASK_SURFACE_READBACK_COVERAGE')
                        atoms+=1;positions+=len(raw)+1;correct+=count['correct'];hits+=count['correct']
                        matched+=len(raw)+1;nlls.append(count['nll'])
                        if negative_sample is None:negative_sample=(saved,raw,path/trace_ref['path'],trace_ref,arm,references)
                    n+=1
                total=sum(len(v['raw'])+1 for v in references)
                need(arm['generated_atoms']==n and arm['events']==seen and arm['model_ended']==ended
                     and len(arm['atoms'])==min(n,len(references)) and arm['matched_positions']==matched
                     and arm['missing_positions']==total-matched and arm['reference_positions']==total
                     and arm['extra_atoms']==max(0,n-len(references)) and arm['missing_atoms']==max(0,len(references)-n)
                     and arm['full_reference_accuracy']==hits/total and arm['matched_nll_sum']==sum(nlls),
                     'TASK_SURFACE_READBACK_DENOMINATOR')
                trajectories+=1
            lookup={(v['architecture'],v['proposal'],v['mode']):v for v in report['arms']}
            for pair in report['paired']:
                left=lookup[pair['architecture'],pair['proposal'],'correct']
                right=lookup.get((pair['architecture'],pair['proposal'],pair['control']))
                need(pair['available']==(right is not None) and pair['scores']==(None if right is None else paired(left,right,root)),
                     'TASK_SURFACE_READBACK_PAIR')
            rows.append(dict(ordinal=case['case']['ordinal'],paired=report['paired']))
        phases['all_forward_and_score_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        controls={}
        def reject(name,fn,code):
            try:fn()
            except (ValueError,RuntimeError) as exc:need(code in str(exc),'TASK_SURFACE_READBACK_CONTROL_REASON');controls[name]=str(exc)
            else:raise ValueError('TASK_SURFACE_READBACK_CONTROL_PASSED:'+name)
        saved,raw,path,ref,arm,refs=negative_sample
        bad=deepcopy(saved);bad['logits']=bad['logits'][:-1]
        reject('missing_eos_position',lambda:recount(bad,raw),'TASK_SURFACE_READBACK_POSITIONS')
        bad=deepcopy(saved);bad['targets'][0]=(int(bad['targets'][0])+1)%256
        reject('wrong_reference_byte',lambda:recount(bad,raw),'TASK_SURFACE_READBACK_TARGET')
        kw=dict(region_pin=arm['region_pin'],region_sizes=arm['region_sizes'],expected_request=arm['trace_request'])
        reject('wrong_trace_request',lambda:score_trace(path,ref['pin'],decoder.model,refs,
            output=a.output/'wrong-request',**dict(kw,expected_request='0'*64)),'TASK_SURFACE_TRACE_REQUEST')
        altered=deepcopy(refs);altered[0]['id']='0'*64
        reject('wrong_reference_binding',lambda:score_trace(path,ref['pin'],decoder.model,altered,
            output=a.output/'wrong-reference',**kw),'TASK_SURFACE_REFERENCE_PIN')
        original=decoder.model.specs
        try:
            decoder.model.specs=(replace(original[0],identity='0'*64),*original[1:])
            reject('wrong_instrument_identity',lambda:score_trace(path,ref['pin'],decoder.model,refs,
                output=a.output/'wrong-lane',**kw),'MNLM_MODEL_IDENTITY')
        finally:decoder.model.specs=original
        decoder.fresh();need(not torch.cuda.is_initialized(),'TASK_SURFACE_READBACK_GPU')
        phases['controls_and_seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_SURFACE_READBACK_PHASES')
        result=dict(complete=True,input=a.pin,readback_code=file_sha(Path(__file__)),producer_code=r['code'],
            trajectories=trajectories,events=events,matched_atoms=atoms,
            positions=positions,correct=correct,controls=controls,paired=rows,all_forward_logits_exact=True,
            all_reference_bytes_and_eos_recounted=True,all_missing_extra_denominators_recounted=True,
            all_scores_exact=True,decoder_unchanged=True,free_generation_tested=False,capability_demonstrated=False,
            phases=phases,seconds=seconds,acceptance='author-checked')
        pin=write(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,trajectories=trajectories,
            matched_atoms=atoms,positions=positions,correct=correct,seconds=seconds)),flush=True)
    finally:decoder.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
