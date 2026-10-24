"""Measure missing answer/context inputs on CUDA; cold replay changes cohort order."""
import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_native_readback import token_census
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens,weights_sha256,preprocessing_sha256
from lenscommon import take_lease,release_lease
from text.answer_native_plan import AnswerNativePlan,code,read
from text.canonical_windows import contract,forward,exact_reference
from text.constellation_corpus import canonical,digest,file_sha,need
from text.legal_native_store import durable_directory


def arrays(root,report):
    for n,item in report['files'].items():
        need((root/n).stat().st_size==item['bytes'] and file_sha(root/n)==item['sha256'],'ANSWER_NATIVE_BYTES')
    with np.load(root/'reference.npz',allow_pickle=False) as z: values=z['vectors'].copy();bounds=z['bounds'].copy()
    with gzip.open(root/'traces.jsonl.gz','rt',encoding='utf-8') as f: traces=[json.loads(line) for line in f]
    return values,bounds,traces


def lane(plan,name,dest,reference):
    dest.mkdir();started=tick=time.perf_counter();r=plan.report
    lens=LearnedLens(by_name(name),'cuda');identity=contract(lens);lens_id=digest(canonical(identity))
    need(identity==r['contracts'][name] and not lens.model.training and
         all(not p.requires_grad for p in lens.model.parameters()),'ANSWER_NATIVE_INSTRUMENT')
    torch.cuda.reset_peak_memory_stats();counts=plan.tokens[name]['windows'];bounds=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    # Physical resident-output allowance, independent of total work; future larger populations require tiling.
    bytes_needed=int(bounds[-1])*lens.dim*4+bounds.nbytes+r['input_bytes']*64
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    need(bytes_needed<=available//4,'ANSWER_NATIVE_RESIDENT_MEMORY_TILE_REQUIRED')
    values=np.empty((int(bounds[-1]),lens.dim),dtype=np.float32);traces=[None]*len(plan.records)
    if reference is None:
        if r['reused']:
            with np.load(plan.root/name/'reused.npz',allow_pickle=False) as z: old=z['vectors'];o=z['offsets']
            old_traces=json.loads((plan.root/name/'traces.json').read_bytes())
            for j,item in enumerate(r['reused']):
                i=item['index'];values[bounds[i]:bounds[i+1]]=old[o[j]:o[j+1]];traces[i]=old_traces[j]
        order=r['missing'];cohort=32
    else: order=list(reversed(range(len(plan.records))));cohort=8
    phases=dict(load_allocate_and_reuse=time.perf_counter()-tick);tick=time.perf_counter();timings=[]
    for start in range(0,len(order),cohort):
        selected=order[start:start+cohort];v,b,t,perf=forward(lens,[plan.records[i]['text'] for i in selected])
        for j,i in enumerate(selected):
            need(int(b[j+1]-b[j])==counts[i],'ANSWER_NATIVE_WINDOW_COUNT')
            values[bounds[i]:bounds[i+1]]=v[b[j]:b[j+1]];traces[i]=t[j]
        timings.append(perf)
    phases['forward']=time.perf_counter()-tick;tick=time.perf_counter()
    need(all(t is not None for t in traces) and np.isfinite(values).all(),'ANSWER_NATIVE_COVERAGE')
    measured_tokens=0
    for record,trace,n in zip(plan.records,traces,counts):
        need(token_census(lens,record['text'],trace)==n,'ANSWER_NATIVE_TOKEN_CENSUS')
        measured_tokens+=sum(sum(w['attention_mask']) for w in trace['windows'])
    need(measured_tokens==plan.tokens[name]['total_tokens'],'ANSWER_NATIVE_ACTIVE_TOKENS')
    controls={};probes=sorted({0,max(range(len(plan.records)),key=lambda i:(counts[i],plan.records[i]['id'])),
                             *[v['index'] for v in r['reused']]})
    for label,selection in [('mixed',probes),('reversed',list(reversed(probes))),('duplicated',probes+probes)]:
        v,b,t,_=forward(lens,[plan.records[i]['text'] for i in selection])
        controls[label]=exact_reference(values,bounds,traces,v,b,t,selection)
    for i in probes:
        v,b,t,_=forward(lens,[plan.records[i]['text']]);exact_reference(values,bounds,traces,v,b,t,[i])
    controls['singletons']=True
    if reference is not None:
        old=read(reference/name/'report.json',json.loads((reference/'report.json').read_bytes())['instruments'][name]['pin'])
        v,b,t=arrays(reference/name,old)
        controls['all_cold_inputs_exact']=exact_reference(v,b,t,values,bounds,traces,list(range(len(plan.records))))
        need(old['lens_id']==lens_id,'ANSWER_NATIVE_COLD_LENS')
    need(weights_sha256(lens.snap)[0]==lens.sha256 and preprocessing_sha256(lens.snap)==lens.preprocessing,
         'ANSWER_NATIVE_WEIGHTS_CHANGED')
    phases['token_identity_and_cohort_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    np.savez(dest/'reference.npz',vectors=values,bounds=bounds)
    with gzip.open(dest/'traces.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
        for trace in traces:f.write(json.dumps(trace,separators=(',',':'))+'\n')
    for n in ('reference.npz','traces.jsonl.gz'):
        with (dest/n).open('rb') as f:os.fsync(f.fileno())
    files={n:dict(sha256=file_sha(dest/n),bytes=(dest/n).stat().st_size) for n in ('reference.npz','traces.jsonl.gz')}
    phases['persist']=time.perf_counter()-tick;plan.fresh()
    result=dict(name=name,lens_id=lens_id,contract=identity,inputs=len(plan.records),windows=len(values),
        active_tokens=measured_tokens,all_windows_retained=True,measured_inputs=len(order),
        reused_inputs=len(r['reused']) if reference is None else 0,files=files,controls=controls,
        control_inputs=probes,cohort=cohort,timings=timings,phases=phases,seconds=time.perf_counter()-started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
        frozen_unchanged=True,scientific_admission=False)
    pin=write_committed(dest/'report.json',result);durable_directory(dest)
    del lens;torch.cuda.empty_cache();print(json.dumps(dict(name=name,pin=pin,windows=len(values))),flush=True)
    return dict(pin=pin,lens_id=lens_id)


def run(a):
    a.output.mkdir(exist_ok=False);started=time.perf_counter();leased=False;results={};errors=[]
    try:
        plan=AnswerNativePlan(a.plan,a.pin);need(plan.report['code']==code(),'ANSWER_NATIVE_CODE_CHANGED')
        reference=None
        if a.reference:
            previous=read(a.reference/'report.json',a.reference_pin)
            need(previous['complete'] and previous['plan']['pin']==a.pin and previous['code']==code(), 'ANSWER_NATIVE_COLD_PARENT')
            reference=a.reference
        need(torch.cuda.is_available(),'ANSWER_NATIVE_GPU_REQUIRED')
        resident=subprocess.check_output(['nvidia-smi'],text=True)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        need(not active.strip(),'ANSWER_NATIVE_GPU_BUSY');take_lease('29','answer-canonical-native-measurement');leased=True
        (a.output/'gpu-before.txt').write_text(resident)
        torch.set_num_threads(32);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for name in plan.report['roster']:
            try:results[name]=lane(plan,name,a.output/name,reference)
            except Exception as exc:
                failure=dict(name=name,error=repr(exc));errors.append(failure)
                write_committed(a.output/(name+'-failure.json'),failure);torch.cuda.empty_cache()
        plan.fresh()
        report=dict(schema='answer-canonical-native-v1',complete=not errors,errors=errors,plan=dict(path=str(a.plan),pin=a.pin),
            code=code(),instruments=results,roster=plan.report['roster'],inputs=len(plan.records),
            cold_reference=None if reference is None else dict(path=str(reference),pin=a.reference_pin),
            cold_all_inputs_exact=reference is not None and not errors,training_ready=False,scientific_admission=False,
            acceptance='author-checked',seconds=time.perf_counter()-started,
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,complete=not errors)),flush=True)
        need(not errors,'ANSWER_NATIVE_MEASUREMENT_REFUSED')
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=repr(exc),results=results));raise
    finally:
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--reference',type=Path);p.add_argument('--reference-pin')
    run(p.parse_args())
