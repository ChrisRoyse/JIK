"""Attribute the real legal inversion input path before changing its execution."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_inversion as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need

PLAN='legal-inversion-plan-v3-20261024'
PIN='2d548670cf1b40f859262c3789d1defefc9a977649cbbdeeff4fdd481b91668b'


def payload(examples):
    return tuple(dict(occurrence=e.occurrence,uid=e.uid,side=e.side,raw=torch.tensor(list(e.raw),dtype=torch.uint8),
        native={n:dict(identity=x.identity,rows=x.rows,values=x.values,padding=x.padding,
                       absent=x.absent,indices=x.indices) for n,x in e.native.items()}) for e in examples)


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={};state=None
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        state=source.sources(a.root);p=source.authority.pinned(a.root/PLAN,PIN)
        need(p['binding']==state['adapter'].binding and p['code']==state['code'],'INPUT_PROFILE_PLAN')
        train=np.load(a.root/PLAN/'train.npy',mmap_mode='r',allow_pickle=False)
        need(len(train)==p['train']['atoms'],'INPUT_PROFILE_POPULATION')
        starts=[0,128,1605632,1605760,(len(train)//256)*128,(len(train)//256)*128+128,len(train)-256,len(train)-128]
        code=dict(state['code']);code['audit/legal_surface_input_profile.py']=file_sha(__file__)
        phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter();rows=[];pins={}
        for repeat in range(3):
            for start in starts:
                ids=train[start:start+128].tolist();wall=time.perf_counter();cpu=time.process_time()
                examples=state['adapter'].fetch(ids,side='train',fit=True)
                row=dict(start=start,repeat=repeat,seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                         atoms=len(examples),bytes=sum(len(e.raw) for e in examples),
                         windows=sum(x.values.shape[1] for e in examples for x in e.native.values()))
                value=payload(examples);pin=fingerprint(value)
                if repeat==0:
                    name=f'tile-{start}.pt';persist(a.output/name,value);pins[start]=pin
                    need(fingerprint(torch.load(a.output/name,map_location='cpu',weights_only=True))==pin,'INPUT_PROFILE_READBACK')
                need(pin==pins[start],'INPUT_PROFILE_WARM_BYTES');rows.append(row)
                print(json.dumps(row),flush=True)
        phases['unprofiled_fetch_and_persistence']=time.perf_counter()-tick;tick=time.perf_counter()
        profiler=cProfile.Profile();profiled=[]
        for start in starts:
            ids=train[start:start+128].tolist();profiler.enable()
            examples=state['adapter'].fetch(ids,side='train',fit=True)
            profiler.disable();need(fingerprint(payload(examples))==pins[start],'INPUT_PROFILE_INSTRUMENTED_BYTES')
        profiler.dump_stats(str(a.output/'input.prof'));stats=pstats.Stats(profiler)
        for (path,line,name),(primitive,calls,own,cumulative,callers) in stats.stats.items():
            profiled.append(dict(path=path,line=line,name=name,primitive_calls=primitive,calls=calls,
                                 self_seconds=own,cumulative_seconds=cumulative))
        write_committed(a.output/'calls.json',sorted(profiled,key=lambda x:-x['cumulative_seconds']))
        phases['profile_and_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        counts=lambda:(state['reader'].stats['calls'],state['surfaces'].stats['calls'])
        before=counts();held=int(state['data'].documents[p['evaluation']['query']['documents'][0],0])
        controls=dict(held_out=rejected(lambda:state['adapter'].fetch([held],side='train',fit=True),'LEGAL_LEARNING_CROSS_SPLIT'),
            empty=rejected(lambda:state['adapter'].fetch([],side='train',fit=True),'LEGAL_LEARNING_OCCURRENCES'),
            physical_tile=rejected(lambda:state['adapter'].fetch(train[:129].tolist(),side='train',fit=True),'LEGAL_SURFACE_MEMORY_TILE'))
        need(counts()==before,'INPUT_PROFILE_EARLY_REFUSAL')
        counters=dict(native=dict(state['reader'].stats),handles=dict(state['reader'].io_stats),surface=dict(state['surfaces'].stats))
        source.close(state);need(not torch.cuda.is_initialized(),'INPUT_PROFILE_GPU')
        base=Path(__file__).resolve().parents[1]
        need(all(file_sha(base/n)==h for n,h in code.items()),'INPUT_PROFILE_CODE')
        phases['controls_and_closure']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'INPUT_PROFILE_PHASE_COVERAGE')
        files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
        report=dict(complete=True,plan=PIN,code=code,files=files,train=p['train'],starts=starts,
            rows=rows,payload_pins=pins,controls=controls,counters=counters,phases=phases,seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            top_calls=sorted(profiled,key=lambda x:-x['cumulative_seconds'])[:25],
            gpu_used=False,training_updates=0,acceptance='author-checked',optimized=False)
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)
    finally:
        if state is not None:
            if not state['reader'].closed:state['reader'].close()
            if not state['surfaces'].closed:state['surfaces'].close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists():write_committed(args.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
