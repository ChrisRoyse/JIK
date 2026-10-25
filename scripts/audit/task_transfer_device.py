"""Resident declared-device task generation and a separate CPU reference scorer."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import task_transfer_contract as contract
from audit.complete_answer_evaluation import artifact,CACHE
from audit.complete_answer_device_cohort import access_guard,gpu_ready,validate
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model import task_answer_transfer as transfer
from model.task_answer_public_pack import TaskPublicPack
from model.complete_answer_evaluation import ARCHITECTURES,EvaluationCheckpoint,METADATA
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha,need


def opened_request(a):
    m=contract.read(a.request,a.pin);contract.request_scope(m,m['device'])
    need(file_sha(a.request)==a.pin,'TASK_DEVICE_REQUEST_CHANGED')
    return m


def worker(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);m=opened_request(a)
    device=m['device'];need(a.inspect or device==('cuda:0' if a.cuda else 'cpu'),'TASK_DEVICE_CLI_DEVICE')
    a.request_pin=a.pin;allowed,opened=access_guard(a,m);pack=None;leased=False;phases={}
    try:
        runtime=artifact(m['runtime']);target_pin,pressure=validate(m,runtime,device)
        pack=TaskPublicPack(m['pack']['path'],m['pack']['pin'])
        need(transfer.schedule(pack)==m['plan'],'TASK_DEVICE_PUBLIC_SCHEDULE')
        if a.inspect:
            for name,row in m['checkpoints'].items():
                saved=artifact(row['reference'])
                need(set(saved)=={*METADATA,'model','optimizer'} and all(saved[k]==row['metadata'][k] for k in METADATA),
                     'TASK_DEVICE_INSPECT_METADATA')
                del saved
            result=dict(complete=True,model_loads=0,target_loads=0,cases=len(m['plan']['rows']) if m['ordinals'] is None else len(m['ordinals']))
            phases['inspect']=time.perf_counter()-tick
        else:
            if a.cuda:
                gpu_ready(a.output);take_lease('29','original-task-transfer');leased=True
                need(torch.cuda.is_available(),'TASK_DEVICE_CUDA_REQUIRED');torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
            target,actual_pin=extension(runtime);target.to(device);need(actual_pin==target_pin,'TASK_DEVICE_TARGET')
            loaded={}
            for name in ARCHITECTURES:
                row=m['checkpoints'][name]
                loaded[name]=EvaluationCheckpoint(row['reference']['path'],row['reference'],row['metadata'],runtime,target,target_pin,device=device)
                if pressure:loaded[name].model.forecaster.predictor.cache_bytes=CACHE
            phases['resident_load']=time.perf_counter()-tick;tick=time.perf_counter()
            path,pin=transfer.run(loaded,pack,m['plan'],target=target,output=a.output/'run',authority=m['authority'],ordinals=m['ordinals'])
            if a.cuda:torch.cuda.synchronize()
            phases['generation_readback']=time.perf_counter()-tick
            result=dict(complete=True,run=dict(path=str(path),pin=pin),model_loads=3,target_loads=1)
        tick=time.perf_counter();pack.close();pack=None
        for n,h in m['code'].items():need(file_sha(contract.REPO/'scripts'/n)==h,'TASK_DEVICE_FINAL_CODE')
        need(file_sha(a.request)==a.pin and (a.cuda or not torch.cuda.is_initialized()),'TASK_DEVICE_FINAL_STATE')
        private=contract.ROOT/'answer-native-plan-v1-20261023/report.json'
        try:private.read_bytes()
        except ValueError as exc:need('DEVICE_COHORT_UNDECLARED_ARTIFACT' in str(exc),'TASK_DEVICE_WRONG_ACCESS_REFUSAL')
        else:raise ValueError('TASK_DEVICE_PRIVATE_READ_PASSED')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_DEVICE_PHASES')
        result.update(request=a.pin,device=device,mode='inspect' if a.inspect else 'cuda' if a.cuda else 'cpu-proof',
            full_training=m['full_training'],full_query=m['ordinals'] is None,intervention=m['intervention'],optimizer_updates=0,
            encoder_calls=0,gpu_used=a.cuda,private_read_refused=True,opened_inputs=sorted(opened),
            access_guard='Python open audit hook over ZFS, not OS sandbox',phases=phases,seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.cuda else 0,capability_demonstrated=False,architecture_winner=False)
        pin=write(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)
    finally:
        if pack is not None:pack.close()
        if leased:release_lease()


def score(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);m=opened_request(a)
    generated=contract.read(a.generated,a.generated_pin)
    need(generated['complete'] and generated['request']==a.pin and generated['mode'] in ('cuda','cpu-proof')
         and generated['device']==m['device'] and generated['full_query']==(m['ordinals'] is None),'TASK_DEVICE_SCORE_GENERATION')
    run_ref=generated['run'];r=contract.read(run_ref['path'],run_ref['pin']);root=Path(run_ref['path']).parent;b=r['binding']
    selected=list(range(len(m['plan']['rows']))) if m['ordinals'] is None else m['ordinals']
    need(b['authority']==m['authority'] and b['ordinals']==selected and [x['ordinal'] for x in r['results']]==selected
         and b['plan']==m['plan_pin'],'TASK_DEVICE_SCORE_DENOMINATOR')
    for n,v in b['checkpoints'].items():
        ref=m['checkpoints'][n]['reference'];meta=m['checkpoints'][n]['metadata']
        need(v['checkpoint']==ref['pin'] and v['state']==ref['state'] and v['counts']==meta['counts']
             and v['step']==meta['step'] and v['target']==meta['binding']['target'],'TASK_DEVICE_SCORE_CHECKPOINT')
    pack=TaskPublicPack(m['pack']['path'],m['pack']['pin']);checked=[];phases={}
    try:
        for ref in r['results']:
            row=m['plan']['rows'][ref['ordinal']];_,_,public,unavailable=transfer.inputs(pack,row)
            kw=dict(row=row,plan_pin=m['plan_pin'],identities=b['checkpoints'],expected=b['expected'],public=public,
                unavailable=unavailable,target_state=b['frozen'],caches=b['cache_bytes'],
                authority=dict(m['authority'],plan=m['plan_pin'],task=row['task'],public=row['public'],ordinal=row['ordinal']))
            transfer.check_case(root/ref['path'],ref['pin'],**kw);checked.append((ref,kw))
        phases['authenticate_generation']=time.perf_counter()-tick;tick=time.perf_counter()
        from audit.task_answer_public_pack import source
        schedule,store,binding=source();need(binding==pack.report['binding'],'TASK_DEVICE_SCORE_SOURCE')
        phases['reference_source']=time.perf_counter()-tick;tick=time.perf_counter();results=[]
        for ref,kw in checked:
            row=kw['row'];private=schedule.supervision(row['task'],fitting=False)
            future=tuple(store.frame(i) for i in private['targets'])
            value=transfer.score_case(root/ref['path'],ref['pin'],**kw,future=future,roster=pack.roster)
            path=a.output/f"case-{row['ordinal']:06d}.json";pin=write(path,value)
            results.append(dict(ordinal=row['ordinal'],path=path.name,pin=pin))
        phases['score']=time.perf_counter()-tick;tick=time.perf_counter();store.fresh();pack.close();pack=None
        for n,h in m['code'].items():need(file_sha(contract.REPO/'scripts'/n)==h,'TASK_DEVICE_SCORE_CODE_CHANGED')
        need(not torch.cuda.is_initialized(),'TASK_DEVICE_SCORE_CUDA')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_DEVICE_SCORE_PHASES')
        value=dict(complete=True,request=a.pin,generation=a.generated_pin,run=run_ref,cases=results,private_binding=binding,
            query_denominator=170,evaluated=len(results),full_query=m['ordinals'] is None,full_training=m['full_training'],
            model_loads=0,encoder_calls=0,optimizer_updates=0,gpu_used=False,phases=phases,seconds=seconds,
            semantic_validity_established=False,architecture_winner=False)
        pin=write(a.output/'report.json',value);print(json.dumps(dict(complete=True,pin=pin,cases=len(results),seconds=seconds)),flush=True)
    finally:
        if pack is not None:pack.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for n in ('inspect','cpu-proof','cuda','score'):g.add_argument('--'+n,action='store_true')
    for n in ('request','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--generated',type=Path);p.add_argument('--generated-pin')
    p.add_argument('--owner-fd',type=int);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if a.owner_fd is not None:os.fstat(a.owner_fd)
        score(a) if a.score else worker(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
