"""Wait for the actual pipeline exit, then run every original task on CUDA."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import select
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import task_transfer_contract as contract
from audit.task_transfer_proof import LOCK,execute,compare
from audit.surface_device_queue import check_owner,identity,need,sha,write


def run(a):
    cfg,source,fresh=contract.authenticate(a.authority,a.pin)
    proof=contract.read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['code']==cfg['code'] and proof['all_metrics_exact']
         and proof['cold']==dict(cases=3,trajectories=56,events=600,ends=0,refusals=56,all_events_exact=True)
         and len(proof['controls'])==12 and not proof['full_cuda_executed'],'TASK_QUEUE_PROOF')
    need('torch' not in sys.modules,'TASK_QUEUE_EARLY_MODEL_IMPORT')
    with LOCK.open('a+b') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(exist_ok=False)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            driver_pin=sha(__file__),proof=a.proof_pin,wait=cfg['wait']))
        def event(stage,**values):
            with (a.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        wait=cfg['wait'];fresh();check_owner(wait)
        if identity(wait['pid'])==wait['starttime']:
            try:fd=os.pidfd_open(wait['pid'])
            except ProcessLookupError:pass
            else:
                try:
                    check_owner(wait);event('waiting',owner=wait,wait='kernel_pidfd',torch_loaded=False)
                    select.select([fd],[],[])
                finally:os.close(fd)
        began=tick=time.perf_counter();fresh();m=contract.request(cfg,a.pin,source,full=True)
        write(a.output/'request-full.json',m);fullpin=sha(a.output/'request-full.json')
        q=deepcopy(m);q['ordinals']=cfg['qualification'];write(a.output/'request-qualification.json',q)
        qpin=sha(a.output/'request-qualification.json');phases=dict(terminal_export=time.perf_counter()-tick);outputs=[]
        event('predecessor_complete',handoff=m['authority']['handoff'])
        def child(label,args):
            fresh();tick=time.perf_counter();event('launching',label=label)
            ref=execute(label,args,a.output/label,owner_fd=owner.fileno());outputs.append(ref)
            fresh();phases[label]=time.perf_counter()-tick;event('complete',label=label,pin=ref['pin']);return ref
        qargs=['--request',str(a.output/'request-qualification.json'),'--pin',qpin]
        fargs=['--request',str(a.output/'request-full.json'),'--pin',fullpin]
        child('inspect',['--inspect',*fargs])
        first=child('qualification',['--cuda',*qargs]);second=child('qualification-cold',['--cuda',*qargs])
        tick=time.perf_counter();cold=compare(first,second);write(a.output/'qualification-readback.json',cold)
        need(cold['all_events_exact'] and cold['cases']==len(cfg['qualification']),'TASK_QUEUE_COLD')
        phases['qualification_readback']=time.perf_counter()-tick
        full=child('full',['--cuda',*fargs]);generation=contract.read(full['path'],full['pin'])
        need(generation['complete'] and generation['gpu_used'] and generation['full_query'] and generation['full_training']
             and generation['intervention']==dict(kind='trained_runtime_policy'),'TASK_QUEUE_FULL_GENERATION')
        scored=child('score',['--score',*fargs,'--generated',full['path'],'--generated-pin',full['pin']])
        result=contract.read(scored['path'],scored['pin'])
        need(result['complete'] and result['evaluated']==result['query_denominator']==170 and result['full_query']
             and result['full_training'] and result['generation']==full['pin'],'TASK_QUEUE_FULL_SCORE')
        need([r['label'] for r in outputs]==list(contract.STAGES),'TASK_QUEUE_STAGE_ORDER')
        fresh();seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_QUEUE_PHASES')
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),proof=a.proof_pin,
            request=fullpin,qualification_request=qpin,qualification=cold,outputs=outputs,query_cases=170,
            full_population_transfer_measured=True,optimizer_updates=0,encoder_calls=0,
            capability_demonstrated=False,architecture_winner=False,phases=phases,seconds_after_wait=seconds,
            acceptance='author-checked'))
        event('terminal',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    for n in ('authority','proof','output'):p.add_argument('--'+n,type=Path)
    p.add_argument('--pin');p.add_argument('--proof-pin');a=p.parse_args()
    try:
        if a.prepare:print(json.dumps(dict(complete=True,pin=contract.prepare(a.output))),flush=True)
        else:run(a)
    except BaseException as exc:
        if a.output is not None and a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
