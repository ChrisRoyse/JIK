"""Wait on the complete answer chain, then run six serial leased GPU stages."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import protected_corpus_queue_contract as contract
from audit.surface_device_queue import identity,need,sha


def run(a):
    cfg,_,fresh=contract.authenticate(a.authority,a.pin)
    proof=contract.read(a.proof/'report.json',a.proof_pin)
    back=contract.read(a.readback/'report.json',a.readback_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['code']==cfg['code']
         and proof['gpu_children_started']==0 and proof['live_predecessor_refused']
         and proof['three_boundaries_checked'],'PROTECTED_QUEUE_PROOF_REQUIRED')
    need(back['complete'] and back['authority']==a.pin and back['previous']==a.proof_pin
         and back['observation']==proof['observation'] and back['controls']==proof['controls']
         and back['code']==cfg['code'] and back['cold_readback_exact'],'PROTECTED_QUEUE_READBACK_REQUIRED')
    need('torch' not in sys.modules,'PROTECTED_QUEUE_MODEL_IMPORT')
    outputs={};created=False
    with contract.lock() as lock:
        a.output.mkdir(exist_ok=False);created=True
        def event(kind,**values):
            with (a.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(kind=kind,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        try:
            contract.persist(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
                driver_pin=sha(__file__),proof=a.proof_pin,readback=a.readback_pin,wait=cfg['wait']))
            owner=cfg['wait'];current=contract.process_current(owner)
            if current is not None:
                try:fd=os.pidfd_open(owner['pid'])
                except ProcessLookupError:fd=None
                if fd is not None:
                    try:
                        contract.process_current(owner)
                        event('waiting_predecessor',pid=owner['pid'],starttime=owner['starttime'],wait='kernel_pidfd',gpu_children_started=0)
                        select.select([fd],[],[])
                    finally:os.close(fd)
            fresh();prior=contract.predecessor(cfg)
            contract.persist(a.output/'predecessor.json',prior);event('predecessor_complete',pin=prior['pin'])
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            need(env.get('CUDA_VISIBLE_DEVICES')!='','PROTECTED_QUEUE_GPU_HIDDEN')
            for label in contract.STAGES:
                fresh();script,args=contract.arguments(label,cfg,outputs);dest=a.output/label
                command=[sys.executable,'-I',str(contract.REPO/'scripts/audit'/script),*args,'--output',str(dest)]
                began=time.perf_counter()
                with (a.output/(label+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,cwd=contract.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                    event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command)
                    status=child.wait();log.flush();os.fsync(log.fileno())
                event('terminal',label=label,exit_code=status,seconds=time.perf_counter()-began)
                need(status==0,'PROTECTED_QUEUE_CHILD_FAILED:'+label)
                path=dest/'report.json';pin=sha(path);r=contract.read(path,pin)
                contract.child_scope(label,r,cfg,outputs);fresh()
                ref=dict(path=str(dest),pin=pin);outputs[label]=ref
                contract.persist(a.output/f'completed-{label}.json',ref)
            need(tuple(outputs)==contract.STAGES,'PROTECTED_QUEUE_FINAL_STAGES')
            contract.persist(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),
                predecessor=prior,outputs=outputs,full_population_trained=True,full_final_document_readback=True,
                full_gradient_replay=False,scientific_admission=False,capability_demonstrated=False,acceptance='author-checked'))
        except BaseException as e:
            if created:
                contract.persist(a.output/'failure.json',dict(error=type(e).__name__,message=str(e),outputs=outputs))
                event('failed',error=type(e).__name__,message=str(e))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','run'],required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof','readback'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin','readback-pin'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.mode=='prepare':
        a.output.mkdir(exist_ok=False);cfg=contract.build();pin=contract.persist(a.output/'authority.json',cfg)
        contract.authenticate(a.output/'authority.json',pin)
        print(json.dumps(dict(pin=pin,code_files=len(cfg['code']),wait=cfg['wait'])),flush=True)
    else:run(a)
