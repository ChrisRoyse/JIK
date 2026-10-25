"""Wait without GPU/model residency, then train the entire qualified answer source."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import legal_answer_full_authority as authority
from audit.surface_device_queue import check_owner, identity, need, sha, write

LOCK=authority.REPO/'.lease/legal-answer-full-queue'
CONTROLS={'authority_pin','owner_identity','live_predecessor','missing_stage','duplicate_stage','extra_stage',
          'source_counts','architecture','training_scope','code_binding','inspect_is_not_training',
          'duplicate_driver','child_retains_lock','released_after_child_exit'}


def probe(cfg,inspection,fresh,output):
    controls={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'ANSWER_FULL_PROBE_WRONG_REFUSAL:'+str(exc));controls[name]=str(exc)
        else:raise ValueError('ANSWER_FULL_PROBE_ACCEPTED:'+name)
    reject('authority_pin',lambda:authority.read(Path(cfg['inspection']['path'])/'report.json','0'*64),'ANSWER_FULL_PIN')
    reject('owner_identity',lambda:check_owner(dict(cfg['wait'],starttime='0')),'SURFACE_QUEUE_OWNER_IDENTITY')
    reject('live_predecessor',lambda:authority.predecessor(cfg,output),'ANSWER_FULL_PREDECESSOR_LIVE')
    actual=[dict(label=n) for n in ('tile','epoch','source')]
    authority.stages(actual,('tile','epoch','source'))
    for name,rows in [('missing_stage',actual[:-1]),('duplicate_stage',actual+[actual[0]]),
                      ('extra_stage',actual+[dict(label='invented')])]:
        reject(name,lambda rows=rows:authority.stages(rows,('tile','epoch','source')),'ANSWER_FULL_STAGES')
    altered=deepcopy(cfg['counts']);altered['shared']['documents']-=1
    reject('source_counts',lambda:authority.counts(altered),'ANSWER_FULL_COUNTS')
    reject('architecture',lambda:authority.counts({'shared':cfg['counts']['shared']}),'ANSWER_FULL_ARCHITECTURES')
    changed=deepcopy(cfg);changed['training']['mode']='proof';path=output/'bad-authority.json';write(path,changed)
    reject('training_scope',lambda:authority.authenticate(path,sha(path)),'ANSWER_FULL_TRAINING_SCOPE')
    changed=deepcopy(cfg);changed['code']['audit/legal_answer_source_training.py']='0'*64
    changed_path=output/'changed-code.json';write(changed_path,changed)
    reject('code_binding',lambda:authority.authenticate(changed_path,sha(changed_path)),'ANSWER_FULL_CODE_BINDING')
    reject('inspect_is_not_training',lambda:authority.completed(Path(cfg['inspection']['path']),cfg,{}),
           'ANSWER_FULL_RESULT_SCOPE')
    first=LOCK.open('a+');child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+') as second:reject('duplicate_driver',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        code='import os,sys; os.fstat(int(sys.argv[1])); print("inherited",flush=True); sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',code,str(first.fileno())],pass_fds=(first.fileno(),),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','ANSWER_FULL_PROBE_CHILD');first.close()
        with LOCK.open('a+') as second:reject('child_retains_lock',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'ANSWER_FULL_PROBE_EXIT')
        with LOCK.open('a+') as second:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controls['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    fresh();need(set(controls)==CONTROLS and 'torch' not in sys.modules,'ANSWER_FULL_PROBE_SCOPE')
    return controls


def run(a):
    began=tick=time.perf_counter();cfg,inspection,fresh=authority.authenticate(a.authority,a.pin)
    phases=dict(authentication=time.perf_counter()-tick)
    if a.mode=='probe':
        tick=time.perf_counter();controls=probe(cfg,inspection,fresh,a.output)
        phases['controls']=time.perf_counter()-tick
        value=dict(complete=True,mode='probe',authority=a.pin,code={n:cfg['code'][n] for n in authority.NEW},
            controls=controls,plan=authority.planned_arguments(cfg,a.output),counts=cfg['counts'],
            gpu_children_started=0,torch_loaded=False,phases=phases,seconds=time.perf_counter()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked')
        write(a.output/'report.json',value);return
    proof=authority.read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['mode']=='probe' and proof['authority']==a.pin and set(proof['controls'])==CONTROLS
         and proof['code']=={n:cfg['code'][n] for n in authority.NEW} and proof['gpu_children_started']==0
         and proof['torch_loaded'] is False,'ANSWER_FULL_PROBE_REQUIRED')
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            driver_pin=sha(__file__),wait=cfg['wait'],proof=a.proof_pin,torch_loaded='torch' in sys.modules))
        fresh();ref=cfg['wait'];check_owner(ref);tick=time.perf_counter()
        if identity(ref['pid'])==ref['starttime']:
            try:fd=os.pidfd_open(ref['pid'])
            except ProcessLookupError:authority.ready(ref)
            else:
                try:check_owner(ref);event('waiting',owner=ref,wait='kernel_pidfd',torch_loaded='torch' in sys.modules);select.select([fd],[],[])
                finally:os.close(fd)
        phases['predecessor_wait']=time.perf_counter()-tick;tick=time.perf_counter()
        fresh();handoff=authority.predecessor(cfg,a.output);fresh();write(a.output/'handoff.json',handoff)
        phases['qualification']=time.perf_counter()-tick;tick=time.perf_counter()
        write(a.output/'nvidia-before.json',dict(text=subprocess.check_output(['nvidia-smi'],text=True)))
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
             'ANSWER_FULL_GPU_RESIDENT')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','ANSWER_FULL_GPU_HIDDEN')
        command=[sys.executable,'-I',str(authority.WORKER),*handoff['arguments']]
        with (a.output/'training.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=authority.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            event('running',pid=child.pid,starttime=identity(child.pid),command=command,lock_inherited=True)
            status=child.wait();log.flush();os.fsync(log.fileno())
        write(a.output/'training-exit.json',dict(exit_code=status,seconds=time.perf_counter()-tick));event('terminal',exit_code=status)
        need(status==0,'ANSWER_FULL_CHILD_FAILED');phases['training']=time.perf_counter()-tick;tick=time.perf_counter()
        fresh();result=authority.completed(a.output/'training',cfg,handoff);fresh()
        phases['physical_completion_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'ANSWER_FULL_PHASES')
        write(a.output/'report.json',dict(complete=True,mode='run',authority=a.pin,proof=a.proof_pin,
            driver_pin=sha(__file__),training=result,full_population_trained=True,optimizer_updates=3*91540,
            capability_demonstrated=False,scientific_admission=False,promoted=False,phases=phases,seconds=seconds,
            acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','probe','run'],required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin'):p.add_argument('--'+n)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        if a.mode=='prepare':authority.prepare(a.output)
        else:run(a)
        name='authority.json' if a.mode=='prepare' else 'report.json'
        print(json.dumps(dict(path=str(a.output/name),pin=sha(a.output/name))),flush=True)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)));raise
