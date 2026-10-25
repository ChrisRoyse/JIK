"""Wait for actual full answer training, then qualify, generate and score all cases."""
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
from audit import legal_answer_evaluation_handoff as authority
from audit.surface_device_queue import check_owner,identity,need,sha,write

LOCK=authority.REPO/'.lease/legal-answer-evaluation-queue'
CONTROLS={'authority_pin','owner_identity','live_predecessor','changed_code','changed_scope',
          'missing_stage','duplicate_stage','extra_stage','duplicate_driver','child_retains_lock','released_after_child_exit'}


def stages(rows):return authority.training.stages(rows,('inspect','qualification','qualification-cold','score'))


def proofs(a,cfg):
    reports=[]
    for path,pin in ((a.pipeline,a.pipeline_pin),(a.cold,a.cold_pin)):
        r=authority.training.read(path/'report.json',pin);reports.append(r)
        need(r['complete'] and r['mode']=='cpu-proof' and r['authority']==a.pin and r['code']==cfg['code']
             and not r['full_population_evaluated'] and not r['full_population_trained']
             and r['optimizer_updates']==0 and r['private_reference_only_after_generation'],
             'ANSWER_EVALUATION_PIPELINE_PROOF')
        q=r['qualification'];need(q['all_events_exact'] and q['cases']==4 and q['trajectories']==96
            and q['events']==1704 and q['device']=='cpu' and not q['gpu_used'],'ANSWER_EVALUATION_PIPELINE_EVENTS')
        outputs=stages(r['outputs'])
        for label,ref in outputs.items():
            p=Path(ref['path']);need(p.resolve().is_relative_to(path.resolve()),'ANSWER_EVALUATION_PROOF_PATH')
            child=authority.training.read(p,ref['pin']);need(child['complete'] and not child['gpu_used'],
                                                           'ANSWER_EVALUATION_CHILD_PROOF')
        need(r['scored']['cases']==4 and r['scored']['actions']==dict(scored=4,reused=0,gap=0),
             'ANSWER_EVALUATION_PIPELINE_SCORING')
    need(reports[0]['qualification']==reports[1]['qualification']
         and reports[0]['scored']['summary']==reports[1]['scored']['summary'],'ANSWER_EVALUATION_PIPELINE_COLD')
    return dict(producer=dict(path=str(a.pipeline),pin=a.pipeline_pin),cold=dict(path=str(a.cold),pin=a.cold_pin))


def probe(cfg,parent,template,output,fresh):
    controls={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'ANSWER_EVALUATION_PROBE_WRONG_REFUSAL:'+str(exc));controls[name]=str(exc)
        else:raise ValueError('ANSWER_EVALUATION_PROBE_ACCEPTED:'+name)
    reject('authority_pin',lambda:authority.training.read(Path(cfg['template']['path']),'0'*64),'ANSWER_FULL_PIN')
    reject('owner_identity',lambda:check_owner(dict(cfg['wait'],starttime='0')),'SURFACE_QUEUE_OWNER_IDENTITY')
    reject('live_predecessor',lambda:authority.terminal(cfg,parent),'ANSWER_FULL_PREDECESSOR_LIVE')
    rows=[dict(label=n) for n in ('inspect','qualification','qualification-cold','score')];stages(rows)
    for name,changed in (('missing_stage',rows[:-1]),('duplicate_stage',rows+[rows[0]]),('extra_stage',rows+[dict(label='invented')])):
        reject(name,lambda changed=changed:stages(changed),'ANSWER_FULL_STAGES')
    for name in ('changed_code','changed_scope'):
        altered=deepcopy(cfg)
        if name=='changed_code':altered['code'][authority.NEW[1]]='0'*64
        else:altered['plan']['cases']=4
        p=output/(name+'.json');write(p,altered)
        reject(name,lambda p=p:authority.authenticate(p,sha(p)),
               'ANSWER_EVALUATION_CODE_BINDING' if name=='changed_code' else 'ANSWER_EVALUATION_SCOPE')
    first=LOCK.open('a+');child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+') as second:reject('duplicate_driver',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),
                                              'Resource temporarily unavailable')
        code='import os,sys; os.fstat(int(sys.argv[1])); print("inherited",flush=True); sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',code,str(first.fileno())],pass_fds=(first.fileno(),),
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','ANSWER_EVALUATION_PROBE_CHILD');first.close()
        with LOCK.open('a+') as second:reject('child_retains_lock',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),
                                              'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'ANSWER_EVALUATION_PROBE_CHILD_EXIT')
        with LOCK.open('a+') as second:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controls['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    fresh();need(set(controls)==CONTROLS and 'torch' not in sys.modules,'ANSWER_EVALUATION_PROBE_SCOPE')
    return controls


def run(a):
    began=tick=time.perf_counter();cfg,parent,template,fresh=authority.authenticate(a.authority,a.pin)
    prior=proofs(a,cfg);phases=dict(authentication=time.perf_counter()-tick)
    if a.mode=='probe':
        tick=time.perf_counter();controls=probe(cfg,parent,template,a.output,fresh);phases['controls']=time.perf_counter()-tick
        write(a.output/'report.json',dict(complete=True,mode='probe',authority=a.pin,code=cfg['code'],controls=controls,
            pipeline_proofs=prior,plan=authority.planned(a.output),gpu_children_started=0,torch_loaded=False,
            phases=phases,seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        return
    proof=authority.training.read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['mode']=='probe' and proof['authority']==a.pin and proof['code']==cfg['code']
         and proof['pipeline_proofs']==prior and set(proof['controls'])==CONTROLS and not proof['torch_loaded']
         and proof['gpu_children_started']==0,'ANSWER_EVALUATION_PROBE_REQUIRED')
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
            except ProcessLookupError:authority.training.ready(ref)
            else:
                try:check_owner(ref);event('waiting',owner=ref,wait='kernel_pidfd',torch_loaded='torch' in sys.modules);select.select([fd],[],[])
                finally:os.close(fd)
        phases['predecessor_wait']=time.perf_counter()-tick;tick=time.perf_counter();fresh()
        terminal=authority.terminal(cfg,parent);write(a.output/'terminal.json',terminal)
        phases['terminal_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        plan=authority.planned(a.output);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                               CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','ANSWER_EVALUATION_GPU_HIDDEN')
        command=[sys.executable,'-I',plan['pipeline'],'--cuda','--authority',str(a.authority),'--pin',a.pin,
                 '--output',plan['output'],'--owner-fd',str(lock.fileno())]
        with (a.output/'evaluation.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=authority.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            event('running',pid=child.pid,starttime=identity(child.pid),command=command,lock_inherited=True)
            status=child.wait();log.flush();os.fsync(log.fileno())
        event('terminal',exit_code=status);need(status==0,'ANSWER_EVALUATION_PIPELINE_FAILED');fresh()
        path=Path(plan['output'])/'report.json';pin=sha(path);r=authority.training.read(path,pin)
        need(r['complete'] and r['mode']=='cuda' and r['authority']==a.pin and r['full_population_evaluated']
             and r['full_population_trained'] and r['qualification']['gpu_used'] and r['scored']['cases']==23513,
             'ANSWER_EVALUATION_FULL_RESULT')
        phases['pipeline']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'ANSWER_EVALUATION_QUEUE_PHASES')
        write(a.output/'report.json',dict(complete=True,mode='run',authority=a.pin,proof=a.proof_pin,driver_pin=sha(__file__),
            predecessor=terminal['queue'],evaluation=dict(path=str(path),pin=pin),full_population_evaluated=True,
            optimizer_updates=0,admitted_senses=0,capability_demonstrated=False,architecture_winner=False,
            phases=phases,seconds=seconds,acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','probe','run'),required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','pipeline','cold','proof'):p.add_argument('--'+n,type=Path)
    for n in ('pin','pipeline-pin','cold-pin','proof-pin'):p.add_argument('--'+n)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        if a.mode=='prepare':authority.prepare(a.output)
        else:run(a)
        path=a.output/('authority.json' if a.mode=='prepare' else 'report.json')
        print(json.dumps(dict(path=str(path),pin=sha(path))),flush=True)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)));raise
