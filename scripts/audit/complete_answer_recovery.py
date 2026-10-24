"""Resume unchanged numerical workers after an authenticated, empty queue failure."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import complete_answer_schedule as schedule

REPO=schedule.REPO
sha,read,write,need=schedule.sha,schedule.read,schedule.write,schedule.need


def identity(pid):
    try:fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    return None if fields[0]=='Z' else fields[19]


def terminal(root):
    launch=json.loads((root/'launch.json').read_bytes())
    need(identity(launch['pid']) is None,'RECOVERY_OWNER_LIVE_OR_REUSED')
    return launch


def unchanged_except_wait(old,new,wait):
    expected=deepcopy(old);expected['wait']=wait
    need(new==expected,'RECOVERY_CHANGED_WORK')


def prepare(a):
    a.output.mkdir(exist_ok=False)
    c,sources,fresh=schedule.authenticate(a.parent,a.parent_pin)
    old=Path(a.old_queue);launch=terminal(old);failure=json.loads((old/'failure.json').read_bytes())
    need(launch['authority']==a.parent_pin and failure['outputs']==[] and failure['errors']=={}
         and not (old/'report.json').exists(),'RECOVERY_OLD_PARTIAL_WORK')
    previous_runtime=Path(c['runtime']['path']);terminal(previous_runtime)
    failed=json.loads((previous_runtime/'failure.json').read_bytes())
    need(failed['outputs']==[] and not (previous_runtime/'report.json').exists(),'RECOVERY_RUNTIME_PARTIAL_WORK')
    runtime=read(a.runtime/'report.json',a.runtime_pin);new_launch=terminal(a.runtime)
    old_launch=json.loads((previous_runtime/'launch.json').read_bytes())
    need(runtime['complete'] and runtime['driver_pin']==old_launch['driver_pin']==new_launch['driver_pin'],
         'RECOVERY_RUNTIME_DRIVER')
    for k in ('source','cpu','cpu_cold'):
        need(runtime[k]==old_launch[k]==new_launch[k],'RECOVERY_RUNTIME_SOURCE')
    need(all(new_launch[k]==old_launch[k] for k in ('wait_pid','wait_start','wait_root')),'RECOVERY_RUNTIME_PREDECESSOR')
    need({r['label'] for r in runtime['outputs']}=={'gpu','gpu-cold'},'RECOVERY_RUNTIME_SCOPE')
    for ref in runtime['outputs']:
        r=schedule.report(ref)
        need(r['complete'] and r['gpu_qualified'] and r['all_updates_exact'],'RECOVERY_RUNTIME_PROOF')
        for n,pin in r['files'].items():need(sha(Path(ref['path'])/n)==pin,'RECOVERY_RUNTIME_BYTES')
    wait=dict(path=str(a.runtime),pid=new_launch['pid'],starttime=str(new_launch['starttime']),launch_pin=sha(a.runtime/'launch.json'))
    p=sources['priority'];native=read(p['answer_authority']['path'],p['answer_authority']['pin'])
    replacement=deepcopy(native);replacement['wait']=wait;unchanged_except_wait(native,replacement,wait)
    write(a.output/'native-authority.json',replacement)
    group=read(p['authorities']['grouped']['path'],p['authorities']['grouped']['pin'])
    gw={k:v for k,v in wait.items() if k!='launch_pin'};changed=deepcopy(group);changed['wait']=gw
    unchanged_except_wait(group,changed,gw);write(a.output/'grouped-authority.json',changed)
    handoff=read(p['grouped_handoff']['path'],p['grouped_handoff']['pin']);successor=deepcopy(handoff)
    successor.update(authority=sha(a.output/'grouped-authority.json'),inherited_handoff=p['grouped_handoff'],
        original_authority=handoff['authority'],transition='rebind_completed_runtime_after_empty_queue_failure')
    write(a.output/'grouped-handoff.json',successor)
    evidence=[old/'launch.json',old/'failure.json',old/'plan.json',previous_runtime/'launch.json',
        previous_runtime/'failure.json',a.runtime/'launch.json',a.runtime/'report.json']
    for ref in runtime['outputs']:evidence.append(Path(ref['path'])/'report.json')
    cfg=dict(schema='complete-answer-empty-failure-recovery-v1',parent=dict(path=str(a.parent),pin=a.parent_pin),
        failed_queue=str(old),failed_runtime=str(previous_runtime),runtime=dict(path=str(a.runtime),pin=a.runtime_pin),
        files={str(p):sha(p) for p in evidence},driver=sha(__file__),
        replacement={n:dict(path=str(a.output/f),pin=sha(a.output/f)) for n,f in
            [('native','native-authority.json'),('grouped','grouped-authority.json'),('handoff','grouped-handoff.json')]})
    fresh();write(a.output/'authority.json',cfg)
    print(json.dumps(dict(pin=sha(a.output/'authority.json'),path=str(a.output/'authority.json'))),flush=True)


def authenticate(path,pin):
    cfg=read(path,pin);need(cfg['schema']=='complete-answer-empty-failure-recovery-v1' and cfg['driver']==sha(__file__),'RECOVERY_SCHEMA')
    c,sources,old_fresh=schedule.authenticate(Path(cfg['parent']['path']),cfg['parent']['pin'])
    terminal(Path(cfg['failed_queue']));terminal(Path(cfg['failed_runtime']));launch=terminal(Path(cfg['runtime']['path']))
    runtime=schedule.report(cfg['runtime']);need(runtime['complete'],'RECOVERY_RUNTIME_INCOMPLETE')
    wait=dict(path=cfg['runtime']['path'],pid=launch['pid'],starttime=str(launch['starttime']),
        launch_pin=sha(Path(cfg['runtime']['path'])/'launch.json'))
    p=sources['priority'];r=cfg['replacement']
    native=read(r['native']['path'],r['native']['pin']);group=read(r['grouped']['path'],r['grouped']['pin'])
    unchanged_except_wait(read(p['answer_authority']['path'],p['answer_authority']['pin']),native,wait)
    unchanged_except_wait(read(p['authorities']['grouped']['path'],p['authorities']['grouped']['pin']),group,
        {k:v for k,v in wait.items() if k!='launch_pin'})
    previous=read(p['grouped_handoff']['path'],p['grouped_handoff']['pin']);expected=deepcopy(previous)
    expected.update(authority=r['grouped']['pin'],inherited_handoff=p['grouped_handoff'],original_authority=previous['authority'],
        transition='rebind_completed_runtime_after_empty_queue_failure')
    need(read(r['handoff']['path'],r['handoff']['pin'])==expected,'RECOVERY_HANDOFF_LINEAGE')
    adapted=deepcopy(sources);adapted['priority']['answer_authority']=r['native']
    adapted['priority']['authorities']['grouped']=r['grouped'];adapted['priority']['grouped_handoff']=r['handoff']
    old_plan=schedule.plan(sources);new_plan=schedule.plan(adapted)
    for old,new in zip(old_plan,new_plan,strict=True):
        if old['label'] not in ('native','grouped'):need(old==new,'RECOVERY_WORK_CHANGED')
    def fresh():
        need(sha(path)==pin and sha(__file__)==cfg['driver'],'RECOVERY_DRIVER_CHANGED');old_fresh()
        for p,h in cfg['files'].items():need(sha(p)==h,'RECOVERY_EVIDENCE_CHANGED')
        for ref in r.values():need(sha(ref['path'])==ref['pin'],'RECOVERY_CONTRACT_CHANGED')
    fresh();return cfg,adapted,fresh


def run(a):
    a.output.mkdir(exist_ok=False);lock=None;results={};errors={}
    def event(stage,**fields):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**fields),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    try:
        cfg,sources,fresh=authenticate(a.authority,a.pin);work=schedule.plan(sources);schedule.validate_plan(work)
        write(a.output/'plan.json',work);(a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        if a.mode=='probe':
            controls={}
            def reject(name,call):
                try:call()
                except (ValueError,KeyError):controls[name]=True
                else:raise ValueError('RECOVERY_CONTROL_ACCEPTED:'+name)
            reject('live_owner',lambda:need(identity(os.getpid()) is None,'RECOVERY_OWNER_LIVE_OR_REUSED'))
            reject('wrong_authority',lambda:read(a.authority,'0'*64))
            wrong=deepcopy(work);wrong.reverse();reject('reordered_work',lambda:schedule.validate_plan(wrong))
            node=next(v for v in work if v['label']=='task-full')
            reject('missing_prerequisite',lambda:schedule.arguments(node,{},sources))
            ref=cfg['replacement']['native'];old=read(ref['path'],ref['pin']);bad=deepcopy(old)
            bad['code']={};reject('changed_work',lambda:unchanged_except_wait(old,bad,old['wait']))
            wrong_wait=deepcopy(old['wait']);wrong_wait['pid']=os.getpid()
            reject('wrong_owner_binding',lambda:unchanged_except_wait(old,old,wrong_wait))
            fresh();write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),
                plan_pin=sha(a.output/'plan.json'),controls=controls,children_started=0,acceptance='author-checked'));return
        proof=read(a.proof/'report.json',a.proof_pin)
        need(proof['complete'] and proof['children_started']==0 and proof['authority']==a.pin and
             proof['driver_pin']==sha(__file__) and proof['plan_pin']==sha(a.output/'plan.json'),'RECOVERY_PROBE')
        lock=(REPO/'.lease/complete-answer-priority').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            proof=a.proof_pin,driver_pin=sha(__file__),runtime=cfg['runtime']))
        for row in work:
            label=row['label'];missing=[n for n in row['requires'] if n not in results]
            if missing:
                errors[label]=dict(code='RECOVERY_PREREQUISITE_FAILED',missing=missing)
                write(a.output/(label+'-skipped.json'),errors[label]);event('skipped',label=label,**errors[label]);continue
            fresh();dest=a.output/label;args=schedule.arguments(row,results,sources)
            cmd=[sys.executable,'-I',str(REPO/'scripts/audit'/row['script']),*args,'--output',str(dest)]
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            if row['cpu_only']:env['CUDA_VISIBLE_DEVICES']=''
            else:
                need(env.get('CUDA_VISIBLE_DEVICES')!='','RECOVERY_GPU_HIDDEN')
                (a.output/(label+'-nvidia-before.txt')).write_text(subprocess.check_output(['nvidia-smi'],text=True))
                need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'RECOVERY_GPU_BUSY')
            began=time.monotonic()
            try:
                with (a.output/(label+'.log')).open('xb') as log:
                    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=cmd)
                    status=child.wait();log.flush();os.fsync(log.fileno())
                write(a.output/(label+'-exit.json'),dict(exit_code=status,seconds=time.monotonic()-began))
                need(status==0,'RECOVERY_WORKER_FAILED:'+label)
                ref=dict(label=label,path=str(dest),pin=sha(dest/'report.json'))
                schedule.output_check(label,ref,sources);fresh();results[label]=ref;event('terminal',**ref)
            except Exception as e:
                errors[label]=dict(code='RECOVERY_WORKER_REFUSED',error=repr(e))
                write(a.output/(label+'-failure.json'),errors[label]);event('worker_failed',label=label,**errors[label])
        fresh();write(a.output/'report.json',dict(complete=not errors,authority=a.pin,outputs=list(results.values()),
            errors=errors,driver_pin=sha(__file__),original_numerical_workers_unchanged=True,scientific_admission=False,
            capability_demonstrated=False,acceptance='author-checked'));event('complete',success=not errors)
    except BaseException as e:
        write(a.output/'failure.json',dict(error=repr(e),outputs=list(results.values()),errors=errors));event('failed',error=repr(e));raise
    finally:
        if lock:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','probe','run'),required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('parent','old-queue','runtime','authority','proof'):p.add_argument('--'+n,type=Path)
    for n in ('parent-pin','runtime-pin','pin','proof-pin'):p.add_argument('--'+n)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:run(a)
