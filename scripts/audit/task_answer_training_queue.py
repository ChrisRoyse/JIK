"""After phase88: qualify/train complete task answers, then preserve phase91 inversion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.surface_device_queue import sha,write,need,identity,check_owner


def wait(owner):
    check_owner(owner)
    if identity(owner['pid'])==owner['starttime']:
        fd=os.pidfd_open(owner['pid'])
        try:
            need(identity(owner['pid']) in (owner['starttime'],None),'TASK_QUEUE_PID_REUSED')
            select.select([fd],[],[])
        finally:os.close(fd)
    path=Path(owner['path'])/'report.json';need(path.is_file(),'TASK_QUEUE_PREDECESSOR_NO_REPORT')
    r=json.loads(path.read_bytes());need(r['complete'],'TASK_QUEUE_PREDECESSOR_FAILED')
    need(r['driver_pin']==json.loads((Path(owner['path'])/'launch.json').read_bytes())['driver_pin'],'TASK_QUEUE_PREDECESSOR_DRIVER')
    for ref in r['outputs']:
        p=Path(ref['path'])/'report.json';need(sha(p)==ref['pin'] and json.loads(p.read_bytes())['complete'],'TASK_QUEUE_PREDECESSOR_OUTPUT')
    return r


def run(a):
    a.output.mkdir(exist_ok=False);repo=Path(__file__).resolve().parents[2];lock=None;outputs=[]
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    try:
        need(sha(a.authority)==a.pin,'TASK_QUEUE_AUTHORITY');c=json.loads(a.authority.read_bytes())
        need(c['schema']=='complete-task-answer-queue-v1','TASK_QUEUE_SCHEMA')
        def fresh():
            need(sha(a.authority)==a.pin,'TASK_QUEUE_AUTHORITY_CHANGED')
            for n,h in c['code'].items():need(sha(repo/'scripts'/n)==h,'TASK_QUEUE_CODE')
            for ref in c['proofs'].values():need(sha(Path(ref['path'])/'report.json')==ref['pin'],'TASK_QUEUE_PROOF_PIN')
            need(sha(Path(c['inversion']['path']))==c['inversion']['pin'],'TASK_QUEUE_INVERSION_AUTHORITY')
        fresh()
        proof=json.loads((Path(c['proofs']['readback']['path'])/'report.json').read_bytes())
        need(proof['complete'] and proof['device']=='cpu' and proof['source_mode']=='proof' and proof['all_updates_exact']
             and proof['all_trajectories_exact'] and proof['code']==c['code'],'TASK_QUEUE_CPU_QUALIFICATION')
        for owner in (c['wait'],c['consumer'],c['replaces']):check_owner(owner)
        if a.mode=='probe':
            write(a.output/'report.json',dict(complete=True,authority=a.pin,children_started=0,code_files=len(c['code'])));return
        need(a.handoff and sha(a.handoff)==a.handoff_pin,'TASK_QUEUE_HANDOFF_PIN');h=json.loads(a.handoff.read_bytes())
        need(h['old']==c['replaces'] and h['new_authority']==a.pin and h['childless_before_exit']
             and identity(c['replaces']['pid']) is None,'TASK_QUEUE_HANDOFF_NOT_COMPLETE')
        lock=(repo/'.lease/task-answer-priority-queue').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            driver_pin=sha(__file__),wait=c['wait'],handoff=a.handoff_pin))
        event('waiting_for_phase88',owner=c['wait']);prior=wait(c['wait'])
        need({r['label'] for r in prior['outputs']}=={'answer','grouped','surface'},'TASK_QUEUE_PHASE88_SCOPE')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','TASK_QUEUE_GPU_HIDDEN')
        def execute(label,script,args):
            fresh();dest=a.output/label;cmd=[sys.executable,'-I',str(repo/'scripts/audit'/script),*args,'--output',str(dest)]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=cmd)
                status=child.wait();log.flush();os.fsync(log.fileno())
            need(status==0,'TASK_QUEUE_CHILD_FAILED:'+label)
            r=json.loads((dest/'report.json').read_bytes());need(r['complete'],'TASK_QUEUE_CHILD_INCOMPLETE')
            ref=dict(label=label,path=str(dest),pin=sha(dest/'report.json'));outputs.append(ref);event('terminal',**ref);return ref
        task_error=None
        try:
            event('waiting_for_consumer',owner=c['consumer']);consumer=wait(c['consumer'])
            refs={r['label']:r for r in consumer['outputs']};need(set(refs)=={'bind','readback'},'TASK_QUEUE_CONSUMER_SCOPE')
            first=execute('gpu-proof','task_answer_training.py',['--mode','proof'])
            cold=execute('gpu-proof-readback','task_answer_training_readback.py',['--input',first['path'],'--pin',first['pin']])
            native=['--consumer',refs['bind']['path'],'--consumer-pin',refs['bind']['pin'],
                    '--consumer-cold',refs['readback']['path'],'--consumer-cold-pin',refs['readback']['pin']]
            full=execute('full','task_answer_training.py',['--mode','full',*native,'--qualification',cold['path'],'--qualification-pin',cold['pin']])
            execute('full-readback','task_answer_training_readback.py',['--input',full['path'],'--pin',full['pin']])
        except BaseException as exc:
            task_error=repr(exc);write(a.output/'task-failure.json',dict(error=task_error,outputs=outputs));event('task_failed',error=task_error)
        # Inversion is an independently authorized job. Preserve it even if this candidate fails.
        execute('inversion','legal_surface_inversion_queue.py',['--mode','run','--authority',c['inversion']['path'],'--pin',c['inversion']['pin']])
        write(a.output/'report.json',dict(complete=task_error is None,authority=a.pin,outputs=outputs,task_error=task_error,
            inversion_preserved=True,driver_pin=sha(__file__),scientific_admission=False,capability_demonstrated=False,acceptance='author-checked'))
    except BaseException as exc:write(a.output/'failure.json',dict(error=repr(exc),outputs=outputs));event('failed',error=repr(exc));raise
    finally:
        if lock:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['probe','run'],required=True)
    p.add_argument('--authority',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--handoff',type=Path);p.add_argument('--handoff-pin');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
