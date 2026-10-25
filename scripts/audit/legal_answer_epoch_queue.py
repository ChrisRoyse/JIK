"""Queue the complete-answer CUDA comparison behind its actual pinned GPU owner."""
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
from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
LOCK=REPO/'.lease/legal-answer-epoch-gpu-queue'
WORKER=REPO/'scripts/audit/legal_answer_epoch_cuda.py'


def read(path,pin):
    need(sha(path)==pin,'ANSWER_QUEUE_PIN'); return json.loads(Path(path).read_bytes())


def ready(owner):
    current=identity(owner['pid'])
    need(current in (owner['starttime'],None),'ANSWER_QUEUE_PID_REUSED')
    need(current is None,'ANSWER_QUEUE_PREDECESSOR_LIVE')


def prepare(a):
    proof=read(a.inspection/'report.json',a.inspection_pin)
    owner=dict(path=str(a.wait_root),pid=a.wait_pid,starttime=a.wait_start,launch_pin=sha(a.wait_root/'launch.json'))
    check_owner(owner)
    cfg=dict(schema='complete-answer-epoch-cuda-queue-v1',wait=owner,code=proof['code'],
             inspection=dict(path=str(a.inspection),pin=a.inspection_pin))
    write(a.output/'authority.json',cfg); pin=sha(a.output/'authority.json')
    authenticate(a.output/'authority.json',pin)
    print(json.dumps(dict(path=str(a.output/'authority.json'),pin=pin)),flush=True)


def authenticate(path,pin):
    cfg=read(path,pin); need(cfg['schema']=='complete-answer-epoch-cuda-queue-v1','ANSWER_QUEUE_SCHEMA')
    ref=cfg['inspection']; root=Path(ref['path']); proof=read(root/'report.json',ref['pin'])
    need(proof['complete'] and proof['mode']=='inspect' and proof['optimizer_updates']==0
         and not proof['gpu_used'] and proof['code']==cfg['code']
         and set(proof['results'])=={'shared','own_slot','whole_proposals'},'ANSWER_QUEUE_INSPECTION')
    need(all(set(v)=={'pin','cursor','next_document','held_out','binding','moments'} for v in proof['controls'].values()),
         'ANSWER_QUEUE_CONTROL_COVERAGE')
    need(cfg['code']['audit/legal_answer_epoch_queue.py']==sha(__file__)
         and cfg['code']['audit/legal_answer_epoch_cuda.py']==sha(WORKER),'ANSWER_QUEUE_DRIVER')
    check_owner(cfg['wait'])
    files={Path(path):pin,root/'report.json':ref['pin'],Path(cfg['wait']['path'])/'launch.json':cfg['wait']['launch_pin']}
    files.update({root/n:h for n,h in proof['files'].items()})
    for p,h in files.items():need(sha(p)==h,'ANSWER_QUEUE_PHYSICAL_PROOF')
    stamps={p:stamp(p) for p in files}
    def fresh():
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h,'ANSWER_QUEUE_CODE_CHANGED')
        for p,s in stamps.items():need(stamp(p)==s,'ANSWER_QUEUE_AUTHORITY_CHANGED')
    fresh(); return cfg,fresh


def probe(cfg,fresh):
    controls={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'ANSWER_QUEUE_WRONG_REFUSAL:'+str(exc));controls[name]=str(exc)
        else:raise ValueError('ANSWER_QUEUE_NEGATIVE_ACCEPTED:'+name)
    reject('proof_pin',lambda:read(Path(cfg['inspection']['path'])/'report.json','0'*64),'ANSWER_QUEUE_PIN')
    reject('owner_identity',lambda:check_owner(dict(cfg['wait'],starttime='0')),'SURFACE_QUEUE_OWNER_IDENTITY')
    reject('live_owner',lambda:ready(dict(pid=os.getpid(),starttime=identity(os.getpid()))),'ANSWER_QUEUE_PREDECESSOR_LIVE')
    first=LOCK.open('a+'); child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+') as second:
            reject('duplicate_driver',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        # The child must keep the lock alive even after the parent's handle closes.
        code='import os,sys; os.fstat(int(sys.argv[1])); print("inherited",flush=True); sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',code,str(first.fileno())],pass_fds=(first.fileno(),),
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','ANSWER_QUEUE_CHILD_NO_HANDLE');first.close()
        with LOCK.open('a+') as second:
            reject('child_retains_lock',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'ANSWER_QUEUE_LOCK_CHILD')
        with LOCK.open('a+') as second:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controls['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    fresh();return controls


def predecessor(owner):
    ready(owner);check_owner(owner);root=Path(owner['path'])
    launch=json.loads((root/'launch.json').read_bytes());r=json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['driver_pin']==launch['driver_pin'] and r['authority']==launch['authority']
         and r['gpu_qualified'],'ANSWER_QUEUE_PREDECESSOR_FAILED')
    outputs={v['label']:v for v in r['outputs']}
    need(set(outputs)=={'gpu','gpu-cold'},'ANSWER_QUEUE_PREDECESSOR_OUTPUTS')
    results={}
    for name,v in outputs.items():
        path=Path(v['path']);p=read(path/'report.json',v['pin']);results[name]=p
        need(p['complete'] and p['gpu_used'] and p['binding']['device']=='cuda:0'
             and p['all_updates_exact'] and p['independent_restored_continuation_exact'],'ANSWER_QUEUE_PREDECESSOR_QUALIFICATION')
        for n,h in p['files'].items():need(sha(path/n)==h,'ANSWER_QUEUE_PREDECESSOR_BYTES')
    a,b=results['gpu'],results['gpu-cold']
    need(a['binding']==b['binding'] and a['numerical_pins']==b['numerical_pins'] and a['controls']==b['controls']
         and b['previous_pin']==outputs['gpu']['pin'],'ANSWER_QUEUE_PREDECESSOR_COLD')
    return sha(root/'report.json')


def run(a):
    cfg,fresh=authenticate(a.authority,a.pin);owner=cfg['wait'];outputs=[]
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    if a.mode=='probe':
        controls=probe(cfg,fresh)
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),controls=controls,
              owner_identity=identity(owner['pid']),gpu_children_started=0,acceptance='author-checked'))
        print(json.dumps(dict(pin=sha(a.output/'report.json'),controls=controls)),flush=True);return
    proof=read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['driver_pin']==sha(__file__)
         and proof['gpu_children_started']==0
         and set(proof['controls'])=={'proof_pin','owner_identity','live_owner','duplicate_driver','child_retains_lock','released_after_child_exit'},
         'ANSWER_QUEUE_PROBE')
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
              driver_pin=sha(__file__),wait=owner,proof=a.proof_pin))
        if identity(owner['pid'])==owner['starttime']:
            try:fd=os.pidfd_open(owner['pid'])
            except ProcessLookupError:ready(owner)
            else:
                try:
                    need(identity(owner['pid']) in (owner['starttime'],None),'ANSWER_QUEUE_PID_REUSED')
                    event('waiting',pid=owner['pid'],starttime=owner['starttime'],wait='kernel_pidfd');select.select([fd],[],[])
                finally:os.close(fd)
        previous_pin=predecessor(owner);fresh()
        (a.output/'predecessor-report.json').write_bytes((Path(owner['path'])/'report.json').read_bytes())
        event('predecessor_terminal',pin=previous_pin)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','ANSWER_QUEUE_GPU_HIDDEN')
        for label in ('gpu','gpu-cold'):
            fresh();(a.output/(label+'-nvidia-before.txt')).write_text(subprocess.check_output(['nvidia-smi'],text=True))
            need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
                 'ANSWER_QUEUE_GPU_RESIDENT')
            args=[] if not outputs else ['--previous',outputs[0]['path'],'--previous-pin',outputs[0]['pin']]
            cmd=[sys.executable,'-I',str(WORKER),'--mode','cuda','--root',str(a.root),'--output',str(a.output/label),
                 '--inspection',cfg['inspection']['path'],'--inspection-pin',cfg['inspection']['pin'],*args]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       pass_fds=(lock.fileno(),))
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),lock_inherited=True);status=child.wait()
                log.flush();os.fsync(log.fileno())
            event('terminal',label=label,exit_code=status);need(status==0,'ANSWER_QUEUE_CHILD_FAILED')
            path=a.output/label;r=json.loads((path/'report.json').read_bytes())
            need(r['complete'] and r['mode']=='cuda' and r['gpu_used'] and r['all_updates_exact']
                 and r['checkpoint_restores_exact'] and r['optimizer_updates']==30 and r['decisions']==396,'ANSWER_QUEUE_RESULT')
            if outputs:need(r['cold_reproduced'] and r['previous_pin']==outputs[0]['pin'],'ANSWER_QUEUE_COLD_LINEAGE')
            for n,h in r['files'].items():need(sha(path/n)==h,'ANSWER_QUEUE_RESULT_BYTES')
            outputs.append(dict(label=label,path=str(path),pin=sha(path/'report.json')));fresh()
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),outputs=outputs,
            predecessor_pin=previous_pin,gpu_qualified=True,promoted=False,full_population_trained=False,
            capability_demonstrated=False,acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','probe','run'],required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof','inspection','wait-root'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin','inspection-pin','wait-start'):p.add_argument('--'+n)
    p.add_argument('--wait-pid',type=int);a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:prepare(a) if a.mode=='prepare' else run(a)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)));raise
