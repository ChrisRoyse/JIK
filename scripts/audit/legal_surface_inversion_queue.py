"""Wait for the existing GPU chain, qualify continuation, then train every TRAIN atom."""
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


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);lock=None;outputs=[]
    repo=Path(__file__).resolve().parents[2]
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    try:
        need(sha(a.authority)==a.pin,'INVERSION_QUEUE_AUTHORITY');c=json.loads(a.authority.read_bytes())
        need(c['schema']=='legal-inversion-queue-v1','INVERSION_QUEUE_SCHEMA')
        def fresh():
            need(sha(a.authority)==a.pin,'INVERSION_QUEUE_AUTHORITY_CHANGED')
            for n,h in c['code'].items():need(sha(repo/'scripts'/n)==h,'INVERSION_QUEUE_CODE')
            for label,ref in c['proofs'].items():
                p=Path(ref['path'])/'report.json';need(sha(p)==ref['pin'],'INVERSION_QUEUE_PROOF')
        fresh();check_owner(c['wait'])
        proof=json.loads((Path(c['proofs']['readback']['path'])/'report.json').read_bytes())
        need(proof['complete'] and proof['restored_midpoint_continuation_exact'] and proof['every_evaluation_byte_exact']
             and proof['cold_optimizer_updates']==3,'INVERSION_QUEUE_CPU_PROOF')
        if a.mode=='probe':
            write(a.output/'report.json',dict(complete=True,authority=a.pin,gpu_children_started=0,wait=c['wait']));return
        lock=(repo/'.lease/legal-surface-inversion-queue').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,driver_pin=sha(__file__),wait=c['wait']))
        owner=c['wait']
        if identity(owner['pid'])==owner['starttime']:
            fd=os.pidfd_open(owner['pid'])
            try:
                need(identity(owner['pid']) in (owner['starttime'],None),'INVERSION_QUEUE_PID_REUSED')
                event('waiting_for_phase88',pid=owner['pid'],starttime=owner['starttime']);select.select([fd],[],[])
            finally:os.close(fd)
        prior=json.loads((Path(owner['path'])/'report.json').read_bytes())
        need(prior['complete'] and prior['driver_pin']==json.loads((Path(owner['path'])/'launch.json').read_bytes())['driver_pin'],
             'INVERSION_QUEUE_PREDECESSOR_FAILED')
        refs={v['label']:v for v in prior['outputs']};need(set(refs)=={'answer','grouped','surface'},'INVERSION_QUEUE_PREDECESSOR_SCOPE')
        for v in refs.values():
            p=Path(v['path'])/'report.json';need(sha(p)==v['pin'] and json.loads(p.read_bytes())['complete'],'INVERSION_QUEUE_PREDECESSOR_OUTPUT')
        surface=json.loads((Path(refs['surface']['path'])/'report.json').read_bytes())
        readback=next(v for v in surface['outputs'] if v['label']=='readback')
        p=Path(readback['path'])/'report.json';need(sha(p)==readback['pin'],'INVERSION_QUEUE_CUDA_READBACK')
        r=json.loads(p.read_bytes());need(r['all_cold_arm_states_exact'] and r['same_cohort_exact'],'INVERSION_QUEUE_CUDA_ARITHMETIC')
        plan=c['proofs']['plan'];common=['--plan',plan['path'],'--plan-pin',plan['pin']]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','INVERSION_QUEUE_GPU_HIDDEN')
        def execute(label,script,args):
            fresh();path=a.output/label;command=[sys.executable,'-I',str(repo/'scripts/audit'/script),*common,*args,'--output',str(path)]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid));status=child.wait();log.flush();os.fsync(log.fileno())
            need(status==0,'INVERSION_QUEUE_CHILD_FAILED');r=json.loads((path/'report.json').read_bytes());need(r['complete'],'INVERSION_QUEUE_INCOMPLETE')
            ref=dict(label=label,path=str(path),pin=sha(path/'report.json'));outputs.append(ref);event('terminal',**ref);return ref
        proof=execute('gpu-proof','legal_surface_inversion.py',['--mode','proof'])
        q=execute('gpu-proof-readback','legal_surface_inversion_readback.py',['--input',proof['path'],'--input-pin',proof['pin']])
        full=execute('full','legal_surface_inversion.py',['--mode','full','--qualification',q['path'],'--qualification-pin',q['pin']])
        execute('full-readback','legal_surface_inversion_readback.py',['--input',full['path'],'--input-pin',full['pin']])
        write(a.output/'report.json',dict(complete=True,authority=a.pin,outputs=outputs,driver_pin=sha(__file__),
            capability_demonstrated=False,scientific_admission=False,acceptance='author-checked'))
    except BaseException as e:
        write(a.output/'failure.json',dict(error=repr(e),outputs=outputs));event('failed',error=repr(e));raise
    finally:
        if lock:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['probe','run'],required=True)
    p.add_argument('--authority',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
