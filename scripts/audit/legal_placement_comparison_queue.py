"""Serialize the legal placement comparison behind authenticated GPU qualification."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path,pin):
    if sha(path)!=pin:raise ValueError('PLACEMENT_COMPARISON_QUEUE_PIN')
    return json.loads(Path(path).read_bytes())


def identity(pid):
    try:s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if s[0]=='Z' else s[19]


def run(a):
    repo=Path(__file__).resolve().parents[2];lease=(repo/'.lease/legal-placement-comparison-queue').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (a.output/name).open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    def event(stage,**value):
        value=dict(stage=stage,timestamp=time.time(),**value)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value),flush=True)
    pin=sha(__file__);(a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    write('launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),driver_pin=pin,
        wait_pid=a.wait_pid,wait_start=a.wait_start,wait_root=str(a.wait_root),cpu=a.cpu_pin,cold=a.cold_pin,
        readback=a.readback_pin,cold_readback=a.cold_readback_pin,source_plan=a.source_plan_pin))
    results=[]
    try:
        q=read(a.cpu/'report.json',a.cpu_pin);c=read(a.cold/'report.json',a.cold_pin)
        r=read(a.readback/'report.json',a.readback_pin);s=read(a.cold_readback/'report.json',a.cold_readback_pin)
        plan=read(a.source_plan,a.source_plan_pin)
        assert all(x['complete'] for x in (q,c,r,s))
        assert q['binding']==c['binding'] and c['previous_pin']==a.cpu_pin and q['results_pin']==c['results_pin']
        assert q['binding']['scope']=='small' and q['binding']['device']=='cpu'
        assert r['input_pin']==a.cpu_pin and s['input_pin']==a.cold_pin and r['totals']==s['totals']
        assert r['rows']==s['rows']==3200 and r['checkpoints']==s['checkpoints']==18
        assert plan['complete'] and plan['code']==q['binding']['code']
        assert plan['results']['full']['train_documents']==91540 and plan['results']['full']['expected_training_rows']==6455636
        assert plan['results']['full']['eval_documents']==192 and plan['results']['qualify']['actual_training_rows']==4104
        authenticated={a.cpu/'report.json':a.cpu_pin,a.cold/'report.json':a.cold_pin,
            a.readback/'report.json':a.readback_pin,a.cold_readback/'report.json':a.cold_readback_pin,
            a.source_plan:a.source_plan_pin}
        for root,report in ((a.cpu,q),(a.cold,c)):
            authenticated.update({root/n:p for n,p in report['files'].items()})
        for path,expected in authenticated.items():assert sha(path)==expected,'PLACEMENT_COMPARISON_QUEUE_PROOF_BYTES'
        stamps={p:p.stat() for p in authenticated}
        def check():
            assert sha(__file__)==pin,'PLACEMENT_COMPARISON_QUEUE_DRIVER_CHANGED'
            for n,p in q['binding']['code'].items():assert sha(repo/'scripts'/n)==p,'PLACEMENT_COMPARISON_QUEUE_CODE_CHANGED'
            for p,old in stamps.items():
                now=p.stat()
                assert (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)==(
                    old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns,old.st_ctime_ns),'PLACEMENT_COMPARISON_QUEUE_PROOF_CHANGED'
        owner=json.loads((a.wait_root/'launch.json').read_bytes())
        assert owner['pid']==a.wait_pid and str(owner['starttime'])==a.wait_start,'PLACEMENT_COMPARISON_QUEUE_OWNER'
        check()
        if identity(a.wait_pid)==a.wait_start:
            fd=os.pidfd_open(a.wait_pid)
            try:
                assert identity(a.wait_pid) in (a.wait_start,None),'PLACEMENT_COMPARISON_QUEUE_PID_REUSED'
                event('waiting_for_runtime_owner',pid=a.wait_pid,starttime=a.wait_start,wait='kernel_pidfd')
                select.select([fd],[],[])
            finally:os.close(fd)
        prior=json.loads((a.wait_root/'report.json').read_bytes());assert prior['complete'],'PLACEMENT_COMPARISON_QUEUE_RUNTIME_FAILED'
        runtime={x['label']:x for x in prior['outputs']};assert {'gpu','gpu-cold'}.issubset(runtime)
        check();event('runtime_owner_terminal',report_pin=sha(a.wait_root/'report.json'))
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        assert env.get('CUDA_VISIBLE_DEVICES')!='','PLACEMENT_COMPARISON_QUEUE_GPU_HIDDEN'
        def execute(label,command):
            check()
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command)
                status=child.wait();log.flush();os.fsync(log.fileno())
            event('terminal',label=label,exit_code=status);assert status==0,'PLACEMENT_COMPARISON_QUEUE_CHILD_FAILED'
            path=a.output/label;report=json.loads((path/'report.json').read_bytes());assert report['complete']
            result=dict(label=label,path=str(path),pin=sha(path/'report.json'));results.append(result);check();return result
        common=[]
        for argument,label in (('runtime','gpu'),('runtime-cold','gpu-cold')):
            common+=['--'+argument,runtime[label]['path'],'--'+argument+'-pin',runtime[label]['pin']]
        def train(label,scope,extra=()):
            return execute(label,[sys.executable,'-I',str(repo/'scripts/audit/legal_placement_comparison_training.py'),
                '--scope',scope,'--output',str(a.output/label)]+common+list(extra))
        def recount(label,ref):
            return execute(label,[sys.executable,'-I',str(repo/'scripts/audit/legal_placement_comparison_readback.py'),
                '--input',ref['path'],'--input-pin',ref['pin'],'--output',str(a.output/label)])
        first=train('qualification','qualify');second=train('qualification-cold','qualify',
            ['--previous',first['path'],'--previous-pin',first['pin']])
        recount('qualification-readback',first);recount('qualification-cold-readback',second)
        full=train('full','full',['--qualification',first['path'],'--qualification-pin',first['pin'],
             '--qualification-cold',second['path'],'--qualification-cold-pin',second['pin']])
        recount('full-readback',full)
        write('report.json',dict(complete=True,outputs=results,driver_pin=pin,acceptance='author-checked',
            scientific_admission=False,capability_demonstrated=False))
        event('queue_terminal',complete=True,pin=sha(a.output/'report.json'))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),outputs=results));event('queue_terminal',complete=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cpu','cold','readback','cold-readback'):
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    for n in ('output','wait-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--wait-start',required=True)
    p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--source-plan-pin',required=True)
    run(p.parse_args())
