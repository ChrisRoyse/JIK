"""Prepare task-native learner inputs as soon as phase 88's answer cold replay lands."""
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
    a.output.mkdir(exist_ok=False);repo=Path(__file__).resolve().parents[2];outputs=[];lock=None;fd=None
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    try:
        need(sha(a.authority)==a.pin,'ANSWER_CONSUMER_QUEUE_PIN');c=json.loads(a.authority.read_bytes())
        need(c['schema']=='canonical-answer-consumer-queue-v1','ANSWER_CONSUMER_QUEUE_SCOPE')
        def fresh():
            need(sha(a.authority)==a.pin,'ANSWER_CONSUMER_QUEUE_CHANGED')
            for n,pin in c['code'].items():need(sha(repo/'scripts'/n)==pin,'ANSWER_CONSUMER_QUEUE_CODE')
            for ref in c['proofs'].values():need(sha(Path(ref['path'])/'report.json')==ref['pin'],'ANSWER_CONSUMER_QUEUE_PROOF')
            check_owner(c['wait'])
        fresh();p=json.loads((Path(c['proofs']['plan']['path'])/'report.json').read_bytes())
        q=json.loads((Path(c['proofs']['readback']['path'])/'report.json').read_bytes())
        need(p['complete'] and p['mode']=='plan' and not p['experimental_items_ready']
             and q['complete'] and q['source']['pin']==c['proofs']['plan']['pin']
             and q['summary']==p['summary'] and c['code']==p['code'],'ANSWER_CONSUMER_QUEUE_PLAN')
        root=Path(c['answer_root']);need(root==Path(c['wait']['path'])/'answer','ANSWER_CONSUMER_QUEUE_TARGET')
        if a.mode=='probe':
            write(a.output/'report.json',dict(complete=True,authority=a.pin,code_files=len(c['code']),
                children_started=0,waiting_for=str(root/'report.json')));return
        lock=(repo/'.lease/canonical-answer-consumer-queue').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            driver_pin=sha(__file__),wait=c['wait']))
        owner=c['wait']
        if identity(owner['pid'])==owner['starttime']:
            fd=os.pidfd_open(owner['pid']);need(identity(owner['pid']) in (owner['starttime'],None),'ANSWER_CONSUMER_QUEUE_PID')
        event('waiting_for_answer_cold',path=str(root/'report.json'),mechanism='pidfd_with_30_second_receipt_check')
        while not (root/'report.json').exists():
            need(not (root/'failure.json').exists() and not (Path(owner['path'])/'failure.json').exists(),
                 'ANSWER_CONSUMER_QUEUE_UPSTREAM_FAILURE')
            need(fd is not None and identity(owner['pid'])==owner['starttime'],'ANSWER_CONSUMER_QUEUE_UPSTREAM_EXIT')
            select.select([fd],[],[],30)
        fresh();prior=json.loads((root/'report.json').read_bytes())
        need(prior['complete'] and prior['authority']==c['answer_authority']
             and prior['driver_pin']==sha(repo/'scripts/audit/answer_native_queue.py'),'ANSWER_CONSUMER_QUEUE_PARENT')
        refs={v['label']:v for v in prior['outputs']};need(set(refs)=={'native','cold'},'ANSWER_CONSUMER_QUEUE_OUTPUTS')
        for ref in refs.values():
            need(sha(Path(ref['path'])/'report.json')==ref['pin'],'ANSWER_CONSUMER_QUEUE_NATIVE_PIN')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        def execute(label,script,args):
            fresh();dest=a.output/label;cmd=[sys.executable,'-I',str(repo/'scripts/audit'/script),*args,'--output',str(dest)]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=cmd)
                status=child.wait();log.flush();os.fsync(log.fileno())
            need(status==0,'ANSWER_CONSUMER_QUEUE_CHILD_FAILED');r=json.loads((dest/'report.json').read_bytes())
            need(r['complete'],'ANSWER_CONSUMER_QUEUE_INCOMPLETE')
            ref=dict(label=label,path=str(dest),pin=sha(dest/'report.json'));outputs.append(ref);event('terminal',**ref);return ref
        args=['--mode','bind']
        for n in ('native','cold'):args+=['--'+n,refs[n]['path'],'--'+n+'-pin',refs[n]['pin']]
        result=execute('bind','canonical_answer_consumer.py',args)
        cold=execute('readback','canonical_answer_consumer_readback.py',['--input',result['path'],'--pin',result['pin']])
        readback=json.loads((Path(cold['path'])/'report.json').read_bytes())
        need(readback['all_native_inputs_checked'] and readback['all_learner_items_checked'],'ANSWER_CONSUMER_QUEUE_COVERAGE')
        write(a.output/'report.json',dict(complete=True,authority=a.pin,outputs=outputs,driver_pin=sha(__file__),
            native_parent=sha(root/'report.json'),training_updates=0,scientific_admission=False,acceptance='author-checked'))
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=repr(exc),outputs=outputs));event('failed',error=repr(exc));raise
    finally:
        if fd is not None:os.close(fd)
        if lock:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['probe','run'],required=True)
    p.add_argument('--authority',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
