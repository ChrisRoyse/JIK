"""Pin the answer measurement plan and wait behind the existing GPU queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from text import answer_native_measure  # Import the execution dependency tree before pinning it.
from text.answer_native_plan import AnswerNativePlan,code,read
from text.constellation_corpus import file_sha,need
from claims.process_conditions import write_committed


def identity(pid):
    try:fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return fields[19] if fields[0]!='Z' else None


def code_files():
    root=Path(__file__).resolve().parents[2];result={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if path.is_relative_to(root/'scripts') and path.suffix=='.py':result[str(path.relative_to(root))]=file_sha(path)
    for name in code():result['scripts/'+name]=file_sha(root/'scripts'/name)
    return dict(sorted(result.items()))


def stage(plan,pin,readback,readback_pin,predecessor,output):
    view=AnswerNativePlan(plan,pin);proof=read(readback/'report.json',readback_pin)
    need(proof['complete'] and proof['plan']==dict(path=str(plan),pin=pin) and proof['code']==code()
         and proof['all_reused_native_exact'] and proof['all_token_counts_exact'],'ANSWER_QUEUE_PLAN_PROOF')
    launch=json.loads((predecessor/'launch.json').read_bytes());pid=launch['pid'];start=str(launch['starttime'])
    need(identity(pid)==start,'ANSWER_QUEUE_PREDECESSOR_NOT_LIVE')
    value=dict(schema='answer-native-queue-authority-v1',plan=dict(path=str(plan),pin=pin),
        readback=dict(path=str(readback),pin=readback_pin),code=code_files(),
        wait=dict(path=str(predecessor),pid=pid,starttime=start,launch_pin=file_sha(predecessor/'launch.json')),
        gpu_jobs_per_host=1,training=False,scientific_admission=False)
    need(not output.exists(),'ANSWER_QUEUE_AUTHORITY_EXISTS');pin=write_committed(output,value)
    return dict(pin=pin,wait=value['wait'],code_files=len(value['code']),inputs=view.report['inputs'])


def authenticate(path,pin):
    contract=read(path,pin);repo=Path(__file__).resolve().parents[2]
    need(contract['schema']=='answer-native-queue-authority-v1' and not contract['training'],'ANSWER_QUEUE_SCOPE')
    def fresh():
        need(file_sha(path)==pin,'ANSWER_QUEUE_AUTHORITY_CHANGED')
        for n,h in contract['code'].items():need(file_sha(repo/n)==h,'ANSWER_QUEUE_CODE_CHANGED')
        for name in ('plan','readback'):
            ref=contract[name];need(file_sha(Path(ref['path'])/'report.json')==ref['pin'],'ANSWER_QUEUE_REPORT_CHANGED')
        wait=contract['wait'];need(file_sha(Path(wait['path'])/'launch.json')==wait['launch_pin'],'ANSWER_QUEUE_OWNER_CHANGED')
        need(identity(wait['pid']) in (wait['starttime'],None),'ANSWER_QUEUE_PID_REUSED')
    fresh();view=AnswerNativePlan(contract['plan']['path'],contract['plan']['pin'])
    proof=read(Path(contract['readback']['path'])/'report.json',contract['readback']['pin'])
    need(proof['complete'] and proof['plan']==contract['plan'] and proof['code']==code()
         and proof['source_bytes_exact'] and proof['target_order_exact'] and proof['all_token_counts_exact']
         and proof['all_reused_native_exact'],'ANSWER_QUEUE_READBACK')
    for n,h in proof['files'].items():need(file_sha(Path(contract['readback']['path'])/n)==h,'ANSWER_QUEUE_READBACK_FILE')
    return contract,fresh


def run(a):
    a.output.mkdir(exist_ok=False);outputs=[];lock=None;repo=Path(__file__).resolve().parents[2]
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    try:
        contract,fresh=authenticate(a.authority,a.pin);owner=contract['wait']
        if a.mode=='probe':
            write_committed(a.output/'report.json',dict(complete=True,authority=a.pin,live_predecessor=identity(owner['pid']),
                gpu_children_started=0,code_files=len(contract['code']),scientific_admission=False));return
        lock=(repo/'.lease/answer-native-queue').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        write_committed(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            driver_pin=file_sha(__file__),wait=owner))
        if identity(owner['pid'])==owner['starttime']:
            fd=os.pidfd_open(owner['pid'])
            try:
                need(identity(owner['pid']) in (owner['starttime'],None),'ANSWER_QUEUE_PID_REUSED')
                event('waiting',pid=owner['pid'],starttime=owner['starttime'],mechanism='kernel_pidfd');select.select([fd],[],[])
            finally:os.close(fd)
        prior=Path(owner['path'])/'report.json';preceding=json.loads(prior.read_bytes())
        need(preceding['complete'],'ANSWER_QUEUE_PREDECESSOR_FAILED')
        for result in preceding['outputs']:
            done=read(Path(result['path'])/'report.json',result['pin']);need(done['complete'],'ANSWER_QUEUE_PREDECESSOR_OUTPUT')
        fresh();event('predecessor_complete',pin=file_sha(prior))
        def execute(label,extra):
            fresh();dest=a.output/label
            cmd=[sys.executable,'-I',str(repo/'scripts/text/answer_native_measure.py'),
                 '--plan',contract['plan']['path'],'--pin',contract['plan']['pin'],'--output',str(dest),*extra]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=cmd)
                status=child.wait();log.flush();os.fsync(log.fileno())
            need(status==0,'ANSWER_QUEUE_CHILD_FAILED');result=json.loads((dest/'report.json').read_bytes())
            need(result['complete'] and (label!='cold' or result['cold_all_inputs_exact']),'ANSWER_QUEUE_OUTPUT')
            item=dict(label=label,path=str(dest),pin=file_sha(dest/'report.json'));outputs.append(item);fresh();return item
        first=execute('native',[]);execute('cold',['--reference',first['path'],'--reference-pin',first['pin']])
        write_committed(a.output/'report.json',dict(complete=True,authority=a.pin,outputs=outputs,
            driver_pin=file_sha(__file__),predecessor=file_sha(prior),training_ready=False,
            scientific_admission=False,capability_demonstrated=False,acceptance='author-checked'))
        event('complete')
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(error=repr(exc),outputs=outputs));event('failed',error=repr(exc));raise
    finally:
        if lock:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['probe','full'],required=True)
    parser.add_argument('--authority',type=Path,required=True);parser.add_argument('--pin',required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
