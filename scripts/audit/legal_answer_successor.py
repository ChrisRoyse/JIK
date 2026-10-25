"""Serial successor coordinator; numerical workers and their inner leases are unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import legal_answer_successor_authority as authority
from audit.surface_device_queue import identity,need,sha,write

LOCK=authority.REPO/'.lease/legal-answer-successor'


class Children:
    def __init__(self,root,fresh,fd=None):self.root=root;self.fresh=fresh;self.fd=fd;self.outputs=[]

    def event(self,stage,**values):
        with (self.root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n')
            f.flush();os.fsync(f.fileno())

    def run(self,label,script,args,dest,report='report.json'):
        self.fresh();command=[sys.executable,'-I',str(authority.REPO/'scripts/audit'/script),*map(str,args),
                              '--output',str(dest)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','SUCCESSOR_GPU_HIDDEN')
        if self.fd is not None:os.fstat(self.fd)
        began=time.perf_counter()
        with (self.root/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=authority.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,pass_fds=() if self.fd is None else (self.fd,))
            self.event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command)
            status=child.wait();log.flush();os.fsync(log.fileno())
        self.event('terminal',label=label,exit_code=status,seconds=time.perf_counter()-began)
        need(status==0,'SUCCESSOR_CHILD_FAILED:'+label)
        path=dest/report;pin=sha(path);value=authority.read(path,pin)
        if report!='authority.json':need(value['complete'],'SUCCESSOR_CHILD_INCOMPLETE:'+label)
        ref=dict(label=label,path=str(dest),pin=pin,report=report)
        self.outputs.append(ref);self.fresh();return ref


def paired_args(name,ref):return ['--'+name,ref['path'],'--'+name+'-pin',ref['pin']]


def queue_steps(children,cfg,label,predecessor,root):
    q=cfg['queues'][label];old=authority.read(q['authority']['path'],q['authority']['pin'])
    script=q['script'];dest=root/label;auth=root/(label+'-authority');proof=root/(label+'-probe')
    args=['--mode','prepare','--root',str(authority.ROOT),'--wait-root',predecessor['path'],
          '--wait-pid',str(predecessor['pid']),'--wait-start',predecessor['starttime']]
    if label!='tile':args+=paired_args('inspection',old['inspection'])
    a=children.run(label+'-prepare',script,args,auth,'authority.json')
    value=authority.read(auth/'authority.json',a['pin'])
    need(value==dict(old,wait=predecessor),'SUCCESSOR_QUEUE_NUMERICS_CHANGED')
    args=['--root',str(authority.ROOT),'--authority',str(auth/'authority.json'),'--pin',a['pin']]
    p=children.run(label+'-probe',script,['--mode','probe',*args],proof)
    return children.run(label,script,['--mode','run',*args,*paired_args('proof',p)],dest)


def run(a):
    began=time.perf_counter();cfg,_,_,fresh=authority.authenticate(a.authority,a.pin)
    # No child, model import or GPU admission until all nine old owners are terminal.
    authority.handoff(cfg,a.pin,a.handoff,a.handoff_pin)
    proof=authority.read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['code']==cfg['code']
         and proof['mode']=='successor-wiring-proof' and proof['live_launch_refused']
         and not proof['gpu_used'],'SUCCESSOR_PROOF_REQUIRED')
    back=authority.read(a.readback/'report.json',a.readback_pin)
    need(back['complete'] and back['producer']==a.proof_pin and back['authority']==a.pin
         and back['source']==cfg['code']['audit/legal_answer_successor_readback.py']
         and back['all_event_tensors_exact'] and back['all_score_summaries_exact']
         and back['events']==3408 and back['trajectories']==192 and len(back['owners'])==9,
         'SUCCESSOR_READBACK_REQUIRED')
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(exist_ok=False)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            handoff=a.handoff_pin,driver_pin=sha(__file__),proof=a.proof_pin))
        children=Children(a.output,fresh,lock.fileno());outputs={}
        common=['--authority',str(a.authority),'--pin',a.pin,'--owner-fd',str(lock.fileno())]
        for stage in authority.STAGES:
            inputs=a.output/f'inputs-{stage}.json';write(inputs,outputs)
            extra=['--inputs',str(inputs),'--inputs-pin',sha(inputs)]
            if stage in ('prior-readback','placement','capacity','training-readback'):
                args=['--stage',stage,*common,*extra]
                if stage=='placement':args+=['--handoff',str(a.handoff),'--handoff-pin',a.handoff_pin]
                ref=children.run(stage,'legal_answer_successor_worker.py',args,a.output/stage)
            elif stage in ('tile','epoch','source'):
                previous={'tile':'placement','epoch':'tile','source':'epoch'}[stage]
                ref=queue_steps(children,cfg,stage,authority.owner(Path(outputs[previous]['path'])),a.output)
            elif stage=='training':
                cap=outputs['capacity'];value=authority.read(Path(cap['path'])/'report.json',cap['pin'])
                need(value['capacity_qualified'] and not value['full_population_trained'],'SUCCESSOR_CAPACITY_SCOPE')
                ref=children.run(stage,'legal_answer_source_training.py',value['arguments'],a.output/stage)
            else:
                ref=children.run(stage,'legal_answer_successor_evaluation.py',['--cuda',*common,*extra],a.output/stage)
            outputs[stage]=ref;write(a.output/f'completed-{stage}.json',ref)
        need(tuple(outputs)==authority.STAGES,'SUCCESSOR_FINAL_STAGES');fresh()
        value=authority.read(a.output/'evaluation/report.json',outputs['evaluation']['pin'])
        need(value['full_population_trained'] and value['full_population_evaluated']
             and value['scored']['cases']==23513,'SUCCESSOR_EVALUATION_INCOMPLETE')
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),outputs=outputs,
            full_population_trained=True,full_population_evaluated=True,original_complete=False,
            scientific_admission=False,architecture_winner=False,capability_demonstrated=False,
            seconds=time.perf_counter()-began,acceptance='author-checked'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','run'),required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','handoff','proof','readback'):p.add_argument('--'+n,type=Path)
    for n in ('pin','handoff-pin','proof-pin','readback-pin'):p.add_argument('--'+n)
    a=p.parse_args()
    try:
        if a.mode=='prepare':
            a.output.mkdir(exist_ok=False)
            for name in authority.NEW:
                dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((authority.REPO/'scripts'/name).read_bytes())
            pin=authority.prepare(a.output)
            authority.authenticate(a.output/'authority.json',pin);print(json.dumps(dict(pin=pin)),flush=True)
        else:run(a)
    except BaseException as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise
