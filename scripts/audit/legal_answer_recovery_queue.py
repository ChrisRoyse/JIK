"""Reconcile one retained reader failure, then execute unchanged CUDA qualifications."""
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
from audit import legal_answer_recovery_authority as authority
from audit.surface_device_queue import check_owner, identity, need, sha, write

REPO=authority.REPO;ROOT=authority.ROOT;LOCK=REPO/'.lease/legal-answer-reader-recovery'
CONTROLS={'proof_pin','correction_scope','missing_stage','duplicate_stage','extra_stage','extra_error',
          'owner_identity','live_owner','reused_owner','waiter_failure','changed_queue_numerics','duplicate_driver',
          'manifest_size','manifest_pin','child_retains_lock','released_after_child_exit'}


def probe(cfg,records,error,fresh,output):
    controls={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'ANSWER_RECOVERY_WRONG_REFUSAL:'+str(exc));controls[name]=str(exc)
        else:raise ValueError('ANSWER_RECOVERY_NEGATIVE_ACCEPTED:'+name)
    reject('proof_pin',lambda:authority.read(Path(cfg['correction']['cold']['path'])/'report.json','0'*64),
           'ANSWER_RECOVERY_PIN')
    bad=deepcopy(records);bad['cold']['comparison_tolerance']=1e-6
    reject('correction_scope',lambda:authority.correction_scope(bad),'ANSWER_RECOVERY_CORRECTION_SCOPE')
    labels=[x for x in authority.schedule.ORDER if x!='inversion-cold']
    for name,value in [('missing_stage',labels[:-1]),('duplicate_stage',labels+[labels[0]]),
                       ('extra_stage',labels+['invented'])]:
        reject(name,lambda value=value:authority.labels(value),'ANSWER_RECOVERY_MISSING_OR_EXTRA_WORK')
    reject('extra_error',lambda:authority.errors({'inversion-cold':error,'grouped':error},error),
           'ANSWER_RECOVERY_ADDITIONAL_FAILURE')
    reject('owner_identity',lambda:check_owner(dict(cfg['parent'],starttime='0')),'SURFACE_QUEUE_OWNER_IDENTITY')
    me=authority.owner(output)
    reject('live_owner',lambda:authority.ready(me),'ANSWER_RECOVERY_PREDECESSOR_LIVE')
    # The launch matches the forged identity, while the live kernel PID does not.
    reused=output/'reused';reused.mkdir();write(reused/'launch.json',dict(pid=os.getpid(),starttime='0'))
    reject('reused_owner',lambda:authority.ready(authority.owner(reused)),'SURFACE_QUEUE_PID_REUSED')
    reject('waiter_failure',lambda:authority.waiter_failure('tile',dict(error='ValueError',detail='GPU_FAILURE'),
                                                         cfg['parent']['path']),'ANSWER_RECOVERY_WAITER_FAILURE')
    old=authority.read(cfg['queues']['tile']['authority']['path'],cfg['queues']['tile']['authority']['pin'])
    changed=deepcopy(old);changed['code']['audit/legal_tile_gpu_queue.py']='0'*64
    reject('changed_queue_numerics',lambda:queue_binding(old,dict(changed,wait=me),me),
           'ANSWER_RECOVERY_NUMERICAL_AUTHORITY_CHANGED')
    root=Path(cfg['parent']['path'])/'bind';report=json.loads((root/'report.json').read_bytes())
    name,entry=next(iter(report['files'].items()));authority.payload(root,name,entry)
    reject('manifest_size',lambda:authority.payload(root,name,dict(entry,bytes=entry['bytes']+1)),
           'ANSWER_RECOVERY_OUTPUT_SIZE')
    reject('manifest_pin',lambda:authority.payload(root,name,dict(entry,sha256='0'*64)),
           'ANSWER_RECOVERY_OUTPUT_BYTES')
    first=LOCK.open('a+');child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+') as second:
            reject('duplicate_driver',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        code='import os,sys; os.fstat(int(sys.argv[1])); print("inherited",flush=True); sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',code,str(first.fileno())],pass_fds=(first.fileno(),),
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','ANSWER_RECOVERY_CHILD_HANDLE');first.close()
        with LOCK.open('a+') as second:
            reject('child_retains_lock',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'ANSWER_RECOVERY_LOCK_CHILD')
        with LOCK.open('a+') as second:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controls['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    need(set(controls)==CONTROLS,'ANSWER_RECOVERY_CONTROL_COVERAGE');fresh();return controls


def proof(a):
    r=authority.read(a.proof/'report.json',a.proof_pin)
    need(r['complete'] and r['mode']=='probe' and r['authority']==a.pin and r['driver_pin']==sha(__file__)
         and r['gpu_children_started']==0 and not r['terminal_reconciliation_performed']
         and set(r['controls'])==CONTROLS,'ANSWER_RECOVERY_PROBE')
    return r


def launch(a):
    write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
          mode=a.mode,driver_pin=sha(__file__),proof=a.proof_pin))


def wait(ref,event):
    check_owner(ref)
    if identity(ref['pid'])==ref['starttime']:
        try:fd=os.pidfd_open(ref['pid'])
        except ProcessLookupError:authority.ready(ref)
        else:
            try:
                check_owner(ref);event('waiting',owner=ref,wait='kernel_pidfd');select.select([fd],[],[])
            finally:os.close(fd)
    authority.ready(ref);event('owner_terminal',owner=ref)


def queue_binding(before,after,predecessor):
    need(after==dict(before,wait=predecessor),'ANSWER_RECOVERY_NUMERICAL_AUTHORITY_CHANGED')


def final_capacity(path):
    # The numerical worker's existing gate authenticates both capacity runs.
    from audit.legal_answer_source_training import capacity
    r=json.loads((path/'report.json').read_bytes());refs={v['label']:v for v in r['outputs']}
    need(r['complete'] and r['capacity_qualified'] and not r['full_population_trained']
         and set(refs)=={'proof','proof-cold'},'ANSWER_RECOVERY_CAPACITY_SCOPE')
    a,b=(refs[n] for n in ('proof','proof-cold'))
    p=authority.read(Path(a['path'])/'report.json',a['pin'])
    capacity(Path(a['path']),a['pin'],Path(b['path']),b['pin'],p['source'],p['binding'])
    return r


def run(a):
    start=time.perf_counter();cfg,sources,records,error,fresh=authority.authenticate(a.authority,a.pin)
    auth_seconds=time.perf_counter()-start
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'authority-helper.py').write_bytes(Path(authority.__file__).read_bytes())
    if a.mode=='probe':
        launch(a);controls=probe(cfg,records,error,fresh,a.output)
        write(a.output/'report.json',dict(complete=True,mode='probe',authority=a.pin,driver_pin=sha(__file__),
            controls=controls,gpu_children_started=0,terminal_reconciliation_performed=False,
            original_owner_states={n:identity(v['pid']) for n,v in dict(parent=cfg['parent'],
                **{n:q['owner'] for n,q in cfg['queues'].items()}).items()},
            authentication_seconds=auth_seconds,seconds=time.perf_counter()-start,
            rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked'))
        return
    proof(a)
    if a.mode=='seal':
        launch(a);result=authority.reconcile(cfg,sources,error);fresh()
        write(a.output/'report.json',dict(complete=True,mode='seal',authority=a.pin,driver_pin=sha(__file__),
            **result,authentication_seconds=auth_seconds,seconds=time.perf_counter()-start,
            scientific_admission=False,capability_demonstrated=False,acceptance='author-checked'))
        return
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);launch(a)
        event('authenticated',seconds=auth_seconds)
        for ref in [cfg['parent'],*[q['owner'] for q in cfg['queues'].values()]]:wait(ref,event)
        fresh()
        def child(label,script,args,dest):
            fresh();cmd=[sys.executable,'-I',str(script),*args,'--output',str(dest)]
            tick=time.perf_counter()
            with (a.output/(label+'.log')).open('xb') as log:
                proc=subprocess.Popen(cmd,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                      pass_fds=(lock.fileno(),))
                event('running',label=label,pid=proc.pid,starttime=identity(proc.pid),command=cmd,lock_inherited=True)
                status=proc.wait();log.flush();os.fsync(log.fileno())
            write(a.output/(label+'-exit.json'),dict(exit_code=status,seconds=time.perf_counter()-tick))
            event('terminal',label=label,exit_code=status);need(status==0,'ANSWER_RECOVERY_CHILD_FAILED:'+label);fresh()
        common=['--authority',str(a.authority),'--pin',a.pin,'--proof',str(a.proof),'--proof-pin',a.proof_pin]
        barrier=a.output/'barrier';child('barrier',Path(__file__),['--mode','seal',*common],barrier)
        b=json.loads((barrier/'report.json').read_bytes())
        need(b['complete'] and b['mode']=='seal' and b['authority']==a.pin and b['driver_pin']==sha(__file__),
             'ANSWER_RECOVERY_BARRIER')
        previous=authority.owner(barrier);authority.ready(previous);outputs=[]
        for name in ('tile','epoch','source'):
            q=cfg['queues'][name];old=authority.read(q['authority']['path'],q['authority']['pin'])
            script=REPO/'scripts/audit'/q['script'];dest=a.output/name;auth=a.output/(name+'-authority');test=a.output/(name+'-probe')
            args=['--mode','prepare','--root',str(ROOT),'--wait-root',previous['path'],
                  '--wait-pid',str(previous['pid']),'--wait-start',previous['starttime']]
            if name!='tile':args+=['--inspection',old['inspection']['path'],'--inspection-pin',old['inspection']['pin']]
            child(name+'-prepare',script,args,auth)
            next_cfg=json.loads((auth/'authority.json').read_bytes());queue_binding(old,next_cfg,previous)
            pin=sha(auth/'authority.json');args=['--root',str(ROOT),'--authority',str(auth/'authority.json'),'--pin',pin]
            child(name+'-probe',script,['--mode','probe',*args],test)
            child(name+'-run',script,['--mode','run',*args,'--proof',str(test),'--proof-pin',sha(test/'report.json')],dest)
            previous=authority.owner(dest);authority.ready(previous)
            if name=='tile':authority.epoch.predecessor(previous)
            elif name=='epoch':authority.source.predecessor(previous)
            else:final_capacity(dest)
            outputs.append(dict(label=name,path=str(dest),pin=sha(dest/'report.json')))
        fresh();write(a.output/'report.json',dict(complete=True,mode='run',authority=a.pin,driver_pin=sha(__file__),
            barrier=dict(path=str(barrier),pin=sha(barrier/'report.json')),outputs=outputs,
            original_workers_unchanged=True,capacity_qualified=True,full_population_trained=False,
            capability_demonstrated=False,scientific_admission=False,seconds=time.perf_counter()-start,acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','probe','seal','run'],required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin'):p.add_argument('--'+n)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    (a.output/'source-driver.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'source-helper.py').write_bytes(Path(authority.__file__).read_bytes())
    try:
        if a.mode=='prepare':
            pin=authority.prepare(a.output);authority.authenticate(a.output/'authority.json',pin)
            print(json.dumps(dict(path=str(a.output/'authority.json'),pin=pin)),flush=True)
        else:
            run(a);print(json.dumps(dict(path=str(a.output/'report.json'),pin=sha(a.output/'report.json'))),flush=True)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)));raise
