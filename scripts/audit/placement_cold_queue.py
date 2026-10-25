"""Wait for the exact long-answer process, cold replay it, then read both states."""
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
from audit.surface_device_queue import identity,need,sha,write

REPO=Path(__file__).resolve().parents[2];ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
LAUNCH=ROOT/'phase148-long-launch.json';PRODUCER=ROOT/'placement-long-v1-20261024/producer'
PROBE=ROOT/'placement-long-probe-v1-20261024'
PROBE_PIN='adc8c1397d16d7a81cb3ecf0259cf4d6a370fd4eb2a66ca2e525bf5b44d3a8e7'
READER=ROOT/'placement-long-probe-readback-v1-20261024'
NEW=('audit/placement_cold_queue.py','audit/placement_long_readback.py','audit/surface_device_queue.py')
LOCK=REPO/'.lease/native-placement-cold'
CONTROLS={'launch_pin','owner_identity','live_predecessor','probe_not_long','wrong_architecture',
          'code_changed','reader_pin','duplicate_driver','child_retains_lock','released_after_child_exit'}


def read(path,pin):
    need(sha(path)==pin,'PLACEMENT_HANDOFF_PIN');return json.loads(Path(path).read_bytes())


def owner(cfg):
    launch=read(LAUNCH,cfg['launch_pin']);ref=cfg['wait']
    need(launch['pid']==ref['pid'] and launch['starttime']==ref['starttime']
        and Path(launch['output'])==PRODUCER,'PLACEMENT_HANDOFF_OWNER')
    need(identity(ref['pid']) in (ref['starttime'],None),'PLACEMENT_HANDOFF_PID_REUSED')
    return launch


def ready(cfg):
    owner(cfg);need(identity(cfg['wait']['pid']) is None,'PLACEMENT_HANDOFF_PREDECESSOR_LIVE')


def qualify(cfg,report,mode):
    need(report['complete'] and report['mode']==mode and report['binding']==cfg['producer_binding']
        and report['all_tensors_exact'] and report['checkpoint_restarts_exact']
        and report['counts']['updates']==(1 if mode=='probe' else 3)
        and report['counts']['decisions']==(7 if mode=='probe' else 3217)
        and all(report[n] is False for n in ('gpu_used','promoted','cuda_qualified','capability_demonstrated')),
        'PLACEMENT_HANDOFF_RESULT_SCOPE')


def prepare(output):
    launch=json.loads(LAUNCH.read_bytes());binding=json.loads((PRODUCER/'binding.json').read_bytes())
    probe=read(PROBE/'report.json',PROBE_PIN);reader_pin=sha(READER/'report.json');reader=read(READER/'report.json',reader_pin)
    need(binding==probe['binding'] and binding['schema']=='native-placement-long-v1'
        and binding['architecture']=='shared' and binding['device']=='cpu','PLACEMENT_HANDOFF_BINDING')
    need(reader['complete'] and reader['mode']=='probe' and reader['producer']==PROBE_PIN
        and reader['all_numerical_bytes_exact'] and reader['counts']['updates']==2 and reader['counts']['decisions']==14,
        'PLACEMENT_HANDOFF_READER_PROOF')
    code={**binding['code'],**{n:sha(REPO/'scripts'/n) for n in NEW}}
    need(code['audit/placement_long_readback.py']==reader['reader_source'],'PLACEMENT_HANDOFF_READER_SOURCE')
    cfg=dict(schema='native-placement-cold-v1',wait=dict(pid=launch['pid'],starttime=launch['starttime']),
        launch_pin=sha(LAUNCH),binding_pin=sha(PRODUCER/'binding.json'),producer_binding=binding,
        probe=dict(path=str(PROBE),pin=PROBE_PIN),reader_proof=dict(path=str(READER),pin=reader_pin),code=code,
        plan=['cold_long','physical_readback'],gpu_children=0,automatic_retry=False)
    owner(cfg);write(output/'authority.json',cfg)


def authenticate(path,pin):
    cfg=read(path,pin)
    need(cfg['schema']=='native-placement-cold-v1' and cfg['plan']==['cold_long','physical_readback']
        and cfg['gpu_children']==0 and cfg['automatic_retry'] is False
        and cfg['probe']==dict(path=str(PROBE),pin=PROBE_PIN),'PLACEMENT_HANDOFF_AUTHORITY')
    binding=read(PRODUCER/'binding.json',cfg['binding_pin']);need(binding==cfg['producer_binding'],'PLACEMENT_HANDOFF_BINDING')
    expected={**binding['code'],**{n:sha(REPO/'scripts'/n) for n in NEW}}
    need(cfg['code']==expected,'PLACEMENT_HANDOFF_CODE_BINDING');owner(cfg)
    reader=read(Path(cfg['reader_proof']['path'])/'report.json',cfg['reader_proof']['pin'])
    need(reader['complete'] and reader['mode']=='probe' and reader['producer']==PROBE_PIN
        and reader['reader_source']==cfg['code']['audit/placement_long_readback.py']
        and reader['all_numerical_bytes_exact'] and reader['counts']['updates']==2 and reader['counts']['decisions']==14,
        'PLACEMENT_HANDOFF_READER_PROOF')
    probe=read(PROBE/'report.json',PROBE_PIN);qualify(cfg,probe,'probe')
    def fresh():
        need(sha(path)==pin and sha(LAUNCH)==cfg['launch_pin'] and sha(PRODUCER/'binding.json')==cfg['binding_pin'],
             'PLACEMENT_HANDOFF_CHANGED')
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h,'PLACEMENT_HANDOFF_CODE_CHANGED')
    fresh();return cfg,probe,fresh


def controls(cfg,report,fresh,output):
    result={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'PLACEMENT_HANDOFF_WRONG_REFUSAL:'+str(exc));result[name]=str(exc)
        else:raise ValueError('PLACEMENT_HANDOFF_NEGATIVE_ACCEPTED:'+name)
    reject('launch_pin',lambda:read(LAUNCH,'0'*64),'PLACEMENT_HANDOFF_PIN')
    altered=deepcopy(cfg);altered['wait']['starttime']='0'
    reject('owner_identity',lambda:owner(altered),'PLACEMENT_HANDOFF_OWNER')
    reject('live_predecessor',lambda:ready(cfg),'PLACEMENT_HANDOFF_PREDECESSOR_LIVE')
    reject('probe_not_long',lambda:qualify(cfg,report,'long'),'PLACEMENT_HANDOFF_RESULT_SCOPE')
    changed=deepcopy(report);changed['binding']['architecture']='whole_proposals'
    reject('wrong_architecture',lambda:qualify(cfg,changed,'probe'),'PLACEMENT_HANDOFF_RESULT_SCOPE')
    changed=deepcopy(cfg);changed['code'][NEW[0]]='0'*64;bad=output/'changed-code.json';write(bad,changed)
    reject('code_changed',lambda:authenticate(bad,sha(bad)),'PLACEMENT_HANDOFF_CODE_BINDING')
    reject('reader_pin',lambda:read(Path(cfg['reader_proof']['path'])/'report.json','0'*64),'PLACEMENT_HANDOFF_PIN')
    first=LOCK.open('a+');child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+') as other:
            reject('duplicate_driver',lambda:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        program='import os,sys; os.fstat(int(sys.argv[1])); print("inherited",flush=True); sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',program,str(first.fileno())],pass_fds=(first.fileno(),),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','PLACEMENT_HANDOFF_CONTROL_CHILD');first.close()
        with LOCK.open('a+') as other:
            reject('child_retains_lock',lambda:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'PLACEMENT_HANDOFF_CONTROL_EXIT')
        with LOCK.open('a+') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    fresh();need(set(result)==CONTROLS and 'torch' not in sys.modules,'PLACEMENT_HANDOFF_CONTROL_SCOPE')
    return result


def run(a):
    began=time.perf_counter();cfg,probe,fresh=authenticate(a.authority,a.pin)
    if a.mode=='probe':
        negative=controls(cfg,probe,fresh,a.output)
        write(a.output/'report.json',dict(complete=True,mode='probe',authority=a.pin,code=cfg['code'],controls=negative,
            torch_loaded=False,gpu_children_started=0,seconds=time.perf_counter()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));return
    proof=read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['mode']=='probe' and proof['authority']==a.pin and proof['code']==cfg['code']
        and set(proof['controls'])==CONTROLS and not proof['torch_loaded'] and proof['gpu_children_started']==0,
        'PLACEMENT_HANDOFF_CONTROL_REQUIRED')
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
            wait=cfg['wait'],driver_pin=sha(__file__),proof=a.proof_pin,torch_loaded='torch' in sys.modules))
        ref=cfg['wait'];owner(cfg)
        if identity(ref['pid'])==ref['starttime']:
            try:fd=os.pidfd_open(ref['pid'])
            except ProcessLookupError:ready(cfg)
            else:
                try:owner(cfg);event('waiting',owner=ref,method='kernel_pidfd',torch_loaded='torch' in sys.modules);select.select([fd],[],[])
                finally:os.close(fd)
        ready(cfg);fresh()
        need(not (PRODUCER/'failure.json').exists(),'PLACEMENT_HANDOFF_PRODUCER_REFUSED')
        need((PRODUCER/'report.json').exists(),'PLACEMENT_HANDOFF_TERMINAL_MISSING')
        producer_pin=sha(PRODUCER/'report.json');producer=read(PRODUCER/'report.json',producer_pin);qualify(cfg,producer,'long')
        write(a.output/'producer-terminal.json',dict(path=str(PRODUCER),pin=producer_pin,identity_terminal=True))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        stages=[]
        def execute(label,script,args):
            fresh();command=[sys.executable,'-I',str(REPO/'scripts/audit'/script),*args];tick=time.perf_counter()
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command,lock_inherited=True)
                status=child.wait();log.flush();os.fsync(log.fileno())
            write(a.output/(label+'-exit.json'),dict(exit_code=status,seconds=time.perf_counter()-tick))
            event('terminal',label=label,exit_code=status);need(status==0,'PLACEMENT_HANDOFF_CHILD_FAILED:'+label);fresh()
            path=a.output/label;pin=sha(path/'report.json');result=read(path/'report.json',pin)
            stages.append(dict(label=label,path=str(path),pin=pin));return result,pin
        cold,cold_pin=execute('cold','placement_long_answer.py',['--mode','long','--output',str(a.output/'cold'),
            '--probe',str(PROBE),'--probe-pin',PROBE_PIN,'--previous',str(PRODUCER),'--previous-pin',producer_pin])
        qualify(cfg,cold,'long');need(cold['previous_pin']==producer_pin and cold['cold_reproduced'],'PLACEMENT_HANDOFF_COLD')
        reader,reader_pin=execute('readback','placement_long_readback.py',['--mode','long','--root',str(ROOT),
            '--producer',str(PRODUCER),'--producer-pin',producer_pin,'--cold',str(a.output/'cold'),
            '--cold-pin',cold_pin,'--output',str(a.output/'readback')])
        need(reader['complete'] and reader['mode']=='long' and reader['all_numerical_bytes_exact']
            and reader['checkpoint_states_exact'] and reader['counts']['updates']==6 and reader['counts']['decisions']==6434
            and reader['producer']==producer_pin and reader['cold']==cold_pin and not reader['gpu_used'],
            'PLACEMENT_HANDOFF_READBACK')
        fresh();write(a.output/'report.json',dict(complete=True,mode='run',authority=a.pin,proof=a.proof_pin,stages=stages,
            producer=producer_pin,cold=cold_pin,reader=reader_pin,counts=reader['counts'],gpu_children_started=0,
            cuda_qualified=False,promoted=False,capability_demonstrated=False,torch_loaded='torch' in sys.modules,
            seconds=time.perf_counter()-began,acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','probe','run'),required=True)
    p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin'):p.add_argument('--'+n)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        prepare(a.output) if a.mode=='prepare' else run(a)
        path=a.output/('authority.json' if a.mode=='prepare' else 'report.json')
        print(json.dumps(dict(path=str(path),pin=sha(path))),flush=True)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)));raise
