"""After actual task-transfer completion, qualify and cold-recount both decoders."""
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
from audit import paired_surface_contract as contract
from audit.surface_device_queue import check_owner,identity,need,sha,write


def execute(label,script,args,output,fd):
    contract.owner_fd(fd)
    command=[sys.executable,'-I',str(contract.REPO/'scripts/audit'/script),*args,
             '--owner-fd',str(fd),'--output',str(output)]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    need(env.get('CUDA_VISIBLE_DEVICES')!='','PAIRED_QUEUE_GPU_HIDDEN')
    with output.with_suffix('.log').open('xb') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,))
        write(output.with_suffix('.launch.json'),dict(pid=p.pid,starttime=identity(p.pid),command=command))
        status=p.wait()
    need(status==0,'PAIRED_QUEUE_CHILD_FAILED:'+label)
    path=output/'report.json';pin=sha(path);need(contract.read(path,pin)['complete'],'PAIRED_QUEUE_CHILD_INCOMPLETE')
    return dict(label=label,path=str(path),pin=pin)


def run(a):
    cfg,source,fresh=contract.authenticate(a.authority,a.pin);proof=contract.read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['code']==cfg['code']
         and proof['all_cpu_scores_exact'] and proof['controls_count']==13 and not proof['cuda_executed'],
         'PAIRED_QUEUE_PROOF')
    need('torch' not in sys.modules,'PAIRED_QUEUE_EARLY_MODEL_IMPORT')
    with contract.LOCK.open('a+b') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(exist_ok=False)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
              driver_pin=sha(__file__),proof=a.proof_pin,wait=cfg['wait']))
        def event(stage,**values):
            with (a.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        wait=cfg['wait'];fresh();check_owner(wait)
        if identity(wait['pid'])==wait['starttime']:
            try:fd=os.pidfd_open(wait['pid'])
            except ProcessLookupError:pass
            else:
                try:
                    check_owner(wait);event('waiting',owner=wait,wait='kernel_pidfd',torch_loaded=False)
                    select.select([fd],[],[])
                finally:os.close(fd)
        began=tick=time.perf_counter();fresh();inputs=contract.terminal(cfg)
        write(a.output/'handoff.json',inputs);phases=dict(handoff=time.perf_counter()-tick);outputs=[]
        event('predecessor_complete',pin=inputs['parent'])
        for scope in ('qualification','full'):
            flags=['--cuda']+(['--qualification'] if scope=='qualification' else [])
            args=contract.arguments(inputs['pairs'][scope],inputs['corpus']);fresh();tick=time.perf_counter()
            event('launching',label=scope)
            first=execute(scope,'paired_surface_scoring.py',flags+args,a.output/scope,owner.fileno());outputs.append(first)
            phases[scope]=time.perf_counter()-tick;fresh();tick=time.perf_counter();label=scope+'-readback'
            event('launching',label=label)
            cold=execute(label,'paired_surface_readback.py',flags+['--input',str(Path(first['path']).parent),
                '--pin',first['pin']],a.output/label,owner.fileno());outputs.append(cold)
            phases[label]=time.perf_counter()-tick;tick=time.perf_counter()
            r=contract.read(first['path'],first['pin']);c=contract.read(cold['path'],cold['pin'])
            need(c['input_pin']==first['pin'],'PAIRED_QUEUE_READBACK_INPUT')
            contract.check_scores(r,c,cuda=True,qualification=scope=='qualification');fresh()
            phases[scope+'-gate']=time.perf_counter()-tick;event('complete',label=scope,score=first['pin'],readback=cold['pin'])
        need([v['label'] for v in outputs]==list(contract.STAGES),'PAIRED_QUEUE_STAGES')
        seconds=time.perf_counter()-began;need(.9<=sum(phases.values())/seconds<=1.001,'PAIRED_QUEUE_PHASES')
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),proof=a.proof_pin,
            handoff=sha(a.output/'handoff.json'),outputs=outputs,query_cases=170,full_training=True,
            teacher_forced=True,free_generation_tested=False,capability_demonstrated=False,architecture_winner=False,
            optimizer_updates=0,encoder_calls=0,phases=phases,seconds_after_wait=seconds,acceptance='author-checked'))
        event('terminal',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    for name in ('authority','proof','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--pin');p.add_argument('--proof-pin');a=p.parse_args()
    try:
        if a.prepare:print(json.dumps(dict(pin=contract.prepare(a.output))),flush=True)
        else:run(a)
    except BaseException as exc:
        if a.output is not None and a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
