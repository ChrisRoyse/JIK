"""Run the qualified tile-update assay on CUDA after a bound training owner exits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
PROOFS = dict(cpu=('legal-tile-update-proof-v1-20261024',
                  '7e9de129fcdc4b327c0663c655edef80a641ae89e0d24ae2621e82e9eef559ca'),
              cold=('legal-tile-update-proof-cold-v1-20261024',
                    '9d4ac3319799f411f9930ff26e4b06dc5327c2a1dd4196ba994489095b3acc59'))
NEW = ('audit/legal_tile_gpu_queue.py', 'audit/surface_device_queue.py')


def read(path, pin):
    need(sha(path) == pin, 'TILE_GPU_PIN'); return json.loads(Path(path).read_bytes())


def ready(owner):
    current = identity(owner['pid'])
    need(current in (owner['starttime'], None), 'TILE_GPU_PID_REUSED')
    need(current is None, 'TILE_GPU_PREDECESSOR_LIVE')


def prepare(a):
    first = read(a.root/PROOFS['cpu'][0]/'report.json', PROOFS['cpu'][1])
    owner = dict(path=str(a.wait_root), pid=a.wait_pid, starttime=a.wait_start,
                 launch_pin=sha(a.wait_root/'launch.json'))
    check_owner(owner)
    code = dict(first['code'], **{n:sha(REPO/'scripts'/n) for n in NEW})
    authority = dict(schema='legal-tile-gpu-queue-v1', wait=owner, code=code,
        proofs={n:dict(path=str(a.root/d), pin=pin) for n,(d,pin) in PROOFS.items()})
    write(a.output/'authority.json', authority)
    pin = sha(a.output/'authority.json'); authenticate(a.output/'authority.json', pin)
    print(json.dumps(dict(authority=str(a.output/'authority.json'),pin=pin)), flush=True)


def authenticate(path, pin):
    cfg = read(path,pin); need(cfg['schema']=='legal-tile-gpu-queue-v1', 'TILE_GPU_SCHEMA')
    files={Path(path):pin}; reports={}
    for label,(name,expected_pin) in PROOFS.items():
        ref=cfg['proofs'][label]; need(ref['pin']==expected_pin and Path(ref['path']).name==name, 'TILE_GPU_CPU_AUTHORITY')
        root=Path(ref['path']); r=read(root/'report.json',expected_pin); reports[label]=r
        need(r['complete'] and r['all_updates_exact'] and r['independent_restored_continuation_exact']
             and not r['gpu_used'] and r['optimizer_updates']==8 and r['binding']['device']=='cpu', 'TILE_GPU_CPU_SCOPE')
        need(set(cfg['code'])==set(r['code'])|set(NEW)
             and all(cfg['code'][n]==h for n,h in r['code'].items()), 'TILE_GPU_CPU_CODE')
        files[root/'report.json']=expected_pin; files.update({root/n:h for n,h in r['files'].items()})
    first,cold=reports['cpu'],reports['cold']
    need(cold['previous_pin']==PROOFS['cpu'][1] and first['binding']==cold['binding']
         and first['numerical_pins']==cold['numerical_pins'] and first['controls']==cold['controls'], 'TILE_GPU_CPU_COLD')
    check_owner(cfg['wait']); files[Path(cfg['wait']['path'])/'launch.json']=cfg['wait']['launch_pin']
    for p,h in files.items():need(sha(p)==h, 'TILE_GPU_PHYSICAL_PROOF')
    snapshots={p:stamp(p) for p in files}
    def fresh():
        for n,h in cfg['code'].items():need(sha(REPO/'scripts'/n)==h, 'TILE_GPU_CODE_CHANGED')
        for p,s in snapshots.items():need(stamp(p)==s, 'TILE_GPU_AUTHORITY_CHANGED')
    fresh(); return cfg,fresh


def probe(cfg, fresh):
    controls={}
    def reject(name,call,code):
        try:call()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc), 'TILE_GPU_WRONG_REFUSAL'); controls[name]=str(exc)
        else:raise ValueError('TILE_GPU_NEGATIVE_ACCEPTED:'+name)
    reject('proof_pin',lambda:read(Path(cfg['proofs']['cpu']['path'])/'report.json','0'*64),'TILE_GPU_PIN')
    reject('owner_identity',lambda:check_owner(dict(cfg['wait'],starttime='0')),'SURFACE_QUEUE_OWNER_IDENTITY')
    reject('live_owner',lambda:ready(dict(pid=os.getpid(),starttime=identity(os.getpid()))),'TILE_GPU_PREDECESSOR_LIVE')
    with (REPO/'.lease/legal-tile-gpu-queue').open('a+') as first:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (REPO/'.lease/legal-tile-gpu-queue').open('a+') as second:
            reject('duplicate_owner',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
    fresh(); return controls


def run(a):
    cfg,fresh=authenticate(a.authority,a.pin); owner=cfg['wait']; outputs=[]
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    if a.mode=='probe':
        controls=probe(cfg,fresh)
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),
              controls=controls,owner_identity=identity(owner['pid']),gpu_children_started=0,acceptance='author-checked'))
        print(json.dumps(dict(pin=sha(a.output/'report.json'),controls=controls)),flush=True); return
    proof=read(a.proof/'report.json',a.proof_pin)
    need(proof['complete'] and proof['authority']==a.pin and proof['driver_pin']==sha(__file__)
         and proof['gpu_children_started']==0 and set(proof['controls'])=={'proof_pin','owner_identity','live_owner','duplicate_owner'}, 'TILE_GPU_PROBE')
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    with (REPO/'.lease/legal-tile-gpu-queue').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),authority=a.pin,
              driver_pin=sha(__file__),wait=owner,proof=a.proof_pin))
        if identity(owner['pid'])==owner['starttime']:
            try:fd=os.pidfd_open(owner['pid'])
            except ProcessLookupError:ready(owner)
            else:
                try:
                    need(identity(owner['pid']) in (owner['starttime'],None),'TILE_GPU_PID_REUSED')
                    event('waiting',pid=owner['pid'],starttime=owner['starttime'],wait='kernel_pidfd')
                    select.select([fd],[],[])
                finally:os.close(fd)
        ready(owner); check_owner(owner); fresh()
        terminal=Path(owner['path'])/'report.json'; previous=json.loads(terminal.read_bytes())
        launch=json.loads((Path(owner['path'])/'launch.json').read_bytes())
        need(previous['complete'] and previous['driver_pin']==launch['driver_pin'], 'TILE_GPU_PREDECESSOR_FAILED')
        (a.output/'predecessor-report.json').write_bytes(terminal.read_bytes()); event('predecessor_terminal',pin=sha(terminal))
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        need(env.get('CUDA_VISIBLE_DEVICES')!='','TILE_GPU_HIDDEN')
        for label in ('gpu','gpu-cold'):
            fresh(); (a.output/(label+'-nvidia-before.txt')).write_text(subprocess.check_output(['nvidia-smi'],text=True))
            need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'TILE_GPU_RESIDENT')
            args=[] if not outputs else ['--previous',outputs[0]['path'],'--previous-pin',outputs[0]['pin']]
            cmd=[sys.executable,'-I',str(REPO/'scripts/audit/legal_tile_update_proof.py'),'--root',str(a.root),
                 '--output',str(a.output/label),*args]
            with (a.output/(label+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                event('running',label=label,pid=child.pid,starttime=identity(child.pid)); status=child.wait()
                log.flush(); os.fsync(log.fileno())
            event('terminal',label=label,exit_code=status); need(status==0,'TILE_GPU_CHILD_FAILED')
            path=a.output/label; r=json.loads((path/'report.json').read_bytes())
            need(r['complete'] and r['gpu_used'] and r['binding']['device']=='cuda:0' and r['all_updates_exact']
                 and r['independent_restored_continuation_exact'],'TILE_GPU_RESULT')
            if outputs:need(r['previous_pin']==outputs[0]['pin'],'TILE_GPU_COLD_LINEAGE')
            for n,h in r['files'].items():need(sha(path/n)==h,'TILE_GPU_RESULT_BYTES')
            outputs.append(dict(label=label,path=str(path),pin=sha(path/'report.json'))); fresh()
        write(a.output/'report.json',dict(complete=True,authority=a.pin,driver_pin=sha(__file__),outputs=outputs,
              predecessor_pin=sha(terminal),gpu_qualified=True,promoted=False,capability_demonstrated=False,acceptance='author-checked'))
        event('complete',pin=sha(a.output/'report.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=['prepare','probe','run'],required=True)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--output',type=Path,required=True)
    for n in ('authority','proof','wait-root'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin','wait-start'):p.add_argument('--'+n)
    p.add_argument('--wait-pid',type=int); a=p.parse_args(); a.output.mkdir(exist_ok=False)
    try:prepare(a) if a.mode=='prepare' else run(a)
    except BaseException as exc:
        write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc))); raise
