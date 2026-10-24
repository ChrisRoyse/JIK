"""Supersede only an identified, childless, waiting comparison queue owner."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from legal_grouped_placement_queue import authenticate, identity, need, read, sha, write

PRIOR_DRIVER='cfdb997beb5b81331ac1eb515718e3faa7b2ca89b734b9ded950b2a915b56f1d'


def validate(snapshot, expected):
    need(snapshot['pid']==expected['pid'] and snapshot['starttime']==str(expected['starttime']),
         'GROUPED_HANDOFF_OWNER_IDENTITY')
    need(snapshot['state'] in ('S','T') and snapshot['children']==[], 'GROUPED_HANDOFF_ACTIVE_OWNER')
    need(snapshot['driver_pin']==PRIOR_DRIVER and snapshot['events']
         and all(e['stage']=='waiting_for_runtime_owner' for e in snapshot['events']), 'GROUPED_HANDOFF_NOT_WAITING')
    need(not snapshot['terminal'] and 'scripts/audit/legal_placement_comparison_queue.py' in snapshot['command'],
         'GROUPED_HANDOFF_PRIOR_SCOPE')


def snapshot(root, owner):
    pid=owner['pid']; proc=Path(f'/proc/{pid}'); fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    result=dict(pid=pid,starttime=fields[19],state=fields[0],
        children=[int(x) for x in (proc/'task'/str(pid)/'children').read_text().split()],
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),driver_pin=sha(root/'driver.py'),
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()],
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},
        terminal=any((root/n).exists() for n in ['report.json','failure.json']))
    validate(result,owner)
    return result


def rejects(fn):
    try: fn()
    except ValueError as error: return str(error)
    raise ValueError('GROUPED_HANDOFF_NEGATIVE_ACCEPTED')


def run(a):
    a.output.mkdir(parents=True,exist_ok=False); repo=Path(__file__).resolve().parents[2]
    fd=None; stopped=False; terminated=False
    try:
        contract,fresh=authenticate(a.authority,a.authority_pin)
        owner=json.loads((a.prior/'launch.json').read_bytes())
        need(owner['driver_pin']==PRIOR_DRIVER,'GROUPED_HANDOFF_DRIVER_IDENTITY')
        need((owner['wait_pid'],str(owner['wait_start']),owner['wait_root'])==(
             contract['wait']['pid'],contract['wait']['starttime'],contract['wait']['path']), 'GROUPED_HANDOFF_WAIT_AUTHORITY')
        before=snapshot(a.prior,owner); controls={}
        for name,field,value in [('wrong_pid','pid',-1),('reused_pid','starttime','0'),
                                 ('live_child','children',[1]),('running','events',[{'stage':'running'}]),
                                 ('terminal','terminal',True)]:
            bad=deepcopy(before); bad[field]=value; controls[name]=rejects(lambda:validate(bad,owner))
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        write(a.output/'before.json',before); write(a.output/'controls.json',controls)
        if a.mode=='probe':
            write(a.output/'report.json',dict(complete=True,authority=a.authority_pin,prior=str(a.prior),
                owner=owner,controls=controls,signals_sent=0,driver_pin=sha(__file__),scientific_admission=False))
            return
        need(a.proof and a.proof_pin and a.queue_probe and a.queue_probe_pin and a.next_output,
             'GROUPED_HANDOFF_PROOF_REQUIRED')
        proof=read(a.proof/'report.json',a.proof_pin); probe=read(a.queue_probe/'report.json',a.queue_probe_pin)
        need(proof['complete'] and proof['authority']==a.authority_pin and proof['owner']==owner
             and proof['driver_pin']==sha(__file__) and proof['signals_sent']==0
             and proof['controls']==controls and probe['complete'] and probe['authority']==a.authority_pin
             and probe['code']==contract['code'] and probe['gpu_children_started']==0,'GROUPED_HANDOFF_PROOF_BINDING')
        need(not a.next_output.exists(),'GROUPED_HANDOFF_NEXT_EXISTS')
        fd=os.pidfd_open(owner['pid']); fresh()
        need(identity(owner['pid'])==str(owner['starttime']),'GROUPED_HANDOFF_PID_REUSED')
        write(a.output/'intent.json',dict(action='supersede_waiting_owner',authority=a.authority_pin,
            old_pid=owner['pid'],old_starttime=str(owner['starttime']),prior=str(a.prior),next_output=str(a.next_output),
            proof=a.proof_pin,queue_probe=a.queue_probe_pin,driver_pin=sha(__file__)))
        # Freeze the waiting owner briefly so it cannot spawn after the child census.
        signal.pidfd_send_signal(fd,signal.SIGSTOP); stopped=True
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            state=Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')',1)[1].split()[0]
            if state=='T': break
            time.sleep(.01)
        need(state=='T','GROUPED_HANDOFF_STOP_NOT_OBSERVED')
        frozen=snapshot(a.prior,owner)
        need(frozen['events']==before['events'] and frozen['files']==before['files'],'GROUPED_HANDOFF_QUEUE_ADVANCED')
        write(a.output/'frozen.json',frozen)
        signal.pidfd_send_signal(fd,signal.SIGTERM); signal.pidfd_send_signal(fd,signal.SIGCONT); stopped=False
        need(bool(select.select([fd],[],[],10)[0]),'GROUPED_HANDOFF_OWNER_NOT_TERMINAL'); terminated=True
        need(identity(owner['pid'])!=str(owner['starttime']),'GROUPED_HANDOFF_OWNER_STILL_LIVE')
        handoff=dict(complete=True,status='superseded_waiting_owner',authority=a.authority_pin,
            old_pid=owner['pid'],old_starttime=str(owner['starttime']),children_before=frozen['children'],
            original_queue_complete=False,original_artifacts_unchanged=True,driver_pin=sha(__file__),
            acceptance='author-checked',scientific_admission=False)
        need(sha(a.prior/'driver.py')==PRIOR_DRIVER
             and {str(p.relative_to(a.prior)):sha(p) for p in a.prior.rglob('*') if p.is_file()}==before['files']
             and [json.loads(x) for x in (a.prior/'events.jsonl').read_text().splitlines()]==before['events']
             and not any((a.prior/n).exists() for n in ['report.json','failure.json']), 'GROUPED_HANDOFF_PRIOR_CHANGED')
        path=a.output/'handoff.json'; write(path,handoff)
        cmd=[sys.executable,'-I',str(repo/'scripts/audit/legal_grouped_placement_queue.py'),'--mode','full',
            '--authority',str(a.authority),'--authority-pin',a.authority_pin,'--handoff',str(path),'--handoff-pin',sha(path),
            '--output',str(a.next_output)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (a.output/'successor.log').open('xb') as log:
            child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        write(a.output/'successor.json',dict(pid=child.pid,starttime=identity(child.pid),command=cmd))
        write(a.output/'report.json',dict(complete=True,status='handoff_launched',handoff_pin=sha(path),
            successor_pid=child.pid,successor_starttime=identity(child.pid),successor_output=str(a.next_output),
            successor_readback_pending=True,controls=controls,acceptance='author-checked',scientific_admission=False))
    except BaseException as error:
        write(a.output/'failure.json',dict(error=repr(error),old_owner_terminated=terminated)); raise
    finally:
        if stopped and fd is not None: signal.pidfd_send_signal(fd,signal.SIGCONT)
        if fd is not None: os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=['probe','full'],required=True)
    for n in ['authority','prior','output']: p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--authority-pin',required=True)
    for n in ['proof','queue-probe']:
        p.add_argument('--'+n,type=Path); p.add_argument('--'+n+'-pin')
    p.add_argument('--next-output',type=Path)
    run(p.parse_args())
