"""Read the actual frozen boundary, retirement decision, terminal owners and released lease."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import legal_answer_successor_authority as authority
from claims import process_handoff as process
from claims import process_retirement as retirement


def run(a):
    cfg,_,_,fresh=authority.authenticate(a.authority,a.pin)
    handoff=authority.handoff(cfg,a.pin,a.input/'handoff.json',a.handoff_pin)
    manifest,back=authority.sealed(cfg,a.pin,handoff['boundary'],handoff['readback'])
    root=Path(manifest['quiescence']['path']);contract=authority.read(root/'contract.json',manifest['quiescence']['pin'])
    decision=retirement.decision(root,manifest['quiescence']['pin'])
    authority.need(decision is not None,'RETIREMENT_READER_DECISION')
    events=[json.loads(p.read_bytes()) for p in sorted(root.glob('event-*.json'))]
    order=contract['order']
    authority.need([r['pid'] for r in events if r['stage']=='stop_intent']==order
        and [r['pid'] for r in events if r['stage']=='retire_intent']==list(reversed(order))
        and events[-1]['stage']=='all_retired','RETIREMENT_READER_ORDER')
    originals={}
    for name,row in cfg['owners'].items():
        path=Path('/proc')/str(row['pid'])
        # os.kill(pid,0), used by the GPU lease, also sees unreaped zombies.
        authority.need(not path.exists(),'RETIREMENT_READER_PID_NOT_REAPED')
        originals[name]=dict(pid=row['pid'],starttime=row['starttime'],terminal=True)
    total=0
    for rel,h in manifest['files'].items():
        path=Path(handoff['boundary']['path'])/rel
        authority.need(process.sha(path)==h,'RETIREMENT_READER_SEAL_CHANGED');total+=path.stat().st_size
    for name,h in manifest['source_pins'].items():authority.need(process.sha(name)==h,'RETIREMENT_READER_SOURCE_CHANGED')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
    authority.need(not gpu.strip(),'RETIREMENT_READER_GPU_NOT_EMPTY')
    with (authority.REPO/'.lease/gpu.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=authority.REPO/'.lease/gpu'
        value=json.loads(lease.read_bytes()) if lease.exists() else None
        authority.need(value is None or value['pid']==cfg['owners']['producer']['pid'],'RETIREMENT_READER_LEASE_OWNER')
    fresh();a.output.mkdir(exist_ok=False)
    report=dict(complete=True,authority=a.pin,handoff=a.handoff_pin,decision=handoff['decision'],
        owners=originals,source=process.sha(__file__),files=len(manifest['files']),bytes=total,
        all_sealed_and_original_bytes_exact=True,all_original_pids_absent=True,signals_parent_first_children_retired_first=True,
        gpu_residents=gpu,gpu_lease_lock_available=True,stale_lease=value,successors_started=0,
        original_complete=False,acceptance='author-checked',number_status='measured')
    pin=process.write(a.output/'report.json',report);print(json.dumps(dict(pin=pin,files=report['files'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','input','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','handoff-pin'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
