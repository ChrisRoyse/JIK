"""Real CPU score/readback and fail-closed scope/ownership controls before queueing."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import paired_surface_contract as contract
from audit.paired_surface_queue import execute
from audit.surface_device_queue import need,sha,write


def controls(cfg,source,score,cold,output,fd):
    checks={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,RuntimeError) as exc:
            need(code in str(exc),'PAIRED_PROOF_WRONG_REFUSAL:'+name);checks[name]=str(exc)
        else:raise ValueError('PAIRED_PROOF_CONTROL_ACCEPTED:'+name)
    reject('live_predecessor',lambda:contract.terminal(cfg),'PAIRED_PREDECESSOR_LIVE')
    m=contract.read(**source['request'])
    reject('cpu_as_full',lambda:contract.worker_scope(m,cuda=True,qualification=False),'PAIRED_SCOPE_DEVICE')
    reject('cpu_as_qualification',lambda:contract.worker_scope(m,cuda=False,qualification=True),'PAIRED_SCOPE_QUALIFICATION_DEVICE')
    bad=deepcopy(m);bad['ordinals']=None
    reject('cpu_all_tasks',lambda:contract.worker_scope(bad,cuda=False,qualification=False),'PAIRED_SCOPE_POPULATION')
    # Synthetic metadata is only a negative-control mutation; never a result or model input.
    bad=deepcopy(m);bad.update(device='cuda:0',full_training=True)
    reject('partial_as_full',lambda:contract.worker_scope(bad,cuda=True,qualification=False),'PAIRED_SCOPE_POPULATION')
    bad=deepcopy(cold);bad['positions_per_decoder']=0
    reject('empty_qualification',lambda:contract.check_scores(score,bad,cuda=False,qualification=False),'PAIRED_SCORE_QUALIFICATION')
    bad=deepcopy(cold);bad['all_scored_logits_exact']=False
    reject('inexact_logits',lambda:contract.check_scores(score,bad,cuda=False,qualification=False),'PAIRED_SCORE_QUALIFICATION')
    bad=deepcopy(score);bad['evaluated_cases']=170
    reject('wrong_denominator',lambda:contract.check_scores(bad,cold,cuda=False,qualification=False),'PAIRED_SCORE_SCOPE')
    bad=deepcopy(score);bad['qualification']=True
    reject('relabeled_qualification',lambda:contract.check_scores(bad,cold,cuda=False,qualification=False),'PAIRED_SCORE_SCOPE')
    with (output/'wrong-owner').open('xb') as other:
        reject('wrong_owner_fd',lambda:contract.owner_fd(other.fileno()),'PAIRED_OWNER_FD')
    with contract.LOCK.open('a+b') as other:
        try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:checks['duplicate_owner']='REFUSED'
        else:raise ValueError('PAIRED_PROOF_DUPLICATE_OWNER')
    # Hold the actual open-file description in a child, release it in this process,
    # and prove that another owner still cannot enter until the child exits.
    command=[sys.executable,'-I','-c','import os,sys;os.fstat(int(sys.argv[1]));print("ready",flush=True);sys.stdin.buffer.read(1)',str(fd)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,pass_fds=(fd,))
    try:
        need(child.stdout.readline()==b'ready\n','PAIRED_PROOF_LOCK_CHILD')
        os.close(fd)
        with contract.LOCK.open('a+b') as other:
            try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:checks['inherited_child_owner']='REFUSED'
            else:raise ValueError('PAIRED_PROOF_INHERITED_OWNER')
    finally:
        child.stdin.write(b'x');child.stdin.flush();child.wait();child.stdin.close();child.stdout.close()
    with contract.LOCK.open('a+b') as other:
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);checks['owner_released_after_exit']='ACQUIRED'
    need(len(checks)==13,'PAIRED_PROOF_CONTROL_COUNT');return checks


def run(a):
    began=tick=time.perf_counter();cfg,source,fresh=contract.authenticate(a.authority,a.pin)
    a.output.mkdir(exist_ok=False);fd=os.open(contract.LOCK,os.O_RDWR|os.O_CREAT,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pair={k:source[k] for k in ('request','generated')};args=contract.arguments(pair,source['corpus'])
        phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
        first=execute('cpu','paired_surface_scoring.py',['--cpu-proof',*args],a.output/'cpu',fd)
        phases['cpu']=time.perf_counter()-tick;tick=time.perf_counter()
        second=execute('readback','paired_surface_readback.py',['--cpu-proof','--input',str(a.output/'cpu'),
            '--pin',first['pin']],a.output/'readback',fd)
        phases['readback']=time.perf_counter()-tick;tick=time.perf_counter()
        r=contract.read(first['path'],first['pin']);cold=contract.read(second['path'],second['pin'])
        need(cold['input_pin']==first['pin'],'PAIRED_PROOF_READBACK_INPUT')
        contract.check_scores(r,cold,cuda=False,qualification=False)
        need(cold['trajectories']==56 and cold['events']==600 and cold['matched_atoms']==88
             and cold['positions_per_decoder']==2216 and cold['missing_positions_per_decoder']==67872,
             'PAIRED_PROOF_REAL_CORPUS')
        # The ownership control closes fd deliberately. Never close a recycled descriptor.
        owned=fd;fd=None;checks=controls(cfg,source,r,cold,a.output,owned);fresh()
        need('torch' not in sys.modules,'PAIRED_PROOF_EARLY_MODEL_IMPORT')
        phases['negative_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'PAIRED_PROOF_PHASES')
        write(a.output/'report.json',dict(complete=True,authority=a.pin,code=cfg['code'],outputs=[first,second],
            all_cpu_scores_exact=True,controls=checks,controls_count=len(checks),cuda_executed=False,
            trajectories=56,events=600,positions_per_decoder=2216,full_training=False,
            qualification_cuda_executed=False,phases=phases,seconds=seconds,acceptance='author-checked'))
        print(json.dumps(dict(complete=True,pin=sha(a.output/'report.json'),seconds=seconds)),flush=True)
    finally:
        if fd is not None:os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('authority','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
