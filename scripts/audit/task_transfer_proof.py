"""Execute the new device boundary and compare with retained real task traces."""
import argparse
from copy import deepcopy
import fcntl
import json
from itertools import zip_longest
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import task_transfer_contract as contract
from audit.surface_device_queue import identity,need,sha,write

BASELINE=contract.ROOT/'task-answer-transfer-proof-v3-20261025'
BASELINE_PIN='622dbea4a831e2cd6f36327be9745ebac04f21bfdb4798a5fc9051f5862ecf38'
BASE_SCORE=contract.ROOT/'task-answer-transfer-score-v1-20261025'
BASE_SCORE_PIN='57611743928435e5a74b4c09b67d3ab632a38b1d74aaa333a4a62c73e1c80736'
LOCK=contract.REPO/'.lease/task-answer-transfer-queue'


def execute(label,args,output,*,owner_fd=None):
    command=[sys.executable,'-I',str(contract.REPO/'scripts/audit/task_transfer_device.py'),*args,'--output',str(output)]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if owner_fd is not None:command.extend(['--owner-fd',str(owner_fd)])
    with output.with_suffix('.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=contract.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=env,pass_fds=() if owner_fd is None else (owner_fd,))
        write(output.with_suffix('.launch.json'),dict(pid=p.pid,starttime=identity(p.pid),command=command))
        status=p.wait();log.flush();os.fsync(log.fileno())
    need(status==0,'TASK_HANDOFF_CHILD_FAILED:'+label)
    ref=dict(label=label,path=str(output/'report.json'),pin=sha(output/'report.json'))
    r=contract.read(ref['path'],ref['pin']);need(r['complete'],'TASK_HANDOFF_CHILD_INCOMPLETE')
    return ref


def compare(first,second):
    # Only import numerical code after worker termination; the waiting queue never imports this function's dependencies.
    from model.complete_answer_trace import replay
    from model.native_population_training import fingerprint
    a=contract.read(first['path'],first['pin']);b=contract.read(second['path'],second['pin'])
    x=contract.read(a['run']['path'],a['run']['pin']);y=contract.read(b['run']['path'],b['run']['pin'])
    need(x['binding']['checkpoints']==y['binding']['checkpoints'] and x['binding']['ordinals']==y['binding']['ordinals'],
         'TASK_HANDOFF_COLD_CHECKPOINTS')
    paths=[Path(z['run']['path']).parent for z in (a,b)];events=trajectories=ends=refusals=0
    for u,v in zip(x['results'],y['results'],strict=True):
        need(u['ordinal']==v['ordinal'],'TASK_HANDOFF_COLD_CASE');pp=paths[0]/u['path'];qq=paths[1]/v['path']
        c=contract.read(pp/'report.json',u['pin']);d=contract.read(qq/'report.json',v['pin'])
        need(c['public']==d['public'] and c['unavailable']==d['unavailable'],'TASK_HANDOFF_COLD_INPUT')
        for f,g in zip(c['traces'],d['traces'],strict=True):
            need(all(f[k]==g[k] for k in ('architecture','proposal','mode')),'TASK_HANDOFF_COLD_ARM')
            missing=object()
            for aa,bb in zip_longest(replay(pp/f['path'],f['pin']),replay(qq/g['path'],g['pin']),fillvalue=missing):
                need(aa is not missing and bb is not missing and fingerprint(aa)==fingerprint(bb),'TASK_HANDOFF_COLD_EVENT')
                events+=1
            ar=contract.read(pp/f['path']/'report.json',f['pin']);br=contract.read(qq/g['path']/'report.json',g['pin'])
            need(all(ar[k]==br[k] for k in ('status','refusal','generated_atoms','events','model_ended')),'TASK_HANDOFF_COLD_TERMINAL')
            trajectories+=1;ends+=ar['model_ended'];refusals+=not ar['model_ended']
    return dict(cases=len(x['results']),trajectories=trajectories,events=events,ends=ends,refusals=refusals,all_events_exact=True)


def controls(a,cfg,source):
    out={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,BlockingIOError) as exc:
            need(code in str(exc),'TASK_HANDOFF_WRONG_REFUSAL:'+name+':'+str(exc));out[name]=str(exc)
        else:raise ValueError('TASK_HANDOFF_NEGATIVE_PASSED:'+name)
    reject('authority_pin',lambda:contract.authenticate(a.authority,'0'*64),'TASK_DEVICE_PIN')
    reject('live_predecessor',lambda:contract.terminal(cfg),'TASK_DEVICE_PREDECESSOR_LIVE')
    for name in ('scope','code','owner'):
        changed=deepcopy(cfg)
        if name=='scope':changed['policy']['query_cases']=169
        elif name=='code':changed['code'][contract.NEW[0]]='0'*64
        else:changed['wait']['starttime']='0'
        path=a.output/(name+'.json');write(path,changed)
        reject(name,lambda:contract.authenticate(path,sha(path)),{'scope':'TASK_DEVICE_SCOPE','code':'TASK_DEVICE_CODE_BINDING','owner':'TASK_DEVICE_WAIT'}[name])
    m=contract.request(cfg,a.pin,source,full=False)
    for name in ('device','full_cpu','checkpoint','source'):
        q=deepcopy(m)
        if name=='device':q['device']='cuda:0'
        elif name=='full_cpu':q['ordinals']=None
        elif name=='checkpoint':q['checkpoints']['shared']['metadata']['counts']['documents']-=1
        else:q['pack']['pin']='0'*64
        reject(name,lambda:contract.request_scope(q,'cpu'),{'device':'TASK_DEVICE_REQUEST_DEVICE','full_cpu':'TASK_DEVICE_REQUEST_SCOPE',
            'checkpoint':'TASK_DEVICE_PROOF_SOURCE','source':'TASK_DEVICE_REQUEST_SCOPE'}[name])
    first=LOCK.open('a+b');child=None
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with LOCK.open('a+b') as second:reject('duplicate_owner',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        code='import os,sys;os.fstat(int(sys.argv[1]));print("inherited",flush=True);sys.stdin.readline()'
        child=subprocess.Popen([sys.executable,'-I','-c',code,str(first.fileno())],pass_fds=(first.fileno(),),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(child.stdout.readline().strip()=='inherited','TASK_HANDOFF_CHILD_LOCK');first.close()
        with LOCK.open('a+b') as second:reject('inherited_owner',lambda:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB),'Resource temporarily unavailable')
        child.stdin.write('exit\n');child.stdin.flush();need(child.wait()==0,'TASK_HANDOFF_CHILD_EXIT')
        with LOCK.open('a+b') as second:fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out['released_after_child_exit']=True
    finally:
        first.close()
        if child is not None:
            if child.poll() is None:child.stdin.close();child.wait()
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
    return out


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);cfg,source,fresh=contract.authenticate(a.authority,a.pin)
    m=contract.request(cfg,a.pin,source,full=False);write(a.output/'request.json',m);pin=sha(a.output/'request.json')
    phases=dict(authenticate=time.perf_counter()-tick);tick=time.perf_counter();checks=controls(a,cfg,source)
    write(a.output/'controls.json',checks);phases['controls']=time.perf_counter()-tick;tick=time.perf_counter()
    args=['--request',str(a.output/'request.json'),'--pin',pin]
    inspected=execute('inspect',['--inspect',*args],a.output/'inspect');fresh()
    produced=execute('cpu',['--cpu-proof',*args],a.output/'cpu');fresh()
    scored=execute('score',['--score',*args,'--generated',produced['path'],'--generated-pin',produced['pin']],a.output/'score');fresh()
    phases['real_children']=time.perf_counter()-tick;tick=time.perf_counter()
    cold=compare(dict(path=str(BASELINE/'report.json'),pin=BASELINE_PIN),produced)
    need(cold==dict(cases=3,trajectories=56,events=600,ends=0,refusals=56,all_events_exact=True),'TASK_HANDOFF_BASELINE_RECOUNT')
    old=contract.read(BASE_SCORE/'report.json',BASE_SCORE_PIN);new=contract.read(scored['path'],scored['pin'])
    for x,y in zip(old['cases'],new['cases'],strict=True):
        aa=contract.read(BASE_SCORE/x['path'],x['pin']);bb=contract.read(a.output/'score'/y['path'],y['pin'])
        # Trace seals change with the new authority; numerical metrics and pairings must not.
        for v in (aa,bb):
            for arm in v['arms']:arm.pop('trace')
        need(aa==bb,'TASK_HANDOFF_SCORE_CHANGED')
    phases['cold_score_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'TASK_HANDOFF_PHASES');fresh()
    r=dict(complete=True,authority=a.pin,code=cfg['code'],request=pin,controls=checks,outputs=[inspected,produced,scored],
        baseline=dict(path=str(BASELINE),pin=BASELINE_PIN),cold=cold,all_metrics_exact=True,full_cuda_executed=False,
        optimizer_updates=0,encoder_calls=0,capability_demonstrated=False,phases=phases,seconds=seconds,acceptance='author-checked')
    write(a.output/'report.json',r);print(json.dumps(dict(complete=True,pin=sha(a.output/'report.json'),controls=len(checks),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('authority','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
