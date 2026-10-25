"""Final checkpoint export, cold device comparison, full public generation, private scoring."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
from audit import legal_answer_evaluation_handoff as handoff
from audit.surface_device_queue import identity,need,sha,write


def read(path,pin):return handoff.training.read(path,pin)


def traces(root,row):
    p=root/'run'/row['path'];r=read(p/'case.json',row['pin'])
    need(r['complete'] and r['status']=='generated','ANSWER_EVALUATION_QUALIFICATION_GAP')
    p=p/r['generation']['path'];g=read(p/'report.json',r['generation']['pin'])
    return p,g


def compare(left,left_pin,right,right_pin):
    from model.complete_answer_trace import replay
    from model.native_population_training import fingerprint
    a=read(left/'report.json',left_pin);b=read(right/'report.json',right_pin)
    need(a['complete'] and b['complete'] and a['device']==b['device'] and a['gpu_used']==b['gpu_used']
         and a['request']==b['request'] and a['memory_policy']==b['memory_policy'], 'ANSWER_EVALUATION_COLD_SCOPE')
    events=ends=refusals=0
    need(len(a['results'])==len(b['results'])==4,'ANSWER_EVALUATION_QUALIFICATION_CASES')
    for x,y in zip(a['results'],b['results'],strict=True):
        need(x['document']==y['document'] and x['ordinal']==y['ordinal'],'ANSWER_EVALUATION_COLD_CASE')
        p,r=traces(left,x);q,s=traces(right,y)
        need(r['checkpoints']==s['checkpoints'] and len(r['traces'])==len(s['traces'])==24,
             'ANSWER_EVALUATION_COLD_CHECKPOINTS')
        for u,v in zip(r['traces'],s['traces'],strict=True):
            need(all(u[k]==v[k] for k in ('architecture','proposal','mode')),'ANSWER_EVALUATION_COLD_GRID')
            pp=p/u['path'];qq=q/v['path'];aa=read(pp/'report.json',u['pin']);bb=read(qq/'report.json',v['pin'])
            need(all(aa[k]==bb[k] for k in ('status','refusal','generated_atoms','events','model_ended')),
                 'ANSWER_EVALUATION_COLD_TERMINAL')
            ends+=aa['model_ended'];refusals+=not aa['model_ended']
            for aa,bb in zip(replay(pp,u['pin']),replay(qq,v['pin']),strict=True):
                need(fingerprint(aa)==fingerprint(bb),'ANSWER_EVALUATION_COLD_EVENT');events+=1
    return dict(cases=4,trajectories=96,events=events,ends=ends,refusals=refusals,all_events_exact=True,
                device=a['device'],gpu_used=a['gpu_used'],memory_policy=a['memory_policy'])


def score_check(path,pin,cohort,cohort_pin,cases,scope):
    from model.complete_answer_cohort_scores import authority
    from model.legal_answer_public_pack import PublicPack
    r=read(path,pin);g=read(cohort/'report.json',cohort_pin)
    pack=PublicPack(g['pack']['path'],g['pack']['pin'])
    try:
        _,_,rows=authority(cohort,cohort_pin,pack,scope)
        need(len(rows)==cases and r['complete'] and r['declared_cases']==cases and len(r['cases'])==cases
             and r['model_loads']==r['encoder_calls']==r['optimizer_updates']==0 and not r['gpu_used'],
             'ANSWER_EVALUATION_SCORE_COVERAGE')
        b=read(path.parent/'scores/binding.json',r['binding'])
        need(b['cohort']==cohort_pin and b['scope']==scope,'ANSWER_EVALUATION_SCORE_BINDING')
        for ref,row in zip(r['cases'],rows,strict=True):
            c=read(path.parent/'scores'/ref['path'],ref['pin'])
            need(c['ordinal']==row['ordinal'] and c['document']==row['document'],'ANSWER_EVALUATION_SCORE_CASE')
            read(path.parent/'scores'/c['payload']['path'],c['payload']['pin'])
    finally:pack.close()
    return dict(path=str(path),pin=pin,cases=cases,actions=r['actions'],summary=r['summary'])


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);outputs=[];phases={}
    cfg,parent,template,fresh=handoff.authenticate(a.authority,a.pin)
    device='cuda:0' if a.cuda else 'cpu';terminal=None
    if a.cuda:
        terminal=handoff.terminal(cfg,parent);refs=terminal['training']['checkpoints'];training_ref=terminal['training']['pin']
    else:
        refs={n:v['reference'] for n,v in template['checkpoints'].items()};training_ref=template['authority']['training']
    m=handoff.exported(cfg,template,refs,training_ref,device=device,full=a.cuda,
                      finals=terminal['finals'] if terminal else None)
    need('torch' in sys.modules and not sys.modules['torch'].cuda.is_initialized(),'ANSWER_EVALUATION_PARENT_CUDA')
    mode='cuda' if a.cuda else 'cpu-proof'
    write(a.output/'binding.json',dict(authority=a.pin,code=cfg['code'],mode=mode,terminal=terminal,
        checkpoint_refs=refs,training=training_ref,plan=handoff.planned(a.output.parent)))
    def request(label,value):
        p=a.output/label;p.mkdir();write(p/'request.json',value);return p,sha(p/'request.json')
    q=deepcopy(m);q['ordinals']=template['ordinals'];qualification,qpin=request('qualification',q)
    cold,cpin=request('qualification-cold',q);need(qpin==cpin,'ANSWER_EVALUATION_REQUEST_COPY')
    if a.cuda:full,fullpin=request('full',m)
    phases['authentication_export']=time.perf_counter()-tick
    def event(stage,**values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,time=time.time(),**values),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def execute(label,script,args,report):
        fresh();tick=time.perf_counter();command=[sys.executable,'-I',str(handoff.REPO/'scripts/audit'/script),*args]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        if a.owner_fd is not None:os.fstat(a.owner_fd)
        with (a.output/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=handoff.REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,pass_fds=() if a.owner_fd is None else (a.owner_fd,))
            event('running',label=label,pid=child.pid,starttime=identity(child.pid),command=command)
            status=child.wait();log.flush();os.fsync(log.fileno())
        event('terminal',label=label,exit_code=status);need(status==0,'ANSWER_EVALUATION_CHILD_FAILED:'+label)
        r=read(report,sha(report));need(r['complete'],'ANSWER_EVALUATION_CHILD_INCOMPLETE')
        ref=dict(label=label,path=str(report),pin=sha(report));outputs.append(ref);fresh()
        phases[label]=time.perf_counter()-tick;return ref
    worker='complete_answer_device_cohort.py'
    args=['--inspect','--inspect-device',device,'--request',str(qualification/'request.json'),
          '--request-pin',qpin,'--output',str(a.output/'inspect')]
    execute('inspect',worker,args,a.output/'inspect/report.json')
    generated=[]
    for label,root,pin in (('qualification',qualification,qpin),('qualification-cold',cold,cpin)):
        args=['--'+mode,'--request',str(root/'request.json'),'--request-pin',pin,'--output',str(root/'worker')]
        generated.append(execute(label,worker,args,root/'worker/report.json'))
    tick=time.perf_counter();check=compare(qualification/'worker',generated[0]['pin'],cold/'worker',generated[1]['pin'])
    write(a.output/'qualification-readback.json',check);phases['qualification_readback']=time.perf_counter()-tick
    cohort=qualification/'worker';cohort_pin=generated[0]['pin'];scope='proof';cases=4
    if a.cuda:
        need(check['gpu_used'] and check['device']=='cuda:0' and check['memory_policy']['intervention']==dict(kind='trained_runtime_policy'),
             'ANSWER_EVALUATION_CUDA_QUALIFICATION')
        args=['--cuda','--request',str(full/'request.json'),'--request-pin',fullpin,'--output',str(full/'worker')]
        result=execute('full',worker,args,full/'worker/report.json');cohort=full/'worker';cohort_pin=result['pin'];scope='all';cases=23513
    args=['--cohort',str(cohort),'--cohort-pin',cohort_pin,'--root',str(handoff.ROOT),
          '--output',str(a.output/'score'),'--receipt','result','--scope',scope]
    scored=execute('score','legal_answer_cohort_scores.py',args,a.output/'score/result.json')
    tick=time.perf_counter();summary=score_check(Path(scored['path']),scored['pin'],cohort,cohort_pin,cases,scope);fresh()
    need(not sys.modules['torch'].cuda.is_initialized(),'ANSWER_EVALUATION_PARENT_CUDA')
    phases['score_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'ANSWER_EVALUATION_PHASES')
    value=dict(complete=True,mode=mode,authority=a.pin,code=cfg['code'],outputs=outputs,qualification=check,
        scored=summary,full_population_evaluated=a.cuda,full_population_trained=a.cuda,
        optimizer_updates=0,admitted_senses=0,architecture_winner=False,capability_demonstrated=False,
        private_reference_only_after_generation=True,phases=phases,seconds=seconds,acceptance='author-checked')
    write(a.output/'report.json',value);event('complete',pin=sha(a.output/'report.json'))
    print(json.dumps(dict(complete=True,pin=sha(a.output/'report.json'),cases=cases,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-proof',action='store_true');g.add_argument('--cuda',action='store_true')
    p.add_argument('--authority',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owner-fd',type=int)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
