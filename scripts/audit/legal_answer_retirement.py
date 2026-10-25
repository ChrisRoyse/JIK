"""Connect the real frozen nine-owner seal to crash-safe retirement; never launch a GPU job."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import legal_answer_successor_authority as authority
from claims import process_handoff as process
from claims import process_retirement as retirement

ROOT=authority.ROOT;REPO=authority.REPO
LOCK=REPO/'.lease/legal-answer-retirement'
FIXED={
 'initial':('placement-driver-live-seal-v1-20261024/boundary.json','2828a9e97fd558bfb9f70d2c4764a1c67108dd06d9effb2de90819329da7036f'),
 'tail':('placement-tail-proof-v1-20261024/report.json','a4d36d66c7b7ed149aeb84af9deab0817aa7c125d6231114ca19aefc04976c6a'),
 'retirement':('process-retirement-proof-v3-20261024/report.json','1b56c932b977cc2fcfdde6da011e86376c4e2fc2bd324099a5a87669e18809f4'),
 'retirement_readback':('process-retirement-readback-v2-20261024/report.json','c681e822f9152720ddf679d139c8841883af32fbfd90600e8498d2736b149b0b')}


def need(ok,code):
    if not ok:raise ValueError('ANSWER_RETIREMENT_'+code)


def ref(path,pin=None):return dict(path=str(path),pin=process.sha(path) if pin is None else pin)
def read(value):return authority.read(value['path'],value['pin'])


def qualifications(plan,cfg):
    need(plan['kind']=='legal-answer-retirement-plan-v1' and plan['source']==process.sha(__file__),'PLAN')
    need(plan['fixed']=={n:ref(ROOT/p,h) for n,(p,h) in FIXED.items()},'FIXED_PROOFS')
    initial=read(plan['fixed']['initial']);tail=read(plan['fixed']['tail'])
    need(initial['complete'] and initial['source_job']==cfg['owners']['producer']['output']
         and initial['binding']['scope']=='full' and initial['binding']['device']=='cuda','INITIAL')
    need(tail['complete'] and tail['kind']=='placement-tail-proof-v1' and not tail['gpu_used']
         and tail['replayed_records_per_run']==9 and tail['evaluation_rows']==3200
         and tail['exact_comparison']==dict(bytes=1583118240,tensors=433820),'TAIL_PROOF')
    need(all(process.sha(REPO/'scripts'/n)==h for n,h in tail['execution'].items())
         and tail['source']==process.sha(REPO/'scripts/audit/legal_placement_tail_proof.py'),'TAIL_SOURCE')
    rp=read(plan['fixed']['retirement']);rb=read(plan['fixed']['retirement_readback'])
    need(rp['complete'] and rp['coordinator_boundaries']==45 and rp['recovery_boundaries']==42
         and not rp['original_jobs_signalled'] and rp['successors_started']==0,'PROCESS_PROOF')
    need(all(process.sha(p)==h for p,h in rp['sources'].items()),'PROCESS_SOURCE')
    need(rb['complete'] and rb['proof']==plan['fixed']['retirement']['pin'] and rb['fixture_identities_terminal']
         and rb['source']==process.sha(REPO/'scripts/audit/process_retirement_readback.py'),'PROCESS_READBACK')
    proof=read(plan['proof']);back=read(plan['readback']);pin=plan['authority']['pin']
    need(proof['complete'] and proof['authority']==pin and proof['code']==cfg['code']
         and proof['mode']=='successor-wiring-proof' and proof['live_launch_refused'] and not proof['gpu_used'],'WIRING_PROOF')
    need(back['complete'] and back['producer']==plan['proof']['pin'] and back['authority']==pin
         and back['source']==cfg['code']['audit/legal_answer_successor_readback.py']
         and back['all_event_tensors_exact'] and back['all_score_summaries_exact']
         and back['events']==3408 and back['trajectories']==192,'WIRING_READBACK')
    for name,row in cfg['owners'].items():
        need(all(row[k]==rb['originals'][name][k] for k in ('pid','starttime','parent','command')),'PROCESS_OWNER')
    need(plan['branches'] and len(set(x.split(':')[0] for x in plan['branches']))==len(plan['branches']),'BRANCHES')
    for item in plan['branches']:
        name,step=item.split(':');need(name in ('shared','own_slot','whole_proposals') and int(step)>0,'BRANCHES')


def authenticate(path,pin):
    plan=authority.read(path,pin);a=plan['authority']
    cfg,_,_,fresh=authority.authenticate(Path(a['path']),a['pin']);qualifications(plan,cfg)
    return plan,cfg,fresh


def live(cfg):
    rows=process.capture(list(cfg['owners'].values()),[cfg['owners'][n]['pid'] for n in cfg['stop_order']])
    expected={v['pid']:v for v in cfg['owners'].values()}
    need(all(process.binding(r)==process.binding(expected[r['pid']]) and r['state'] in ('R','S','D') for r in rows),'LIVE_CLOSURE')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
    need([int(l.split(',')[0]) for l in gpu.strip().splitlines()]==[cfg['owners']['producer']['pid']],'GPU_OWNER')
    return dict(owners=rows,gpu=gpu)


def prepare(a):
    a.output.mkdir(exist_ok=False);cfg,_,_,fresh=authority.authenticate(a.authority,a.pin)
    value=dict(complete=True,kind='legal-answer-retirement-plan-v1',source=process.sha(__file__),
        authority=ref(a.authority,a.pin),proof=ref(a.proof/'report.json',a.proof_pin),
        readback=ref(a.readback/'report.json',a.readback_pin),branches=a.branch,
        fixed={n:ref(ROOT/p,h) for n,(p,h) in FIXED.items()})
    qualifications(value,cfg);before=live(cfg);controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:need(code in str(error),'CONTROL_CODE');controls[name]=str(error)
        else:raise ValueError('ANSWER_RETIREMENT_CONTROL_ACCEPTED:'+name)
    bad=deepcopy(value);bad['source']='0'*64
    reject('changed_coordinator',lambda:qualifications(bad,cfg),'PLAN')
    reject('wrong_authority_pin',lambda:authority.authenticate(a.authority,'0'*64),'SUCCESSOR_PIN')
    bad_cfg=deepcopy(cfg);bad_cfg['owners'].pop('tile_waiter')
    reject('missing_owner',lambda:authority.scope(bad_cfg),'SUCCESSOR_OWNER_SCOPE')
    bad=deepcopy(value);bad['proof']=bad['fixed']['tail']
    # A valid but different artifact cannot substitute for the wiring proof.
    reject('wrong_proof_kind',lambda:proof_kind(read(bad['proof'])),'WIRING_KIND')
    b=ref(ROOT/'placement-successor-live-seal-v1-20261024','f76d0d8b660d2ca61ddb1125a5a45a11714ac4c05d00d8a8ff801dc69330b8cb')
    reject('live_snapshot_not_retirement',lambda:authority.sealed(cfg,a.pin,b,{}),'SUCCESSOR_BOUNDARY_SCOPE')
    after=live(cfg);fresh();need('torch' not in sys.modules,'MODEL_IMPORTED')
    pin=process.write(a.output/'plan.json',value)
    process.write(a.output/'preflight.json',dict(complete=True,plan=pin,before=before,after=after,
        controls=controls,signals_sent=0,gpu_used=False,acceptance='author-checked'))
    print(json.dumps(dict(pin=pin,controls=len(controls))),flush=True)


def proof_kind(v):need(v.get('mode')=='successor-wiring-proof','WIRING_KIND')


def child(root,label,script,args,fresh):
    fresh();dest=root/label;command=[sys.executable,'-I',str(REPO/'scripts/audit'/script),*map(str,args),'--output',str(dest)]
    with (root/(label+'.log')).open('xb') as log:
        p=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        process.write(root/(label+'-launch.json'),dict(pid=p.pid,starttime=authority.identity(p.pid),command=command))
        status=p.wait();log.flush();os.fsync(log.fileno())
    need(status==0,'CHILD_FAILED:'+label);fresh();return dest


def finish(root,plan,cfg):
    transition=root/'transition';pin=process.sha(transition/'contract.json');decision=retirement.decision(transition,pin)
    need(decision is not None,'NO_RETIREMENT_DECISION');authority.terminal_owners(cfg)
    b=decision['receipts']['boundary'];back=decision['receipts']['readback']
    value=dict(complete=True,kind='coordinated-placement-retirement-v1',authority=plan['authority']['pin'],
        census=cfg['census']['pin'],owners=cfg['owners'],all_terminal=True,original_complete=False,
        boundary=dict(path=str(Path(b['path']).parent),pin=b['pin']),readback=back,
        decision=process.sha(transition/'decision.json'),coordinator_source=process.sha(__file__),
        successors_started=0,acceptance='author-checked',number_status='measured')
    path=root/'handoff.json'
    if path.exists():need(authority.read(path,process.sha(path))==value,'EXISTING_HANDOFF')
    else:process.write(path,value)
    authority.handoff(cfg,plan['authority']['pin'],path,process.sha(path))
    return path


def run(a):
    began=time.perf_counter();plan,cfg,fresh=authenticate(a.plan,a.plan_pin);before=live(cfg)
    a.output.mkdir(exist_ok=False);process.write(a.output/'plan-ref.json',ref(a.plan,a.plan_pin))
    process.write(a.output/'before.json',before)
    rows=list(cfg['owners'].values());order=[cfg['owners'][n]['pid'] for n in cfg['stop_order']]
    turn=retirement.Transition(a.output/'transition',rows,order,LOCK)
    try:
        turn.pause();fresh()
        initial=plan['fixed']['initial'];common=['--authority',plan['authority']['path'],'--pin',plan['authority']['pin']]
        args=[*common,'--initial',str(Path(initial['path']).parent),'--initial-pin',initial['pin'],
              '--transition',str(turn.root),'--transition-pin',turn.pin]
        for branch in plan['branches']:args+=['--branch',branch]
        boundary=child(a.output,'seal','legal_placement_successor_seal.py',args,fresh)
        bp=process.sha(boundary/'boundary.json')
        back=child(a.output,'seal-readback','legal_placement_successor_seal_readback.py',['--input',str(boundary),'--pin',bp],fresh)
        bref=dict(path=str(boundary),pin=bp);rref=ref(back/'report.json')
        manifest,_=authority.sealed(cfg,plan['authority']['pin'],bref,rref)
        need(manifest['quiescence']==dict(path=str(turn.root),pin=turn.pin),'TRANSITION_BINDING')
        turn.quiescent();fresh();qualifications(plan,cfg)
        turn.commit(dict(boundary=ref(boundary/'boundary.json',bp),readback=rref))
        process.write(a.output/'resolution.json',turn.resolve())
    except BaseException as error:
        try:process.write(a.output/'failure.json',dict(error=repr(error),decision_exists=(turn.root/'decision.json').exists()))
        finally:
            # Even failure-log I/O cannot prevent resolving the declared process state.
            result=turn.resolve()
            process.write(a.output/'failure-resolution.json',result)
        raise
    finally:turn.close()
    handoff=finish(a.output,plan,cfg);fresh()
    need('torch' not in sys.modules,'MODEL_IMPORTED')
    process.write(a.output/'report.json',dict(complete=True,plan=a.plan_pin,handoff=ref(handoff),
        all_terminal=True,original_complete=False,successors_started=0,seconds=time.perf_counter()-began,
        source=process.sha(__file__),acceptance='author-checked',number_status='measured'))
    print(json.dumps(dict(handoff=ref(handoff),seconds=time.perf_counter()-began)),flush=True)


def recover(a):
    plan,cfg,fresh=authenticate(a.plan,a.plan_pin)
    need(read(dict(path=str(a.input/'plan-ref.json'),pin=process.sha(a.input/'plan-ref.json')))==ref(a.plan,a.plan_pin),'RECOVERY_PLAN')
    root=a.input/'transition';pin=process.sha(root/'contract.json')
    result=retirement.recover(root,pin,a.output);fresh()
    if result['all_terminal']:finish(a.input,plan,cfg)
    print(json.dumps(dict(action=result['action'],all_terminal=result['all_terminal'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','run','recover'),required=True)
    for n in ('authority','proof','readback','plan','input','output'):p.add_argument('--'+n,type=Path)
    for n in ('pin','proof-pin','readback-pin','plan-pin'):p.add_argument('--'+n)
    p.add_argument('--branch',action='append');a=p.parse_args()
    {'prepare':prepare,'run':run,'recover':recover}[a.mode](a)
