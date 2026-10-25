"""Read-only census of every live placement owner and its remaining scientific work."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims import process_handoff as process

REPO=Path(__file__).resolve().parents[2];ROOT=Path('/zfs/archive/mnlm/evidence/fsv/29')
ORDER=('evaluation','full','recovery','driver','grouped','producer')
ROLES={
 'driver':(3213181,'404196319','complete_answer_recovery.py','phase115-complete-answer-recovery-v1-20261024'),
 'grouped':(3305473,'406019943','legal_grouped_placement_queue.py','phase115-complete-answer-recovery-v1-20261024/grouped'),
 'producer':(3314683,'406282314','legal_grouped_placement_training.py','phase115-complete-answer-recovery-v1-20261024/grouped/full'),
 'recovery':(3319183,'406373595','legal_answer_recovery_queue.py','legal-answer-recovery-queue-v1-20261024'),
 'full':(3371933,'407520305','legal_answer_full_queue.py','legal-answer-full-queue-v1-20261024'),
 'evaluation':(3381658,'407746221','legal_answer_evaluation_queue.py','legal-answer-evaluation-queue-v1-20261024')}
EDGES={'driver':['grouped'],'grouped':['producer'],'producer':[],'recovery':[],'full':[],'evaluation':[]}
TOPOLOGY=('phase158-topology.json','6d0e5ec4ae637cfa7763f564826be0f5323b9fdec263dc62c532c926205a8446')
BOUNDARY=('placement-driver-live-seal-v1-20261024','2828a9e97fd558bfb9f70d2c4764a1c67108dd06d9effb2de90819329da7036f')
CORRECTION=('legal-inversion-final-readback-v1-20261024','02e63b0d17f059a83dd43541a8590d9842a627c825ca12e2a33a252c53018582')
ARCHES=('shared','own_slot','whole_proposals')
STAGES=('native','bind','bind-cold','task-proof','task-proof-cold','task-full','task-cold','surface-proof',
        'surface-reverse','surface-cold','inversion-proof','inversion-proof-cold','inversion-full','inversion-cold','grouped')
REMAINING=(
 dict(name='placement_cuda_continuation',worker='legal_placement_resume_training.py',
      obligations=['original_small','boundary','resumed','cold','exact_match','source_readback']),
 dict(name='placement_full',worker='legal_placement_resume_training.py',documents=91540,targets=6455636,
      architectures=list(ARCHES),resume_from='new_quiescent_checkpoint_and_committed_prefix'),
 dict(name='placement_full_readback',worker='legal_grouped_placement_readback.py'),
 dict(name='prior_stage_reconciliation',worker=None,preserve_original_failures=True,reuse_corrected_inversion_readback=True),
 dict(name='tile_cuda',worker='legal_tile_gpu_queue.py'),
 dict(name='answer_epoch_cuda',worker='legal_answer_epoch_queue.py'),
 dict(name='answer_source_cuda_capacity',worker='legal_answer_source_queue.py'),
 dict(name='answer_full',worker='legal_answer_source_training.py',documents=91540,answers=6368511,decisions=6460051,
      architectures=list(ARCHES),subset=None,length_limit=None),
 dict(name='answer_evaluation',worker='legal_answer_evaluation_pipeline.py',cases=23513,trajectories=564312,
      qualification_cases=4,qualification_trajectories=192,scoring_scope='all',best_of_proposals=False))


def read(path,pin):
    process.need(process.sha(path)==pin,'CENSUS_PIN');return json.loads(Path(path).read_bytes())


def scope(value):
    process.need(value['order']==list(ORDER) and set(value['owners'])==set(ROLES),'CENSUS_ROSTER')
    by_pid={r['pid']:n for n,r in value['owners'].items()}
    for role,row in value['owners'].items():
        pid,start,script,root=ROLES[role]
        process.need(row['pid']==pid and row['starttime']==start and row['script']==script
                     and row['output']==str(ROOT/root),'CENSUS_IDENTITY')
        process.need([by_pid.get(p) for p in row['children']]==EDGES[role],'CENSUS_CLOSURE')
    process.need(value['remaining_work']==list(REMAINING),'CENSUS_WORK')
    process.need(set(value['completed_original'])==set(STAGES)-{'inversion-cold','grouped'},'CENSUS_COMPLETED_SCOPE')
    process.need(not value['retirement_authorized'] and not value['gpu_qualified']
                 and not value['replacement_stage_chain_wired'],'CENSUS_NOT_AUTHORIZATION')


def run(a):
    began=time.monotonic();a.output.mkdir(exist_ok=False);sources={}
    for name,path in (('source.py',Path(__file__)),('process_handoff.py',Path(process.__file__))):
        (a.output/name).write_bytes(path.read_bytes())
    def pinned(path,pin=None):
        path=Path(path);pin=process.sha(path) if pin is None else pin
        value=read(path,pin);sources[str(path)]=dict(pin=pin,bytes=path.stat().st_size);return value
    topology=pinned(ROOT/TOPOLOGY[0],TOPOLOGY[1]);old=pinned(ROOT/BOUNDARY[0]/'boundary.json',BOUNDARY[1])
    expected=[dict(pid=ROLES[n][0],starttime=ROLES[n][1]) for n in ORDER];order=[r['pid'] for r in expected]
    rows=process.capture(expected,order);owners={};journals={}
    for role,row in zip(ORDER,rows,strict=True):
        pid,start,script,directory=ROLES[role];output=ROOT/directory
        prior=topology['processes'][str(pid)]
        process.need(prior['starttime']==start and prior['children']==row['children'],'CENSUS_TOPOLOGY_CHANGED')
        argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
        process.need(sum(Path(v).name==script for v in argv)==1 and '--output' in argv
                     and Path(argv[argv.index('--output')+1]).resolve()==output.resolve(),'CENSUS_COMMAND')
        code=REPO/'scripts/audit'/script;sources[str(code)]=dict(pin=process.sha(code),bytes=code.stat().st_size)
        if role!='producer':
            recorded=topology['artifacts'][str(output)];launch=pinned(output/'launch.json',recorded['launch.json']['pin'])
            process.need(launch['pid']==pid and str(launch['starttime'])==start,'CENSUS_LAUNCH')
            path=output/'events.jsonl';process.need(process.sha(path)==recorded['events.jsonl']['pin'],'CENSUS_STAGE_ADVANCED')
            sources[str(path)]=dict(pin=process.sha(path),bytes=path.stat().st_size)
            journals[role]=[json.loads(v) for v in path.read_bytes().splitlines()]
            key='--authority-pin' if '--authority-pin' in argv else '--pin'
            cfg=pinned(Path(argv[argv.index('--authority')+1]),argv[argv.index(key)+1])
            process.need(launch['authority']==sources[argv[argv.index('--authority')+1]]['pin'],'CENSUS_AUTHORITY')
        owners[role]=dict(row,script=script,output=str(output))
    process.need(journals['driver'][-1]['stage']=='running' and journals['driver'][-1]['pid']==ROLES['grouped'][0]
                 and journals['grouped'][-1]['stage']=='running' and journals['grouped'][-1]['pid']==ROLES['producer'][0],
                 'CENSUS_ACTIVE_CHAIN')
    for role,parent in (('recovery','driver'),('full','recovery'),('evaluation','full')):
        last=journals[role][-1]
        process.need(last['stage']=='waiting' and last['owner']['pid']==ROLES[parent][0]
                     and last['owner']['starttime']==ROLES[parent][1],'CENSUS_WAITER_CHAIN')
    parent=ROOT/ROLES['driver'][3];plan=pinned(parent/'plan.json',topology['artifacts'][str(parent)]['plan.json']['pin'])
    process.need(tuple(r['label'] for r in plan)==STAGES,'CENSUS_ORIGINAL_PLAN')
    completed={}
    for event in journals['driver']:
        if event['stage']!='terminal':continue
        label=event['label'];process.need(label not in completed and Path(event['path'])==parent/label,'CENSUS_OUTPUT_PATH')
        r=pinned(parent/label/'report.json',event['pin']);process.need(r['complete'],'CENSUS_OUTPUT_INCOMPLETE')
        completed[label]=dict(path=event['path'],pin=event['pin'])
    failure=pinned(parent/'inversion-cold-failure.json')
    process.need(failure==dict(code='RECOVERY_WORKER_REFUSED',error="ValueError('RECOVERY_WORKER_FAILED:inversion-cold')"),
                 'CENSUS_RETAINED_FAILURE')
    correction=pinned(ROOT/CORRECTION[0]/'report.json',CORRECTION[1])
    process.need(correction['complete'] and correction['original_failure_preserved'] and not correction['training_restarted'],
                 'CENSUS_CORRECTION')
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:
            process.need(code in str(error),'CENSUS_WRONG_REFUSAL');controls[name]=str(error)
        else:raise ValueError('PROCESS_HANDOFF_CENSUS_CONTROL_ACCEPTED:'+name)
    five_order=[p for p in order if p in {r['pid'] for r in old['owners']}]
    reject('old_five_owner_snapshot',lambda:process.capture(old['owners'],five_order),'UNDECLARED_CHILD')
    value=dict(complete=True,order=list(ORDER),owners=owners,completed_original=completed,remaining_work=list(REMAINING),
        retained_failure=failure,correction=dict(path=str(ROOT/CORRECTION[0]),pin=CORRECTION[1]),
        original_checkpoint=dict(path=str(ROOT/BOUNDARY[0]),pin=BOUNDARY[1],sufficient_ownership=False),
        retirement_authorized=False,gpu_qualified=False,replacement_stage_chain_wired=False)
    scope(value)
    bad=deepcopy(value);del bad['owners']['grouped'];reject('omitted_owner',lambda:scope(bad),'CENSUS_ROSTER')
    bad=deepcopy(value);bad['remaining_work']=bad['remaining_work'][:-1]
    reject('omitted_evaluation',lambda:scope(bad),'CENSUS_WORK')
    bad=deepcopy(value);bad['retirement_authorized']=True
    reject('census_is_not_handoff',lambda:scope(bad),'CENSUS_NOT_AUTHORIZATION')
    process.need([process.binding(r) for r in process.capture(rows,order)]==[process.binding(r) for r in rows],
                 'CENSUS_FINAL_OWNERS')
    for path,ref in sources.items():process.need(process.sha(path)==ref['pin'],'CENSUS_SOURCE_CHANGED')
    for name,path in (('source.py',Path(__file__)),('process_handoff.py',Path(process.__file__))):
        process.need(process.sha(a.output/name)==process.sha(path),'CENSUS_CODE_CHANGED')
    value.update(sources=sources,controls=controls,files={n:process.sha(a.output/n) for n in ('source.py','process_handoff.py')},
        source_bytes=sum(v['bytes'] for v in sources.values()),seconds=time.monotonic()-began,
        gpu_residents=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
        signals_sent=0,gpu_used=False,optimizer_updates=0,payloads_reauthenticated=False,
        acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=process.write(a.output/'report.json',value),owners=len(owners),completed=len(completed))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():process.write(a.output/'failure.json',dict(error=repr(error)))
        raise
