"""Independently recount retirement proof artifacts and inspect live kernel owners."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.process_conditions import write_committed

NAMES=('evaluation','full','recovery','source_waiter','epoch_waiter','tile_waiter','driver','grouped','producer')


def need(ok, code):
    if not ok:raise ValueError('RETIREMENT_READBACK_'+code)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def bound(path,pin):
    need(sha(path)==pin,'PIN');return read(path)


def kernel(pid):
    root=Path('/proc')/str(pid)
    try:
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return dict(starttime=stat[19],state=stat[0],parent=int(stat[1]),
                    command=hashlib.sha256((root/'cmdline').read_bytes()).hexdigest())
    except FileNotFoundError:return None


def transition(before,after,retiring):
    need(set(before)==set(after)==set((*NAMES,'unrelated')),'ROSTER')
    for name,row in after.items():
        prior=before[name]
        need((row['pid'],row['starttime'])==(prior['pid'],prior['starttime']),'IDENTITY')
        if name!='unrelated' and retiring:need(row['state'] in ('absent','Z'),'NOT_TERMINAL')
        else:need(row['state'] in ('R','S','D') and row['heartbeat']>prior['heartbeat'],'NO_PROGRESS')


def run(a):
    began=time.perf_counter();v=bound(a.proof/'report.json',a.proof_pin)
    need(v['complete'] and v['kind']=='process-retirement-proof-v1' and v['owners']==9
         and v['unrelated']==1 and v['successors_started']==0 and not v['gpu_used']
         and not v['original_jobs_signalled'] and not v['scientific_checkpoint_qualified'],'SCOPE')
    for path,pin in v['sources'].items():need(sha(path)==pin,'SOURCE_CHANGED')
    total=0
    for rel,pin in v['files'].items():
        path=a.proof/rel;need(path.resolve().is_relative_to(a.proof.resolve()),'PATH_ESCAPE')
        need(sha(path)==pin,'FILE_CHANGED');total+=path.stat().st_size
    expected=['contract']+[name for _ in NAMES for name in ('stop_intent','stopped')]
    expected+=['quiescent','boundary_written','readback_written','decision_prepared','decision_linked','decision_committed']
    expected+=[name for _ in NAMES for name in ('retire_intent','retired')]+['all_retired','complete']
    need(read(a.proof/'normal/transition/report.json')['markers']==expected,'BOUNDARY_CENSUS')
    normal=read(a.proof/'normal/readback.json')
    normal_contract=read(a.proof/'normal/transition/contract.json')
    normal_events=[read(p) for p in sorted((a.proof/'normal/transition').glob('event-*.json'))]
    need([r['pid'] for r in normal_events if r['stage']=='stop_intent']==normal_contract['order']
         and [r['pid'] for r in normal_events if r['stage']=='retire_intent']==list(reversed(normal_contract['order'])),
         'PARENT_CHILD_SIGNAL_ORDER')
    for name in NAMES:
        need(normal['before'][name]['state'] in ('R','S','D') and normal['paused'][name]['state']=='T'
             and normal['held'][name]['state']=='T'
             and normal['paused'][name]['heartbeat']==normal['held'][name]['heartbeat'],'PAUSE_SCOPE')
    transition(normal['held'],normal['after'],True)
    need(normal['held']['unrelated']['heartbeat']>normal['paused']['unrelated']['heartbeat'],'UNRELATED_PAUSED')
    need(len(v['results'])==v['coordinator_boundaries']==len(expected),'COORDINATOR_COVERAGE')
    cases=[];fixture_ids=set()
    for index,item in enumerate(v['results'],1):
        root=a.proof/f'crash-{index:03d}'
        need(item['path']==str(root) and item['boundary']==index and item['marker']==expected[index-1],'CASE_BINDING')
        row=read(root/'readback.json');committed=index>=expected.index('decision_linked')+1
        need(row['committed']==item['committed']==committed,'DECISION_CLASSIFICATION')
        need((root/'transition/decision.json').exists()==committed,'PHYSICAL_DECISION')
        report=bound(root/'recovery/report.json',row['recovery'])
        need(report['action']==('retire' if committed else 'resume') and report['all_terminal']==committed
             and report['successors_started']==0,'RECOVERY_SCOPE')
        transition(row['stopped'],row['after'],committed)
        contract=bound(root/'transition/contract.json',report['contract'])
        need(contract['order']==[r['pid'] for r in contract['owners']] and len(contract['owners'])==9,'CONTRACT')
        if committed:
            d=read(root/'transition/decision.json');need(d['contract']==report['contract'] and d['action']=='retire','DECISION')
            for ref in d['receipts'].values():need(sha(ref['path'])==ref['pin'],'READINESS_CHANGED')
        fixture_ids.update((r['pid'],r['starttime']) for r in row['after'].values())
        cases.append(dict(index=index,marker=item['marker'],action=report['action']))
    need(len(v['recovery_results'])==44 and v['recovery_boundaries']==42,'RECOVERY_COVERAGE')
    for mode in ('resume','retire'):
        selected=[r for r in v['recovery_results'] if r['mode']==mode]
        need([r['boundary'] for r in selected]==list(range(22)),'RECOVERY_BOUNDARIES')
        for item in selected:
            root=a.proof/f"recovery-{mode}-{item['boundary']:03d}";need(item['path']==str(root),'RECOVERY_PATH')
            row=read(root/'readback.json');transition(row['middle'],row['after'],mode=='retire')
            done=read(root/('second' if item['boundary'] else 'first')/'report.json')
            need(done['action']==mode and done['successors_started']==0,'RECOVERY_RESULT')
            fixture_ids.update((r['pid'],r['starttime']) for r in row['after'].values())
    negative=read(a.proof/'negative-recovery/readback.json')
    for name in NAMES:
        need(negative['before'][name]['state']==negative['held'][name]['state']=='T'
             and negative['before'][name]['heartbeat']==negative['held'][name]['heartbeat'],'NEGATIVE_SIGNALLED')
    transition(negative['held'],negative['after'],False)
    fixture_ids.update((r['pid'],r['starttime']) for r in normal['after'].values())
    fixture_ids.update((r['pid'],r['starttime']) for r in negative['after'].values())
    for pid,start in fixture_ids:
        actual=kernel(pid);need(actual is None or actual['starttime']!=start,'FIXTURE_LEAK')
    authority=bound(a.authority,a.authority_pin);need(set(authority['owners'])==set(NAMES),'ORIGINAL_ROSTER')
    originals={}
    for name,ref in authority['owners'].items():
        now=kernel(ref['pid']);need(now is not None and now['starttime']==ref['starttime']
            and now['command']==ref['command'] and now['state'] in ('R','S','D'),'ORIGINAL_CHANGED')
        originals[name]=dict(pid=ref['pid'],**now)
    controls={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as error:need(code in str(error),'CONTROL_CODE');controls[name]=str(error)
        else:raise ValueError('RETIREMENT_READBACK_CONTROL_ACCEPTED:'+name)
    reject('wrong_report_pin',lambda:bound(a.proof/'report.json','0'*64),'PIN')
    bad=deepcopy(normal['after']);bad['producer']['state']='S'
    reject('live_owner_called_retired',lambda:transition(normal['held'],bad,True),'NOT_TERMINAL')
    bad=deepcopy(negative['after']);bad['unrelated']['heartbeat']=negative['held']['unrelated']['heartbeat']
    reject('unrelated_stalled',lambda:transition(negative['held'],bad,False),'NO_PROGRESS')
    need('torch' not in sys.modules,'MODEL_IMPORTED')
    a.output.mkdir(exist_ok=False)
    report=dict(complete=True,proof=a.proof_pin,source=sha(__file__),files=len(v['files']),bytes=total,
        coordinator_cases=cases,recovery_boundaries=42,fixture_identities=len(fixture_ids),fixture_identities_terminal=True,
        originals=originals,authority=a.authority_pin,controls=controls,gpu_used=False,original_jobs_signalled=False,
        gpu_residents=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
        scientific_checkpoint_qualified=False,seconds=time.perf_counter()-began,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),files=report['files'],
        coordinator_boundaries=len(cases),recovery_boundaries=42)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('proof','authority','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('proof-pin','authority-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args();run(a)
