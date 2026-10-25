"""Independent persisted-byte and event readback of successor wiring evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint


def need(ok,code):
    if not ok:raise ValueError('SUCCESSOR_READBACK_'+code)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path,pin):
    need(sha(path)==pin,'PIN');return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())


def kernel(row):
    root=Path('/proc')/str(row['pid']);s=(root/'stat').read_text().rsplit(')',1)[1].split()
    need(s[19]==row['starttime'] and s[0] in ('R','S','D'),'LIVE_IDENTITY')
    need(hashlib.sha256((root/'cmdline').read_bytes()).hexdigest()==row['command'],'LIVE_COMMAND')
    children=set()
    for task in (root/'task').iterdir():
        try:children.update(map(int,(task/'children').read_text().split()))
        except FileNotFoundError:pass
    need(sorted(children)==row['children'],'LIVE_CHILDREN')
    return dict(pid=row['pid'],starttime=s[19],state=s[0],children=sorted(children))


def coverage(cfg):
    need(cfg['stages']==['prior-readback','placement','tile','epoch','source','capacity','training','training-readback','evaluation'],
         'STAGES')
    need(cfg['placement_stages']==['original','boundary','resumed','cold','match','source-readback','full','full-readback'],
         'PLACEMENT_STAGES')
    need(len(cfg['owners'])==9 and set(cfg['owners'])=={'driver','grouped','producer','recovery','full','evaluation',
                                                      'tile_waiter','epoch_waiter','source_waiter'},'OWNERS')
    need(set(cfg['counts'])=={'shared','own_slot','whole_proposals'},'ARCHITECTURES')
    for row in cfg['counts'].values():
        need(row['documents']==row['ends']==91540 and row['atoms']==6547176
             and row['answers']==6368511 and row['decisions']==6460051,'COUNTS')
    need(cfg['evaluation']['cases']==23513 and cfg['evaluation']['trajectories']==564312
         and cfg['evaluation']['scoring_scope']=='all' and not cfg['evaluation']['best_of_proposals'],'EVALUATION')


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    p=read(a.producer/'report.json',a.pin);need(p['complete'] and p['mode']=='successor-wiring-proof','PRODUCER')
    count=total=0
    for name,h in p['files'].items():
        path=a.producer/name;need(path.resolve().is_relative_to(a.producer.resolve()) and sha(path)==h,'FILE')
        count+=1;total+=path.stat().st_size
    cfg=read(a.authority,p['authority']);coverage(cfg)
    need(p['code']==cfg['code'] and p['owner_count']==9 and p['live_launch_refused']
         and not p['gpu_used'] and not p['retirement_performed'] and p['signals_sent']==0,'SCOPE')
    snapshots={n:json.loads((a.producer/(n+'.json')).read_bytes()) for n in ('before','after')}
    fields=('pid','starttime','parent','command','children')
    need([{k:r[k] for k in fields} for r in snapshots['before']]==[{k:r[k] for k in fields} for r in snapshots['after']],
         'UNCHANGED_OWNERS')
    live=[kernel(r) for r in snapshots['after']]
    for n,h in cfg['code'].items():need(sha(Path(__file__).resolve().parents[1]/n)==h,'CODE')
    for name,ref in p['prepared'].items():
        q=read(Path(ref['path'])/'authority.json',ref['pin']);old=cfg['queues'][name]['authority']
        need(q==read(old['path'],old['pin']),'QUEUE_BINDING')
    prior=read(Path(p['prior']['path'])/'report.json',p['prior']['pin'])
    need(len(prior['completed_original'])==13 and not prior['original_complete']
         and not prior['numerical_payloads_reauthenticated'],'PRIOR_SCOPE')
    left=Path(p['original']['path']);right=Path(p['pipeline']['path'])
    old=read(left/'report.json',p['original']['pin']);new=read(right/'report.json',p['pipeline']['pin'])
    need(old['qualification']==new['qualification'] and old['scored']['summary']==new['scored']['summary'],'SUMMARY')
    events=trajectories=0
    # Compare every generated event against the earlier, separately executed pipeline.
    for label in ('qualification','qualification-cold'):
        paths=[];reports=[]
        for root,r in ((left,old),(right,new)):
            ref=next(v for v in r['outputs'] if v['label']==label)
            path=Path(ref['path']);cohort=read(path,ref['pin']);paths.append(path.parent);reports.append(cohort)
        need(len(reports[0]['results'])==len(reports[1]['results'])==4,'CASES')
        for x,y in zip(reports[0]['results'],reports[1]['results'],strict=True):
            need(x['document']==y['document'] and x['ordinal']==y['ordinal'],'CASE_ORDER')
            generated=[]
            for root,row in zip(paths,(x,y),strict=True):
                case=root/'run'/row['path'];c=read(case/'case.json',row['pin'])
                need(c['complete'] and c['status']=='generated','CASE_GAP')
                g=case/c['generation']['path'];report=read(g/'report.json',c['generation']['pin']);generated.append((g,report))
            (g,gr),(h,hr)=generated
            need(len(gr['traces'])==len(hr['traces'])==24,'TRACE_GRID')
            for u,v in zip(gr['traces'],hr['traces'],strict=True):
                need(all(u[k]==v[k] for k in ('architecture','proposal','mode')),'TRACE_ORDER')
                gu=g/u['path'];hv=h/v['path'];ur=read(gu/'report.json',u['pin']);vr=read(hv/'report.json',v['pin'])
                need(all(ur[k]==vr[k] for k in ('status','refusal','generated_atoms','events','model_ended')),'TERMINAL')
                for aa,bb in zip(replay(gu,u['pin']),replay(hv,v['pin']),strict=True):
                    need(fingerprint(aa)==fingerprint(bb),'EVENT');events+=1
                trajectories+=1
    need(events==3408 and trajectories==192,'EVENT_COUNTS')
    controls={}
    for name,alter,code in (
        ('missing_stage',lambda v:v['stages'].pop(),'STAGES'),
        ('six_owners_only',lambda v:v['owners'].pop('tile_waiter'),'OWNERS'),
        ('missing_decision',lambda v:v['counts']['shared'].update(decisions=6460050),'COUNTS')):
        bad=deepcopy(cfg);alter(bad)
        try:coverage(bad)
        except ValueError as error:
            need(code in str(error),'CONTROL_CODE');controls[name]=str(error)
        else:raise ValueError('SUCCESSOR_READBACK_CONTROL_ACCEPTED')
    need(not torch.cuda.is_initialized(),'GPU_INITIALIZED')
    value=dict(complete=True,producer=a.pin,authority=p['authority'],source=sha(__file__),files=count,bytes=total,
        owners=live,events=events,trajectories=trajectories,all_event_tensors_exact=True,all_score_summaries_exact=True,
        controls=controls,gpu_used=False,optimizer_updates=0,signals_sent=0,retirement_performed=False,
        seconds=time.perf_counter()-began,acceptance='author-checked',number_status='measured')
    write(a.output/'report.json',value);print(json.dumps(dict(pin=sha(a.output/'report.json'),events=events)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('producer','authority','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error)))
        raise
