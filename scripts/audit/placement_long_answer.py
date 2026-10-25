"""Qualify native-only checkpoints on the complete longest legal TRAIN answer."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import captured_long_answer as proof
from claims.process_conditions import write_committed
from model import scalar_placement_checkpoint as candidate
from text.constellation_corpus import file_sha,need

SHORT={
 'producer':('answer-placement-probe-v1-20261024','9894084b55d5274d0169036505972d5675cbc79b8430d89cd597e3f280b15762'),
 'cold':('answer-placement-probe-cold-v1-20261024','0319c8c33ba9dfefeb2e4666683cd12d586d0915d9f95cab5a4d4a6e18d73567'),
 'reader':('answer-placement-readback-v1-20261024','06c5dee2b099e14948789929dc0b2b3b3d6fd2f1d6ed4a5f9b61400e43af087a')}
NEW=('audit/placement_long_answer.py','model/scalar_placement_checkpoint.py','model/scalar_answer_checkpoint.py')


def authenticate(root):
    reports={n:proof.pinned(root/d,h) for n,(d,h) in SHORT.items()}
    a,b,r=(reports[n] for n in ('producer','cold','reader'))
    need(a['all_updates_exact'] and b['all_updates_exact'] and b['cold_reproduced']
        and b['previous']==SHORT['producer'][1] and a['code']==b['code'],'PLACEMENT_LONG_SHORT_PAIR')
    need(r['decision']=='EXACT_SHORT_CPU_PROOF' and r['all_updates_exact'] and r['cold_bytes_exact']
        and r['producer']==SHORT['producer'][1] and r['cold']==SHORT['cold'][1]
        and r['updates_compared']==60 and r['decisions']==792 and r['physical_checkpoints']==24,
        'PLACEMENT_LONG_SHORT_READBACK')
    for name in ('producer','cold'):proof.manifest(root/SHORT[name][0],reports[name])
    code={**a['code'],**r['reader_dependencies'],'audit/answer_placement_readback.py':r['reader_source']}
    code.update({n:file_sha(proof.BASE/n) for n in NEW})
    def fresh():
        for n,h in code.items():need(file_sha(proof.BASE/n)==h,'PLACEMENT_LONG_CODE_CHANGED')
        for d,h in SHORT.values():need(file_sha(root/d/'report.json')==h,'PLACEMENT_LONG_SHORT_CHANGED')
    fresh();return code,fresh


def run(a):
    began=time.perf_counter();a.output.mkdir(exist_ok=False);code,fresh=authenticate(a.root)
    previous=proof.pinned(a.previous,a.previous_pin) if a.previous else None
    binding=json.loads(json.dumps(dict(schema='native-placement-long-v1',code=code,short=SHORT,
        architecture='shared',device='cpu',scope='captured_backward_with_native_only_checkpoints')))
    if previous:
        need(previous['complete'] and previous['binding']==binding and previous['mode']==a.mode,
             'PLACEMENT_LONG_PREVIOUS');proof.manifest(a.previous,previous)
    probe=None
    if a.mode=='long':
        need(a.probe is not None and a.probe_pin is not None,'PLACEMENT_LONG_PROBE_REQUIRED')
        probe=proof.pinned(a.probe,a.probe_pin)
        need(probe['complete'] and probe['mode']=='probe' and probe['binding']==binding,
             'PLACEMENT_LONG_PROBE');proof.manifest(a.probe,probe)
    write_committed(a.output/'binding.json',binding)
    for n,h in code.items():
        path=a.output/'source'/n;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((proof.BASE/n).read_bytes());need(file_sha(path)==h,'PLACEMENT_LONG_SOURCE')
    original=proof.learner.update;prior_new=proof.NEW;observations=[]
    def update(*args,**kwargs):
        model,_,item,step,_=args
        if item['side']!='train':return original(*args,**kwargs)
        tick=time.perf_counter()
        with candidate.scalar_heads(model) as replays:
            try:value=original(*args,**kwargs)
            except BaseException as exc:
                write_committed(a.output/'update-refusal.json',dict(error=type(exc).__name__,detail=str(exc),
                    document=item['document'],step=step,seconds=time.perf_counter()-tick,
                    forward_checkpoints=len(replays),recompute_calls=sum(r.calls==2 for r in replays),
                    storage_at_exception=replays[0].owner.report() if replays else None))
                raise
        row=dict(document=item['document'],step=step,decisions=value[0]['decisions'],
                 seconds=time.perf_counter()-tick,storage=value[0]['storage'],replays=[r.report() for r in replays])
        need(len(replays)==row['decisions']-1 and all(r.calls==2 and r.cursor==len(r.expected) for r in replays),
             'PLACEMENT_LONG_REPLAYS')
        observations.append(row);write_committed(a.output/f'observation-{len(observations):02}.json',row)
        return value
    proof.learner.update=update;proof.NEW=prior_new+NEW
    args=argparse.Namespace(root=a.root,output=a.output/'run',mode=a.mode,
        probe=a.probe/'run' if probe else None,probe_pin=probe['run_pin'] if probe else None,
        previous=a.previous/'run' if previous else None,previous_pin=previous['run_pin'] if previous else None)
    try:proof.run(args)
    finally:
        intact=proof.learner.update is update;proof.learner.update=original;proof.NEW=prior_new
        need(intact and proof.learner.SavedStorage is candidate.STORAGE,'PLACEMENT_LONG_RESTORE')
    run_pin=file_sha(args.output/'report.json');raw=proof.pinned(args.output,run_pin);proof.manifest(args.output,raw)
    need(raw['complete'] and raw['all_tensors_exact'] and raw['checkpoint_restarts_exact']
        and not raw['gpu_used'] and len(observations)==(1 if a.mode=='probe' else 3),'PLACEMENT_LONG_RESULT')
    if previous:
        need(len(previous['observations'])==len(observations),'PLACEMENT_LONG_COLD_COUNT')
        for x,y in zip(observations,previous['observations']):
            need(all(x[n]==y[n] for n in ('document','step','decisions','replays')),'PLACEMENT_LONG_COLD_REPLAY')
    fresh()
    report=dict(complete=True,mode=a.mode,binding=binding,code=code,short=SHORT,run_pin=run_pin,
        observations=observations,counts=raw['counts'],all_tensors_exact=True,checkpoint_restarts_exact=True,
        previous_pin=a.previous_pin,probe_pin=a.probe_pin,cold_reproduced=previous is not None,
        seconds=time.perf_counter()-began,gpu_used=False,promoted=False,cuda_qualified=False,
        capability_demonstrated=False,acceptance='author-checked')
    report['files']={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),counts=raw['counts'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=proof.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('probe','long'),required=True)
    for n in ('probe','previous'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
