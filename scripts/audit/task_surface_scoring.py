"""Frozen-decoder diagnostic after authenticated original-task generation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import task_transfer_contract as contract
from audit.complete_answer_device_cohort import gpu_ready
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model import task_answer_transfer as transfer
from model import task_surface_scoring as scoring
from model.answer_training_schedule import AnswerSchedule
from model.task_answer_public_pack import TaskPublicPack
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha,need

P=contract.ROOT;BASE=contract.REPO/'scripts'
DECODER=(
 ('phase115-complete-answer-recovery-v1-20261024/inversion-full','1fe1850ea1cedfe834259073f8ba71b56b969808499492bc7aaf83a98dc2b125'),
 ('legal-inversion-plan-v3-20261024','2d548670cf1b40f859262c3789d1defefc9a977649cbbdeeff4fdd481b91668b'),
 ('inversion-canonical-repair-v2-20261024/readback','2680402106463fbfe867f307946cb85f0e94400a129706f6c89ada471f7b1b60'),
 ('legal-inversion-final-readback-v1-20261024','02e63b0d17f059a83dd43541a8590d9842a627c825ca12e2a33a252c53018582'))
NEW=('model/task_surface_scoring.py','audit/task_surface_scoring.py','audit/task_surface_readback.py')


def authenticate(request,pin,generated,generated_pin,ordinals):
    m=contract.read(request,pin);contract.request_scope(m,m['device'])
    g=contract.read(generated,generated_pin)
    need(g['complete'] and g['request']==pin and g['private_read_refused'] and g['optimizer_updates']==0
         and g['mode'] in ('cpu-proof','cuda') and g['device']==m['device'], 'TASK_SURFACE_GENERATION')
    r=contract.read(g['run']['path'],g['run']['pin']);root=Path(g['run']['path']).parent;b=r['binding']
    all_ordinals=list(range(170)) if m['ordinals'] is None else m['ordinals']
    need(r['complete'] and b['authority']==m['authority'] and b['plan']==m['plan_pin']
         and b['ordinals']==all_ordinals and [v['ordinal'] for v in r['results']]==all_ordinals,
         'TASK_SURFACE_RUN_BINDING')
    selected=all_ordinals if ordinals is None else ordinals
    need(selected==sorted(set(selected)) and selected and set(selected)<=set(all_ordinals), 'TASK_SURFACE_SELECTION')
    pack=TaskPublicPack(m['pack']['path'],m['pack']['pin']);cases=[]
    try:
        need(transfer.schedule(pack)==m['plan'],'TASK_SURFACE_SCHEDULE')
        for name,v in b['checkpoints'].items():
            ref=m['checkpoints'][name]['reference'];meta=m['checkpoints'][name]['metadata']
            need(v['checkpoint']==ref['pin'] and v['state']==ref['state'] and v['step']==meta['step']
                 and v['counts']==meta['counts'] and v['target']==meta['binding']['target'],'TASK_SURFACE_CHECKPOINT')
        for ref in r['results']:
            if ref['ordinal'] not in selected:continue
            row=m['plan']['rows'][ref['ordinal']];_,_,public,unavailable=transfer.inputs(pack,row)
            kw=dict(row=row,plan_pin=m['plan_pin'],identities=b['checkpoints'],expected=b['expected'],
                public=public,unavailable=unavailable,target_state=b['frozen'],caches=b['cache_bytes'],
                authority=dict(m['authority'],plan=m['plan_pin'],task=row['task'],public=row['public'],ordinal=row['ordinal']))
            case=transfer.check_case(root/ref['path'],ref['pin'],**kw)
            cases.append((root/ref['path'],case))
        return m,r,cases,pack.report['binding']
    finally:pack.close()


def references(schedule,key,texts):
    private=schedule.supervision(key,fitting=False)
    return [dict(id=k,raw=texts[k].encode()) for k in private['targets']]


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);decoder=None;leased=False
    code={n:file_sha(BASE/n) for n in NEW}
    try:
        m,run,cases,binding=authenticate(a.request,a.pin,a.generated,a.generated_pin,None if a.cuda else [0,1,3])
        need(a.cuda== (m['device']=='cuda:0'),'TASK_SURFACE_DEVICE')
        if a.cuda:need(m['full_training'] and m['ordinals'] is None,'TASK_SURFACE_FULL_CUDA')
        phases=dict(authenticate_generation=time.perf_counter()-tick);tick=time.perf_counter()
        source=binding['plan'];schedule=AnswerSchedule(source['path'],source['pin'])
        texts={r['id']:r['text'] for r in schedule.plan.records}
        refs={case['case']['task']:references(schedule,case['case']['task'],texts) for _,case in cases}
        phases['private_references']=time.perf_counter()-tick;tick=time.perf_counter()
        if a.cuda:gpu_ready(a.output);take_lease('29','task-surface-scoring');leased=True
        args=[]
        for path,pin in DECODER:args.extend((P/path,pin))
        decoder=TrainedSurfaceDecoder(*args,device='cuda:0' if a.cuda else 'cpu',cpu_diagnostic=not a.cuda)
        runtime_ref=m['runtime'];need(file_sha(runtime_ref['path'])==runtime_ref['pin'],'TASK_SURFACE_RUNTIME')
        runtime=torch.load(runtime_ref['path'],map_location='cpu',weights_only=True)
        sizes=list(runtime['sizes']);del runtime
        phases['resident_decoder']=time.perf_counter()-tick;tick=time.perf_counter();results=[]
        for path,case in cases:
            dest=a.output/f"case-{case['case']['ordinal']:06d}";dest.mkdir();arms=[]
            for i,ref in enumerate(case['traces']):
                src=path/ref['path'];trace=contract.read(src/'report.json',ref['pin'])
                arm,arm_pin=scoring.score_trace(src,ref['pin'],decoder.model,refs[case['case']['task']],
                    output=dest/f'arm-{i:03d}',region_pin=case['checkpoints'][ref['architecture']]['target'],
                    region_sizes=sizes,expected_request=trace['files']['request.json'])
                arms.append(dict(arm,architecture=ref['architecture'],proposal=ref['proposal'],mode=ref['mode'],
                                 path=f'arm-{i:03d}',pin=arm_pin))
            lookup={(v['architecture'],v['proposal'],v['mode']):v for v in arms};pairs=[]
            for name,proposals in case['expected'].items():
                for proposal in proposals:
                    left=lookup[name,proposal,'correct']
                    for mode in ('reversed','train_donor'):
                        right=lookup.get((name,proposal,mode))
                        pairs.append(dict(architecture=name,proposal=proposal,control=mode,available=right is not None,
                            reason=case['unavailable'].get(mode),scores=None if right is None else scoring.paired(left,right,dest)))
            value=dict(complete=True,case=case['case'],arms=arms,paired=pairs,unavailable=case['unavailable'])
            rp=write(dest/'report.json',value);results.append(dict(path=dest.name,pin=rp))
            print(json.dumps(dict(case=case['case']['ordinal'],arms=len(arms))),flush=True)
        if a.cuda:torch.cuda.synchronize()
        phases['surface_scoring']=time.perf_counter()-tick;tick=time.perf_counter();decoder.fresh()
        for _,case in cases:schedule.fresh(case['case']['task'])
        for n,h in {**m['code'],**code}.items():need(file_sha(BASE/n)==h,'TASK_SURFACE_CODE_CHANGED')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_SURFACE_PHASES')
        r=dict(complete=True,request=dict(path=str(a.request),pin=a.pin),generated=dict(path=str(a.generated),pin=a.generated_pin),
            decoder=decoder.receipt,decoder_refs=DECODER,code=code,source=binding,results=results,
            declared_cases=170,evaluated_cases=len(cases),full_query=a.cuda,full_training=m['full_training'],
            teacher_forced=True,free_generation_tested=False,semantic_validity_established=False,architecture_winner=False,
            encoder_calls=0,optimizer_updates=0,decoder_loads=1,gpu_used=a.cuda,phases=phases,seconds=seconds)
        rp=write(a.output/'report.json',r);print(json.dumps(dict(complete=True,pin=rp,seconds=seconds)),flush=True)
    finally:
        if decoder is not None:decoder.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu-proof',action='store_true');group.add_argument('--cuda',action='store_true')
    for n in ('request','generated','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','generated-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
