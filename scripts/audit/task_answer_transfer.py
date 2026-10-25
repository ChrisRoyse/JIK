"""Qualify public-only task transfer against saved CPU mechanics checkpoints."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.complete_answer_evaluation import artifact, read as read_path, CACHE
from audit.complete_answer_device_cohort import validate
from claims.process_conditions import write_committed as write
from model import task_answer_transfer as transfer
from model.task_answer_public_pack import TaskPublicPack
from model.complete_answer_evaluation import ARCHITECTURES, EvaluationCheckpoint
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import canonical, digest, file_sha, need

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29'); BASE = Path(__file__).resolve().parents[1]
PACK = dict(path=str(ROOT/'task-public-pack-v1-20261025/pack'),pin='eee854d0c42cfac6d9c2b002334144b7dc1f58a5f4478ecadee0c8997f8b46d5')
TEMPLATE = dict(path=str(ROOT/'complete-answer-device-proof-v1-20261024/request.json'),
                pin='9e36c4df6f41a4c89616c4361c8169a3349a7880a2777643d3d95e649510389b')
BANK = Path('/zfs/hot/mnlm/readings/legal-source-v1-20261022/configuration.json')
BANK_PIN = '2dd93334c150ed38dc6be22efa219cdf95528642335edab445ef1ba909fbc35d'
NEW = ('model/task_answer_transfer.py','audit/task_answer_transfer.py','model/task_answer_public_pack.py')


def read(path,pin):return read_path(Path(path),pin)


def bank_present(value):
    if isinstance(value,dict):
        return value.get('bank')==BANK_PIN or value.get('native_bank')==BANK_PIN or any(bank_present(v) for v in value.values())
    if isinstance(value,list):return any(bank_present(v) for v in value)
    return False


def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError) as exc:return str(exc)
    raise ValueError('TASK_TRANSFER_NEGATIVE_PASSED')


def prepare(a):
    a.output.mkdir(exist_ok=False); began=time.perf_counter()
    m=read(TEMPLATE['path'],TEMPLATE['pin']); bank=read(BANK,BANK_PIN)
    pack=TaskPublicPack(PACK['path'],PACK['pin'])
    try:
        need({n:v['lens_id'] for n,v in bank['instruments'].items()}==pack.report['binding']['native']['lens_ids'],
             'TASK_TRANSFER_LENS_IDENTITY')
        need(all(bank_present(v['metadata']['binding']) for v in m['checkpoints'].values()), 'TASK_TRANSFER_TRAINING_BANK')
        plan=transfer.schedule(pack); selected={}
        for row in plan['rows']:
            selected.setdefault(('family',row['family']),row['ordinal'])
            selected.setdefault(('donor',row['donor'] is not None),row['ordinal'])
        ordinals=sorted(set(selected.values()))
        code=dict(m['code']);code.update({n:file_sha(BASE/n) for n in NEW})
        for n,h in code.items():need(file_sha(BASE/n)==h,'TASK_TRANSFER_CODE')
        value=dict(schema='task-answer-transfer-proof-request-v1',plan=plan,plan_pin=digest(canonical(plan)),pack=PACK,
            runtime=m['runtime'],checkpoints=m['checkpoints'],code=code,ordinals=ordinals,
            intervention=m['intervention'],template=TEMPLATE,
            authority=dict(public_pack=PACK['pin'],training=m['authority']['training'],template=TEMPLATE['pin'],
                training_bank=BANK_PIN,lens_ids=pack.report['binding']['native']['lens_ids'],code=fingerprint(code),
                scope='CPU_transfer_mechanics_exposed_original_tasks',semantic_validity_established=False))
        pin=write(a.output/'request.json',value)
        refs=[dict(ordinal=i,task=plan['rows'][i]['task'],family=plan['rows'][i]['family'],donor=plan['rows'][i]['donor'] is not None) for i in ordinals]
        pack.fresh(physical=True)
        r=dict(complete=True,request=pin,schedule=plan['summary'],proof=refs,all_six_native_identities_exact=True,
               model_loads=0,encoder_calls=0,optimizer_updates=0,seconds=time.perf_counter()-began)
        write(a.output/'report.json',r); print(json.dumps(r),flush=True)
    finally:pack.close()


def checked_request(a):
    m=read(a.request,a.pin)
    need(m['schema']=='task-answer-transfer-proof-request-v1' and m['pack']==PACK and m['template']==TEMPLATE
         and digest(canonical(m['plan']))==m['plan_pin'] and m['authority']['code']==fingerprint(m['code']),
         'TASK_TRANSFER_REQUEST')
    for n,h in m['code'].items():need(file_sha(BASE/n)==h,'TASK_TRANSFER_CODE')
    return m


def worker(a):
    a.output.mkdir(exist_ok=False); began=tick=time.perf_counter(); m=checked_request(a); phases={}
    allowed={a.request.resolve(),Path(m['runtime']['path']).resolve()}
    allowed.update(Path(v['reference']['path']).resolve() for v in m['checkpoints'].values())
    pr=read(Path(PACK['path'])/'report.json',PACK['pin'])
    allowed.update((Path(PACK['path'])/n).resolve() for n in ('report.json',*pr['files']))
    opened=set(); destination=a.output.resolve()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to('/zfs'):return
        need(p in allowed or p.is_relative_to(destination),'TASK_TRANSFER_PRIVATE_ARTIFACT')
        if p in allowed:opened.add(str(p))
    sys.addaudithook(guard)
    pack=TaskPublicPack(PACK['path'],PACK['pin'])
    try:
        need(transfer.schedule(pack)==m['plan'],'TASK_TRANSFER_SCHEDULE')
        runtime=artifact(m['runtime']); target_pin,pressure=validate(m,runtime,'cpu')
        need(pressure and m['intervention']==dict(kind='CPU_memory_pressure_mechanics',cache_bytes=CACHE),
             'TASK_TRANSFER_CPU_PROOF')
        target,pin=extension(runtime);need(pin==target_pin,'TASK_TRANSFER_TARGET')
        loaded={}
        for name in ARCHITECTURES:
            row=m['checkpoints'][name]
            loaded[name]=EvaluationCheckpoint(row['reference']['path'],row['reference'],row['metadata'],runtime,target,target_pin,device='cpu')
            loaded[name].model.forecaster.predictor.cache_bytes=CACHE
        phases['resident_load']=time.perf_counter()-tick;tick=time.perf_counter()
        path,pin=transfer.run(loaded,pack,m['plan'],target=target,output=a.output/'run',authority=m['authority'],ordinals=m['ordinals'])
        phases['generation_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        private=rejected(lambda:(ROOT/'answer-native-plan-v1-20261023/report.json').read_bytes())
        need(private=='MNLM_TEXT_CORPUS_TASK_TRANSFER_PRIVATE_ARTIFACT','TASK_TRANSFER_ACCESS_CONTROL')
        need(not torch.cuda.is_initialized(),'TASK_TRANSFER_UNEXPECTED_CUDA')
        pack.close();pack=None
        for n,h in m['code'].items():need(file_sha(BASE/n)==h,'TASK_TRANSFER_CODE_CHANGED')
        need(file_sha(a.request)==a.pin,'TASK_TRANSFER_REQUEST_CHANGED')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_TRANSFER_WORKER_PHASES')
        report=dict(complete=True,request=a.pin,run=dict(path=str(path),pin=pin),model_loads=3,target_loads=1,
            optimizer_updates=0,encoder_calls=0,gpu_used=False,mode='CPU_mechanics_memory_pressure',cache_bytes=CACHE,
            private_read_control=private,opened_inputs=sorted(opened),guard='Python open audit hook over ZFS, not OS sandbox',
            phases=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            capability_demonstrated=False,architecture_winner=False)
        report_pin=write(a.output/'report.json',report);print(json.dumps(dict(complete=True,pin=report_pin,seconds=seconds)),flush=True)
    finally:
        if pack is not None:pack.close()


def score(a):
    # Authentication of the sealed generation cohort precedes opening private references.
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);m=checked_request(a)
    generated=read(a.generated,a.generated_pin);need(generated['complete'] and generated['request']==a.pin,'TASK_TRANSFER_SCORE_GENERATION')
    run_ref=generated['run'];run=read(run_ref['path'],run_ref['pin']);root=Path(run_ref['path']).parent
    need(run['binding']['ordinals']==m['ordinals'] and [r['ordinal'] for r in run['results']]==m['ordinals'],
         'TASK_TRANSFER_SCORE_DENOMINATOR')
    pack=TaskPublicPack(PACK['path'],PACK['pin']);checks=[]
    try:
        for ref in run['results']:
            row=m['plan']['rows'][ref['ordinal']];_,_,public,unavailable=transfer.inputs(pack,row)
            auth=dict(m['authority'],plan=m['plan_pin'],task=row['task'],public=row['public'],ordinal=row['ordinal'])
            kw=dict(row=row,plan_pin=m['plan_pin'],identities=run['binding']['checkpoints'],expected=run['binding']['expected'],
                    public=public,unavailable=unavailable,authority=auth,target_state=run['binding']['frozen'],caches=run['binding']['cache_bytes'])
            transfer.check_case(root/ref['path'],ref['pin'],**kw);checks.append((ref,kw))
        phases=dict(generation_authentication=time.perf_counter()-tick);tick=time.perf_counter()
        from audit.task_answer_public_pack import source
        schedule,store,binding=source()
        need(binding==pack.report['binding'],'TASK_TRANSFER_PRIVATE_SOURCE')
        phases['private_source']=time.perf_counter()-tick;tick=time.perf_counter();cases=[]
        for ref,kw in checks:
            row=kw['row'];private=schedule.supervision(row['task'],fitting=False)
            future=tuple(store.frame(i) for i in private['targets'])
            case=transfer.score_case(root/ref['path'],ref['pin'],**kw,future=future,roster=pack.roster)
            path=a.output/f"case-{row['ordinal']:06d}.json";pin=write(path,case)
            cases.append(dict(ordinal=row['ordinal'],path=path.name,pin=pin))
        phases['scoring']=time.perf_counter()-tick;tick=time.perf_counter()
        store.fresh();pack.close();pack=None
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'TASK_TRANSFER_SCORE_PHASES')
        report=dict(complete=True,request=a.pin,generation=a.generated_pin,run=run_ref,cases=cases,private_binding=binding,
            query_denominator=len(m['plan']['rows']),evaluated=len(cases),mode='CPU_mechanics_only',model_loads=0,encoder_calls=0,
            optimizer_updates=0,gpu_used=False,phases=phases,seconds=seconds,semantic_validity_established=False)
        pin=write(a.output/'report.json',report);print(json.dumps(dict(complete=True,pin=pin,cases=len(cases),seconds=seconds)),flush=True)
    finally:
        if pack is not None:pack.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','worker','score'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--pin')
    p.add_argument('--generated',type=Path);p.add_argument('--generated-pin');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:globals()[a.mode](a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise


if __name__=='__main__':main()
