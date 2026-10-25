"""Paired parent/corpus-correction scores over every original-task proposal."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

if '--cuda' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import task_surface_scoring as previous
from audit import paired_surface_contract as contract
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model import paired_surface_scoring as scoring
from model.answer_training_schedule import AnswerSchedule
from model.corpus_protected_decoder import CorpusProtectedDecoder
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha,need

P=previous.P;BASE=previous.BASE
NEW=('model/paired_surface_scoring.py','audit/paired_surface_scoring.py','audit/paired_surface_readback.py',
     'model/corpus_protected_decoder.py','audit/paired_surface_contract.py')+previous.NEW


def correction_args(a):
    return dict(plan=a.corpus_plan,plan_pin=a.corpus_plan_pin,run=a.corpus_run,run_pin=a.corpus_run_pin,
                readback=a.corpus_readback,readback_pin=a.corpus_readback_pin,full=a.cuda)


def run(a):
    contract.owner_fd(a.owner_fd)
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);parent=decoder=None;leased=False
    code={n:file_sha(BASE/n) for n in NEW}
    try:
        selected=[0,1,3] if not a.cuda or a.qualification else None
        m,generated,cases,binding=previous.authenticate(a.request,a.pin,a.generated,a.generated_pin,selected)
        contract.worker_scope(m,cuda=a.cuda,qualification=a.qualification)
        phases=dict(authenticate_generation=time.perf_counter()-tick);tick=time.perf_counter()
        source=binding['plan'];schedule=AnswerSchedule(source['path'],source['pin'])
        texts={r['id']:r['text'] for r in schedule.plan.records}
        references={case['case']['task']:previous.references(schedule,case['case']['task'],texts) for _,case in cases}
        phases['reference_inputs']=time.perf_counter()-tick;tick=time.perf_counter()
        if a.cuda:previous.gpu_ready(a.output);take_lease('29','paired-surface-scoring');leased=True
        args=[]
        for path,pin in previous.DECODER:args.extend((P/path,pin))
        parent=TrainedSurfaceDecoder(*args,device='cuda:0' if a.cuda else 'cpu',cpu_diagnostic=not a.cuda)
        decoder=CorpusProtectedDecoder(parent,**correction_args(a))
        runtime_ref=m['runtime'];need(file_sha(runtime_ref['path'])==runtime_ref['pin'],'PAIRED_TASK_RUNTIME')
        runtime=torch.load(runtime_ref['path'],map_location='cpu',weights_only=True);sizes=list(runtime['sizes']);del runtime
        phases['resident_decoders']=time.perf_counter()-tick;tick=time.perf_counter();results=[];calls=0
        for path,case in cases:
            dest=a.output/f"case-{case['case']['ordinal']:06d}";dest.mkdir();arms=[]
            for i,ref in enumerate(case['traces']):
                src=path/ref['path'];trace=previous.contract.read(src/'report.json',ref['pin'])
                arm,arm_pin=scoring.score_trace(src,ref['pin'],decoder.model,references[case['case']['task']],
                    output=dest/f'trajectory-{i:03d}',region_pin=case['checkpoints'][ref['architecture']]['target'],
                    region_sizes=sizes,expected_request=trace['files']['request.json'])
                calls+=sum(v['calls'] for v in arm['atoms'] if v['status']=='scored')
                arms.append(dict(arm,architecture=ref['architecture'],proposal=ref['proposal'],mode=ref['mode'],
                                 path=f'trajectory-{i:03d}',pin=arm_pin))
            lookup={(v['architecture'],v['proposal'],v['mode']):v for v in arms};pairs=[]
            for name,proposals in case['expected'].items():
                for proposal in proposals:
                    left=lookup[name,proposal,'correct']
                    for mode in ('reversed','train_donor'):
                        right=lookup.get((name,proposal,mode))
                        pairs.append(dict(architecture=name,proposal=proposal,control=mode,available=right is not None,
                            reason=case['unavailable'].get(mode),scores=None if right is None else scoring.contrast(left,right,dest)))
            value=dict(complete=True,case=case['case'],arms=arms,paired=pairs,unavailable=case['unavailable'])
            rp=write(dest/'report.json',value);results.append(dict(path=dest.name,pin=rp))
            print(json.dumps(dict(case=case['case']['ordinal'],trajectories=len(arms))),flush=True)
        if a.cuda:torch.cuda.synchronize()
        phases['paired_surface_scoring']=time.perf_counter()-tick;tick=time.perf_counter()
        parent.fresh();decoder.fresh()
        for _,case in cases:schedule.fresh(case['case']['task'])
        for n,h in {**m['code'],**code}.items():need(file_sha(BASE/n)==h,'PAIRED_TASK_CODE_CHANGED')
        if not a.cuda:need(not torch.cuda.is_initialized(),'PAIRED_TASK_CPU_ONLY')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'PAIRED_TASK_PHASES')
        r=dict(complete=True,request=dict(path=str(a.request),pin=a.pin),generated=dict(path=str(a.generated),pin=a.generated_pin),
            decoder=decoder.receipt,decoder_refs=previous.DECODER,corpus={k:str(v) if isinstance(v,Path) else v for k,v in correction_args(a).items()},
            code=code,source=binding,results=results,declared_cases=170,evaluated_cases=len(cases),
            full_query=a.cuda and not a.qualification,qualification=a.qualification,
            full_training=m['full_training'],teacher_forced=True,free_generation_tested=False,
            semantic_validity_established=False,architecture_winner=False,encoder_calls=0,optimizer_updates=0,
            parent_loads=1,correction_loads=1,parent_forward_calls=calls,correction_forward_calls=calls,
            gpu_used=a.cuda,phases=phases,seconds=seconds,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            acceptance='author-checked',number_status='measured')
        rp=write(a.output/'report.json',r);print(json.dumps(dict(complete=True,pin=rp,seconds=seconds,calls_per_component=calls)),flush=True)
    finally:
        if decoder:decoder.close()
        if parent:parent.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-proof',action='store_true');g.add_argument('--cuda',action='store_true')
    p.add_argument('--qualification',action='store_true');p.add_argument('--owner-fd',type=int)
    for n in ('request','generated','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('pin','generated-pin'):p.add_argument('--'+n,required=True)
    for n in ('plan','run','readback'):
        p.add_argument('--corpus-'+n,type=Path,required=True);p.add_argument('--corpus-'+n+'-pin',required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
