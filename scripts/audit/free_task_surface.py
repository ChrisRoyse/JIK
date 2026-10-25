"""Reference-free parent/corpus decoder rendering of sealed original-task trajectories."""
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
from audit import paired_surface_scoring as paired
from claims.process_conditions import write_committed as write
from lenscommon import take_lease,release_lease
from model import free_answer_surface as render
from model.corpus_protected_decoder import CorpusProtectedDecoder
from model.native_population_training import fingerprint
from model.trained_surface_decoder import TrainedSurfaceDecoder
from text.constellation_corpus import file_sha,need

NEW=paired.NEW+('model/free_answer_surface.py','audit/free_task_surface.py','audit/free_task_surface_readback.py')


def private_guard(cases,output,extra=()):
    allowed={Path(path)/ref['path']/name for path,case in cases for ref in case['traces']
             for name in ('report.json','request.json','events.bin','index.jsonl')}
    allowed.update(Path(p) for p in extra);output=Path(output).resolve();opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to('/zfs'):return
        need(path in allowed or path.is_relative_to(output),'FREE_TASK_PRIVATE_OPEN')
        opened.add(str(path))
    sys.addaudithook(hook)
    return opened


def load(a):
    selected=[0,1,3] if not a.cuda or a.qualification else None
    m,g,cases,binding=paired.previous.authenticate(a.request,a.pin,a.generated,a.generated_pin,selected)
    paired.contract.worker_scope(m,cuda=a.cuda,qualification=a.qualification)
    args=[]
    for path,pin in paired.previous.DECODER:args.extend((paired.P/path,pin))
    parent=TrainedSurfaceDecoder(*args,device='cuda:0' if a.cuda else 'cpu',cpu_diagnostic=not a.cuda)
    try:correction=CorpusProtectedDecoder(parent,**paired.correction_args(a))
    except BaseException:parent.close();raise
    ref=m['runtime'];need(file_sha(ref['path'])==ref['pin'],'FREE_TASK_RUNTIME')
    runtime=torch.load(ref['path'],map_location='cpu',weights_only=True);sizes=list(runtime['sizes']);del runtime
    return m,cases,binding,parent,correction,sizes


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);leased=False;parent=correction=None
    code={name:file_sha(paired.BASE/name) for name in NEW}
    try:
        if a.cuda:paired.previous.gpu_ready(a.output);take_lease('29','free-task-surface');leased=True
        m,cases,binding,parent,correction,sizes=load(a)
        models=dict(parent=parent.model,corrected=correction.model)
        pins={name:fingerprint(model.state_dict()) for name,model in models.items()}
        opened=private_guard(cases,a.output)
        phases=dict(authenticate_and_restore=time.perf_counter()-tick);tick=time.perf_counter();results=[]
        for path,case in cases:
            dest=a.output/f"case-{case['case']['ordinal']:06d}";dest.mkdir();traces=[]
            for i,ref in enumerate(case['traces']):
                src=path/ref['path'];native=paired.previous.contract.read(src/'report.json',ref['pin'])
                value,pin=render.collect(src,ref['pin'],models,output=dest/f'trajectory-{i:03d}',
                    region_pin=case['checkpoints'][ref['architecture']]['target'],region_sizes=sizes,
                    expected_request=native['files']['request.json'],decoder_pins=pins)
                traces.append(dict(path=f'trajectory-{i:03d}',pin=pin,architecture=ref['architecture'],
                    proposal=ref['proposal'],mode=ref['mode'],arms={k:{f:v[f] for f in
                        ('answer_ended','generated_bytes','completed_atoms','surface_refusal','native_refusal')} for k,v in value['arms'].items()}))
                print(json.dumps(dict(case=case['case']['ordinal'],trajectory=i,arms=traces[-1]['arms'])),flush=True)
            value=dict(complete=True,case=case['case'],traces=traces,unavailable=case['unavailable'])
            results.append(dict(path=dest.name,pin=write(dest/'report.json',value)))
        if a.cuda:torch.cuda.synchronize()
        phases['free_generation']=time.perf_counter()-tick;tick=time.perf_counter()
        parent.fresh();correction.fresh()
        need(all(fingerprint(model.state_dict())==pins[name] for name,model in models.items()),'FREE_TASK_WEIGHTS_CHANGED')
        for name,h in {**m['code'],**code}.items():need(file_sha(paired.BASE/name)==h,'FREE_TASK_CODE_CHANGED')
        forbidden=paired.P/'answer-native-plan-v1-20261023/report.json'
        try:forbidden.read_bytes()
        except ValueError as exc:need('FREE_TASK_PRIVATE_OPEN' in str(exc),'FREE_TASK_WRONG_PRIVATE_REFUSAL')
        else:raise ValueError('FREE_TASK_PRIVATE_READ_PASSED')
        if not a.cuda:need(not torch.cuda.is_initialized(),'FREE_TASK_CPU_ONLY')
        phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'FREE_TASK_PHASES')
        r=dict(complete=True,request=dict(path=str(a.request),pin=a.pin),generated=dict(path=str(a.generated),pin=a.generated_pin),
            corpus={k:str(v) if isinstance(v,Path) else v for k,v in paired.correction_args(a).items()},code=code,
            source=binding,decoder=correction.receipt,decoder_pins=pins,results=results,
            evaluated_cases=len(cases),declared_cases=170,full_query=a.cuda and not a.qualification,
            qualification=a.qualification,full_training=m['full_training'],gpu_used=a.cuda,
            reference_visible=False,reference_open_refused=True,opened_zfs_paths=sorted(opened),
            free_generation_tested=True,teacher_forced=False,capability_demonstrated=False,architecture_winner=False,
            optimizer_updates=0,encoder_calls=0,phases=phases,seconds=seconds,
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,acceptance='author-checked')
        pin=write(a.output/'report.json',r);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)
    finally:
        if correction:correction.close()
        if parent:parent.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-proof',action='store_true');g.add_argument('--cuda',action='store_true')
    p.add_argument('--qualification',action='store_true')
    for name in ('request','generated','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pin','generated-pin'):p.add_argument('--'+name,required=True)
    for name in ('plan','run','readback'):
        p.add_argument('--corpus-'+name,type=Path,required=True);p.add_argument('--corpus-'+name+'-pin',required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(exc)))
        raise
