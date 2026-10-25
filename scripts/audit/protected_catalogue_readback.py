"""Matched protected-corpus readback using a pinned native authentication catalogue."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_inversion as original
from audit import protected_corpus_program as producer
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model import protected_corpus_lineage as lineage
from model import protected_corpus_readback as reader
from model import protected_corpus_training as training
from model.native_population_training import fingerprint
from senses.contract import digest
from text.constellation_corpus import file_sha,need
from text.legal_native_catalogue import attach


def pinned(path,pin):
    need(path.is_file() and file_sha(path)==pin,'PROTECTED_READBACK_FILE_PIN')
    return json.loads(path.read_bytes())


def controls(a,r,s,source,model):
    first=r['records'][0];b=r['binding'];ids=s['ids'][:384];branches=set(first['row']['branches'])
    kwargs=dict(ids=ids,cursor=0,step=0,previous=digest(b),before=first['row']['before'],branches=branches,source=source)
    out={};bad=deepcopy(first);bad['row']['cursor']+=1
    out['ledger_cursor']=producer.rejected(lambda:reader.row(bad,**kwargs),'PROTECTED_READBACK_LEDGER')
    out['missing_first_row']=producer.rejected(lambda:reader.row(r['records'][1],**kwargs),'PROTECTED_READBACK_LEDGER')
    bad=deepcopy(first);bad['row']['input']='0'*64;bad['pin']=digest(bad['row'])
    out['coherent_wrong_input']=producer.rejected(lambda:reader.row(bad,**kwargs),'PROTECTED_READBACK_INPUT')
    bad=deepcopy(r['checkpoints'][-1]);bad['after']='0'*64
    out['checkpoint_boundary']=producer.rejected(lambda:reader.checkpoint(bad,r['records'][-1],b,model),
                                                 'PROTECTED_READBACK_CHECKPOINT_LEDGER')
    bad=deepcopy(r['checkpoints'][-1]);bad.pop('origin')
    out['missing_origin']=producer.rejected(lambda:lineage.origin(bad,b),'PROTECTED_ORIGIN_REQUIRED')
    bad_binding=deepcopy(b);bad_binding['runtime']['threads']+=1
    out['different_runtime']=producer.rejected(lambda:lineage.origin(r['checkpoints'][-1],bad_binding),'PROTECTED_ORIGIN_BINDING')
    example=source['adapter'].fetch([int(ids[0])],side='train')[0]
    out['missing_byte']=producer.rejected(lambda:reader.complete_scores([
        dict(losses=torch.zeros(len(example.raw)),argmax=torch.zeros(len(example.raw),dtype=torch.int64))],[example]),
        'PROTECTED_READBACK_MISSING_BYTE')
    out['donor_metadata']=producer.rejected(lambda:reader.metadata([], [example],[int(ids[1])],source),
                                            'PROTECTED_READBACK_EVALUATION_METADATA')
    # An ancestry parser witness partitions the real persisted three-update ledger.
    # It is not a new training run or evidence that full-mode resume has executed.
    prior=r['checkpoints'][0];tip=deepcopy(r['checkpoints'][-1]);root=a.output/'ancestry-parser-witness';root.mkdir()
    priorpath=Path(prior['root']).parent/f"checkpoint-{prior['step']:06d}.json"
    origin=dict(schema='protected-corpus-origin-v1',program=b,
                resume=dict(path=str(priorpath),pin=file_sha(priorpath),reference=prior),
                start={k:prior[k] for k in ('cursor','step','ledger')},initial_correction=prior['after'])
    path=root/'origin.json';pin=write_committed(path,origin)
    tip.update(root=str(root/'checkpoint-000003'),origin=dict(path=str(path),pin=pin))
    need(lineage.origin(tip,b)==origin,'PROTECTED_READBACK_ANCESTRY_WITNESS')
    out['real_prefix_ancestry_parser']=True
    badroot=a.output/'negative-ancestry';badroot.mkdir();bad=deepcopy(origin);bad['start']['cursor']+=1
    path=badroot/'origin.json';pin=write_committed(path,bad);badtip=deepcopy(tip)
    badtip.update(root=str(badroot/'checkpoint-000003'),origin=dict(path=str(path),pin=pin))
    out['ancestry_cursor']=producer.rejected(lambda:lineage.origin(badtip,b),'PROTECTED_ORIGIN_CONTINUATION')
    return out


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(exist_ok=False);source=None;s=None;leased=False
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=pinned(a.plan/'report.json',a.plan_pin)
    need(plan['complete'] and plan['schema']=='protected-corpus-plan-v1' and plan['policy']==training.POLICY,
         'PROTECTED_READBACK_PLAN')
    base=Path(__file__).resolve().parents[1]
    implementation=dict(plan['code'])
    implementation.update({n:file_sha(base/n) for n in ('audit/protected_catalogue_readback.py','text/legal_native_catalogue.py',
        'model/protected_corpus_readback.py','audit/protected_corpus_program.py','model/protected_corpus_lineage.py')})
    need(all(file_sha(base/n)==h for n,h in implementation.items()),'PROTECTED_READBACK_CODE')
    try:
        source=original.sources(a.root)
        catalogue=attach(source,a.catalogue,a.catalogue_pin)
        need(source['adapter'].binding==plan['binding'],'PROTECTED_READBACK_SOURCE_BINDING')
        path=Path(plan['train_index']['path']);need(file_sha(path)==plan['train_index']['pin'],'PROTECTED_READBACK_INDEX')
        ids=np.load(path,mmap_mode='r',allow_pickle=False);census=reader.census(source,ids,plan)
        phases=dict(source_authentication_census=time.perf_counter()-tick);tick=time.perf_counter()
        if a.mode=='census':
            selected=reader.smallest_documents(source,plan)
            out=dict(complete=True,plan=a.plan_pin,source_census=census,selected=selected,
                     rule='minimum_complete_positions_then_document_within_original_100_per_side',no_forward=True)
        else:
            need(a.input and a.input_pin,'PROTECTED_READBACK_PRODUCER_REQUIRED')
            r=pinned(a.input/'report.json',a.input_pin)
            need(r['complete'] and r['plan']==a.plan_pin and r['full_training']==(r['binding']['scope']=='full'),
                 'PROTECTED_READBACK_PRODUCER')
            need(not r['full_training'] or (not a.cpu_proof and a.mode=='read' and a.evaluation is None),
                 'PROTECTED_READBACK_FULL_GPU_REQUIRED')
            if not a.cpu_proof:
                (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
                take_lease('29','protected-corpus-readback-'+a.mode);leased=True
                need(torch.cuda.is_available(),'PROTECTED_READBACK_CUDA_REQUIRED')
            device=torch.device('cpu' if a.cpu_proof else 'cuda:0')
            need(r['device']==str(device),'PROTECTED_READBACK_DEVICE')
            loadargs=SimpleNamespace(**vars(a));loadargs.mode='proof';s=producer.load(loadargs,device)
            loadargs.mode='full' if r['full_training'] else 'proof'
            b=producer.binding(loadargs,s,device,len(ids) if r['full_training'] else 384)
            need(r['binding']==b and r['runtime']==b['runtime'] and r['implementation']==s['implementation'],
                 'PROTECTED_READBACK_IMPLEMENTATION_RUNTIME')
            model,opt=training.create(s['parent'].model,s['config'])
            need(r['origin']==r['checkpoints'][-1]['origin'] and lineage.origin(r['checkpoints'][-1],b)['resume']==r['resume'],
                 'PROTECTED_READBACK_REPORTED_ORIGIN')
            phases['model_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
            ledger=reader.ledger(r,source,ids,model)
            training.restore(r['checkpoints'][-1],model,opt,b,size=b['size'])
            phases['ledger_checkpoints_restore']=time.perf_counter()-tick;tick=time.perf_counter()
            if r['full_training']:
                groups={side:v['documents'] for side,v in plan['evaluation'].items()};refs=r['full_evaluation'];selection=None
            else:
                need(a.selection and a.selection_pin,'PROTECTED_READBACK_SELECTION_REQUIRED')
                selection=pinned(a.selection/'report.json',a.selection_pin)
                need(selection['complete'] and selection['no_forward'] and selection['plan']==a.plan_pin
                     and selection['source_census']==census and selection['selected']==reader.smallest_documents(source,plan),
                     'PROTECTED_READBACK_SELECTION')
                groups={side:[v['document']] for side,v in selection['selected'].items()}
            if a.mode=='produce':
                need(not r['full_training'],'PROTECTED_READBACK_PROOF_ONLY')
                refs=original.evaluation(source,model,groups,ids,a.output/'evaluation')
                evaluation=dict(refs=refs,independently_replayed=False)
            else:
                if not r['full_training']:
                    need(a.evaluation and a.evaluation_pin,'PROTECTED_READBACK_EVALUATION_REQUIRED')
                    previous=pinned(a.evaluation/'report.json',a.evaluation_pin)
                    need(previous['complete'] and previous['mode']=='produce' and previous['input_pin']==a.input_pin
                         and previous['selection']==a.selection_pin and previous['runtime']==b['runtime'],
                         'PROTECTED_READBACK_EVALUATION_BINDING')
                    refs=previous['evaluation']['refs']
                evaluation=dict(totals=reader.evaluation(refs,source,model,groups,ids),independently_replayed=True)
            phases['complete_document_evaluation']=time.perf_counter()-tick;tick=time.perf_counter()
            negatives=controls(a,r,s,source,model) if a.mode=='read' and not r['full_training'] else {}
            out=dict(complete=True,input_pin=a.input_pin,plan=a.plan_pin,source_census=census,ledger=ledger,
                     selection=a.selection_pin,producer_implementation=s['implementation'],runtime=b['runtime'],
                     evaluation=evaluation,controls=negatives,device=str(device),full_training=r['full_training'],
                     full_gradient_replay=False,all_final_document_evaluation_bytes_exact=evaluation['independently_replayed'],
                     initial_witness_replayed=False)
            phases['negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
        if s:s['parent'].fresh();s['parent'].close();s=None
        original.close(source);source=None
        phases['finish']=time.perf_counter()-tick
        need(all(file_sha(base/n)==h for n,h in implementation.items()),'PROTECTED_READBACK_CODE_CHANGED')
        out.update(mode=a.mode,catalogue=catalogue,implementation=implementation,phases=phases,wall=time.perf_counter()-began,
                   cpu=time.process_time()-c0,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   capability_demonstrated=False,promoted=False,acceptance='author-checked',number_status='measured')
        if a.cpu_proof or a.mode=='census':need(not torch.cuda.is_initialized(),'PROTECTED_READBACK_CPU_ONLY')
        pin=write_committed(a.output/'report.json',out)
        print(json.dumps(dict(pin=pin,wall=out['wall'],selected=out.get('selected'),evaluation=out.get('evaluation'))),flush=True)
    finally:
        if source:original.close(source)
        if s:s['parent'].close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['census','produce','read'],required=True)
    for n in ('root','plan','output','catalogue'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--catalogue-pin',required=True);p.add_argument('--plan-pin',required=True);p.add_argument('--cpu-proof',action='store_true')
    for n in ('input','selection','evaluation'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise
