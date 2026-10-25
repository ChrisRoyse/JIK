"""Durable complete-source answer training after same-CUDA and full-length qualification."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

VISIBLE=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_epoch_cuda as reference
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model.legal_answer_epoch import AnswerEpoch, persist
from model.legal_answer_run import AnswerRun
from model.native_population_training import fingerprint
from model.task_answer_training import extension
from text.constellation_corpus import file_sha, need

ROOT=reference.ROOT; BASE=reference.BASE; ARCHITECTURES=reference.ARCHITECTURES
RUN_PROOFS=(('legal-answer-run-proof-v2-20261024','f16fe76f6030c323a089b0b62eeb78d43c86e1557f08c0c1bcff4fc9f4defba3'),
 ('legal-answer-run-proof-cold-v2-20261024','86b2dc6ded8ef0cf0123f376c6029dfc568af46b4f7f3e181743657d7da25ff5'),
 ('legal-answer-run-readback-v1-20261024','e19ab08481d2bcd1c4f2c0ef76fb76c3cb8fab3ab959add562c87a004701a0c7'))
NEW=('model/legal_answer_run.py','audit/legal_answer_source_training.py','audit/legal_answer_source_queue.py')
LONG=(5796,105748)


def read(path,pin):
    need(file_sha(path)==pin,'ANSWER_SOURCE_PROOF_PIN');return json.loads(Path(path).read_bytes())


def canonical(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def authenticate(root):
    reports,code,parent_fresh=reference.authenticate(root)
    prior=[read(root/d/'report.json',h) for d,h in RUN_PROOFS]
    a,b,back=prior
    need(a['complete'] and b['complete'] and b['cold_reproduced'] and b['previous_pin']==RUN_PROOFS[0][1]
         and a['numerical']==b['numerical'] and a['controls']==b['controls'],'ANSWER_SOURCE_RUN_PROOF')
    need(back['complete'] and back['producer']==RUN_PROOFS[0][1] and back['cold']==RUN_PROOFS[1][1]
         and back['cold_exact'] and back['all_committed_checkpoints_exact']
         and back['full_schedule_initialized_and_reopened'] and back['full_schedule_optimizer_updates']==0,
         'ANSWER_SOURCE_RUN_READBACK')
    for d,r in zip((v[0] for v in RUN_PROOFS[:2]),prior[:2]):
        for n,h in r['files'].items():need(file_sha(root/d/n)==h,'ANSWER_SOURCE_RUN_BYTES')
    code=dict(code,**{n:file_sha(BASE/n) for n in NEW})
    need(code['model/legal_answer_run.py']==a['code']['model/legal_answer_run.py'],'ANSWER_SOURCE_RUN_CODE')
    def fresh():
        parent_fresh()
        for n,h in code.items():need(file_sha(BASE/n)==h,'ANSWER_SOURCE_CODE_CHANGED')
        for d,h in RUN_PROOFS:need(file_sha(root/d/'report.json')==h,'ANSWER_SOURCE_RUN_CHANGED')
    fresh();return reports,code,fresh


def qualification(root,pin,code):
    q=read(root/'report.json',pin);need(q['complete'] and q['gpu_qualified'],'ANSWER_SOURCE_CUDA_REQUIRED')
    outputs={v['label']:v for v in q['outputs']}
    need(set(outputs)=={'gpu','gpu-cold'},'ANSWER_SOURCE_CUDA_PAIR')
    reports={}
    for label,v in outputs.items():
        path=Path(v['path']);r=read(path/'report.json',v['pin']);reports[label]=r
        need(r['complete'] and r['mode']=='cuda' and r['gpu_used'] and r['all_updates_exact']
             and r['checkpoint_restores_exact'] and r['optimizer_updates']==30 and r['decisions']==396
             and r['binding']['device']=='cuda:0','ANSWER_SOURCE_CUDA_SCOPE')
        need(all(code.get(n)==h for n,h in r['code'].items()),'ANSWER_SOURCE_CUDA_CODE')
        for n,h in r['files'].items():need(file_sha(path/n)==h,'ANSWER_SOURCE_CUDA_BYTES')
    a,b=reports['gpu'],reports['gpu-cold']
    need(b['previous_pin']==outputs['gpu']['pin'] and b['cold_reproduced'] and a['binding']==b['binding']
         and a['controls']==b['controls'],'ANSWER_SOURCE_CUDA_COLD')
    for arch in ARCHITECTURES:need(a['results'][arch]['updates']==b['results'][arch]['updates'],'ANSWER_SOURCE_CUDA_UPDATES')
    return outputs,reports


def bind(path,value):
    value=canonical(value)
    if path.exists():need(json.loads(path.read_bytes())==value,'ANSWER_SOURCE_BINDING_CHANGED')
    else:write_committed(path,value)
    return value


def execute(output,make,*,documents,mode,expected=None):
    output.mkdir(exist_ok=True);epoch=make(documents);interval=256 if mode=='full' else 2
    run=AnswerRun(output/'run',epoch,checkpoint_documents=interval)
    initial_cursor=epoch.cursor;rows=[];restores=[];start=time.perf_counter()
    elapsed=dict(read=0.,update=0.,journal=0.,reopen=0.);peaks=[]
    try:
        while epoch.cursor<len(epoch.documents):
            cursor=epoch.cursor;doc=epoch.documents[cursor];projected=[0]
            def progress(module,args,value):
                projected[0]+=1
                if projected[0]%128==0:
                    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                                       if x.startswith('MemAvailable:')).split()[1])*1024
                    need(available>=8*1024**3,'ANSWER_SOURCE_HOST_PHYSICAL_HEADROOM')
                    row=dict(document=doc,projected_atoms=projected[0],cursor=cursor,
                        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        cuda_allocated=torch.cuda.memory_allocated(),time=time.time())
                    with (output/'progress.jsonl').open('a') as f:
                        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            hook=epoch.model.forecaster.predictor.projectors[epoch.model.roster[0]].register_forward_hook(progress)
            tick=time.perf_counter()
            try:value,row=run.step()
            finally:hook.remove()
            duration=time.perf_counter()-tick
            elapsed['read']+=row['read_seconds'];elapsed['update']+=row['update_seconds']
            elapsed['journal']+=duration-row['read_seconds']-row['update_seconds']
            before,after=epoch.counts(cursor),epoch.counts(cursor+1)
            need(projected[0]==after['atoms']-before['atoms'],'ANSWER_SOURCE_PROJECTED_COVERAGE')
            need(value['record']['storage']['remaining_saved_storage_bytes']==0,'ANSWER_SOURCE_GRAPH_RETAINED')
            if mode!='full':
                if expected is not None:
                    old=torch.load(expected/f'update-{cursor+6}.pt',map_location='cpu',weights_only=True)
                    need(fingerprint(reference.numerical(value))==fingerprint(reference.numerical(old)),
                         'ANSWER_SOURCE_REFERENCE_UPDATE');del old
                pin=persist(output/f'update-{cursor+6}.pt',value)
                rows.append(dict(document=doc,cursor=cursor,numerical=fingerprint(reference.numerical(value)),
                    checkpoint_payload=pin,chain=row['chain'],counts=after))
                peaks.append(dict(document=doc,projected_atoms=projected[0],storage=value['record']['storage'],
                                  phases=value['record']['phases']))
            del value,row
            # The long case checkpoints immediately after the complete long update.
            reopen=(mode=='short' and epoch.cursor==2) or (mode=='long' and epoch.cursor==1)
            if reopen:
                tick=time.perf_counter();run.commit();saved=fingerprint(epoch.snapshot());cursor=epoch.cursor
                run.close();del run,epoch;gc.collect();epoch=make(documents)
                run=AnswerRun(output/'run',epoch,checkpoint_documents=interval)
                need(epoch.cursor==cursor and fingerprint(epoch.snapshot())==saved,'ANSWER_SOURCE_REOPEN_CHANGED')
                restores.append(dict(cursor=cursor,state=saved));elapsed['reopen']+=time.perf_counter()-tick
            if epoch.cursor==run.committed_cursor:
                print(json.dumps(dict(path=str(output),committed_documents=epoch.cursor,counts=epoch.counts(epoch.cursor))),flush=True)
        result=run.finish();saved=fingerprint(epoch.snapshot());run.close();del run,epoch;gc.collect()
        tick=time.perf_counter();epoch=make(documents);run=AnswerRun(output/'run',epoch,checkpoint_documents=interval)
        need(fingerprint(epoch.snapshot())==saved and run.finish()==result,'ANSWER_SOURCE_COMPLETED_REOPEN')
        elapsed['reopen']+=time.perf_counter()-tick
        return dict(result=result,updates=rows,restores=restores,peaks=peaks,final_state=saved,
                    initial_cursor=initial_cursor,seconds=time.perf_counter()-start,phases=elapsed)
    finally:
        if 'run' in locals():run.close()


def signatures(results):
    return {arch:{name:dict(final_state=v['final_state'],chain=v['result']['chain'],counts=v['result']['counts'],
        updates=[{k:x for k,x in row.items() if k!='checkpoint_payload'} for row in v['updates']],restores=v['restores'])
        for name,v in branches.items()} for arch,branches in results.items()}


def capacity(producer,producer_pin,cold,cold_pin,source,binding):
    a=read(producer/'report.json',producer_pin);b=read(cold/'report.json',cold_pin)
    for path,r in ((producer,a),(cold,b)):
        need(r['complete'] and r['mode']=='proof' and r['gpu_used'] and r['source']==source
             and r['binding']==binding and r['optimizer_updates']==21
             and set(r['results'])==set(ARCHITECTURES),'ANSWER_SOURCE_CAPACITY_SCOPE')
        for n,h in r['files'].items():need(file_sha(path/n)==h,'ANSWER_SOURCE_CAPACITY_BYTES')
        for branches in r['results'].values():
            need(set(branches)=={'short','long'} and branches['short']['result']['counts']['documents']==5
                 and branches['long']['result']['counts']['atoms']==3215
                 and branches['long']['result']['counts']['documents']==2,'ANSWER_SOURCE_CAPACITY_COVERAGE')
    need(b['previous_pin']==producer_pin and b['cold_reproduced'] and signatures(a['results'])==signatures(b['results']),
         'ANSWER_SOURCE_CAPACITY_COLD')


def run(a):
    began=tick=time.perf_counter();phases={};corpus=None;leased=False
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==VISIBLE and not torch.cuda.is_initialized(),'ANSWER_SOURCE_IMPORT_DEVICE')
    a.output.mkdir(exist_ok=a.mode=='full')
    need(not (a.output/'report.json').exists(),'ANSWER_SOURCE_ALREADY_COMPLETE')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        reports,code,fresh=authenticate(a.root)
        corpus,labels,continuation,bundle,inherited=reference.prepare(a.root)
        need(all(code.get(n)==h for n,h in inherited.items()),'ANSWER_SOURCE_INHERITED_CODE')
        target,target_pin=extension(bundle);policy=dict(bundle['policy'],device='cpu');initials={};census={};controls={}
        def make(arch,documents):
            return AnswerEpoch(corpus,bundle,target,target_pin,arch,initials[arch],5,policy,code,documents=documents)
        for arch in ARCHITECTURES:
            initials[arch]=torch.load(a.root/reference.PROOFS['initial'][0]/arch/'update-5.pt',map_location='cpu',weights_only=True)
            epoch=make(arch,None);census[arch]=epoch.counts(len(epoch.documents))
            need(len(epoch.documents)==91540 and census[arch]['atoms']==6547176 and census[arch]['decisions']==6460051,
                 'ANSWER_SOURCE_FULL_CENSUS');del epoch
        long_census=[]
        for doc in LONG:
            item,request,_=corpus.example(doc,fitting=True)
            long_census.append(dict(document=doc,atoms=len(item['context'])+len(item['answers']),answers=len(item['answers']),
                                    request=request.pin,native=item['native_pin']))
        need([x['atoms'] for x in long_census]==[3210,5],'ANSWER_SOURCE_LONG_CENSUS')
        source=bind(a.output/'source.json',dict(schema='complete-answer-source-v1',code=code,source=corpus.binding,
            target=target_pin,initial_step=5,initial={n:fingerprint((v['model'],v['optimizer'])) for n,v in initials.items()},
            policy=bundle['policy'],counts=census,long=long_census,short=reference.DOCUMENTS,
            full_checkpoint_documents=256,scope='complete_document_continuation_not_question_answering'))
        phases['source_authentication']=time.perf_counter()-tick;tick=time.perf_counter();qualified=None;previous=None
        if a.mode=='inspect':
            for arch in ARCHITECTURES:
                out=a.output/arch;out.mkdir();controls[arch]=reference.negatives(make(arch,reference.DOCUMENTS),out)
            binding=dict(source=source,device='cpu',torch=str(torch.__version__),cuda=None)
        else:
            inspection=read(a.inspection/'report.json',a.inspection_pin)
            need(inspection['complete'] and inspection['mode']=='inspect' and inspection['source']==source
                 and inspection['optimizer_updates']==0 and not inspection['gpu_used'],'ANSWER_SOURCE_INSPECTION')
            for n,h in inspection['files'].items():need(file_sha(a.inspection/n)==h,'ANSWER_SOURCE_INSPECTION_BYTES')
            controls=inspection['controls'];qualified,_=qualification(a.qualification,a.qualification_pin,code)
            binding=dict(source=source,device='cuda:0',torch=str(torch.__version__),cuda=torch.version.cuda,
                precision='float32_tf32_off',qualification=a.qualification_pin,history='all_prior_positions_with_gradients')
            binding=canonical(binding)
            if a.mode=='full':capacity(a.proof,a.proof_pin,a.cold,a.cold_pin,source,binding)
            if a.previous:
                previous=read(a.previous/'report.json',a.previous_pin)
                need(a.mode=='proof' and previous['complete'] and previous['mode']=='proof' and previous['binding']==binding
                     and previous['optimizer_updates']==21,'ANSWER_SOURCE_PREVIOUS_BINDING')
                for n,h in previous['files'].items():need(file_sha(a.previous/n)==h,'ANSWER_SOURCE_PREVIOUS_BYTES')
            need(VISIBLE!='','ANSWER_SOURCE_GPU_HIDDEN')
            suffix=0
            while (a.output/f'nvidia-before-{suffix:06d}.txt').exists():suffix+=1
            (a.output/f'nvidia-before-{suffix:06d}.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
            need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
                 'ANSWER_SOURCE_GPU_RESIDENT')
            take_lease('29','complete-answer-source');leased=True
            need(torch.cuda.is_available(),'ANSWER_SOURCE_CUDA_REQUIRED')
            target.to('cuda:0');policy=dict(bundle['policy'],device='cuda:0');torch.cuda.reset_peak_memory_stats()
        binding=bind(a.output/'binding.json',binding);phases['qualification']=time.perf_counter()-tick;results={}
        if a.mode!='inspect':
            for arch in ARCHITECTURES:
                tick=time.perf_counter();out=a.output/arch;out.mkdir(exist_ok=a.mode=='full');fresh()
                factory=lambda docs:make(arch,docs)
                if a.mode=='proof':
                    short=execute(out/'short',factory,documents=reference.DOCUMENTS,mode='short',
                                  expected=Path(qualified['gpu']['path'])/arch)
                    long=execute(out/'long',factory,documents=LONG,mode='long')
                    results[arch]=dict(short=short,long=long)
                else:results[arch]=dict(full=execute(out/'full',factory,documents=None,mode='full'))
                phases[arch]=time.perf_counter()-tick;gc.collect()
            if previous:need(signatures(results)==signatures(previous['results']),'ANSWER_SOURCE_COLD_NUMERICAL')
        tick=time.perf_counter();corpus.close();corpus.cache.close();corpus.reader.close();labels.fresh();continuation.fresh();fresh()
        need(fingerprint(target.state_dict())==bundle['target_fingerprint'],'ANSWER_SOURCE_TARGET_CHANGED')
        if a.mode=='inspect':need(not torch.cuda.is_initialized(),'ANSWER_SOURCE_INSPECTION_GPU')
        # Proof archives include all payloads. Full-run checkpoint/row receipts are already in the journal.
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()} if a.mode!='full' else \
              {n:file_sha(a.output/n) for n in ('source.json','binding.json')}
        phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001,'ANSWER_SOURCE_PHASES')
        report=dict(complete=True,mode=a.mode,source=source,binding=binding,code=code,files=files,controls=controls,results=results,
            optimizer_updates=0 if a.mode=='inspect' else (21 if a.mode=='proof' else 3*91540),
            cold_reproduced=previous is not None,previous_pin=a.previous_pin,phases=phases,seconds=seconds,gpu_used=leased,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if leased else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if leased else 0,
            full_population_trained=a.mode=='full',native_encoder_calls=0,promoted=False,capability_demonstrated=False,
            acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),mode=a.mode,seconds=seconds)),flush=True)
    finally:
        if corpus is not None and not corpus.closed:corpus.close();corpus.cache.close();corpus.reader.close()
        if leased:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['inspect','proof','full'],required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    for n in ('inspection','qualification','previous','proof','cold'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():
            index=0
            while (a.output/f'failure-{index:06d}.json').exists():index+=1
            write_committed(a.output/f'failure-{index:06d}.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
