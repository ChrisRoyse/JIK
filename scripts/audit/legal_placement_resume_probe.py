"""Physical original/continuation proof, with a separate process at the saved boundary."""
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
import numpy as np
import torch

from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as worker
from claims.process_conditions import write_committed
from model.legal_placement_resume import Census, PlacementContinuation, recount
from model.native_population_training import cpu, fingerprint
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need


NAMES = ('model/legal_placement_resume.py','audit/legal_placement_resume_probe.py')


def append(stream, row):
    stream.write(canonical(row)+b'\n'); stream.flush(); os.fsync(stream.fileno())


def payload(model, optimizer, record, gradients, noise):
    return cpu(dict(model=dict(model.state_dict()),optimizer=optimizer.state_dict(),
                    record=record,gradients=gradients,noise=noise))


def state(worker):
    return fingerprint((worker.model.state_dict(),worker.optimizer.state_dict(),worker.step,worker.chain))


def controls(engine, path, reference, prefix, initial, census, item, output):
    results = {}; before = state(engine)
    def reject(name, fn, code):
        try: fn()
        except (ValueError,RuntimeError) as error:
            need(code in str(error), 'PLACEMENT_RESUME_WRONG_REFUSAL'); results[name] = str(error)
        else: raise ValueError('PLACEMENT_RESUME_NEGATIVE_ACCEPTED:'+name)
        need(state(engine) == before, 'PLACEMENT_RESUME_NEGATIVE_MUTATED')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    def restore(p=path, r=reference, step=saved['step'], chain=saved['chain'], ledger=prefix):
        return engine.restore(p,r,ledger,initial,cursor=step,chain=chain)
    reject('wrong_cursor',lambda:restore(step=saved['step']+1),'CHECKPOINT_BINDING')
    reject('wrong_chain',lambda:restore(chain='0'*64),'CHECKPOINT_BINDING')
    reject('physical_pin',lambda:restore(r=dict(reference,pin='0'*64)),'CHECKPOINT_PIN')
    reject('state_pin',lambda:restore(r=dict(reference,state='0'*64)),'CHECKPOINT_STATE')
    reject('wrong_document',lambda:engine.update(dict(item,document=census.documents[0])),'NEXT_DOCUMENT')
    rows = prefix.read_bytes().splitlines(keepends=True)
    for name, raw, code in (
        ('torn_prefix',b''.join(rows)[:-1],'TORN_PREFIX'),
        ('missing_prefix',b''.join(rows[:-1]),'TORN_PREFIX'),
        ('extra_prefix',b''.join(rows+[rows[-1]]),'EXTRA_PREFIX'),
        ('reordered_prefix',b''.join(reversed(rows)),'CHAIN')):
        p=output/(name+'.jsonl'); p.write_bytes(raw)
        reject(name,lambda p=p:restore(ledger=p),code)
    # Alter moments and recompute the checkpoint state pin. A valid checkpoint
    # hash alone cannot make it match the authenticated numerical ledger.
    broken = deepcopy(saved); first = next(iter(broken['optimizer']['state']))
    del broken['optimizer']['state'][first]
    p=output/'missing-moments.pt'; receipt=worker.measure.persist(p,broken)
    reject('missing_moments',lambda:restore(p,receipt),'PLACEMENT_RESUME_STATE')
    # The old runtime's own controls additionally exercise optimizer validation.
    runtime = worker.runtime.controls(engine.model,engine.optimizer,item,
        dict(item,side='query'),engine.target,engine.step,engine.policy)
    need(state(engine) == before, 'PLACEMENT_RESUME_RUNTIME_CONTROLS_MUTATED')
    results['runtime'] = runtime
    return results


def run(a):
    began=time.perf_counter(); cpu_start=time.process_time(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    source,bundle=worker.source_bundle(a.root); old_code=worker.code(source)
    execution={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in NAMES}
    for n,pin in {**old_code,**execution}.items():
        f=a.output/'source'/n; f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes()); need(file_sha(f)==pin,'PLACEMENT_RESUME_SOURCE_COPY')
    data=worker.load_data(argparse.Namespace(scope='small',root=a.root),bundle)
    items=data['items']; fitting=[i for i in items if i['side']=='train']; labels=data['labels']
    target=deepcopy(labels.model).eval().requires_grad_(False); frozen=fingerprint(target.state_dict())
    policy=dict(bundle['policy'],device='cpu')
    census=Census([i['document'] for i in fitting],np.asarray([len(i['codes']) for i in fitting],dtype=np.int64),
        np.asarray([[sum(len(f[n]) for f in i['frames']) for n in target.roster] for i in fitting],dtype=np.int64),target.roster)
    need(len(fitting)==5 and census.counts(5)['rows']==66,'PLACEMENT_RESUME_PROOF_SCOPE')
    worker.measure.persist(a.output/'census.pt',census.value())
    metadata=worker.measure.metadata(items); donors=worker.measure.donor_plan(metadata)
    worker.measure.validate_donors(metadata,donors)
    donor_pin=write_committed(a.output/'donors.json',dict(metadata=metadata,donors=donors))
    binding=dict(code=old_code,policy=policy,source=data['binding'],target=bundle['target_ref'],scope='small',
                 device='cpu',torch_version=str(torch.__version__),donors=donor_pin,reader_match=None)
    previous=None
    if a.previous:
        previous=worker.pinned(a.previous,a.previous_pin); worker.physical(a.previous,previous)
        need(previous['mode']=='seed' and previous['binding']==binding and previous['execution']==execution
             and previous['census']==census.pin and previous['pid']!=os.getpid(), 'PLACEMENT_RESUME_PROCESS_BOUNDARY')
    need((a.mode=='continue') == bool(previous), 'PLACEMENT_RESUME_PREVIOUS_REQUIRED')
    results={}; timings=[]; matched=dict(tensors=0,bytes=0)
    for name in (tuple(reversed(worker.runtime.original.ARCHITECTURES)) if previous else worker.runtime.original.ARCHITECTURES):
        branch=dict(binding,architecture=name); out=a.output/name; out.mkdir()
        engine=PlacementContinuation(bundle,target,branch,census)
        initial=engine.snapshot(); initial_ref=worker.measure.persist(out/'checkpoint-0.pt',initial)
        if not previous:
            model,optimizer=worker.runtime.make(bundle,target,name,'cpu'); chain=digest(branch)
            refdir=out/'uninterrupted'; refdir.mkdir()
            with (refdir/'steps.jsonl').open('xb') as ledger:
                for cursor,item in enumerate(fitting):
                    tick=time.perf_counter()
                    update,gradients,noise=worker.runtime.original.update(model,optimizer,item,cursor,policy,target=target)
                    need(update==bundle['cpu_reference'][name]['updates'][cursor]['record'], 'PLACEMENT_RESUME_PRIOR_CPU')
                    record=dict(parent=chain,cursor=cursor,update=update); chain=digest(record); record['hash']=chain
                    append(ledger,record)
                    worker.measure.persist(refdir/f'update-{cursor+1}.pt',payload(model,optimizer,record,gradients,noise))
                    timings.append(dict(architecture=name,arm='uninterrupted',step=cursor+1,seconds=time.perf_counter()-tick))
            worker.checkpoint(refdir/'checkpoint-5.pt',model,optimizer,branch,5,66,chain,policy)
            records,_=worker.measure.evaluate(model,target,labels,items,donors,bundle['statistics'],refdir/'final')
            worker.measure.persist(refdir/'metrics.pt',records)
            del model,optimizer
            start,stop=0,2
        else:
            ref=previous['results'][name]['checkpoint']; old=a.previous/name
            prefix=old/'steps.jsonl'; cp=old/'checkpoint-2.pt'
            initial= torch.load(old/'checkpoint-0.pt',map_location='cpu',weights_only=True)
            rd.exact(initial,engine.snapshot())
            saved=torch.load(cp,map_location='cpu',weights_only=True)
            imported=engine.restore(cp,ref,prefix,initial,cursor=2,chain=saved['chain'])
            worker.measure.persist(out/'restored.pt',engine.snapshot())
            control_dir=out/'controls';control_dir.mkdir()
            negatives=controls(engine,cp,ref,prefix,initial,census,fitting[2],control_dir)
            start,stop=2,5
        with (out/'steps.jsonl').open('xb') as ledger:
            if previous:
                with (a.previous/name/'steps.jsonl').open('rb') as f:
                    for raw in f:ledger.write(raw)
                ledger.flush();os.fsync(ledger.fileno())
            for cursor in range(start,stop):
                tick=time.perf_counter(); record,gradients,noise=engine.update(fitting[cursor])
                append(ledger,record); value=payload(engine.model,engine.optimizer,record,gradients,noise)
                worker.measure.persist(out/f'update-{cursor+1}.pt',value)
                base=(a.previous/name if previous else out)/'uninterrupted'/f'update-{cursor+1}.pt'
                comparison=rd.exact(value,torch.load(base,map_location='cpu',weights_only=True))
                for key in matched:matched[key]+=comparison[key]
                timings.append(dict(architecture=name,arm='resumed' if previous else 'prefix',step=cursor+1,seconds=time.perf_counter()-tick))
        ref=worker.measure.persist(out/f'checkpoint-{stop}.pt',engine.snapshot())
        counts=recount(out/'steps.jsonl',torch.load(out/f'checkpoint-{stop}.pt',map_location='cpu',weights_only=True),initial,census)
        result=dict(checkpoint=ref,counts=counts)
        if previous:
            final=torch.load(a.previous/name/'uninterrupted/checkpoint-5.pt',map_location='cpu',weights_only=True)
            rd.exact(engine.snapshot(),final)
            records,_=worker.measure.evaluate(engine.model,target,labels,items,donors,bundle['statistics'],out/'final')
            worker.measure.persist(out/'metrics.pt',records)
            comparison=rd.exact(records,torch.load(a.previous/name/'uninterrupted/metrics.pt',map_location='cpu',weights_only=True))
            for key in matched:matched[key]+=comparison[key]
            result.update(imported=imported,controls=negatives,final_metrics_exact=True,metrics=fingerprint(records))
        results[name]=result
        print(json.dumps(dict(architecture=name,mode=a.mode,step=stop,exact=True)),flush=True)
    data['close'](); need(fingerprint(target.state_dict())==frozen and not torch.cuda.is_initialized(),'PLACEMENT_RESUME_FROZEN_OR_DEVICE')
    for n,pin in {**old_code,**execution}.items():need(file_sha(Path(__file__).resolve().parents[1]/n)==pin,'PLACEMENT_RESUME_SOURCE_CHANGED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,mode=a.mode,binding=binding,execution=execution,census=census.pin,
        previous_pin=a.previous_pin,pid=os.getpid(),results=results,comparison=matched,timings=timings,files=files,
        seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu_start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,
        cpu_continuation_exact=bool(previous),cuda_continuation_qualified=False,live_processes_changed=False,
        acceptance='author-checked',number_status='measured',model_promotion=False,scientific_admission=False)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),comparison=matched)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('seed','continue'),required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
