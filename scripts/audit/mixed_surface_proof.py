"""Characterize mixed-atom arithmetic and optimizer state on retained legal atoms."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import legal_surface_decoder_proof as previous
from claims.process_conditions import write_committed
from model.contracts import Instrument,ModelConfig
from model.legal_surface_inputs import SurfaceExample
from model.native import NativeInput
from model.native_population_training import cpu,fingerprint
from model.stream_surface import StreamSurfaceDecoder,batches as original_batches
from model import mixed_surface_batches as mixed
from text.constellation_corpus import file_sha,need

SOURCE=('legal-surface-decoder-proof-v1-20261023','3400b9c5813d56859603a89d5f0fe3182370b94a68e014199b50a2ff605c8d45')
READBACK=('legal-surface-decoder-readback-v1-20261023','61e974430b13b259135eee4123a26c13957d8545361d7a37a2255258f23d2c8f')
COHORTS=(8,32,128,512)


def code():
    root=Path(__file__).resolve().parents[1]
    names=set(previous.code())|{'model/mixed_surface_batches.py','audit/mixed_surface_proof.py','audit/mixed_surface_readback.py'}
    return {n:file_sha(root/n) for n in sorted(names)}


def load(root):
    p=root/SOURCE[0];need(file_sha(p/'report.json')==SOURCE[1],'MIXED_PROOF_SOURCE_PIN')
    r=json.loads((p/'report.json').read_bytes())
    q=root/READBACK[0]/'report.json';need(file_sha(q)==READBACK[1],'MIXED_PROOF_READBACK_PIN')
    readback=json.loads(q.read_bytes())
    need(r['complete'] and readback['complete'] and readback['input_pin']==SOURCE[1]
         and r['code']==previous.code(),'MIXED_PROOF_AUTHORITY')
    for name,pin in r['files'].items():need(file_sha(p/name)==pin,'MIXED_PROOF_PHYSICAL')
    config=ModelConfig(**r['config']);specs=tuple(Instrument(**s).validate() for s in r['instruments'])
    saved=torch.load(p/'observations.pt',map_location='cpu',weights_only=True)
    examples=[];metadata=[]
    for d in saved:
        for v in d['observations']:
            native={s.name:NativeInput(s.identity,(v['row'],),v['native'][s.name][None],
                torch.zeros((1,len(v['native'][s.name])),dtype=torch.bool),torch.zeros(1,dtype=torch.bool)) for s in specs}
            examples.append(SurfaceExample(v['occurrence'],v['uid'],d['side'],bytes(v['raw'].tolist()),native))
            metadata.append(dict(document=d['document'],side=d['side'],train_present=v['train_present']))
    initial=torch.load(p/'initial.pt',map_location='cpu',weights_only=True)
    return r,config,specs,examples,metadata,initial


def packed_state(batch):
    return dict(prefix=batch.prefix,labels=batch.labels,offsets=batch.offsets,denominators=batch.denominators,
        native={n:dict(identity=x.identity,rows=x.rows,values=x.values,padding=x.padding,absent=x.absent) for n,x in batch.native.items()},
        records=batch.records,atoms=batch.atoms)


def source_batches(examples,config,specs,rows):
    if rows is not None:
        # The source tile is a physical bound, not an epoch/data cap.
        for lo in range(0,len(examples),128):
            part=examples[lo:lo+128]
            for batch in mixed.batches(part,config,specs,rows=rows):
                # Preserve global logical-group normalization across physical tiles.
                records=tuple(dict(r,atom=r['atom']+lo) for r in batch.records)
                yield mixed.MixedSurfaceBatch(batch.prefix,batch.labels,batch.offsets,batch.denominators,
                                               batch.native,records,len(examples))
    else:
        for i,e in enumerate(examples):
            for prefix,labels,offsets,native,items in original_batches(e.raw,e.native,config):
                records=tuple(dict(atom=i,occurrence=e.occurrence,uid=e.uid,offset=x['offset'],
                                   scored=x['scored'],prefix_length=len(x['prefix'])) for x in items)
                yield mixed.MixedSurfaceBatch(prefix,labels,offsets,torch.full((len(items),),len(e.raw)+1,dtype=torch.float32),
                                               native,records,len(examples))


def evaluate(model,examples,rows,*,backward=False):
    if backward:need(all(e.side=='train' for e in examples),'MIXED_PROOF_TRAIN_ONLY')
    result=[dict(losses=torch.full((len(e.raw)+1,),float('nan')),argmax=torch.full((len(e.raw)+1,),-1,dtype=torch.int64)) for e in examples]
    calls=0;objective=0.;started=time.perf_counter()
    for batch in source_batches(examples,model.config,model.specs,rows):
        logits=model(batch.prefix,batch.native,offsets=batch.offsets)['logits']
        if rows is None:
            loss=F.cross_entropy(logits.reshape(-1,model.config.vocabulary),batch.labels.reshape(-1),
                                ignore_index=-100,reduction='sum')/float(batch.denominators[0])/len(examples)
            token=F.cross_entropy(logits.reshape(-1,model.config.vocabulary),batch.labels.reshape(-1),
                                 ignore_index=-100,reduction='none').reshape(batch.labels.shape)
        else:loss,token=mixed.loss(logits,batch)
        if backward:loss.backward()
        objective+=float(loss.detach());calls+=1
        for row,r in enumerate(batch.records):
            i=r['atom'];a,b=r['scored'];left=a-r['offset'];right=b-r['offset']
            need(bool(torch.isnan(result[i]['losses'][a:b]).all()),'MIXED_PROOF_DUPLICATE_SCORE')
            result[i]['losses'][a:b]=token[row,left:right].detach()
            result[i]['argmax'][a:b]=logits[row,left:right].argmax(-1).detach()
    need(all(bool(v['losses'].isfinite().all()) and bool((v['argmax']>=0).all()) for v in result),'MIXED_PROOF_MISSING_SCORE')
    return result,dict(objective=objective,calls=calls,positions=sum(len(e.raw)+1 for e in examples),seconds=time.perf_counter()-started)


def difference(left,right):
    need(len(left)==len(right),'MIXED_PROOF_COMPARISON_ROWS')
    return dict(max_token_loss_difference=max(float((a['losses']-b['losses']).abs().max()) for a,b in zip(left,right)),
        differing_token_losses=sum(int((a['losses']!=b['losses']).sum()) for a,b in zip(left,right)),
        differing_argmax=sum(int((a['argmax']!=b['argmax']).sum()) for a,b in zip(left,right)),
        exact=fingerprint(left)==fingerprint(right))


def tensor_difference(left,right):
    need(left.keys()==right.keys(),'MIXED_PROOF_PARAMETER_KEYS')
    return dict(exact=fingerprint(left)==fingerprint(right),
        maximum=max(float((left[n]-right[n]).abs().max()) for n in left),
        differing_elements=sum(int((left[n]!=right[n]).sum()) for n in left))


def run(a):
    started=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pins=code();r,config,specs,examples,metadata,initial=load(a.root)
    train=[e for e in examples if e.side=='train'];selected=[next(e for e in train if e.occurrence==v['occurrence']) for v in r['gradients']]
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    schedules={}
    for rows in COHORTS:
        published=[packed_state(b) for b in mixed.batches(selected,config,specs,rows=rows)]
        torch.save(published,a.output/f'packed-{rows}.pt')
        manifest=[]
        for b in source_batches(examples,config,specs,rows):
            state=packed_state(b);manifest.append(dict(records=b.records,state=fingerprint(state),shape=list(b.prefix.shape),
                native_shapes={n:list(x.values.shape) for n,x in b.native.items()}))
        write_committed(a.output/f'schedule-{rows}.json',manifest);schedules[str(rows)]=len(manifest)
    phases['packing_and_persistence']=time.perf_counter()-tick;tick=time.perf_counter()
    # One fixed different-document TRAIN donor per source; indices never enter the model.
    train_ids=[i for i,e in enumerate(examples) if e.side=='train'];shuffled=[];donors=[]
    for i,e in enumerate(examples):
        j=next(j for j in train_ids if metadata[j]['document']!=metadata[i]['document'] and examples[j].uid!=e.uid)
        donor=examples[j];donors.append(j)
        shuffled.append(SurfaceExample(e.occurrence,e.uid,e.side,e.raw,donor.native))
    results={};baseline=None
    for label,rows in [('reference',None)]+[(f'rows-{n}',n) for n in COHORTS]:
        arm_started=time.perf_counter();model=StreamSurfaceDecoder(config,specs);model.load_state_dict(initial,strict=True);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
        model.zero_grad(set_to_none=True);training,fit=evaluate(model,train,rows,backward=True)
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        need(gradients and all(bool(v.isfinite().all()) for v in gradients.values()),'MIXED_PROOF_GRADIENT_FINITE')
        need(all(any(bool(v.abs().any()) for n,v in gradients.items() if n.startswith(prefix))
                 for s in specs for prefix in ['projectors.'+s.name+'.','blocks.0.branches.'+s.name+'.']),
             'MIXED_PROOF_DISCONNECTED_BRANCH')
        optimizer.step();after=cpu(model.state_dict());moments=cpu(optimizer.state_dict())
        with torch.no_grad():
            evaluation,score=evaluate(model,examples,rows);control,shuf=evaluate(model,shuffled,rows)
        artifact=dict(training=training,evaluation=evaluation,shuffled=control,gradients=gradients,after=after,optimizer=moments)
        torch.save(artifact,a.output/(label+'.pt'))
        need(fingerprint(torch.load(a.output/(label+'.pt'),weights_only=True))==fingerprint(artifact),'MIXED_PROOF_SAVED_STATE')
        stats=dict(train=fit,evaluation=score,shuffled=shuf,seconds=time.perf_counter()-arm_started)
        value=dict(state=fingerprint(artifact),profile=stats,optimizer_steps=1,
            conditioning_delta=[float((b['losses']-c['losses']).mean()) for b,c in zip(control,evaluation)])
        if baseline is None:baseline=artifact
        else:
            value['difference']=dict(training=difference(training,baseline['training']),evaluation=difference(evaluation,baseline['evaluation']),
                shuffled=difference(control,baseline['shuffled']),gradients=tensor_difference(gradients,baseline['gradients']),
                parameters=tensor_difference(after,baseline['after']),optimizer_exact=fingerprint(moments)==fingerprint(baseline['optimizer']))
        results[label]=value;print(json.dumps(dict(arm=label,calls=fit['calls'],seconds=stats['seconds'],difference=value.get('difference'))),flush=True)
    phases['matched_decoder_execution']=time.perf_counter()-tick;tick=time.perf_counter()
    stable=dict(results={k:{n:v for n,v in r.items() if n!='profile'} for k,r in results.items()},donors=donors,metadata=metadata,
                observations=len(examples),train_atoms=len(train),selected=[e.occurrence for e in selected],schedules=schedules)
    write_committed(a.output/'results.json',stable)
    need(code()==pins and not torch.cuda.is_initialized(),'MIXED_PROOF_ENVIRONMENT_CHANGED')
    for name,pin in pins.items():
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(p)==pin,'MIXED_PROOF_CODE_COPY')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(.9<=sum(phases.values())/seconds<=1.000001,'MIXED_PROOF_PHASE_COVERAGE')
    report=dict(complete=True,source=SOURCE,readback=READBACK,code=pins,files=files,results=results,phases=phases,
        seconds=seconds,cpu_seconds=time.process_time()-process,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        numerical_equivalence=all(v['state']==results['reference']['state'] for k,v in results.items() if k!='reference'),
        promoted=False,gpu_characterized=False,scientific_admission=False,learned_capability=False,
        number_status='measured',acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
