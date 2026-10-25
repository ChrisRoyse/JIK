"""Full-size protected decoder updates and source-bound cursor checkpoints."""
from dataclasses import asdict
import re
import time

import torch
from model import protected_surface as protected
from model import protected_surface_state as state
from model.native_population_training import cpu,fingerprint
from model.surface_device_runtime import evaluate,sync
from senses.contract import digest
from text.constellation_corpus import need

POLICY=dict(epochs=1,atoms=128,mode='gathered',rows=512,evaluation_rows=128,lr=1e-4,
            checkpoint_steps=256,weighting='mean_of_complete_atom_byte_means',
            order='ascending_TRAIN_occurrence',initialization='zero_correction_seed_171_over_frozen_full_parent',
            dtype='float32',tf32=False,label='source_reconstruction_not_world_truth')


def create(parent,config):
    model=protected.ProtectedSurfaceDecoder(parent,config,seed=171)
    return model,protected.optimizer_for(model)


def update(model,optimizer,examples,step):
    need(0<len(examples)<=POLICY['atoms'] and all(e.side=='train' for e in examples),'PROTECTED_CORPUS_TRAIN_ONLY')
    device=next(model.parameters()).device;sync(device);began=tick=time.perf_counter()
    protected.check(model,optimizer,step);before=fingerprint(model.correction.state_dict())
    native=fingerprint([{n:x.values for n,x in e.native.items()} for e in examples])
    phases=dict(precheck=time.perf_counter()-tick);tick=time.perf_counter()
    model.train();optimizer.zero_grad(set_to_none=True)
    scores,profile=evaluate(model,examples,POLICY['mode'],POLICY['rows'],backward=True)
    sync(device);phases['forward_backward']=time.perf_counter()-tick;tick=time.perf_counter()
    # Capture each gradient once. All branch checks and hashes use this same CPU snapshot.
    gradients=cpu({n:p.grad for n,p in model.correction.named_parameters()})
    need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()),'PROTECTED_CORPUS_GRADIENT')
    branches={}
    for spec in model.specs:
        for prefix in ['projectors.'+spec.name+'.',*[f'blocks.{i}.branches.{spec.name}.' for i in range(model.config.layers)]]:
            values={n:v for n,v in gradients.items() if n.startswith(prefix)}
            nonzero=any(bool(v.abs().any()) for v in values.values())
            need(values and (step==0 or nonzero),'PROTECTED_CORPUS_DISCONNECTED')
            branches[prefix]=dict(pin=fingerprint(values),nonzero=nonzero)
    if step==0:
        need(all(not bool(v.abs().any()) for n,v in gradients.items() if not n.startswith('output.'))
             and any(bool(v.abs().any()) for n,v in gradients.items() if n.startswith('output.')),'PROTECTED_CORPUS_ZERO_START')
    gradient=fingerprint(gradients);del gradients
    phases['gradient_capture_checks']=time.perf_counter()-tick;tick=time.perf_counter()
    optimizer.step();protected.check(model,optimizer,step+1);sync(device)
    phases['optimizer_check']=time.perf_counter()-tick;tick=time.perf_counter()
    need(native==fingerprint([{n:x.values for n,x in e.native.items()} for e in examples]),'PROTECTED_CORPUS_NATIVE_CHANGED')
    row=dict(step=step+1,occurrences=[e.occurrence for e in examples],
             input=digest([dict(occurrence=e.occurrence,uid=e.uid,raw=e.raw.hex()) for e in examples]),
             native=native,before=before,after=fingerprint(model.correction.state_dict()),
             gradient=gradient,branches=branches,optimizer=fingerprint(optimizer.state_dict()),
             scores=fingerprint(scores),positions=profile['positions'],objective=profile['objective'])
    phases['state_hashes']=time.perf_counter()-tick;wall=time.perf_counter()-began
    need(.9<=sum(phases.values())/wall<=1.000001,'PROTECTED_CORPUS_PHASE_COVERAGE')
    return row,dict(wall=wall,phases=phases,batches=profile),scores


def progress(binding,cursor,step,ledger,size):
    need(binding['policy']==POLICY and type(size) is int and size>0,'PROTECTED_CORPUS_POLICY')
    need(type(cursor) is int and type(step) is int and 0<=cursor<=size and step>=0
         and cursor==min(step*POLICY['atoms'],size) and step<=(size+POLICY['atoms']-1)//POLICY['atoms'],
         'PROTECTED_CORPUS_CURSOR')
    need(type(ledger) is str and re.fullmatch('[0-9a-f]{64}',ledger),'PROTECTED_CORPUS_LEDGER')
    return dict(program=binding,cursor=cursor,step=step,ledger=ledger,size=size)


def checkpoint(path,model,optimizer,binding,cursor,step,ledger,size):
    b=progress(binding,cursor,step,ledger,size)
    ref=state.save(path,model,optimizer,step=step,binding=b)
    # Physical readback of every shard before publishing a resumable reference.
    payload,m=state.read(path,ref['pin'],binding=b,parent=model.parent_pin)
    need(fingerprint(payload)==ref['state']==m['state'],'PROTECTED_CORPUS_CHECKPOINT_READBACK')
    return dict(ref,cursor=cursor,ledger=ledger,size=size)


def restore(ref,model,optimizer,binding,*,size):
    need(ref['size']==size,'PROTECTED_CORPUS_RESUME_SIZE')
    b=progress(binding,ref['cursor'],ref['step'],ref['ledger'],size)
    step=state.restore(ref,model,optimizer,binding=b)
    return ref['cursor'],step,ref['ledger']
