"""Read real full-answer updates against the original arithmetic before memory scaling."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import recomputed_answer_training as previous
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_full_history as learner
from model.native_population_training import cpu, fingerprint
from model.transient_answer_attention import transient_attention, TransientFold
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import file_sha, need

REFERENCE = 'recomputed-answer-v2-20261024'
PINS = dict(shared='792d66ae4c4eb2ac41d44f173176778c17c76dff7dc122c6c15e0b6724ab9234',
            own_slot='63aa668719a8337a64e830192230970e6ec907e5e786c2806e6926c4ebdc56e0',
            whole_proposals='8f8a8c993f66daeb5b8fc1fda0ca4bb7c53e928ca792b9ab9f1fa4ecaac2cdac')
NEW = (*previous.NEW,'model/transient_answer_attention.py','audit/transient_answer_training.py')


def differences(a,b,path=''):
    rows=[]
    if isinstance(a,torch.Tensor):
        need(isinstance(b,torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, 'TRANSIENT_COMPARE_LAYOUT')
        count=int((a != b).sum())
        if count: rows.append(dict(path=path,changed=count,total=a.numel(),max_abs=float((a.double()-b.double()).abs().max())))
    elif isinstance(a,dict):
        need(a.keys() == b.keys(), 'TRANSIENT_COMPARE_KEYS')
        for k in a: rows.extend(differences(a[k],b[k],path+'/'+str(k)))
    elif isinstance(a,(tuple,list)):
        need(type(a) == type(b) and len(a) == len(b), 'TRANSIENT_COMPARE_SEQUENCE')
        for i,(x,y) in enumerate(zip(a,b)): rows.extend(differences(x,y,path+'/'+str(i)))
    elif a != b: rows.append(dict(path=path,reference=a,candidate=b))
    return rows


def controls(state,heads):
    model,opt=previous.source.prior.make(state,'shared');item,request=previous.source.prior.item(state,state['train'][0])
    origin=fingerprint((model.state_dict(),opt.state_dict()));out={}
    def update(value=item,policy=None):
        with transient_attention(model,heads=heads):
            return learner.update(model,opt,value,5,policy or state['policy'],request=request,target=state['target'],full=True)
    held=next(k for k,d in state['items'].items() if d['side'] != 'train')
    out['held_out_fit']=rejected(lambda:update(previous.source.prior.item(state,held)[0]),'CONDITIONED_NONTRAIN')
    for field,code in (('native','CONDITIONED_NATIVE_PIN'),('label','CONDITIONED_LABEL_PIN'),('binding','CONDITIONED_SOURCE_BINDING')):
        bad=deepcopy(item)
        if field=='native':next(iter(bad['answers'][0].values())).add_(1)
        elif field=='label':bad['codes'][0,0]=(bad['codes'][0,0]+1)%4
        else:bad['provenance']['observed']+=1
        out[field]=rejected(lambda:update(bad),code)
    out['physical_memory']=rejected(lambda:update(policy=dict(state['policy'],working_bytes=1)),'GRADIENT_STORAGE_MEMORY')
    need(origin == fingerprint((model.state_dict(),opt.state_dict())) and all(p.grad is None for p in model.parameters()),
         'TRANSIENT_REFUSAL_MUTATION')
    with transient_attention(model,heads=heads):
        stream=model.stream(request);first=item['context'][:1]
        hidden,cache=stream.step(first,context='control',offset=0,detach_history=False)
        out['context']=rejected(lambda:stream.step(first,context='wrong',offset=1,state=cache,detach_history=False),
                                'PREFIX_STREAM_WRONG_CONTEXT')
        out['offset']=rejected(lambda:stream.step(first,context='control',offset=2,state=cache,detach_history=False),
                               'PREFIX_STREAM_OFFSET')
        del hidden,cache,stream
    # Derive the fold input from real native projections, never a synthetic example.
    with transient_attention(model):
        stream=model.stream(request);x,atoms,_=stream.pack(item['context'][:1],0)
        block=stream.encoder.blocks[0];heads_count=stream.encoder.config.heads;depth=x.shape[-1]//heads_count
        z=block.first(x)
        q,k,v=(m(z).reshape(1,len(atoms),heads_count,depth).transpose(1,2).detach().requires_grad_() for m in (block.q,block.k,block.v))
        storage=SavedStorage(state['policy']['working_bytes'],'cpu')
        with storage.hooks():value=TransientFold.apply(stream.fold,q,atoms,k,v,atoms)
        with torch.no_grad():k.add_(1)
        out['saved_input_mutation']=rejected(lambda:value.sum().backward(),'GRADIENT_STORAGE_MUTATED')
        del value;gc.collect();need(storage.current == 0,'TRANSIENT_SAVED_LEAK')
        value=TransientFold.apply(stream.fold,q,atoms,k,v,atoms)
        out['higher_order']=rejected(lambda:torch.autograd.grad(value.sum(),q,create_graph=True),'TRANSIENT_HIGHER_ORDER_UNSUPPORTED')
        del value,q,k,v,x,z,stream;gc.collect()
    need('stream' not in vars(model) and 'losses' not in vars(model.forecaster),'TRANSIENT_POLICY_RESTORE')
    need(origin == fingerprint((model.state_dict(),opt.state_dict())),'TRANSIENT_CONTROLS_MUTATED')
    return out


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    state=previous.source.load(a.root);base=Path(__file__).resolve().parents[1]
    code=dict(state['code'],**{n:file_sha(base/n) for n in NEW})
    reference=a.root/REFERENCE/(a.architecture+'-reference')
    need(file_sha(reference/'report.json') == PINS[a.architecture],'TRANSIENT_REFERENCE_PIN')
    ref=json.loads((reference/'report.json').read_bytes())
    need(all(file_sha(base/n) == h for n,h in ref['code'].items()),'TRANSIENT_REFERENCE_CODE')
    prior=None
    if a.previous:
        need(file_sha(a.previous/'report.json') == a.previous_pin,'TRANSIENT_PREVIOUS_PIN')
        prior=json.loads((a.previous/'report.json').read_bytes())
        need(prior['code'] == code and prior['architecture'] == a.architecture and prior['heads'] == a.heads,
             'TRANSIENT_PREVIOUS_BINDING')
    model,opt=previous.source.prior.make(state,a.architecture);model.train()
    initial=fingerprint((model.state_dict(),opt.state_dict()));need(initial == ref['initial'],'TRANSIENT_INITIAL_STATE')
    persist(a.output/'initial.pt',dict(model=model.state_dict(),optimizer=opt.state_dict()))
    phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter();rows=[];qualified=True;negative={}
    if a.long_document is not None:
        need(a.qualification and file_sha(a.qualification/'report.json') == a.qualification_pin,'TRANSIENT_LONG_QUALIFICATION')
        qual=json.loads((a.qualification/'report.json').read_bytes())
        need(qual['qualified'] and qual['architecture'] == a.architecture and qual['heads'] == a.heads and qual['code'] == code,
             'TRANSIENT_LONG_UNQUALIFIED')
        item,request=previous.long_item(a.root,a.long_document)
        phases['consumer_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        with transient_attention(model,heads=a.heads):
            long=previous.memory_probe(state,model,opt,item,request,a.output,'reference')
        phases['complete_document_memory_probe']=time.perf_counter()-tick
    else:
        long=None
        for cursor,doc in enumerate(state['train']):
            item,request=previous.source.prior.item(state,doc)
            with transient_attention(model,heads=a.heads) as streams:
                row,gradient,early=learner.update(model,opt,item,5+cursor,state['policy'],request=request,target=state['target'],full=True)
                calls=[dict(forward=s.forward_folds,calls=s.fold_calls) for s in streams]
            value=dict(record=row,gradients=gradient,early=early,model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()))
            name=f'update-{6+cursor}.pt';persist(a.output/name,value)
            saved=torch.load(a.output/name,map_location='cpu',weights_only=True);previous.exact(value,saved)
            need(file_sha(reference/name) == ref['files'][name],'TRANSIENT_REFERENCE_UPDATE')
            original=torch.load(reference/name,map_location='cpu',weights_only=True)
            equal=fingerprint(previous.numerical(value)) == fingerprint(previous.numerical(original))
            if not equal:
                diff=differences(previous.numerical(original),previous.numerical(value))
                write_committed(a.output/'differences.json',diff);qualified=False
            if prior:
                need(file_sha(a.previous/name) == prior['files'][name],'TRANSIENT_COLD_UPDATE_PIN')
                previous.exact(value,torch.load(a.previous/name,map_location='cpu',weights_only=True))
            rows.append(dict(document=doc,decisions=row['decisions'],storage=row['storage'],phases=row['phases'],
                             numerical_pin=fingerprint(previous.numerical(value)),exact=equal,recomputation=calls))
            print(json.dumps(dict(architecture=a.architecture,heads=a.heads,**rows[-1])),flush=True)
            del saved,original,value,gradient,early
            if not equal:break
        phases['updates_and_readback']=time.perf_counter()-tick;tick=time.perf_counter()
        if a.architecture=='shared':negative=controls(state,a.heads)
        phases['negative_controls']=time.perf_counter()-tick
    tick=time.perf_counter();state['fresh']()
    need(all(file_sha(base/n) == h for n,h in code.items()),'TRANSIENT_CODE_CHANGED')
    need(not torch.cuda.is_initialized() and 'stream' not in vars(model) and 'losses' not in vars(model.forecaster),
         'TRANSIENT_RUNTIME_SCOPE')
    if prior:
        need(prior['qualified'] == qualified and prior['controls'] == negative,'TRANSIENT_COLD_VERDICT')
        if long is not None:
            for k in ('status','projected_atoms','storage','optimizer_updates','original_state_unchanged'):
                need(long[k] == prior['long'][k],'TRANSIENT_COLD_LONG_RESULT')
    files={p.name:file_sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001,'TRANSIENT_PHASE_ATTRIBUTION')
    report=dict(complete=True,qualified=qualified,architecture=a.architecture,heads=a.heads,code=code,files=files,
        reference_pin=PINS[a.architecture],previous_pin=a.previous_pin,qualification_pin=a.qualification_pin,
        initial=initial,rows=rows,controls=negative,long=long,phases=phases,seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch_version=torch.__version__,
        gpu_used=False,promoted=False,capability_demonstrated=False,acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),qualified=qualified,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=previous.source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--architecture',choices=tuple(PINS),required=True)
    p.add_argument('--heads',action='store_true');p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--long-document',type=int);p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
