"""Independent directional derivatives and mutation controls on actual native folds."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import transient_answer_training as candidate
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.native_population_training import cpu, fingerprint
from model.transient_answer_attention import TransientFold, transient_attention
from model.guarded_transient_answer import guarded_transient_attention
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import file_sha, need


def changed_head(model,opt,item,request,state):
    negative={}
    with guarded_transient_attention(model,heads=True):
        stream=model.stream(request);cache=None
        for position,frame in enumerate(item['context']):
            hidden,cache=stream.step((frame,),context='mutation',offset=position,state=cache,detach_history=False)
        p=state['policy'];noise=torch.randn(len(item['answers']),p['samples'],p['noise_width'],
                                          generator=torch.Generator().manual_seed(p['noise_seed']+5))[:1]
        storage=SavedStorage(p['working_bytes'],'cpu')
        with storage.hooks():
            result=model.forecaster.losses(hidden,item['codes'][:1],{n:t[:1] for n,t in item['counts'].items()},
                                           item['answers'][:1],noise,target=state['target'])
            name,parameter=next((n,t) for n,t in model.named_parameters() if 'placement.assembled.residual.' in n)
            saved=parameter.detach().clone()
            with torch.no_grad():parameter.add_(1)
            try:negative['changed_head']=rejected(lambda:result['loss'].backward(),'TRANSIENT_HEAD_WEIGHTS_CHANGED')
            finally:
                with torch.no_grad():parameter.copy_(saved)
                opt.zero_grad(set_to_none=True)
        negative['changed_parameter']=name
    return negative,storage


def run(a):
    began=tick=time.perf_counter();a.output.mkdir(exist_ok=False);phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    state=candidate.previous.source.load(a.root);base=Path(__file__).resolve().parents[1]
    code=dict(state['code'],**{n:file_sha(base/n) for n in (*candidate.NEW,'model/guarded_transient_answer.py',
                                                         'audit/transient_answer_controls.py')})
    model,opt=candidate.previous.source.prior.make(state,'shared')
    item,request=candidate.previous.source.prior.item(state,27195)
    initial=fingerprint((model.state_dict(),opt.state_dict()));phases['authentication']=time.perf_counter()-tick
    tick=time.perf_counter();probes=[]
    with transient_attention(model):
        stream=model.stream(request);cache=None
        for position,frame in enumerate(item['context']):
            hidden,cache=stream.step((frame,),context='direction',offset=position,state=cache,detach_history=False)
        x,atoms,_=stream.pack(item['answers'][:1],2);block=stream.encoder.blocks[0]
        heads=stream.encoder.config.heads;depth=x.shape[-1]//heads;z=block.first(x)
        q,k,v=(m(z).reshape(1,len(atoms),heads,depth).transpose(1,2) for m in (block.q,block.k,block.v))
        history=tuple(t for b in cache.layers[0] for t in (b.key,b.value,b.atoms))+(k,v,atoms)
        inputs=tuple(t.detach().double().requires_grad_() if t.is_floating_point() else t.detach() for t in (q,atoms,*history))
        selected=[i for i,t in enumerate(inputs) if t.requires_grad]
        out=TransientFold.apply(stream.fold,*inputs);loss=out.square().mean()
        gradients=torch.autograd.grad(loss,tuple(inputs[i] for i in selected))
        for index,gradient in zip(selected,gradients):
            need(bool(gradient.abs().any()),'TRANSIENT_DIRECTION_DISCONNECTED')
            direction=gradient/gradient.norm();analytic=float((gradient*direction).sum())
            for epsilon in (1e-4,1e-5,1e-6):
                values=[]
                for sign in (-1,1):
                    args=list(inputs);args[index]=args[index]+sign*epsilon*direction
                    with torch.no_grad():values.append(float(stream.fold(*args).square().mean()))
                central=(values[1]-values[0])/(2*epsilon);tolerance=1e-7+1e-5*abs(analytic)
                need(abs(central-analytic)<=tolerance,'TRANSIENT_DIRECTIONAL_DERIVATIVE')
                probes.append(dict(input=index,epsilon=epsilon,analytic=analytic,central=central,
                                   absolute_error=abs(central-analytic),tolerance=tolerance,losses=values))
        persist(a.output/'direction.pt',dict(document=item['document'],native=item['native_pin'],inputs=inputs,
                                            gradients=gradients,probes=probes))
        del hidden,cache,x,z,q,k,v,inputs,out,loss,gradients,stream;gc.collect()
    phases['directional_derivatives']=time.perf_counter()-tick;tick=time.perf_counter();negative={}
    negative,storage=changed_head(model,opt,item,request,state)
    gc.collect()
    need(storage.current==0,'TRANSIENT_HEAD_MUTATION_STORAGE:'+json.dumps(storage.report()))
    negative['released_saved_storage']=storage.report()
    # The comparison guard itself must detect a corrupted persisted gradient.
    reference=a.root/candidate.REFERENCE/'shared-reference';report=json.loads((reference/'report.json').read_bytes())
    path=reference/'update-6.pt';need(file_sha(path)==report['files'][path.name],'TRANSIENT_CONTROL_REFERENCE_PIN')
    original=torch.load(path,map_location='cpu',weights_only=True);bad=deepcopy(original)
    next(iter(bad['gradients'].values())).view(-1)[0]+=1
    negative['changed_gradient']=rejected(lambda:candidate.previous.exact(original,bad),'RECOMPUTE_NUMERICAL_MISMATCH')
    need(initial==fingerprint((model.state_dict(),opt.state_dict())),'TRANSIENT_CONTROL_STATE_CHANGED')
    phases['mutation_controls']=time.perf_counter()-tick;tick=time.perf_counter();state['fresh']()
    need(all(file_sha(base/n)==h for n,h in code.items()) and not torch.cuda.is_initialized(),'TRANSIENT_CONTROL_SCOPE')
    verdict=dict(document=item['document'],native=item['native_pin'],probes=probes,negative=negative,
                 first_order_only=True,original_state_restored=True)
    persist(a.output/'controls.pt',verdict)
    if a.previous:
        need(file_sha(a.previous/'report.json')==a.previous_pin,'TRANSIENT_CONTROL_PREVIOUS_PIN')
        previous=json.loads((a.previous/'report.json').read_bytes())
        need(previous['code']==code,'TRANSIENT_CONTROL_PREVIOUS_CODE')
        for name in ('direction.pt','controls.pt'):
            need(file_sha(a.previous/name)==previous['files'][name],'TRANSIENT_CONTROL_COLD_PIN')
            left=torch.load(a.previous/name,map_location='cpu',weights_only=True)
            right=torch.load(a.output/name,map_location='cpu',weights_only=True)
            need(fingerprint(left)==fingerprint(right),'TRANSIENT_CONTROL_COLD_REPLAY')
    files={p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'TRANSIENT_CONTROL_PHASES')
    report=dict(complete=True,code=code,files=files,previous_pin=a.previous_pin,directions=len(probes),
                negative=negative,phases=phases,seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                acceptance='author-checked',gpu_used=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds,directions=len(probes))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=candidate.previous.source.ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
