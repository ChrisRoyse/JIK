"""Full-answer robustness learning with explicit, unverified generated inputs."""
import time

import torch
from model.breadth_answer_transfer import validate
from model.native_answer_sequence import versions
from model.native_conditioned_answer import check
from model.native_device_transport import transfer
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest
from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import need


def recipe(model,probability):
    need(model.forecaster.architecture=='shared','FEEDBACK_ARCHITECTURE')
    need(type(probability) is float and probability in (0.,.5,1.),'FEEDBACK_PROBABILITY')


def forward(model,item,cursor,policy,request,target,probability):
    recipe(model,probability);schedule=check(item)
    need(isinstance(request,NativeRequest) and request.observed==len(item['context'])
         and request.pin==item['provenance']['request'],'FEEDBACK_REQUEST')
    device=next(model.parameters()).device;c,a=len(item['context']),len(item['answers'])
    frames=transfer((*item['context'],*item['answers']),device)
    codes=item['codes'].to(device);counts={n:v.to(device) for n,v in item['counts'].items()}
    noise=torch.randn(a,policy['samples'],policy['noise_width'],device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+cursor))
    mask=(torch.rand(a,generator=torch.Generator().manual_seed(policy['seed']+2000000+cursor))<probability).tolist()
    feedback_noise=torch.randn(a,1,policy['noise_width'],device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+1000000+cursor))
    stream=model.stream(request);state=None;losses=[];records=[];next_frame=None
    for position in range(c+a):
        frame=frames[position] if position<c else next_frame
        hidden,state=stream.step((frame,),context='feedback-answer:'+str(item['document']),offset=position,
                                 state=state,detach_history=False)
        if position<c-1:continue
        index=position-c+1;row=schedule[index];stop=model.stop_loss(hidden,row['end']);loss=stop['loss']/(a+1)
        continuation=None;feedback=None
        if not row['end']:
            continuation=model.forecaster.losses(hidden,codes[index:index+1],{n:v[index:index+1] for n,v in counts.items()},
                frames[position+1:position+2],noise[index:index+1],target=target)
            loss=loss+continuation['loss']/a
            if mask[index]:
                # Candidate generation receives hidden state and its own random draw only.
                # The real next code/count/native readings enter the loss, never this call.
                with torch.no_grad():
                    forecast=model.forecaster.predictor.forecast(hidden.detach())
                    samples=model.forecaster.placement['assembled'](hidden.detach(),forecast['codes'],forecast['counts'],
                        feedback_noise[index:index+1],target=target)[0]
                    next_frame={n:x[0].detach() for n,x in samples.items()}
                feedback=dict(source='generated_unverified',slots=fingerprint(next_frame),codes=cpu(forecast['codes']),
                    counts=cpu(forecast['counts']),noise=fingerprint(feedback_noise[index:index+1]),
                    detached=all(not x.requires_grad for x in next_frame.values()))
                need(feedback['detached'],'FEEDBACK_SAMPLE_GRAPH')
            else:
                next_frame=frames[position+1]
                feedback=dict(source='reference',slots=fingerprint(next_frame))
            feedback['training_continuation_override']=bool(stop['proposals']['assembled']['logits'].argmax(-1).item())
        need(bool(torch.isfinite(loss)),'FEEDBACK_NONFINITE')
        losses.append(loss)
        records.append(dict(**row,stop=cpu(stop),continuation=None if continuation is None else float(continuation['loss'].detach()),
            weighted_loss=float(loss.detach()),input_pin=fingerprint(frame),feedback=feedback))
    need(len(records)==a+1 and sum(r['end'] for r in records)==1,'FEEDBACK_COVERAGE')
    return torch.stack(losses).sum(),losses[-1],records


def update(model,opt,item,cursor,policy,binding,*,request,target,probability):
    recipe(model,probability);check(item);validate(model,opt,policy,binding,cursor)
    before=fingerprint(model.state_dict());frozen=versions(target);opt.zero_grad(set_to_none=True)
    first={};handles=[];phases={}
    def capture(name):
        def hook(module,args,output):
            if name not in first:first[name]=output
        return hook
    for name,module in model.forecaster.predictor.projectors.items():handles.append(module.register_forward_hook(capture(name)))
    storage=SavedStorage(policy['working_bytes'],next(model.parameters()).device)
    try:
        with storage.hooks():
            tick=time.perf_counter();total,end,records=forward(model,item,cursor,policy,request,target,probability)
            phases['forward']=time.perf_counter()-tick;tick=time.perf_counter()
            need(set(first)==set(model.roster) and all(v.requires_grad for v in first.values()),'FEEDBACK_CONTEXT_GRAPH')
            ends=torch.autograd.grad(end,tuple(first.values()),retain_graph=True)
            gradients=torch.autograd.grad(total,tuple(first.values()),retain_graph=True)
            early={n:dict(value=cpu(v),gradient=cpu(g),end_gradient=cpu(e)) for (n,v),g,e in zip(first.items(),gradients,ends)}
            need(all(bool(v['gradient'].abs().any()) and bool(v['end_gradient'].abs().any()) for v in early.values()),
                 'FEEDBACK_CONTEXT_CREDIT')
            phases['credit_probe']=time.perf_counter()-tick;tick=time.perf_counter();total.backward()
            gradients=cpu({n:p.grad for n,p in model.named_parameters()});phases['backward']=time.perf_counter()-tick
        need(storage.current==0,'FEEDBACK_RETAINED_GRAPH')
        need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()),'FEEDBACK_GRADIENTS')
        prefixes=['forecaster.predictor.projectors.'+n+'.' for n in model.roster]
        prefixes+=['request_projection.lanes.'+n+'.' for n in model.roster]
        prefixes+=['ending.assembled.'+n+'.' for n in model.roster]
        branches={}
        for prefix in prefixes:
            group={n:v for n,v in gradients.items() if n.startswith(prefix)}
            need(group and any(bool(v.abs().any()) for v in group.values()),'FEEDBACK_BRANCH:'+prefix)
            branches[prefix]=fingerprint(group)
        check(item);need(versions(target)==frozen,'FEEDBACK_FROZEN_CHANGED')
        tick=time.perf_counter();opt.step();validate(model,opt,policy,binding,cursor+1);phases['optimizer']=time.perf_counter()-tick
        need(all(bool(v.isfinite().all()) for v in model.state_dict().values()),'FEEDBACK_PARAMETERS')
    except BaseException:
        opt.zero_grad(set_to_none=True);raise
    finally:
        for handle in handles:handle.remove()
        first.clear()
    return dict(document=item['document'],request=request.pin,native=item['native_pin'],labels=item['label_pin'],
        binding=item['binding'],before=before,after=fingerprint(model.state_dict()),optimizer=fingerprint(opt.state_dict()),
        gradients=fingerprint(gradients),branch_gradients=branches,early=fingerprint(early),records=records,
        decisions=len(records),cursor=cursor+1,probability=probability,storage=storage.report(),phases=phases),early
