"""Complete native answers conditioned on public operation, roles and clock evidence."""
import torch

from model.native_answer_sequence import NativeAnswerSequence, versions
from model.native_conditioned_answer import check
from model.native_device_transport import transfer
from model.native_placement_checked import validate_optimizer
from model.native_population_training import cpu, fingerprint
from model.native_request_input import NativeRequest
from model.native_request_stream import RequestProjection, bind
from text.constellation_corpus import need


class RequestedAnswer(NativeAnswerSequence):
    def __init__(self,forecaster):
        super().__init__(forecaster)
        self.request_projection=RequestProjection(self.roster,forecaster.predictor.config['width'])

    def stream(self,request):
        return bind(self.forecaster.predictor.stream(),self.request_projection,request)

    def rollout(self,prefix,*,request,target,proposal,context,seed):
        need(isinstance(request,NativeRequest) and request.observed==len(prefix),'REQUESTED_ANSWER_CONTEXT')
        need(proposal in self.proposals and isinstance(prefix,(tuple,list)) and prefix
             and type(seed) is int and seed>=0 and type(context) is str and context,'ANSWER_REQUEST')
        need(all(set(f)==set(self.roster) for f in prefix),'ANSWER_PREFIX_ROSTER')
        placement=self.forecaster.placement[proposal];placement.check_target(target)
        pin=versions(self);original,select,readout=self._stream(proposal)
        stream=bind(original,self.request_projection,request);device=next(self.parameters()).device
        rng=torch.Generator(device=device).manual_seed(seed);state=hidden=None;offset=generated=0
        try:
            with torch.no_grad():
                for frame in prefix:
                    hidden,state=stream.step((select(frame),),context=context,offset=offset,state=state,detach_history=True)
                    offset+=1
            while True:
                with torch.no_grad():
                    need(versions(self)==pin,'ANSWER_WEIGHTS_CHANGED');placement.check_target(target)
                    head,value=readout(hidden);logits=self.stop_logits(value,proposal).detach().cpu()
                    ended=bool(logits.argmax(-1).item())
                    if ended:event=dict(kind='end',proposal=proposal,generated=generated,
                                        stop_logits=logits.clone(),validity='unassessed')
                    else:
                        forecast=head.forecast(value)
                        noise=torch.randn(1,1,placement.noise_width,generator=rng,device=device)
                        sample=placement(value,forecast['codes'],forecast['counts'],noise,target=target)[0]
                        frame={n:sample[n][0].detach() for n in self.roster}
                        hidden,state=stream.step((select(frame),),context=context,offset=offset,state=state,detach_history=True)
                        offset+=1
                        event=dict(kind='constellation',proposal=proposal,index=generated,
                            slots={n:x.clone() for n,x in frame.items()},codes=forecast['codes'].clone(),
                            counts={n:x.clone() for n,x in forecast['counts'].items()},stop_logits=logits.clone(),
                            feedback_index=offset-1,source='predicted',validity='unassessed')
                        generated+=1
                yield event
                if ended:return
        finally:state=hidden=stream=original=None


def update(model,optimizer,item,step,policy,*,request,target):
    schedule=check(item)
    need(isinstance(request,NativeRequest) and request.observed==len(item['context'])
         and item['provenance']['request']==request.pin,'REQUESTED_ANSWER_SOURCE_BINDING')
    validate_optimizer(model,optimizer,policy,step);before=fingerprint(model.state_dict());frozen=versions(target)
    optimizer.zero_grad(set_to_none=True);device=next(model.parameters()).device
    frames=transfer((*item['context'],*item['answers']),device);c,a=len(item['context']),len(item['answers'])
    codes=item['codes'].to(device);counts={n:x.to(device) for n,x in item['counts'].items()}
    noise=torch.randn(a,policy['samples'],policy['noise_width'],device=device,
        generator=torch.Generator(device=device).manual_seed(policy['noise_seed']+step))
    stream=model.stream(request);state=None;records=[]
    for position,frame in enumerate(frames):
        with torch.set_grad_enabled(position>=c-1):
            hidden,state=stream.step((frame,),context='requested-answer:'+str(item['document']),offset=position,
                                    state=state,detach_history=True)
        if position<c-1:continue
        index=position-(c-1);row=schedule[index];stop=model.stop_loss(hidden,row['end'])
        loss=stop['loss']/(a+1);continuation=None
        if not row['end']:
            continuation=model.forecaster.losses(hidden,codes[index:index+1],{n:x[index:index+1] for n,x in counts.items()},
                frames[position+1:position+2],noise[index:index+1],target=target)
            loss=loss+continuation['loss']/a
        need(bool(torch.isfinite(loss)),'REQUESTED_ANSWER_NONFINITE')
        records.append(dict(**row,stop=cpu(stop),continuation=None if continuation is None else
                            float(continuation['loss'].detach()),weighted_loss=float(loss.detach())))
        loss.backward()
    need(len(records)==a+1 and sum(r['end'] for r in records)==1,'REQUESTED_ANSWER_COVERAGE')
    gradients=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()),'REQUESTED_ANSWER_GRADIENTS')
    optimizer.step();validate_optimizer(model,optimizer,policy,step+1)
    need(all(bool(x.isfinite().all()) for x in model.state_dict().values()),'REQUESTED_ANSWER_PARAMETERS')
    need(versions(target)==frozen,'REQUESTED_ANSWER_FROZEN_CHANGED');check(item)
    return dict(document=item['document'],request=request.pin,context_positions=c,answer_targets=a,decisions=a+1,
        end_labels=1,prompt_transition_labels=0,before=before,after=fingerprint(model.state_dict()),
        gradients=fingerprint(gradients),optimizer=fingerprint(optimizer.state_dict()),native=item['native_pin'],
        labels=item['label_pin'],binding=item['binding'],records=records,backward_history='detached_prior_positions'),gradients
