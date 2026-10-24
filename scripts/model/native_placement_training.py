"""Connected region/count/native-sample learning with source-bound document updates."""
import torch
from torch import nn

from model.loop_training_checkpoint import optimizer_for,validate_optimizer
from model.native_branch_predictor import IndependentNativeRegionPredictor,PerInstrumentConstellationPredictor
from model.native_population_training import cpu,fingerprint
from model.native_region_predictor import NativeRegionPredictor
from model.native_stochastic_placement import NativeStochasticPlacement,energy_loss
from text.constellation_corpus import need

ARCHITECTURES=dict(shared=NativeRegionPredictor,own_slot=IndependentNativeRegionPredictor,
                   whole_proposals=PerInstrumentConstellationPredictor)


class NativePlacementForecaster(nn.Module):
    def __init__(self,target,*,target_pin,architecture,seed,placement_seed,cache_bytes=16<<20,output_bytes=16<<20):
        super().__init__();need(architecture in ARCHITECTURES,'PLACEMENT_TRAIN_ARCHITECTURE')
        self.architecture=architecture
        self.predictor=ARCHITECTURES[architecture](target,target_pin=target_pin,seed=seed,cache_bytes=cache_bytes)
        self.roster,self.sizes=tuple(target.roster),tuple(target.sizes)
        self.proposals=self.roster if architecture=='whole_proposals' else ('assembled',)
        self.placement=nn.ModuleDict({n:NativeStochasticPlacement(target,target_pin=target_pin,seed=placement_seed,
                                      output_bytes=output_bytes) for n in self.proposals})

    def contexts(self,hidden):
        if self.architecture=='whole_proposals':
            return {n:(self.predictor.readouts[n],self.predictor.proposal_context(hidden,n)) for n in self.roster}
        return {'assembled':(self.predictor,hidden)}

    def losses(self,hidden,codes,counts,targets,noise,*,target):
        rows={}
        for name,(head,context) in self.contexts(hidden).items():
            discrete=head.loss(context,codes,counts)
            samples=self.placement[name](context,codes,counts,noise,target=target)
            native=energy_loss(samples,targets,self.placement[name].variances())
            rows[name]=dict(loss=discrete['loss']+native['loss'],discrete=discrete,native=native,samples=samples)
        return dict(loss=torch.stack([r['loss'] for r in rows.values()]).mean(),proposals=rows)


def check_item(item):
    need(item['side']=='train','PLACEMENT_TRAIN_NONTRAIN')
    need(len(item['frames'])>=2 and len(item['codes'])==len(item['frames'])-1,'PLACEMENT_TRAIN_TRANSITIONS')
    need(fingerprint(item['frames'])==item['native_pin'],'PLACEMENT_TRAIN_NATIVE_PIN')
    need(fingerprint((item['codes'],item['counts']))==item['label_pin'],'PLACEMENT_TRAIN_LABEL_PIN')


def branch_pins(values,model):
    prefixes=['predictor.encoder.']+['predictor.projectors.'+n+'.' for n in model.roster]
    prefixes += ['placement.'+p+'.'+kind+'.'+n+'.' for p in model.proposals
                 for kind in ('context','region','noise','position','residual') for n in model.roster]
    return {p:fingerprint({n:v for n,v in values.items() if n.startswith(p)}) for p in prefixes}


def microtiles(frames,lo,hi,model,samples,working_bytes):
    """Memory scope includes every proposal and its live backward differences."""
    need(type(working_bytes) is int and working_bytes>0 and 2<=samples<=8,'PLACEMENT_TRAIN_MEMORY_POLICY')
    count=len(model.proposals);head=next(iter(model.placement.values()));start=lo;elements=0
    for i in range(lo,hi):
        size=sum(x.numel() for x in frames[i+1].values())
        candidate=elements+size
        estimated=count*4*8*candidate*(samples+samples*(samples-1)//2)
        output=count*4*candidate*samples
        if estimated>working_bytes or output>head.output_bytes:
            need(i>start,'PLACEMENT_TRAIN_SINGLE_TARGET_MEMORY')
            yield start,i
            start=i;elements=size
            need(count*4*8*size*(samples+samples*(samples-1)//2)<=working_bytes
                 and count*4*size*samples<=head.output_bytes,'PLACEMENT_TRAIN_SINGLE_TARGET_MEMORY')
        else:elements=candidate
    if start<hi:yield start,hi


def _complete(state,atoms):
    if hasattr(state,'states'):return all(s.next_atom==atoms for s in state.states)
    return state.next_atom==atoms


def update(model,optimizer,item,step,policy,*,target):
    check_item(item);validate_optimizer(model,optimizer,policy,step)
    device=next(model.parameters()).device;total=len(item['codes']);samples=policy['samples']
    # Validate all physical microtiles before any backward action.
    plan=[(lo,min(lo+policy['tile'],len(item['frames'])),
           list(microtiles(item['frames'],lo,min(lo+policy['tile'],total),model,samples,policy['working_bytes'])))
          for lo in range(0,len(item['frames']),policy['tile'])]
    before=fingerprint(model.state_dict());optimizer.zero_grad(set_to_none=True)
    frames=tuple({n:x.to(device) for n,x in f.items()} for f in item['frames'])
    codes=item['codes'].to(device);counts={n:x.to(device) for n,x in item['counts'].items()}
    # Fresh per-update draws with a persisted cursor; no dependence on labels.
    noise_seed=policy['noise_seed']+step
    noise=torch.randn(total,samples,policy['noise_width'],generator=torch.Generator(device=device).manual_seed(noise_seed),device=device)
    stream=model.predictor.stream();state=None;context='placement-document:'+str(item['document']);losses=[];used=0
    for lo,hi,tiles in plan:
        hidden,state=stream.step(frames[lo:hi],context=context,offset=lo,state=state,detach_history=True)
        for index,(start,stop) in enumerate(tiles):
            out=model.losses(hidden[start-lo:stop-lo],codes[start:stop],{n:x[start:stop] for n,x in counts.items()},
                             frames[start+1:stop+1],noise[start:stop],target=target)
            weighted=out['loss']*((stop-start)/total)
            need(bool(torch.isfinite(weighted)),'PLACEMENT_TRAIN_NONFINITE_LOSS')
            weighted.backward(retain_graph=index+1<len(tiles));used+=stop-start
            losses.append(dict(start=start,stop=stop,loss=float(weighted.detach()),
                discrete={n:float(r['discrete']['loss'].detach()) for n,r in out['proposals'].items()},
                native={n:float(r['native']['loss'].detach()) for n,r in out['proposals'].items()}))
    need(used==total and _complete(state,len(frames)),'PLACEMENT_TRAIN_COVERAGE')
    gradients=cpu({n:p.grad for n,p in model.named_parameters()})
    need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()),'PLACEMENT_TRAIN_GRADIENT')
    for lane in model.roster:
        need(any(bool(g.abs().max()>0) for n,g in gradients.items() if n.startswith('predictor.projectors.'+lane+'.')),
             'PLACEMENT_TRAIN_DISCONNECTED_HISTORY')
        for proposal in model.proposals:
            need(any(bool(g.abs().max()>0) for n,g in gradients.items() if n.startswith('placement.'+proposal+'.residual.'+lane+'.')),
                 'PLACEMENT_TRAIN_DISCONNECTED_PLACEMENT')
    optimizer.step();validate_optimizer(model,optimizer,policy,step+1)
    need(all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()),'PLACEMENT_TRAIN_PARAMETER')
    check_item(item)
    try:stream.step(frames[-1:],context=context,offset=len(frames),state=state,detach_history=True)
    except (ValueError,RuntimeError) as e:stale=str(e)
    else:raise AssertionError('PLACEMENT_TRAIN_STALE_HISTORY_ACCEPTED')
    need('WEIGHTS_CHANGED' in stale,'PLACEMENT_TRAIN_STALE_HISTORY_REASON')
    return dict(step=step+1,document=item['document'],targets=total,noise_seed=noise_seed,noise=fingerprint(noise),
                loss=sum(x['loss'] for x in losses),tiles=losses,before=before,after=fingerprint(model.state_dict()),
                gradient=fingerprint(gradients),optimizer=fingerprint(optimizer.state_dict()),
                branch_gradient=branch_pins(gradients,model),stale_cache=stale,
                native=item['native_pin'],labels=item['label_pin'],backward_history='detached_prior_tiles'),gradients,noise
