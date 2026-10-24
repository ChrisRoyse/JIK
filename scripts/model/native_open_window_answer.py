"""Full-answer predictor with position-wise native count endings and TRAIN-only tail baseline."""
import torch
from torch import nn
from torch.nn import functional as F

from model.native_branch_predictor import IndependentNativeEncoder,IndependentNativeStream,PerInstrumentConstellationPredictor
from model.native_placement_training import NativePlacementForecaster
from model.native_region_predictor import NativeRegionPredictor
from model.native_requested_answer import RequestedAnswer
from model.native_rq_packed import PackedNativeRQ
from model.native_stochastic_placement import NativeStochasticPlacement
from model.native_window_end import WindowEnd,physical_limits
from text.constellation_corpus import need


class OpenPositionRQ(PackedNativeRQ):
    """Weights/codes unchanged. Unseen positions use the immutable TRAIN mean prior."""
    def baseline(self,lane,windows):
        need(type(windows) is int and windows>0,'RQ_RESIDUAL_WINDOW_COUNT')
        name=self.roster[lane]
        need(windows<=physical_limits(self,self.config['max_native_elements']*4)[name],
             'OPEN_POSITION_PHYSICAL_MEMORY')
        original=getattr(self,'position_'+str(lane))
        if windows<=len(original):return original[:windows]
        tail=getattr(self,'center_'+str(lane)).expand(windows-len(original),-1)
        return torch.cat((original,tail),0)


class OpenReadout:
    def install_counts(self,target,seed,output_bytes):
        self.max_windows=physical_limits(target,output_bytes)
        # Keep all other modules' initialization unchanged across the matched arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.counts=nn.ModuleDict({n:WindowEnd(self.config['width'],target.encoder.frequency,self.max_windows[n])
                                      for n in self.roster})

    def forecast(self,hidden,*,generator=None):
        self.check_hidden(hidden);logits,codes=self.region(hidden,generator=generator)
        results={n:self.counts[n].forecast(hidden[:,i],generator=generator) for i,n in enumerate(self.roster)}
        return dict(logits=logits,codes=codes,counts={n:r['counts'] for n,r in results.items()},
                    count_logits={n:r['logits'] for n,r in results.items()},count_law='window_end_hazard')

    def loss(self,hidden,codes,counts):
        self.check_hidden(hidden)
        need(codes.device==hidden.device and set(counts)==set(self.roster),'REGION_PREDICTOR_TARGET')
        logits,_=self.region(hidden,codes)
        region=tuple(F.cross_entropy(score.double(),codes[:,i]) for i,score in enumerate(logits))
        window={n:self.counts[n].loss(hidden[:,i],counts[n])['loss'] for i,n in enumerate(self.roster)}
        return dict(loss=torch.stack(region).mean()+torch.stack(list(window.values())).mean(),
                    region=region,counts=window,logits=logits,count_law='window_end_hazard')


class OpenRegionPredictor(OpenReadout,NativeRegionPredictor):
    def __init__(self,target,*,count_seed,output_bytes,**kwargs):
        super().__init__(target,**kwargs);self.install_counts(target,count_seed,output_bytes)


class OpenIndependentPredictor(OpenRegionPredictor):
    def __init__(self,target,**kwargs):
        super().__init__(target,**kwargs);self.encoder=IndependentNativeEncoder(self.encoder)

    def stream(self):return IndependentNativeStream(self.encoder,self.projectors,cache_bytes=self.cache_bytes)


class OpenProposalReadout(OpenReadout,nn.Module):
    def __init__(self,original,target,seed,output_bytes):
        super().__init__()
        for n in ('target_pin','roster','_target','_target_versions','config','sizes'):
            setattr(self,n,getattr(original,n))
        self.region=original.region;self.install_counts(target,seed,output_bytes)

    check_hidden=NativeRegionPredictor.check_hidden
    render=NativeRegionPredictor.render


class OpenProposalPredictor(PerInstrumentConstellationPredictor):
    def __init__(self,target,*,count_seed,output_bytes,**kwargs):
        super().__init__(target,**kwargs)
        self.readouts=nn.ModuleDict({n:OpenProposalReadout(head,target,count_seed+i,output_bytes)
                                    for i,(n,head) in enumerate(self.readouts.items())})


class OpenPlacement(NativeStochasticPlacement):
    def __init__(self,target,**kwargs):
        need(isinstance(target,OpenPositionRQ),'OPEN_PLACEMENT_TARGET_RUNTIME')
        super().__init__(target,**kwargs);self.max_windows=physical_limits(target,self.output_bytes)


class OpenWindowForecaster(nn.Module):
    def __init__(self,target,*,target_pin,architecture,seed,placement_seed,count_seed,
                 cache_bytes=16<<20,output_bytes=16<<20):
        super().__init__();need(isinstance(target,OpenPositionRQ),'OPEN_FORECASTER_TARGET_RUNTIME')
        types=dict(shared=OpenRegionPredictor,own_slot=OpenIndependentPredictor,whole_proposals=OpenProposalPredictor)
        need(architecture in types,'PLACEMENT_TRAIN_ARCHITECTURE');self.architecture=architecture
        self.predictor=types[architecture](target,target_pin=target_pin,seed=seed,cache_bytes=cache_bytes,
                                           count_seed=count_seed,output_bytes=output_bytes)
        self.roster,self.sizes=tuple(target.roster),tuple(target.sizes)
        self.proposals=self.roster if architecture=='whole_proposals' else ('assembled',)
        self.placement=nn.ModuleDict({n:OpenPlacement(target,target_pin=target_pin,seed=placement_seed,
                                                     output_bytes=output_bytes) for n in self.proposals})

    contexts=NativePlacementForecaster.contexts
    losses=NativePlacementForecaster.losses


def answer_model(target,bundle,architecture,*,target_pin,count_seed=20261024):
    p=bundle['policy']
    return RequestedAnswer(OpenWindowForecaster(target,target_pin=target_pin,architecture=architecture,
        seed=p['seed'],placement_seed=p['placement_seed'],count_seed=count_seed,
        cache_bytes=p['cache_bytes'],output_bytes=p['output_bytes']))
