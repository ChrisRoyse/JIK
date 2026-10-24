"""Region-conditioned native-window samples and a typed product-space energy loss.

The noise is explicit and shared across slots of a candidate. No target reading
enters generation. Region membership must still be checked after generation.
"""
from itertools import combinations
import math

import torch
from torch import nn

from model.attention import Linear
from text.constellation_corpus import need


def versions(target):
    return tuple((id(x),x._version) for x in target.state_dict(keep_vars=True).values())


class NativeStochasticPlacement(nn.Module):
    def __init__(self, target, *, target_pin, seed, noise_width=8, output_bytes=16<<20):
        super().__init__()
        need(not target.training and not any(p.requires_grad for p in target.parameters())
             and bool(target.initialized), 'PLACEMENT_TARGET_NOT_FROZEN')
        need(type(target_pin) is str and len(target_pin)==64 and type(noise_width) is int and noise_width>0
             and type(output_bytes) is int and output_bytes>0, 'PLACEMENT_CONTRACT')
        self.roster,self.dimensions=tuple(target.roster),dict(target.dimensions)
        self.width,self.sizes=target.config['width'],tuple(target.sizes)
        self.noise_width,self.output_bytes=noise_width,output_bytes
        self.target_pin,self.target_identity,self.target_versions=target_pin,id(target),versions(target)
        self.max_windows={n:len(getattr(target,'position_'+str(i))) for i,n in enumerate(self.roster)}
        torch.manual_seed(seed);h=self.width
        self.context=nn.ModuleDict({n:Linear(h,h) for n in self.roster})
        self.region=nn.ModuleDict({n:Linear(h,h) for n in self.roster})
        self.noise=nn.ModuleDict({n:Linear(noise_width,h) for n in self.roster})
        self.position=nn.ModuleDict({n:Linear(h,h) for n in self.roster})
        self.residual=nn.ModuleDict({n:nn.Sequential(nn.LayerNorm(h),Linear(h,h),nn.GELU(),Linear(h,d))
                                    for n,d in self.dimensions.items()})
        for i,n in enumerate(self.roster):
            variance=getattr(target,'variance_'+str(i)).detach().clone()
            need(variance.numel()==1 and bool(torch.isfinite(variance)) and float(variance)>0,'PLACEMENT_VARIANCE')
            self.register_buffer('variance_'+str(i),variance)

    def check_target(self,target):
        need(id(target)==self.target_identity and versions(target)==self.target_versions
             and tuple(target.roster)==self.roster and not target.training
             and not any(p.requires_grad for p in target.parameters()), 'PLACEMENT_TARGET_CHANGED')

    def validate(self,hidden,codes,counts,noise,target):
        self.check_target(target);device=next(self.parameters()).device
        need(target.encoder.query.device==device,'PLACEMENT_TARGET_DEVICE')
        need(isinstance(hidden,torch.Tensor) and hidden.dtype==torch.float32 and hidden.device==device
             and hidden.ndim==3 and hidden.shape[1:]==(len(self.roster),self.width)
             and 0<len(hidden)<=64 and bool(torch.isfinite(hidden).all()), 'PLACEMENT_CONTEXT')
        b=len(hidden)
        need(isinstance(codes,torch.Tensor) and codes.dtype==torch.int64 and codes.device==device
             and codes.shape==(b,len(self.sizes)), 'PLACEMENT_REGION')
        need(all(bool(((codes[:,i]>=0)&(codes[:,i]<s)).all()) for i,s in enumerate(self.sizes)), 'PLACEMENT_REGION_RANGE')
        need(isinstance(noise,torch.Tensor) and noise.dtype==torch.float32 and noise.device==device
             and noise.ndim==3 and noise.shape[0]==b and 1<=noise.shape[1]<=8
             and noise.shape[2]==self.noise_width and not noise.requires_grad
             and bool(torch.isfinite(noise).all()), 'PLACEMENT_NOISE')
        need(isinstance(counts,dict) and set(counts)==set(self.roster),'PLACEMENT_COUNT_ROSTER')
        for n,x in counts.items():
            need(isinstance(x,torch.Tensor) and x.shape==(b,) and x.dtype==torch.int64 and x.device==device,
                 'PLACEMENT_COUNT_RANGE')
        # One small metadata transfer, rather than B*L CUDA scalar synchronizations.
        native_counts=torch.stack([counts[n] for n in self.roster],dim=1).detach().cpu().tolist()
        need(all(1<=w<=self.max_windows[n] for row in native_counts for n,w in zip(self.roster,row)),
             'PLACEMENT_COUNT_RANGE')
        rows=[dict(zip(self.roster,row)) for row in native_counts]
        # Check before allocating native samples; never truncate a window to fit.
        required=noise.shape[1]*4*sum(row[n]*self.dimensions[n] for row in rows for n in self.roster)
        need(required<=self.output_bytes,'PLACEMENT_OUTPUT_MEMORY')
        return b,noise.shape[1],rows

    def forward(self,hidden,codes,counts,noise,*,target):
        b,s,rows=self.validate(hidden,codes,counts,noise,target)
        # Immutable reconstruction is computed once, not once per sample.
        with torch.no_grad():
            regions=target.values(codes).detach()
            base=target.reconstruct(regions,rows)
            positions={w:target.encoder.position(torch.arange(w,device=hidden.device)).detach()
                       for w in {w for row in rows for w in row.values()}}
        outputs=[{} for _ in range(b)]
        for lane,n in enumerate(self.roster):
            fixed=self.context[n](hidden[:,lane])+self.region[n](regions)
            random=self.noise[n](noise)
            positional={w:self.position[n](positions[w]) for w in {row[n] for row in rows}}
            scale=getattr(self,'variance_'+str(lane)).sqrt()
            for i,row in enumerate(rows):
                value=fixed[i][None,None,:]+random[i][:,None,:]+positional[row[n]][None,:,:]
                outputs[i][n]=base[i][n][None,:,:]+scale*self.residual[n](value)
        need(bool(torch.stack([torch.isfinite(x).all() for row in outputs for x in row.values()]).all()),
             'PLACEMENT_NONFINITE_OUTPUT')
        return tuple(outputs)

    def variances(self):
        return {n:getattr(self,'variance_'+str(i)) for i,n in enumerate(self.roster)}


def energy_loss(samples,targets,variances,*,empirical=False,working_bytes=512<<20):
    """Alpha=1 energy loss; per-slot terms plus a joint typed-product metric.

    empirical=False: unbiased distinct-pair estimate from independent draws.
    empirical=True: exact finite empirical-distribution score, including the
    zero diagonals in its denominator. This mode is for deterministic controls.
    Counts are fixed conditioning for this score, not a random ragged metric.
    """
    need(type(empirical) is bool and type(working_bytes) is int and working_bytes>0,'PLACEMENT_SCORE_POLICY')
    need(isinstance(samples,(tuple,list)) and 0<len(samples)<=64
         and isinstance(targets,(tuple,list)) and len(samples)==len(targets), 'PLACEMENT_SCORE_BATCH')
    roster=tuple(variances);scales={n:float(v) for n,v in variances.items()}
    need(roster and all(math.isfinite(v) and v>0 for v in scales.values()), 'PLACEMENT_SCORE_VARIANCE')
    first=samples[0][roster[0]] if set(samples[0])==set(roster) else None
    need(isinstance(first,torch.Tensor) and first.ndim==3 and 2<=len(first)<=8,'PLACEMENT_SCORE_SAMPLES')
    count=len(first);device=first.device;elements=0;finite=[]
    for row,target in zip(samples,targets):
        need(set(row)==set(target)==set(roster),'PLACEMENT_SCORE_ROSTER')
        for n in roster:
            x,y=row[n],target[n]
            need(isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and x.ndim==3
                 and x.shape==(count,*y.shape) and y.ndim==2 and y.numel()>0
                 and x.dtype in (torch.float32,torch.float64) and y.dtype in (torch.float32,torch.float64)
                 and x.device==y.device==device and not y.requires_grad,'PLACEMENT_SCORE_LAYOUT')
            elements+=y.numel()
            finite.extend((torch.isfinite(x).all(),torch.isfinite(y).all()))
    need(bool(torch.stack(finite).all()),'PLACEMENT_SCORE_LAYOUT')
    # Conservative live difference/gradient tensor allowance, before pair work.
    need(4*8*elements*(count+count*(count-1)//2)<=working_bytes,'PLACEMENT_SCORE_MEMORY')
    pair_indices=tuple(combinations(range(count),2));rows=[]
    for samples_row,target in zip(samples,targets):
        target_distances,pair_distances={},{}
        for n in roster:
            x=samples_row[n].double();y=target[n].double()
            scale=math.sqrt(y.numel()*scales[n])
            target_distances[n]=torch.linalg.vector_norm((x-y[None]).flatten(1),dim=1)/scale
            pair_distances[n]=torch.stack([torch.linalg.vector_norm(x[a]-x[b])/scale for a,b in pair_indices])
        # For exact empirical distributions the S diagonal pairs have distance 0.
        factor=(count-1)/count if empirical else 1.
        lane={n:target_distances[n].mean()-.5*factor*pair_distances[n].mean() for n in roster}
        td=torch.linalg.vector_norm(torch.stack(list(target_distances.values())),dim=0)/math.sqrt(len(roster))
        pd=torch.linalg.vector_norm(torch.stack(list(pair_distances.values())),dim=0)/math.sqrt(len(roster))
        joint=td.mean()-.5*factor*pd.mean()
        rows.append(dict(lanes=lane,joint=joint,loss=torch.stack(list(lane.values())).mean()+joint))
    return dict(loss=torch.stack([r['loss'] for r in rows]).mean(),
                joint=torch.stack([r['joint'] for r in rows]),
                lanes={n:torch.stack([r['lanes'][n] for r in rows]) for n in roster})
