"""Declared, separately reproducible training interventions for native RQ.

Raw instrument values remain the measurement authority. All changes below are
inside the derived student index, never new instruments or admission evidence.
"""
import math

import torch
from torch.nn import functional as F

from model.native_residual_codebook import NativeResidualCodebook
from text.constellation_corpus import need

RECIPES = ('baseline', 'lloyd_initialization', 'input_dimension_scale',
           'squared_norm_losses', 'continuous_warmup', 'mean_residual')


class NativeRQTrainingModel(NativeResidualCodebook):
    def __init__(self, dimensions, config, sizes, recipe):
        need(recipe in RECIPES, 'RQ_TRAINING_RECIPE')
        super().__init__(dimensions, config, sizes)
        self.recipe = recipe
        for i, dims in enumerate(dimensions.values()): self.register_buffer('center_'+str(i), torch.zeros(dims))

    def latent(self, frames):
        if self.recipe != 'input_dimension_scale': return super().latent(frames)
        need(isinstance(frames, (tuple,list)) and frames, 'NATIVE_RQ_EMPTY')
        projectors = {n: (lambda x, n=n: self.projectors[n](x*math.sqrt(self.dimensions[n]))) for n in self.roster}
        output=[]
        for frame in frames:
            self.validate(frame)
            hidden=self.encoder.encode((frame,), projectors)[0]
            output.append(self.to_latent(hidden.reshape(1,-1))[0])
        return torch.stack(output)

    @torch.no_grad()
    def prepare(self, frames):
        if self.recipe=='mean_residual':
            for i,n in enumerate(self.roster):
                getattr(self,'center_'+str(i)).copy_(torch.stack([f[n].mean(0) for f in frames]).mean(0))
                self.decoders[n][-1].weight.zero_(); self.decoders[n][-1].bias.zero_()
        self.initialize(frames)

    @torch.no_grad()
    def initialize(self, frames):
        if self.recipe!='lloyd_initialization': return super().initialize(frames)
        # Same deterministic seeds; change only Lloyd refinement before the next residual.
        need(not bool(self.initialized), 'NATIVE_RQ_ALREADY_INITIALIZED')
        residual=self.latent(frames); need(len(residual)>=max(self.sizes), 'NATIVE_RQ_INITIALIZATION_SUPPORT')
        self.initialization=[]
        for book in self.books:
            chosen=[0]; distance=(residual-residual[0]).square().sum(-1)
            for _ in range(1,len(book)):
                distance[chosen]=-torch.inf; idx=int(distance.argmax())
                need(bool(distance[idx]>0),'NATIVE_RQ_INITIALIZATION_COLLAPSED')
                chosen.append(idx); distance=torch.minimum(distance,(residual-residual[idx]).square().sum(-1))
            centroids=residual[chosen].clone(); history=[]
            for _ in range(16):
                squared=(residual[:,None]-centroids[None]).square().sum(-1); ids=squared.argmin(-1)
                counts=torch.bincount(ids,minlength=len(book)); history.append(float(squared.gather(1,ids[:,None]).mean()))
                for i in range(len(book)):
                    if int(counts[i]): centroids[i]=residual[ids==i].mean(0)
                # Empty cells retain their last declared centroid; never invent an observation.
            book.copy_(centroids)
            ids=(residual[:,None]-book[None]).square().sum(-1).argmin(-1)
            self.initialization.append(dict(objective=history,counts=torch.bincount(ids,minlength=len(book)).tolist()))
            residual=residual-book[ids]
        self.initialized.fill_(True)

    def reconstruct(self, latent, counts):
        values=super().reconstruct(latent,counts)
        if self.recipe!='mean_residual': return values
        return tuple({n:row[n]+getattr(self,'center_'+str(i)) for i,n in enumerate(self.roster)} for row in values)

    def training_loss(self, frames, step):
        need(type(step) is int and step>=0, 'RQ_TRAINING_STEP')
        latent=self.latent(frames)
        continuous=self.recipe=='continuous_warmup' and step<256
        if continuous:
            values=latent; penalty=latent.sum()*0; codes=None
        else: codes,values,penalty=self.quantize(latent)
        decoded=self.reconstruct(values,[{n:len(f[n]) for n in self.roster} for f in frames])
        lanes={n:torch.stack([F.mse_loss(d[n],f[n]) for d,f in zip(decoded,frames)]).mean() for n in self.roster}
        if self.recipe=='squared_norm_losses':
            reconstruction=torch.stack([v*self.dimensions[n] for n,v in lanes.items()]).mean()
            penalty=penalty*self.config['width']
        else: reconstruction=torch.stack(list(lanes.values())).mean()
        return dict(loss=reconstruction+penalty,reconstruction=reconstruction,quantizer=penalty,
                    per_lane=lanes,decoded=decoded,codes=codes,latent=latent,continuous=continuous)

    @torch.no_grad()
    def finish_warmup(self, frames, optimizer):
        need(self.recipe=='continuous_warmup','RQ_TRAINING_NOT_WARMUP')
        need(all(p not in optimizer.state for p in self.books),'RQ_TRAINING_WARMUP_BOOK_MOMENTS')
        self.initialized.fill_(False); self.initialize(frames)
