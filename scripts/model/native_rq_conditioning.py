"""TRAIN-calibrated student inputs/losses; native instrument readings never change."""
import torch
from torch.nn import functional as F

from model.native_residual_codebook import NativeResidualCodebook
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import digest
from text.constellation_corpus import need

RECIPES = ('baseline', 'normalized', 'normalized_contrast')


class ConditionedNativeRQ(NativeResidualCodebook):
    def __init__(self, dimensions, config, sizes, recipe, stats):
        need(recipe in RECIPES, 'RQ_CONDITIONING_RECIPE')
        super().__init__(dimensions, config, sizes); self.recipe = recipe
        need(set(stats['centers']) == set(stats['variance']) == set(self.roster), 'RQ_CONDITIONING_STATISTICS')
        for i, n in enumerate(self.roster):
            need(stats['centers'][n].shape == (dimensions[n],) and bool(torch.isfinite(stats['centers'][n]).all())
                 and stats['variance'][n].numel() == 1 and bool(torch.isfinite(stats['variance'][n]) & (stats['variance'][n] > 0)),
                 'RQ_CONDITIONING_STATISTICS')
            self.register_buffer('center_'+str(i), stats['centers'][n].float().clone())
            self.register_buffer('variance_'+str(i), stats['variance'][n].float().clone())
        self.statistics_pin = self.calibration_pin()

    def calibration_pin(self):
        return digest(tensor_fingerprint({n:x for n,x in self.named_buffers() if n.startswith(('center_','variance_'))}))

    def check_statistics(self):
        need(self.calibration_pin() == self.statistics_pin, 'RQ_CONDITIONING_STATISTICS_CHANGED')

    def latent(self, frames):
        if self.recipe == 'baseline': return super().latent(frames)
        projectors = {n: (lambda x, n=n, i=i: self.projectors[n](
            (x-getattr(self,'center_'+str(i)))/getattr(self,'variance_'+str(i)).sqrt()))
            for i,n in enumerate(self.roster)}
        out = []
        for frame in frames:
            self.validate(frame)
            hidden = self.encoder.encode((frame,), projectors)[0]
            out.append(self.to_latent(hidden.reshape(1, -1))[0])
        return torch.stack(out)

    @torch.no_grad()
    def initialize_latents(self, latent):
        need(not bool(self.initialized) and len(latent) >= max(self.sizes), 'RQ_CONDITIONING_INITIALIZATION')
        residual = latent
        for book in self.books:
            selected = [0]; distance = (residual-residual[0]).square().sum(-1)
            for _ in range(1, len(book)):
                distance[selected] = -torch.inf; j = int(distance.argmax())
                need(bool(distance[j] > 0), 'RQ_CONDITIONING_INITIAL_COLLAPSE')
                selected.append(j); distance = torch.minimum(distance, (residual-residual[j]).square().sum(-1))
            book.copy_(residual[selected])
            codes = torch.cat([(residual[lo:lo+128,None]-book[None]).square().sum(-1).argmin(-1)
                               for lo in range(0, len(residual), 128)])
            residual = residual-book[codes]
        self.initialized.fill_(True)

    def training_loss(self, frames, donors, eligible):
        out = super().forward(frames)
        if self.recipe == 'baseline':
            return dict(**out, reconstruction=torch.stack(list(out['per_lane'].values())).mean(),
                        contrast=out['loss']*0, contrast_rows=0)
        losses = torch.stack([out['per_lane'][n]/getattr(self,'variance_'+str(i)) for i,n in enumerate(self.roster)])
        reconstruction = losses.mean(); contrast = reconstruction*0
        if self.recipe == 'normalized_contrast' and bool(eligible.any()):
            need(donors.shape == eligible.shape == (len(frames),) and donors.dtype == torch.int64
                 and eligible.dtype == torch.bool and bool(((donors >= 0)&(donors < len(frames))).all()),
                 'RQ_CONDITIONING_DONORS')
            _, quantized, _ = self.quantize(out['latent'])
            wrong = self.reconstruct(quantized[donors], [{n:len(f[n]) for n in self.roster} for f in frames])
            lane = []
            for i,n in enumerate(self.roster):
                delta = torch.stack([((right[n]-f[n]).square().mean()-(bad[n]-f[n]).square().mean())
                                     /getattr(self,'variance_'+str(i)) for right,bad,f in zip(out['decoded'],wrong,frames)])
                lane.append(F.softplus(delta[eligible]).mean())
            contrast = torch.stack(lane).mean()
        return dict(out, loss=reconstruction+out['quantizer_loss']+contrast,
                    reconstruction=reconstruction, contrast=contrast, contrast_rows=int(eligible.sum()))
