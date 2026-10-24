"""Declared mechanisms for the observed RQ collapse; frozen native slots are unchanged."""
import torch
from torch.nn import functional as F

from model.native_rq_residual import ResidualConditionedNativeRQ
from model.text_sigreg import statistic
from text.constellation_corpus import need

ARMS = ('fixed_encoder', 'decoder_warmup', 'no_encoder_quantizer', 'latent_sigreg', 'continuous_then_rq')
SCHEDULE = dict(updates=3306, decoder_warmup=256, continuous_warmup=1653,
                sigreg_strength=.02, projection_seed=20261021, slices=256, knots=17)
QUALIFY = dict(fixed_encoder=32, decoder_warmup=258, no_encoder_quantizer=32,
               latent_sigreg=32, continuous_then_rq=1655)


class InterventionNativeRQ(ResidualConditionedNativeRQ):
    def __init__(self, dimensions, config, sizes, arm, statistics):
        need(arm in ARMS, 'RQ_INTERVENTION_ARM')
        super().__init__(dimensions, config, sizes, 'position_residual', statistics)
        self.arm = arm
        self.set_step(0)

    def set_step(self, number):
        need(type(number) is int and 0 <= number <= SCHEDULE['updates'], 'RQ_INTERVENTION_CURSOR')
        self.step_number = number

    def stage(self):
        if self.arm == 'fixed_encoder' or (self.arm == 'decoder_warmup' and self.step_number < SCHEDULE['decoder_warmup']):
            return 'decoder_only'
        if self.arm == 'continuous_then_rq' and self.step_number < SCHEDULE['continuous_warmup']:
            return 'continuous'
        return 'quantized_joint'

    def frozen_names(self):
        stage = self.stage()
        return {n for n, _ in self.named_parameters()
                if (stage == 'decoder_only' and not n.startswith('decoders.'))
                or (stage == 'continuous' and n.startswith('books.'))}

    def quantize(self, latent):
        if self.arm != 'no_encoder_quantizer':
            return super().quantize(latent)
        need(bool(self.initialized), 'NATIVE_RQ_UNINITIALIZED')
        residual = latent; codes = []; pieces = []; penalties = []
        for book in self.books:
            ids = (residual[:, None]-book[None]).square().sum(-1).argmin(-1)
            selected = book[ids]
            penalties.append(F.mse_loss(selected, residual.detach()))
            codes.append(ids); pieces.append(selected); residual = residual-selected.detach()
        total = torch.stack(pieces).sum(0)
        return torch.stack(codes, -1), latent+(total-latent).detach(), torch.stack(penalties).sum()

    def projection(self, latent):
        need(latent.device.type == 'cpu' and latent.dtype == torch.float32, 'RQ_INTERVENTION_CPU_LATENT')
        generator = torch.Generator(device='cpu').manual_seed(SCHEDULE['projection_seed']+self.step_number)
        matrix = torch.randn(1, latent.shape[1], SCHEDULE['slices'], generator=generator)
        return matrix/torch.linalg.vector_norm(matrix, dim=1, keepdim=True)

    def training_loss(self, frames, donors, eligible):
        stage = self.stage()
        if stage == 'decoder_only':
            with torch.no_grad():
                latent = self.latent(frames); codes, values, _ = self.quantize(latent)
            penalty = latent.sum()*0
        else:
            latent = self.latent(frames)
            if stage == 'continuous':
                values = latent; penalty = latent.sum()*0; codes = None
            else:
                codes, values, penalty = self.quantize(latent)
        counts = [{n:len(f[n]) for n in self.roster} for f in frames]
        decoded = self.reconstruct(values, counts)
        lanes = {n:torch.stack([F.mse_loss(d[n], f[n]) for d, f in zip(decoded, frames)]).mean() for n in self.roster}
        reconstruction = torch.stack([lanes[n]/getattr(self, 'variance_'+str(i)) for i, n in enumerate(self.roster)]).mean()
        contrast = reconstruction*0
        if bool(eligible.any()):
            need(donors.shape == eligible.shape == (len(frames),) and donors.dtype == torch.int64
                 and eligible.dtype == torch.bool and bool(((donors >= 0) & (donors < len(frames))).all()),
                 'RQ_INTERVENTION_DONORS')
            wrong = self.reconstruct(values[donors], counts); terms = []
            for i, n in enumerate(self.roster):
                delta = torch.stack([((right[n]-f[n]).square().mean()-(bad[n]-f[n]).square().mean())
                    /getattr(self, 'variance_'+str(i)) for right, bad, f in zip(decoded, wrong, frames)])
                terms.append(F.softplus(delta[eligible]).mean())
            contrast = torch.stack(terms).mean()
        regularizer = reconstruction*0
        if self.arm == 'latent_sigreg':
            regularizer = statistic(latent[None], self.projection(latent)).mean()
        loss = reconstruction+contrast+penalty+SCHEDULE['sigreg_strength']*regularizer
        return dict(loss=loss, reconstruction=reconstruction, contrast=contrast, quantizer_loss=penalty,
                    regularizer=regularizer, stage=stage, latent=latent, values=values,
                    codes=codes, decoded=decoded, per_lane=lanes)
