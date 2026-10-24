"""Native reconstruction through an immutable TRAIN baseline and scaled residual.

The original window values remain authoritative. This is a student decoder
parameterization, not a new instrument or a retention guarantee for its outputs.
"""
import torch

from model.native_rq_conditioning import ConditionedNativeRQ
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import digest
from text.constellation_corpus import need

RECIPES = ('mean_residual', 'position_residual')


class ResidualConditionedNativeRQ(ConditionedNativeRQ):
    def __init__(self, dimensions, config, sizes, recipe, statistics):
        need(recipe in RECIPES, 'RQ_RESIDUAL_RECIPE')
        super().__init__(dimensions, config, sizes, 'normalized_contrast', statistics)
        self.output_baseline = recipe
        for i, name in enumerate(self.roster):
            value = statistics['positions'][name]
            need(value.ndim == 2 and value.shape[1] == dimensions[name] and len(value) > 0
                 and bool(torch.isfinite(value).all()), 'RQ_RESIDUAL_POSITION_STATISTICS')
            self.register_buffer('position_'+str(i), value.float().clone())
            with torch.no_grad():
                self.decoders[name][-1].weight.zero_()
                self.decoders[name][-1].bias.zero_()
        self.statistics_pin = self.calibration_pin()

    def calibration_pin(self):
        return digest(tensor_fingerprint({n: x for n, x in self.named_buffers()
                      if n.startswith(('center_', 'variance_', 'position_'))}))

    def baseline(self, lane, windows):
        need(type(windows) is int and windows > 0, 'RQ_RESIDUAL_WINDOW_COUNT')
        if self.output_baseline == 'mean_residual':
            return getattr(self, 'center_'+str(lane)).expand(windows, -1)
        value = getattr(self, 'position_'+str(lane))
        need(windows <= len(value), 'RQ_RESIDUAL_UNSEEN_POSITION')
        return value[:windows]

    def reconstruct(self, latent, counts):
        residual = super().reconstruct(latent, counts)
        return tuple({name: self.baseline(i, len(row[name]))
                      +getattr(self, 'variance_'+str(i)).sqrt()*row[name]
                      for i, name in enumerate(self.roster)} for row in residual)
