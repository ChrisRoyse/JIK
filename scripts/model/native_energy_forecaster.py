"""Opt-in placement forecaster with within-instrument energy-distance batching.

Parameter names, shapes, initialization and generation are inherited unchanged.
Device-specific full-update equality is required before a caller may promote it.
"""
import torch

from model.native_energy_batched import energy_loss
from model.native_placement_training import NativePlacementForecaster


class EnergyBatchedForecaster(NativePlacementForecaster):
    def losses(self, hidden, codes, counts, targets, noise, *, target):
        rows = {}
        for name, (head, context) in self.contexts(hidden).items():
            discrete = head.loss(context, codes, counts)
            samples = self.placement[name](context, codes, counts, noise, target=target)
            native = energy_loss(samples, targets, self.placement[name].variances())
            rows[name] = dict(loss=discrete['loss']+native['loss'], discrete=discrete,
                              native=native, samples=samples)
        return dict(loss=torch.stack([r['loss'] for r in rows.values()]).mean(), proposals=rows)
