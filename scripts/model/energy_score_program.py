"""An unchanged typed energy-score tensor program behind the original input checks.

This is a capture boundary, not a CUDA graph cache. Each invocation still validates
its real inputs. The program owns metadata only and cannot retain prior examples.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
import math

import torch

from model import native_placement_training as training
from model.native_stochastic_placement import energy_loss as ORIGINAL
from text.constellation_corpus import need

ACTIVE = False


@dataclass(frozen=True)
class EnergyProgram:
    roster: tuple
    scales: tuple
    batch: int
    count: int
    empirical: bool
    layouts: tuple

    def __call__(self, *inputs):
        # Shape/dtype/device are host metadata; there is no data-dependent host read.
        need(len(inputs) == len(self.layouts) and all(
            (tuple(x.shape), x.dtype, x.device) == layout for x, layout in zip(inputs, self.layouts)),
            'ENERGY_PROGRAM_LAYOUT')
        pairs = tuple(combinations(range(self.count), 2)); rows = []; cursor = 0
        for _ in range(self.batch):
            target_distances, pair_distances = {}, {}
            for n, variance in zip(self.roster, self.scales):
                x = inputs[cursor].double(); y = inputs[cursor+1].double(); cursor += 2
                scale = math.sqrt(y.numel()*variance)
                target_distances[n] = torch.linalg.vector_norm((x-y[None]).flatten(1), dim=1)/scale
                pair_distances[n] = torch.stack([torch.linalg.vector_norm(x[a]-x[b])/scale for a, b in pairs])
            factor = (self.count-1)/self.count if self.empirical else 1.
            lane = {n: target_distances[n].mean()-.5*factor*pair_distances[n].mean() for n in self.roster}
            td = torch.linalg.vector_norm(torch.stack(list(target_distances.values())), dim=0)/math.sqrt(len(self.roster))
            pd = torch.linalg.vector_norm(torch.stack(list(pair_distances.values())), dim=0)/math.sqrt(len(self.roster))
            joint = td.mean()-.5*factor*pd.mean()
            rows.append(dict(lanes=lane, joint=joint, loss=torch.stack(list(lane.values())).mean()+joint))
        return dict(loss=torch.stack([r['loss'] for r in rows]).mean(),
                    joint=torch.stack([r['joint'] for r in rows]),
                    lanes={n: torch.stack([r['lanes'][n] for r in rows]) for n in self.roster})

    def description(self):
        return dict(roster=self.roster, scales=self.scales, batch=self.batch, count=self.count,
                    empirical=self.empirical,
                    layouts=[dict(shape=s, dtype=str(t), device=str(d)) for s, t, d in self.layouts])


def prepare(samples, targets, variances, *, empirical=False, working_bytes=512 << 20):
    # Preserve reference refusal conditions and order before any numerical program runs.
    need(type(empirical) is bool and type(working_bytes) is int and working_bytes > 0, 'PLACEMENT_SCORE_POLICY')
    need(isinstance(samples, (tuple, list)) and 0 < len(samples) <= 64
         and isinstance(targets, (tuple, list)) and len(samples) == len(targets), 'PLACEMENT_SCORE_BATCH')
    roster = tuple(variances); scales = {n: float(v) for n, v in variances.items()}
    need(roster and all(math.isfinite(v) and v > 0 for v in scales.values()), 'PLACEMENT_SCORE_VARIANCE')
    first = samples[0][roster[0]] if set(samples[0]) == set(roster) else None
    need(isinstance(first, torch.Tensor) and first.ndim == 3 and 2 <= len(first) <= 8, 'PLACEMENT_SCORE_SAMPLES')
    count = len(first); device = first.device; elements = 0; finite = []; inputs = []
    for row, target in zip(samples, targets):
        need(set(row) == set(target) == set(roster), 'PLACEMENT_SCORE_ROSTER')
        for n in roster:
            x, y = row[n], target[n]
            need(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.ndim == 3
                 and x.shape == (count, *y.shape) and y.ndim == 2 and y.numel() > 0
                 and x.dtype in (torch.float32, torch.float64) and y.dtype in (torch.float32, torch.float64)
                 and x.device == y.device == device and not y.requires_grad, 'PLACEMENT_SCORE_LAYOUT')
            elements += y.numel()
            finite.extend((torch.isfinite(x).all(), torch.isfinite(y).all()))
            inputs.extend((x, y))
    need(bool(torch.stack(finite).all()), 'PLACEMENT_SCORE_LAYOUT')
    need(4*8*elements*(count+count*(count-1)//2) <= working_bytes, 'PLACEMENT_SCORE_MEMORY')
    layouts = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)
    return EnergyProgram(roster, tuple(scales.values()), len(samples), count, empirical, layouts), tuple(inputs)


class CheckedEnergy:
    def __init__(self, observe=None):
        self.records = []; self.closed = False; self.observe = observe

    def __call__(self, samples, targets, variances, **kwargs):
        need(not self.closed, 'ENERGY_PROGRAM_CLOSED')
        program, inputs = prepare(samples, targets, variances, **kwargs)
        if self.observe is not None: self.observe(program, inputs, samples, targets, variances, kwargs)
        record = dict(program=program.description(), completed=False); self.records.append(record)
        result = program(*inputs)
        record['completed'] = True
        return result


@contextmanager
def checked_energy(observe=None):
    global ACTIVE
    need(not ACTIVE and training.energy_loss is ORIGINAL, 'ENERGY_PROGRAM_ALREADY_INSTALLED')
    value = CheckedEnergy(observe); ACTIVE = True; training.energy_loss = value
    try: yield value
    finally:
        intact = training.energy_loss is value
        training.energy_loss = ORIGINAL; ACTIVE = False; value.closed = True
        need(intact, 'ENERGY_PROGRAM_OWNERSHIP_CHANGED')
