"""Candidate energy scorer: group equal window shapes within one instrument.

The row/slot reduction order is unchanged. This is an opt-in candidate until
device-specific whole-update equality and performance have been demonstrated.
"""
from itertools import combinations
import math

import torch

from text.constellation_corpus import need


def energy_loss(samples, targets, variances, *, empirical=False, working_bytes=512<<20):
    need(type(empirical) is bool and type(working_bytes) is int and working_bytes > 0,
         'PLACEMENT_SCORE_POLICY')
    need(isinstance(samples, (tuple, list)) and 0 < len(samples) <= 64
         and isinstance(targets, (tuple, list)) and len(samples) == len(targets), 'PLACEMENT_SCORE_BATCH')
    roster = tuple(variances); scales = {n:float(v) for n,v in variances.items()}
    need(roster and all(math.isfinite(v) and v > 0 for v in scales.values()), 'PLACEMENT_SCORE_VARIANCE')
    first = samples[0][roster[0]] if set(samples[0]) == set(roster) else None
    need(isinstance(first, torch.Tensor) and first.ndim == 3 and 2 <= len(first) <= 8,
         'PLACEMENT_SCORE_SAMPLES')
    count = len(first); device = first.device; elements = 0; finite = []
    for row,target in zip(samples,targets):
        need(set(row) == set(target) == set(roster), 'PLACEMENT_SCORE_ROSTER')
        for n in roster:
            x,y = row[n],target[n]
            need(isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and x.ndim == 3
                 and x.shape == (count,*y.shape) and y.ndim == 2 and y.numel() > 0
                 and x.dtype in (torch.float32,torch.float64) and y.dtype in (torch.float32,torch.float64)
                 and x.device == y.device == device and not y.requires_grad, 'PLACEMENT_SCORE_LAYOUT')
            elements += y.numel()
            finite.extend((torch.isfinite(x).all(),torch.isfinite(y).all()))
    need(bool(torch.stack(finite).all()), 'PLACEMENT_SCORE_LAYOUT')
    # Includes retained differences, double copies, grouped inputs and gradients.
    need(4*8*elements*(count+count*(count-1)//2) <= working_bytes, 'PLACEMENT_SCORE_MEMORY')
    pair_indices = tuple(combinations(range(count),2))
    target_distances = [{} for _ in samples]; pair_distances = [{} for _ in samples]
    for n in roster:
        groups = {}
        for i,target in enumerate(targets):
            groups.setdefault(tuple(target[n].shape),[]).append(i)
        for shape,indices in groups.items():
            x = torch.stack([samples[i][n].double() for i in indices])
            y = torch.stack([targets[i][n].double() for i in indices])
            scale = math.sqrt(math.prod(shape)*scales[n])
            td = torch.linalg.vector_norm((x-y[:,None]).flatten(2),dim=2)/scale
            pd = torch.stack([torch.linalg.vector_norm((x[:,a]-x[:,b]).flatten(1),dim=1)/scale
                              for a,b in pair_indices],dim=1)
            for j,i in enumerate(indices):
                target_distances[i][n] = td[j]; pair_distances[i][n] = pd[j]
    rows = []; factor = (count-1)/count if empirical else 1.
    for target_row,pair_row in zip(target_distances,pair_distances):
        lane = {n:target_row[n].mean()-.5*factor*pair_row[n].mean() for n in roster}
        td = torch.linalg.vector_norm(torch.stack(list(target_row.values())),dim=0)/math.sqrt(len(roster))
        pd = torch.linalg.vector_norm(torch.stack(list(pair_row.values())),dim=0)/math.sqrt(len(roster))
        joint = td.mean()-.5*factor*pd.mean()
        rows.append(dict(lanes=lane,joint=joint,loss=torch.stack(list(lane.values())).mean()+joint))
    return dict(loss=torch.stack([r['loss'] for r in rows]).mean(),
                joint=torch.stack([r['joint'] for r in rows]),
                lanes={n:torch.stack([r['lanes'][n] for r in rows]) for n in roster})
