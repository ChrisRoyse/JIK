"""Auxiliary native-vector supervision: reconstruct one instrument from the others.

This trains cross-view associations alongside region-first sequence prediction.
It neither supplies a world-fact verdict nor replaces the region/guard path.
"""
import torch
from torch import nn
from torch.nn import functional as F

from claims.capability_population import need
from model.attention import Attention, Linear
from model.network import positions


class ReciprocalConstellation(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.config, self.specs = core.config, core.specs
        h, count = self.config.width, len(self.specs)
        need(count >= 6, 'RECIPROCAL_VISIBLE_FLOOR')
        extra = count*h + count*2*(4*h*h+4*h) + sum(h*s.dims+s.dims for s in self.specs)
        need(sum(p.numel() for p in core.parameters())+extra <= self.config.max_parameters,
             'RECIPROCAL_PARAMETER_BUDGET')
        # Actual retained intrinsic predictor branches, not a second sensing panel.
        self.projectors = core.projectors
        self.readers = nn.ModuleDict({s.name: Attention(self.config) for s in self.specs})
        self.mix = nn.ModuleDict({s.name: Attention(self.config) for s in self.specs})
        self.output = nn.ModuleDict({s.name: Linear(h, s.dims) for s in self.specs})
        self.seed = nn.Parameter(torch.zeros(count, h))
        self.extra_parameters = extra

    def forward(self, visible, *, hidden, windows):
        names = [s.name for s in self.specs]
        need(hidden in names and set(visible) == set(names)-{hidden}, 'RECIPROCAL_MASK')
        need(type(windows) is int and 0 < windows <= self.config.max_native_tokens,
             'RECIPROCAL_WINDOW_BUDGET')
        rows = None; total = 0
        for s in self.specs:
            if s.name == hidden: continue
            x = visible[s.name]; x.check(s, self.config)
            need(not bool(x.absent.any()) and (rows is None or x.rows == rows), 'RECIPROCAL_ALIGNMENT')
            rows = x.rows; total += x.values.numel()
        need(total <= self.config.max_native_elements, 'RECIPROCAL_NATIVE_BUDGET')
        h, b = self.config.width, len(rows)
        need(b*windows*(len(names)-1)*h <= self.config.max_native_elements, 'RECIPROCAL_BRANCH_BUDGET')
        query = self.seed[names.index(hidden)][None, None, :] + positions(windows, h, self.seed.device)[None]
        query = query.expand(b, -1, -1)
        branches = {}
        for s in self.specs:
            if s.name == hidden: continue
            x = visible[s.name]
            memory = self.projectors[s.name](x.values)
            memory = memory + positions(memory.shape[1], h, memory.device)[None]
            branches[s.name] = self.readers[s.name](query, memory, padding=x.padding)
        # Batch windows, not coordinates: the instrument axis stays explicit in memory.
        views = torch.stack(list(branches.values()), dim=2).reshape(b*windows, len(branches), h)
        mixed = self.mix[hidden](query.reshape(b*windows, 1, h), views).reshape(b, windows, h)
        prediction = F.normalize(self.output[hidden](mixed), dim=-1)
        need(bool(torch.isfinite(prediction).all()), 'RECIPROCAL_NONFINITE')
        return dict(prediction=prediction, visible=tuple(branches), hidden=hidden,
                    target_window_count='teacher_structure_not_predicted',
                    scientific_admission=False, generation_ready=False)

    @staticmethod
    def loss(prediction, target):
        need(prediction.shape == target.shape and not target.requires_grad, 'RECIPROCAL_TARGET')
        return (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
