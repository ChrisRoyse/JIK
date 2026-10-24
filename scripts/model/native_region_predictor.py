"""Predict hierarchical regions/counts from a typed, causal native history.

The frozen target codebook supplies labels and a native reconstruction baseline.
It is never an input containing the next atom. A stochastic within-region placement
head and a surface decoder are separate capabilities, not claimed by this module.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Linear
from model.native_prefix_stream import NativePrefixStream
from model.native_residual_codebook import AutoregressiveSemanticHead
from text.constellation_corpus import need


class NormalizedProjector(nn.Module):
    def __init__(self, projection, center, variance):
        super().__init__()
        self.projection = deepcopy(projection).requires_grad_(True)
        self.register_buffer('center', center.detach().clone())
        self.register_buffer('scale', variance.detach().sqrt().clone())

    def forward(self, value):
        return self.projection((value-self.center)/self.scale)


class NativeRegionPredictor(nn.Module):
    def __init__(self, target, *, target_pin, seed, cache_bytes=16 << 20):
        super().__init__()
        need(not any(p.requires_grad for p in target.parameters()) and bool(target.initialized),
             'REGION_PREDICTOR_TARGET_NOT_FROZEN')
        need(type(target_pin) is str and len(target_pin) == 64, 'REGION_PREDICTOR_TARGET_PIN')
        torch.manual_seed(seed)
        self.target_pin, self.roster = target_pin, tuple(target.roster)
        self._target = id(target)
        self._target_versions = tuple((id(x), x._version) for x in target.state_dict(keep_vars=True).values())
        self.config, self.sizes = dict(target.config), tuple(target.sizes)
        self.encoder = deepcopy(target.encoder).requires_grad_(True)
        self.projectors = nn.ModuleDict({n: NormalizedProjector(target.projectors[n],
            getattr(target, 'center_'+str(i)), getattr(target, 'variance_'+str(i)))
            for i, n in enumerate(self.roster)})
        h = self.config['width']
        self.region = AutoregressiveSemanticHead(len(self.roster), h, self.sizes)
        self.max_windows = {n: len(getattr(target, 'position_'+str(i))) for i, n in enumerate(self.roster)}
        self.counts = nn.ModuleDict({n: Linear(h, self.max_windows[n]) for n in self.roster})
        self.cache_bytes = cache_bytes

    def stream(self):
        return NativePrefixStream(self.encoder, self.projectors, cache_bytes=self.cache_bytes)

    def check_hidden(self, hidden):
        need(hidden.ndim == 3 and hidden.shape[1:] == (len(self.roster), self.config['width'])
             and bool(torch.isfinite(hidden).all()), 'REGION_PREDICTOR_HIDDEN')

    def forecast(self, hidden, *, generator=None):
        """Autonomous forecast: this interface accepts no target values or counts."""
        self.check_hidden(hidden)
        logits, codes = self.region(hidden, generator=generator)
        count_logits = {n: self.counts[n](hidden[:, i]) for i, n in enumerate(self.roster)}
        counts = {}
        for name, score in count_logits.items():
            counts[name] = (score.argmax(-1) if generator is None else
                torch.multinomial(score.softmax(-1), 1, generator=generator).squeeze(-1))+1
        return dict(logits=logits, codes=codes, count_logits=count_logits, counts=counts)

    def loss(self, hidden, codes, counts):
        """Teacher forcing uses earlier code levels only; counts are loss targets only."""
        self.check_hidden(hidden)
        need(codes.device == hidden.device and set(counts) == set(self.roster), 'REGION_PREDICTOR_TARGET')
        logits, _ = self.region(hidden, codes)
        # Small categorical reductions use fp64. fp32 CE and gather/mean differed
        # by 9.54e-7 on a real 12-target document; do not relax the recount gate.
        region = tuple(F.cross_entropy(score.double(), codes[:, level]) for level, score in enumerate(logits))
        window = {}
        for i, name in enumerate(self.roster):
            value = counts[name]
            need(value.shape == (len(hidden),) and value.dtype == torch.int64 and value.device == hidden.device
                 and bool(((value >= 1) & (value <= self.max_windows[name])).all()),
                 'REGION_PREDICTOR_TARGET_COUNTS')
            window[name] = F.cross_entropy(self.counts[name](hidden[:, i]).double(), value-1)
        return dict(loss=torch.stack(region).mean()+torch.stack(list(window.values())).mean(),
                    region=region, counts=window, logits=logits)

    @torch.no_grad()
    def render(self, prediction, target):
        """Decode forecast codes and forecast counts, never reference target counts."""
        need(id(target) == self._target and tuple(target.roster) == self.roster
             and self._target_versions == tuple((id(x), x._version) for x in target.state_dict(keep_vars=True).values())
             and not any(p.requires_grad for p in target.parameters()),
             'REGION_PREDICTOR_RENDER_TARGET')
        codes, counts = prediction['codes'], prediction['counts']
        need(set(counts) == set(self.roster), 'REGION_PREDICTOR_RENDER_COUNTS')
        rows = [{n: int(counts[n][i]) for n in self.roster} for i in range(len(codes))]
        need(all(1 <= row[n] <= self.max_windows[n] for row in rows for n in self.roster),
             'REGION_PREDICTOR_RENDER_RANGE')
        result = []
        for lo in range(0, len(rows), 128):
            result.extend(target.reconstruct(target.values(codes[lo:lo+128]), rows[lo:lo+128]))
        return tuple(result)


@torch.no_grad()
def target_labels(target, frames):
    need(frames and not any(p.requires_grad for p in target.parameters()), 'REGION_LABEL_TARGET_FROZEN')
    device = target.encoder.query.device
    native = tuple({n: x.to(device) for n, x in frame.items()} for frame in frames)
    codes = torch.cat([target.quantize(target.latent(native[lo:lo+128]))[0]
                       for lo in range(0, len(native), 128)])
    counts = {n: torch.tensor([len(f[n]) for f in frames], dtype=torch.int64, device=device) for n in target.roster}
    return codes, counts
