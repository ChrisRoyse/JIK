"""Independent native histories, with assembled or source-specific region forecasts.

Native coordinates remain in their own slots. The common frozen codebook provides
the reconstruction baseline; stochastic native placement is a separate component.
"""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from model.attention import Linear
from model.native_prefix_stream import NativePrefixStream
from model.native_region_predictor import NativeRegionPredictor
from text.constellation_corpus import need


class IndependentNativeEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.specs, self.config = original.specs, original.config
        self.lanes = nn.ModuleDict()
        for index, spec in enumerate(self.specs):
            branch = deepcopy(original)
            branch.specs = (spec,)
            branch.kind = nn.Parameter(original.kind[index:index+1].detach().clone())
            branch.query = nn.Parameter(original.query[index:index+1].detach().clone())
            self.lanes[spec.name] = branch

    def validate(self, frames):
        names = {s.name for s in self.specs}
        need(isinstance(frames, (list, tuple)) and frames
             and all(set(f) == names for f in frames), 'BRANCH_NATIVE_ROSTER')
        for frame in frames:
            for spec in self.specs:
                x = frame[spec.name]
                need(isinstance(x, torch.Tensor) and x.ndim == 2 and len(x) > 0
                     and x.shape[1] == spec.dims and x.dtype == torch.float32
                     and x.device == self.lanes[spec.name].query.device
                     and not x.requires_grad and bool(torch.isfinite(x).all()),
                     'BRANCH_NATIVE_VALUES')

    def encode(self, frames, projectors):
        self.validate(frames)
        need(set(projectors) == set(self.lanes), 'BRANCH_PROJECTOR_ROSTER')
        return torch.cat([encoder.encode(tuple({name:f[name]} for f in frames),
                          {name:projectors[name]}) for name,encoder in self.lanes.items()], dim=1)


@dataclass(frozen=True)
class BranchCache:
    owner: int
    states: tuple


class IndependentNativeStream:
    def __init__(self, encoder, projectors, *, cache_bytes):
        need(type(cache_bytes) is int and cache_bytes >= len(encoder.lanes), 'BRANCH_CACHE_ALLOWANCE')
        self.encoder = encoder
        # Divide one total allowance. Six branches never silently receive 6x memory.
        self.streams = {name:NativePrefixStream(branch, nn.ModuleDict({name:projectors[name]}),
                            cache_bytes=cache_bytes//len(encoder.lanes))
                        for name,branch in encoder.lanes.items()}

    def step(self, frames, *, context, offset, state=None, detach_history):
        self.encoder.validate(frames)
        need(state is None or (isinstance(state, BranchCache) and state.owner == id(self)
             and len(state.states) == len(self.streams)), 'BRANCH_CACHE_OWNER')
        caches = (None,)*len(self.streams) if state is None else state.states
        # Validate every cache before computing any branch. Cache objects are immutable.
        for stream,cache in zip(self.streams.values(), caches):
            stream.check_cache(cache, context, offset, detach_history)
        outputs, updated = [], []
        for (name,stream),cache in zip(self.streams.items(), caches):
            value,new = stream.step(tuple({name:f[name]} for f in frames), context=context,
                                   offset=offset, state=cache, detach_history=detach_history)
            outputs.append(value); updated.append(new)
        return torch.cat(outputs, dim=1), BranchCache(id(self), tuple(updated))


class IndependentNativeRegionPredictor(NativeRegionPredictor):
    """Each history sees one instrument; common region + destination slot/count heads."""
    def __init__(self, target, **kwargs):
        super().__init__(target, **kwargs)
        self.encoder = IndependentNativeEncoder(self.encoder)

    def stream(self):
        return IndependentNativeStream(self.encoder, self.projectors, cache_bytes=self.cache_bytes)


class NativeRegionReadout(nn.Module):
    """Same region/count heads and frozen-target guard used by the shared baseline."""
    def __init__(self, original):
        super().__init__()
        for name in ('target_pin', 'roster', '_target', '_target_versions', 'config', 'sizes', 'max_windows'):
            setattr(self, name, getattr(original, name))
        self.region, self.counts = deepcopy(original.region), deepcopy(original.counts)

    check_hidden = NativeRegionPredictor.check_hidden
    forecast = NativeRegionPredictor.forecast
    loss = NativeRegionPredictor.loss
    render = NativeRegionPredictor.render


class PerInstrumentConstellationPredictor(nn.Module):
    """One complete, separately retained proposal per source-only instrument history."""
    def __init__(self, target, **kwargs):
        super().__init__()
        base = IndependentNativeRegionPredictor(target, **kwargs)
        self.roster, self.config = base.roster, base.config
        self.encoder, self.projectors = base.encoder, base.projectors
        self.cache_bytes = base.cache_bytes
        self.readouts = nn.ModuleDict({n:NativeRegionReadout(base) for n in self.roster})
        width = self.config['width']
        self.destinations = nn.ModuleDict({source:nn.ModuleDict({dest:Linear(width,width)
                    for dest in self.roster}) for source in self.roster})
        # A declared identity initialization: all arms start from the same frozen
        # encoder/region heads, without arbitrary destination distortions.
        with torch.no_grad():
            for branches in self.destinations.values():
                for layer in branches.values():
                    layer.weight.copy_(torch.eye(width))
                    layer.bias.zero_()

    def stream(self):
        return IndependentNativeStream(self.encoder, self.projectors, cache_bytes=self.cache_bytes)

    def proposal_context(self, hidden, source):
        need(source in self.roster, 'BRANCH_PROPOSAL_SOURCE')
        self.readouts[source].check_hidden(hidden)
        value = hidden[:,self.roster.index(source)]
        return torch.stack([self.destinations[source][n](value) for n in self.roster], dim=1)

    def forecast(self, hidden, *, generator=None):
        return {n:self.readouts[n].forecast(self.proposal_context(hidden,n), generator=generator)
                for n in self.roster}

    def loss(self, hidden, codes, counts):
        rows = {n:self.readouts[n].loss(self.proposal_context(hidden,n), codes, counts) for n in self.roster}
        # Only loss scalars combine. Predictions/native slots are never averaged.
        return dict(loss=torch.stack([r['loss'] for r in rows.values()]).mean(), proposals=rows)

    def render(self, predictions, target):
        need(set(predictions) == set(self.roster), 'BRANCH_PROPOSAL_ROSTER')
        return {n:self.readouts[n].render(predictions[n], target) for n in self.roster}
