"""Opt-in recomputation of complete causal attention folds, with unchanged arithmetic.

The reference stream remains frozen for existing publications. This version keeps
all historical K/V graph edges and all physical/cache guards. Tensor savings do
not bound autograd metadata or transient recomputation memory.
"""
from contextlib import contextmanager
import math
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from claims.codata_source import check
from model.native_branch_predictor import IndependentNativeStream
from model.native_prefix_stream import MemoryBlock, PrefixCache, memory_tensors, versions
from model.native_request_stream import RequestStream
from model.native_requested_answer import RequestedAnswer
from text.constellation_corpus import need


class RecomputedRequestStream(RequestStream):
    def __init__(self, original, projection, request):
        super().__init__(original, projection, request)
        self.fold_calls = self.forward_folds = 0

    def fold(self, q, atoms, *history):
        self.fold_calls += 1
        combined = log_mass = None
        depth = q.shape[-1]
        for start in range(0, len(history), 3):
            key, value, prior_atoms = history[start:start+3]
            score = (q@key.transpose(-1, -2))/math.sqrt(depth)
            future = prior_atoms[None, :] > atoms[:, None]
            score = score.masked_fill(future[None, None], -torch.inf)
            mass = torch.logsumexp(score, -1)
            weighted = score.softmax(-1)@value
            if combined is None:
                combined, log_mass = weighted, mass
            else:
                weight = torch.sigmoid(log_mass-mass)
                combined = weight[..., None]*combined+(1-weight[..., None])*weighted
                log_mass = torch.logaddexp(log_mass, mass)
        return combined

    def step(self, frames, *, context, offset, state=None, detach_history):
        self.check_cache(state, context, offset, detach_history)
        encoder = self.encoder; config = encoder.config; heads = config.heads; width = config.width
        check(frames and all(set(f) == set(self.names) for f in frames), 'PREFIX_STREAM_ROSTER')
        size = sum(len(x) for f in frames for x in f.values())+len(frames)*len(self.names)
        old_layers = tuple(() for _ in encoder.blocks) if state is None else state.layers
        prior_bytes = sum(t.numel()*t.element_size() for t in memory_tensors(old_layers))
        check(prior_bytes+len(encoder.blocks)*size*(2*width*4+8) <= self.cache_bytes,
             'PREFIX_STREAM_CACHE_MEMORY')
        check(heads*size*size <= config.max_attention_elements
             and all(heads*size*len(block.atoms) <= config.max_attention_elements
                     for layer in old_layers for block in layer), 'PREFIX_STREAM_ATTENTION_MEMORY')
        x, atoms, selected = self.pack(frames, offset); layers = []
        for block, history in zip(encoder.blocks, old_layers):
            z = block.first(x); depth = width//heads
            q, key, value = (m(z).reshape(1, size, heads, depth).transpose(1, 2) for m in (block.q, block.k, block.v))
            current = MemoryBlock(key, value, atoms)
            inputs = tuple(t for b in (*history, current) for t in (b.key, b.value, b.atoms))
            self.forward_folds += 1
            if torch.is_grad_enabled() and not detach_history:
                combined = checkpoint(self.fold, q, atoms, *inputs, use_reentrant=False,
                                      preserve_rng_state=True, determinism_check='default')
            else:
                combined = self.fold(q, atoms, *inputs)
            x = x+block.out(combined.transpose(1, 2).contiguous().reshape(1, size, width))
            x = x+block.down(torch.nn.functional.gelu(block.up(block.second(x))))
            kept = MemoryBlock(key.detach(), value.detach(), atoms) if detach_history else current
            layers.append((*history, kept))
        hidden = encoder.norm(x)[0, selected]
        check(bool(torch.isfinite(hidden).all()), 'PREFIX_STREAM_NONFINITE')
        layers = tuple(layers)
        return hidden, PrefixCache(id(self), context, offset+len(frames), layers, versions(self.weights()),
                                   versions(memory_tensors(layers)), detach_history)


@contextmanager
def recompute_attention(model):
    """Change only this instance's stream policy; never replace parameters/moments."""
    need(isinstance(model, RequestedAnswer) and 'stream' not in vars(model), 'RECOMPUTE_MODEL_POLICY')
    streams = []
    def create(this, request):
        original = this.forecaster.predictor.stream()
        def wrap(s):
            value = RecomputedRequestStream(s, this.request_projection, request)
            streams.append(value)
            return value
        if isinstance(original, IndependentNativeStream):
            original.streams = {n:wrap(s) for n,s in original.streams.items()}
            return original
        return wrap(original)
    model.stream = MethodType(create, model)
    try:
        yield streams
    finally:
        del model.stream
