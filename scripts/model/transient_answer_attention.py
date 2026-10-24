"""Rebuild each attention-fold graph only for its first-order VJP.

History inputs remain connected through the custom node. No truncation, reduced
precision or historical-gradient stop is introduced. This is an opt-in research
runtime until exact update and physical-memory qualification succeeds.
"""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

from claims.codata_source import check
from model.native_answer_sequence import versions as model_versions
from model.native_branch_predictor import IndependentNativeStream
from model.native_prefix_stream import MemoryBlock, PrefixCache, memory_tensors, versions
from model.native_requested_answer import RequestedAnswer
from model.recomputed_answer_attention import RecomputedRequestStream
from text.constellation_corpus import need


class TransientFold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, *inputs):
        ctx.function = function
        ctx.save_for_backward(*inputs)
        return function(*inputs)

    @staticmethod
    def backward(ctx, derivative):
        need(not torch.is_grad_enabled(), 'TRANSIENT_HIGHER_ORDER_UNSUPPORTED')
        needed = ctx.needs_input_grad[1:]
        # These temporary leaves compute the VJP. Returning every needed VJP
        # reconnects it to its original historical tensor through this node.
        inputs = tuple(x.detach().requires_grad_(yes) for x,yes in zip(ctx.saved_tensors, needed))
        with torch.enable_grad():
            value = ctx.function(*inputs)
            gradients = torch.autograd.grad(value, tuple(x for x,yes in zip(inputs,needed) if yes),
                                            derivative, create_graph=False, allow_unused=False)
        cursor = iter(gradients)
        return (None, *(next(cursor) if yes else None for yes in needed))


class TransientRequestStream(RecomputedRequestStream):
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
            inputs = tuple(t for b in (*history, current) for t in (b.key,b.value,b.atoms))
            self.forward_folds += 1
            combined = TransientFold.apply(self.fold,q,atoms,*inputs) if torch.is_grad_enabled() and not detach_history \
                       else self.fold(q,atoms,*inputs)
            x = x+block.out(combined.transpose(1, 2).contiguous().reshape(1, size, width))
            x = x+block.down(torch.nn.functional.gelu(block.up(block.second(x))))
            kept = MemoryBlock(key.detach(), value.detach(), atoms) if detach_history else current
            layers.append((*history,kept))
        hidden = encoder.norm(x)[0,selected]
        check(bool(torch.isfinite(hidden).all()), 'PREFIX_STREAM_NONFINITE')
        layers = tuple(layers)
        return hidden, PrefixCache(id(self),context,offset+len(frames),layers,versions(self.weights()),
                                   versions(memory_tensors(layers)),detach_history)


@contextmanager
def transient_attention(model, *, heads=False):
    need(isinstance(model,RequestedAnswer) and 'stream' not in vars(model) and type(heads) is bool,
         'TRANSIENT_MODEL_POLICY')
    forecaster = model.forecaster
    need(not heads or 'losses' not in vars(forecaster), 'TRANSIENT_HEAD_POLICY')
    streams = []; original_loss = forecaster.losses
    def create(this,request):
        original = this.forecaster.predictor.stream()
        def wrap(s):
            result = TransientRequestStream(s,this.request_projection,request); streams.append(result)
            return result
        if isinstance(original,IndependentNativeStream):
            original.streams = {n:wrap(s) for n,s in original.streams.items()}; return original
        return wrap(original)
    def loss(this, hidden, codes, counts, targets, noise, *, target):
        roster = this.roster; n = len(roster); b = len(targets); pin = model_versions(this)
        def calculate(h,c,z,*values):
            need(model_versions(this) == pin, 'TRANSIENT_HEAD_WEIGHTS_CHANGED')
            count = dict(zip(roster,values[:n]))
            frames = tuple(dict(zip(roster,values[n+i*n:n+(i+1)*n])) for i in range(b))
            return original_loss(h,c,count,frames,z,target=target)
        return checkpoint(calculate,hidden,codes,noise,*(counts[k] for k in roster),
                          *(row[k] for row in targets for k in roster),use_reentrant=False,
                          preserve_rng_state=True,determinism_check='default')
    model.stream = MethodType(create,model)
    if heads: forecaster.losses = MethodType(loss,forecaster)
    try:
        yield streams
    finally:
        del model.stream
        if heads: del forecaster.losses
