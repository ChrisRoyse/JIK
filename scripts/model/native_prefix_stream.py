"""Carry every earlier typed attention position without re-projecting its prefix.

This wraps the actual NativePrefixTransformer and its existing projectors. It
adds no model parameters. Detached history bounds backward graphs, not forward
context. Weights must remain unchanged until the document state is closed.
"""
from dataclasses import dataclass
import math

import torch

from claims.codata_source import check
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_prefix_transformer import NativePrefixTransformer
from senses.contract import digest


@dataclass(frozen=True)
class MemoryBlock:
    key: torch.Tensor
    value: torch.Tensor
    atoms: torch.Tensor


@dataclass(frozen=True)
class PrefixCache:
    owner: int
    context: str
    next_atom: int
    layers: tuple
    weight_versions: tuple
    tensor_versions: tuple
    detached: bool


def memory_tensors(layers):
    return tuple(t for layer in layers for block in layer for t in (block.key, block.value, block.atoms))


def versions(tensors):
    return tuple((id(t), t._version) for t in tensors)


class NativePrefixStream:
    def __init__(self, encoder, projectors, *, cache_bytes):
        check(isinstance(encoder, NativePrefixTransformer), 'PREFIX_STREAM_ENCODER')
        check(type(cache_bytes) is int and cache_bytes > 0, 'PREFIX_STREAM_MEMORY_POLICY')
        self.encoder, self.projectors, self.cache_bytes = encoder, projectors, cache_bytes
        self.names = tuple(s.name for s in encoder.specs)
        check(set(projectors) == set(self.names), 'PREFIX_STREAM_PROJECTORS')
        self.layout = self.current_layout()

    def current_layout(self):
        config = self.encoder.config
        return dict(width=config.width, heads=config.heads, layers=len(self.encoder.blocks),
                    max_native_tokens=config.max_native_tokens, max_native_elements=config.max_native_elements,
                    max_attention_elements=config.max_attention_elements,
                    specs=[(s.name, s.dims) for s in self.encoder.specs])

    def weights(self):
        return tuple(self.encoder.parameters())+tuple(self.encoder.buffers())+tuple(self.projectors.parameters())+tuple(self.projectors.buffers())

    def weight_pin(self):
        return digest(tensor_fingerprint(dict(encoder=self.encoder.state_dict(), projectors=self.projectors.state_dict())))

    def check_cache(self, state, context, offset, detached):
        check(self.current_layout() == self.layout, 'PREFIX_STREAM_LAYOUT_CHANGED')
        check(type(context) is str and context and type(offset) is int and offset >= 0
              and type(detached) is bool, 'PREFIX_STREAM_CONTEXT')
        if state is None:
            check(offset == 0, 'PREFIX_STREAM_MISSING_HISTORY')
            return
        check(isinstance(state, PrefixCache) and state.owner == id(self) and state.context == context,
              'PREFIX_STREAM_WRONG_CONTEXT')
        check(state.next_atom == offset and state.detached == detached, 'PREFIX_STREAM_OFFSET')
        check(state.weight_versions == versions(self.weights()), 'PREFIX_STREAM_WEIGHTS_CHANGED')
        check(state.tensor_versions == versions(memory_tensors(state.layers)), 'PREFIX_STREAM_CACHE_CHANGED')
        check(len(state.layers) == len(self.encoder.blocks), 'PREFIX_STREAM_LAYERS')

    def pack(self, frames, offset):
        encoder = self.encoder; config = encoder.config; device = encoder.query.device
        check(frames and all(set(f) == set(self.names) for f in frames), 'PREFIX_STREAM_ROSTER')
        windows = sum(len(x) for f in frames for x in f.values())
        elements = sum(x.numel() for f in frames for x in f.values())
        size = windows+len(frames)*len(self.names)
        check(size <= config.max_native_tokens and elements <= config.max_native_elements,
              'PREFIX_STREAM_CURRENT_MEMORY')
        positions = encoder.position(torch.arange(offset, offset+len(frames), device=device))
        tokens, atoms, queries, cursor = [], [], [], 0
        for atom, frame in enumerate(frames):
            selected = []
            for lane, spec in enumerate(encoder.specs):
                values = frame[spec.name]
                check(values.ndim == 2 and values.shape[1] == spec.dims and len(values) > 0
                      and values.device == device and values.dtype == torch.float32
                      and not values.requires_grad and bool(torch.isfinite(values).all()), 'PREFIX_STREAM_NATIVE')
                local = encoder.position(torch.arange(len(values), device=device))
                tokens.append(self.projectors[spec.name](values)+encoder.kind[lane]+positions[atom]+local)
                atoms.extend([offset+atom]*len(values)); cursor += len(values)
                selected.append(cursor)
                tokens.append((encoder.query[lane]+encoder.kind[lane]+positions[atom])[None])
                atoms.append(offset+atom); cursor += 1
            queries.append(selected)
        return torch.cat(tokens, 0)[None], torch.tensor(atoms, device=device), torch.tensor(queries, device=device)

    def step(self, frames, *, context, offset, state=None, detach_history):
        self.check_cache(state, context, offset, detach_history)
        encoder = self.encoder; config = encoder.config; heads = config.heads; width = config.width
        # Preflight native/current layout and persistent cache before allocating projections.
        check(frames and all(set(f) == set(self.names) for f in frames), 'PREFIX_STREAM_ROSTER')
        size = sum(len(x) for f in frames for x in f.values())+len(frames)*len(self.names)
        old_layers = tuple(() for _ in encoder.blocks) if state is None else state.layers
        prior_bytes = sum(t.numel()*t.element_size() for t in memory_tensors(old_layers))
        # Atom addresses are stored per layer, alongside two float32 K/V arrays.
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
            combined = None; log_mass = None
            for memory in (*history, current):
                score = (q@memory.key.transpose(-1, -2))/math.sqrt(depth)
                future = memory.atoms[None, :] > atoms[:, None]
                score = score.masked_fill(future[None, None], -torch.inf)
                mass = torch.logsumexp(score, -1)
                weighted = score.softmax(-1)@memory.value
                if combined is None:
                    combined, log_mass = weighted, mass
                else:
                    weight = torch.sigmoid(log_mass-mass)
                    combined = weight[..., None]*combined+(1-weight[..., None])*weighted
                    log_mass = torch.logaddexp(log_mass, mass)
            x = x+block.out(combined.transpose(1, 2).contiguous().reshape(1, size, width))
            x = x+block.down(torch.nn.functional.gelu(block.up(block.second(x))))
            kept = MemoryBlock(key.detach(), value.detach(), atoms) if detach_history else current
            layers.append((*history, kept))
        hidden = encoder.norm(x)[0, selected]
        check(bool(torch.isfinite(hidden).all()), 'PREFIX_STREAM_NONFINITE')
        layers = tuple(layers)
        result = PrefixCache(id(self), context, offset+len(frames), layers, versions(self.weights()),
                             versions(memory_tensors(layers)), detach_history)
        return hidden, result

    def export(self, state):
        self.check_cache(state, state.context, state.next_atom, state.detached)
        check(state.detached, 'PREFIX_STREAM_CHECKPOINT_REQUIRES_DETACHED')
        layers = [[dict(key=b.key.detach().cpu().clone(), value=b.value.detach().cpu().clone(),
                        atoms=b.atoms.detach().cpu().clone()) for b in layer] for layer in state.layers]
        result = dict(schema='typed-prefix-cache-v1', weight_pin=self.weight_pin(), layout_pin=digest(self.layout), context=state.context,
                      next_atom=state.next_atom, detached=True, cache_bytes=self.cache_bytes, layers=layers)
        return dict(**result, payload_pin=digest(tensor_fingerprint(result)))

    def restore(self, payload, *, context):
        saved = {k: v for k, v in payload.items() if k != 'payload_pin'}
        check(payload['payload_pin'] == digest(tensor_fingerprint(saved)), 'PREFIX_STREAM_CHECKPOINT_PIN')
        check(saved['schema'] == 'typed-prefix-cache-v1' and saved['weight_pin'] == self.weight_pin()
              and saved['layout_pin'] == digest(self.layout) and self.current_layout() == self.layout
              and saved['context'] == context and saved['detached'] is True
              and saved['cache_bytes'] == self.cache_bytes, 'PREFIX_STREAM_CHECKPOINT_BINDING')
        device = self.encoder.query.device; layers = []; expected_sizes = None
        check(type(saved['next_atom']) is int and saved['next_atom'] > 0
              and len(saved['layers']) == len(self.encoder.blocks), 'PREFIX_STREAM_CHECKPOINT_LAYOUT')
        total_bytes = sum(t.numel()*t.element_size() for layer in saved['layers'] for b in layer for t in b.values())
        check(total_bytes <= self.cache_bytes, 'PREFIX_STREAM_CACHE_MEMORY')
        for layer in saved['layers']:
            blocks = []; cursor = 0; sizes = []
            for row in layer:
                key, value, atoms = row['key'], row['value'], row['atoms']
                check(atoms.ndim == 1 and len(atoms) > 0 and atoms.dtype == torch.int64
                      and key.shape == value.shape == (1, self.encoder.config.heads, len(atoms),
                                                       self.encoder.config.width//self.encoder.config.heads)
                      and key.dtype == value.dtype == torch.float32 and torch.isfinite(key).all()
                      and torch.isfinite(value).all(), 'PREFIX_STREAM_CHECKPOINT_LAYOUT')
                check(int(atoms[0]) == cursor and bool((atoms[1:] >= atoms[:-1]).all())
                      and bool((atoms[1:]-atoms[:-1] <= 1).all()), 'PREFIX_STREAM_CHECKPOINT_ORDER')
                cursor = int(atoms[-1])+1; sizes.append(atoms.tolist())
                blocks.append(MemoryBlock(key.to(device).clone(), value.to(device).clone(), atoms.to(device).clone()))
            check(cursor == saved['next_atom'] and (expected_sizes is None or sizes == expected_sizes),
                  'PREFIX_STREAM_CHECKPOINT_COVERAGE')
            expected_sizes = sizes; layers.append(tuple(blocks))
        layers = tuple(layers)
        return PrefixCache(id(self), context, saved['next_atom'], layers, versions(self.weights()),
                           versions(memory_tensors(layers)), True)
