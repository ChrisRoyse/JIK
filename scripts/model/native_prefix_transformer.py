"""Block-causal attention over complete typed native windows, before identity coupling."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check
from model.attention import Linear
from model.native_identity_world import SensoryIdentityCore


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.width
        self.heads, self.width = config.heads, h
        self.q, self.k, self.v, self.out = (Linear(h, h) for _ in range(4))
        self.first, self.second = nn.LayerNorm(h), nn.LayerNorm(h)
        self.up, self.down = Linear(h, 4*h), Linear(4*h, h)

    def forward(self, x, mask):
        z = self.first(x)
        batch, length, depth = z.shape[0], z.shape[1], self.width//self.heads
        q, k, v = (m(z).reshape(batch, length, self.heads, depth).transpose(1, 2)
                   for m in (self.q, self.k, self.v))
        scores = (q@k.transpose(-1, -2))/math.sqrt(depth)
        weights = scores.masked_fill(mask[None, None], -torch.inf).softmax(-1)
        x = x+self.out((weights@v).transpose(1, 2).contiguous().reshape(batch, length, self.width))
        return x+self.down(F.gelu(self.up(self.second(x))))


class NativePrefixTransformer(nn.Module):
    def __init__(self, config, specs):
        super().__init__()
        self.config, self.specs = config, tuple(specs)
        k, h = len(specs), config.width
        self.kind = nn.Parameter(torch.randn(k, h)*.02)
        self.query = nn.Parameter(torch.randn(k, h)*.02)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(h)
        self.register_buffer("frequency", torch.exp(torch.arange(0, h, 2)*(-math.log(10000.)/h)))

    def position(self, indices):
        phase = indices[:, None]*self.frequency[None]
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(1)

    def encode(self, frames, projectors):
        check(frames and all(set(f) == {s.name for s in self.specs} for f in frames), "SEMANTIC_PREFIX_ROSTER")
        windows = sum(sum(len(x) for x in f.values()) for f in frames)
        elements = sum(x.numel() for f in frames for x in f.values())
        size = windows+len(frames)*len(self.specs)
        check(size <= self.config.max_native_tokens and elements <= self.config.max_native_elements
              and self.config.heads*size*size <= self.config.max_attention_elements, "SEMANTIC_PREFIX_BUDGET")
        device = self.query.device
        positions = self.position(torch.arange(len(frames), device=device))
        tokens, atoms, queries, cursor = [], [], [], 0
        for atom, frame in enumerate(frames):
            row = []
            for lane, spec in enumerate(self.specs):
                values = frame[spec.name]
                check(values.ndim == 2 and values.shape[1] == spec.dims and values.device == device
                      and values.dtype == torch.float32, "SEMANTIC_PREFIX_NATIVE")
                local = self.position(torch.arange(len(values), device=device))
                tokens.append(projectors[spec.name](values)+self.kind[lane]+positions[atom]+local)
                atoms.extend([atom]*len(values))
                cursor += len(values)
                row.append(cursor)
                tokens.append((self.query[lane]+self.kind[lane]+positions[atom])[None])
                atoms.append(atom)
                cursor += 1
            queries.append(row)
        x = torch.cat(tokens, 0)[None]  # typed positions; native vector dimensions never concatenate
        order = torch.tensor(atoms, device=device)
        mask = order[None, :] > order[:, None]
        for block in self.blocks:
            x = block(x, mask)
        selected = torch.tensor(queries, device=device)
        return self.norm(x)[0, selected]

    def forward(self, native, projectors):
        native.validate_prefix()
        return self.encode(native.frames, projectors)[-1:]


class PrefixSensoryCore(SensoryIdentityCore):
    def __init__(self, base):
        nn.Module.__init__(self)
        # Reuse the actual initialized modules. Adding attention does not redraw
        # the recurrent/identity model or its recognition and placement heads.
        for name in ("identity", "initial", "projectors", "temporal_projectors", "observe", "update",
                     "mix", "to_identity", "from_identity", "norm"):
            setattr(self, name, getattr(base, name))
        for name in ("specs", "config", "standard_id", "source_id", "_temporal", "temporal_id"):
            setattr(self, name, getattr(base, name))
        h, k = self.config.width, len(self.specs)
        added = 2*k*h+self.config.layers*(12*h*h+13*h)+2*h
        check(sum(p.numel() for p in base.parameters())+added <= self.config.max_parameters,
              "SEMANTIC_TRANSFORMER_PARAMETER_BUDGET")
        self.sequence_transformer = NativePrefixTransformer(self.config, self.specs)
        check(sum(p.numel() for p in self.sequence_transformer.parameters()) == added,
              "SEMANTIC_TRANSFORMER_PARAMETER_COUNT")

    def observe_native(self, native, prior, phase_value):
        return super().observe_native(native, prior, phase_value)+self.sequence_transformer(native, self.projectors)
