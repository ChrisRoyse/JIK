"""Slot-conditioned surface decoder (§25.5), not the §25.3 meaning predictor.

Native memories remain separately keyed. Only language rendering integrates them.
This component is engineering-only until the full model/admission pipeline exists.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from senses.contract import require
from model.contracts import MIN_INSTRUMENTS, parameter_count, validate_bank
from model.attention import Attention, Linear


def positions(length, width, device):
    phase = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    phase = phase * torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000) / width))
    out = torch.empty((length, width), device=device)
    out[:, 0::2], out[:, 1::2] = phase.sin(), phase.cos()
    return out


class CrossBranch(nn.Module):
    def __init__(self, config, gain):
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, h, memory, native):
        active = ~native.absent
        if not bool(active.any()):
            return h
        # All-masked attention produces NaNs. Absent rows never enter a branch.
        q = self.norm(h[active])
        update = self.attention(q, memory[active], padding=native.padding[active])
        delta = torch.zeros_like(h)
        delta[active] = update * self.gain
        return h + delta


class DecoderBlock(nn.Module):
    def __init__(self, config, specs):
        super().__init__()
        h = config.width
        self.self_attention = Attention(config)
        self.norm = nn.LayerNorm(h)
        self.ff_norm = nn.LayerNorm(h)
        self.ff = nn.Sequential(Linear(h, 4 * h), nn.GELU(), Linear(4 * h, h))
        self.branches = nn.ModuleDict({s.name: CrossBranch(config, 1) for s in specs})

    def forward(self, h, memories, native, order):
        q = self.norm(h)
        h = h + self.self_attention(q, q, causal=True)
        for name in order:
            h = self.branches[name](h, memories[name], native[name])
        return h + self.ff(self.ff_norm(h))


class SensoryLanguageModel(nn.Module):
    def __init__(self, config, specs):
        super().__init__()
        config.validate()
        validate_bank(specs)
        require(parameter_count(config, specs) <= config.max_parameters, "MNLM_MODEL_PARAMETER_BUDGET",
                "model exceeds declared parameter budget before allocation")
        self.config, self.specs = config, tuple(specs)
        self.projectors = nn.ModuleDict({s.name: self.projector(s) for s in specs})
        self.tokens = nn.Embedding(config.vocabulary, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config, specs) for _ in range(config.layers)])
        self.output = Linear(config.width, config.vocabulary)
        require(sum(p.numel() for p in self.parameters()) == parameter_count(config, specs),
                "MNLM_MODEL_COST_DRIFT", "parameter accounting changed; repair preflight")

    def projector(self, spec):
        if spec.layout == "sparse":
            return nn.Embedding(spec.dims, self.config.width)
        return Linear(spec.dims, self.config.width)

    def validate_inputs(self, native):
        require(set(native) == {s.name for s in self.specs}, "MNLM_MODEL_ROSTER",
                "every registered instrument must supply an explicit typed input, including absence")
        rows, count, channels = None, None, {}
        total = 0
        for spec in self.specs:
            x = native[spec.name]
            x.check(spec, self.config)
            require(rows is None or rows == x.rows, "MNLM_MODEL_ROWS", "instrument observations are misaligned")
            require(x.values.device == self.tokens.weight.device, "MNLM_MODEL_DEVICE", "move the whole batch explicitly")
            rows = x.rows
            total += x.values.numel()
            present = ~x.absent
            count = present.long() if count is None else count + present.long()
            channels[spec.channel] = present if spec.channel not in channels else channels[spec.channel] | present
        require(total <= self.config.max_native_elements, "MNLM_MODEL_INPUT_BUDGET", "total native batch exceeds budget")
        require(bool((count > 0).all()), "MNLM_MODEL_NO_OBSERVATION", "all instruments absent for an observation")
        return count, torch.stack(list(channels.values())).sum(0)

    def memories(self, native):
        counts = self.validate_inputs(native)
        result = {}
        for spec in self.specs:
            x = native[spec.name]
            projected = self.projectors[spec.name](x.indices) * x.values if spec.layout == "sparse" else self.projectors[spec.name](x.values)
            # Sparse coordinates carry identity; they are not a token sequence.
            if spec.layout == "multi":
                projected = projected + positions(projected.shape[1], self.config.width, projected.device)
            result[spec.name] = projected.masked_fill(x.padding[..., None], 0)
        return result, counts

    def forward(self, prefix, native):
        memories, (instruments, channels) = self.memories(native)
        require(prefix.dtype == torch.int64 and prefix.ndim == 2 and prefix.shape[0] == len(next(iter(native.values())).rows)
                and 0 < prefix.shape[1] <= self.config.max_prefix and prefix.device == self.tokens.weight.device
                and bool(((prefix >= 0) & (prefix < self.config.vocabulary)).all()),
                "MNLM_MODEL_PREFIX", "explicit aligned causal token prefix within vocabulary/budget required")
        h = self.tokens(prefix) + positions(prefix.shape[1], self.config.width, prefix.device)
        for block in self.blocks:
            h = block(h, memories, native, [s.name for s in self.specs])
        logits = self.output(h)
        require(bool(torch.isfinite(logits).all()), "MNLM_MODEL_NONFINITE", "model state produced nonfinite logits")
        return {"logits": logits, "instrument_count": instruments, "channel_count": channels,
                "meaning_asserted": False, "purpose": "engineering_surface_decoder"}

    def assert_meaning(self, native):
        count, _ = self.validate_inputs(native)
        require(bool((count >= MIN_INSTRUMENTS).all()),
                "MNLM_MODEL_MEANING_COVERAGE", "meaning requires five present admitted instruments for every atom")
        require(False, "MNLM_MODEL_UNCOMMISSIONED", "wire accepted panel, predictor and guard before asserting meaning")

    def extend(self, spec, *, freeze_existing=True):
        specs = (*self.specs, spec)
        validate_bank(specs)
        require(parameter_count(self.config, specs) <= self.config.max_parameters,
                "MNLM_MODEL_PARAMETER_BUDGET", "expanded bank exceeds declared budget")
        # Prepare first: a failed allocation must not freeze or mutate the old model.
        device = self.tokens.weight.device
        projection = self.projector(spec).to(device)
        branches = [CrossBranch(self.config, 0).to(device) for _ in self.blocks]
        if freeze_existing:
            self.requires_grad_(False)
        self.projectors[spec.name] = projection
        for block, branch in zip(self.blocks, branches):
            block.branches[spec.name] = branch
        self.specs = specs
        # Caller must rebuild optimizer; training_step enforces exact membership.


def training_step(model, optimizer, prefix, targets, native):
    actual = [p for g in optimizer.param_groups for p in g["params"]]
    expected = [p for p in model.parameters() if p.requires_grad]
    require(len(actual) == len(expected) and {id(p) for p in actual} == {id(p) for p in expected},
            "MNLM_MODEL_STALE_OPTIMIZER", "rebuild optimizer for the current trainable bank before training")
    require(targets.shape == prefix.shape and targets.dtype == torch.int64 and targets.device == prefix.device
            and bool(((targets == -100) | ((targets >= 0) & (targets < model.config.vocabulary))).all())
            and bool((targets != -100).any()), "MNLM_MODEL_TARGET", "aligned nonempty target tokens required")
    optimizer.zero_grad(set_to_none=True)
    prediction = model(prefix, native)
    loss = F.cross_entropy(prediction["logits"].reshape(-1, model.config.vocabulary), targets.reshape(-1), ignore_index=-100)
    require(bool(torch.isfinite(loss)), "MNLM_MODEL_NONFINITE", "nonfinite training loss")
    loss.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in expected),
            "MNLM_MODEL_NONFINITE", "nonfinite gradient; optimizer not advanced")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in expected), "MNLM_MODEL_NONFINITE",
            "optimizer produced nonfinite state; reject this state and restore the preceding checkpoint")
    return float(loss.detach())
