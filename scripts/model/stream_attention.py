"""Exact blockwise native observation attention, with bounded GPU activations.

Native arrays stay frozen and separately typed at the caller. This only computes
derived model state. PC-07/16/35: project inside a recomputed block; never retain
the whole projected sequence on GPU or relax an allocation limit.
"""
from dataclasses import dataclass
import math

import torch
from torch.utils.checkpoint import checkpoint

from senses.contract import require

BLOCK_ROWS = 256


@dataclass(frozen=True)
class NativeBlock:
    offset: int
    values: torch.Tensor
    padding: torch.Tensor
    positions: torch.Tensor | None = None


def blocks(values, padding, config, *, positions=None):
    """Views preserve global row offsets; no projection, pooling or truncation."""
    width = min(BLOCK_ROWS, config.max_native_tokens)
    for offset in range(0, values.shape[1], width):
        yield NativeBlock(offset, values[:, offset:offset+width],
                          padding[:, offset:offset+width],
                          None if positions is None else positions[:, offset:offset+width])


def observe(attention, projector, query, stream, *, counts, config, positioner=None):
    """Match Attention(query, projector(native)) without a full GPU memory bank.

Counts and offsets bind complete ordered coverage, including per-row padding.
The caller binds the immutable source/identity/window before constructing stream.
CPU block inputs remain on CPU in the backward graph. Checkpoint recomputation
uses the same projector and attention parameters as the original forward.
"""
    require(query.ndim == 3 and query.dtype == torch.float32
            and query.shape[2] == config.width and bool(torch.isfinite(query).all()),
            "MNLM_STREAM_QUERY", "finite typed fp32 query required")
    b, length, width = query.shape
    require(b > 0 and 0 < length <= config.max_prefix
            and query.numel() <= config.max_native_elements,
            "MNLM_STREAM_QUERY_BUDGET", "query exceeds fixed allocation limits")
    require(type(counts) is tuple and len(counts) == b
            and all(type(n) is int and n > 0 for n in counts),
            "MNLM_STREAM_COUNTS", "declare every nonempty native row length")
    require(attention.width == width and attention.heads == config.heads
            and projector.out_features == width
            and projector.weight.device == attention.q.weight.device == query.device,
            "MNLM_STREAM_MODULE", "projection and attention must share width and device")
    heads, depth = attention.heads, width//attention.heads
    q = attention.q(query).reshape(b, length, heads, depth).transpose(1, 2)
    combined = query.new_zeros((b, heads, length, depth))
    log_mass = query.new_zeros((b, heads, length))
    seen = torch.zeros(b, dtype=torch.bool, device=query.device)
    cursor = 0

    def partial(q, native, padding, positions):
        # No randomness, mutable loop closure or native gradients in recomputation.
        memory = projector(native.to(query.device))
        if positioner is not None:
            memory = memory + positioner(positions.to(query.device))
        pad = padding.to(query.device)
        active = (~pad).any(1)
        n = memory.shape[1]
        k = attention.k(memory).reshape(b, n, heads, depth).transpose(1, 2)
        v = attention.v(memory).reshape(b, n, heads, depth).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1))/math.sqrt(depth)
        scores = scores.masked_fill(pad[:, None, None, :], -torch.inf)
        # Empty trailing blocks for a shorter row never enter an all-inf softmax.
        safe = torch.where(active[:, None, None, None], scores, torch.zeros_like(scores))
        mass = torch.logsumexp(safe, -1)
        # Use the same stabilized softmax primitive as the dense path. Rebuilding
        # it as exp(score-logsumexp(score)) loses gradient precision on large logits.
        value = safe.softmax(-1) @ v
        value = torch.where(active[:, None, None, None], value, torch.zeros_like(value))
        return value, mass, active

    for block in stream:
        require(isinstance(block, NativeBlock) and type(block.offset) is int
                and block.offset == cursor, "MNLM_STREAM_ORDER",
                "missing, repeated or reordered native block")
        native, padding = block.values, block.padding
        require(native.ndim == 3 and native.shape[0] == b
                and native.shape[2] == projector.in_features and native.dtype == torch.float32
                and not native.requires_grad, "MNLM_STREAM_NATIVE", "frozen native layout required")
        tokens = native.shape[1]
        require(0 < tokens <= min(BLOCK_ROWS, config.max_native_tokens)
                and native.numel() <= config.max_native_elements
                and b*heads*length*tokens <= config.max_attention_elements,
                "MNLM_STREAM_BLOCK_BUDGET", "block exceeds fixed allocation limits")
        require(cursor+tokens <= max(counts), "MNLM_STREAM_COVERAGE", "extra native rows")
        require(padding.dtype == torch.bool and padding.shape == native.shape[:2]
                and padding.device == native.device, "MNLM_STREAM_PADDING", "aligned native mask required")
        expected = (torch.arange(cursor, cursor+tokens, device=native.device)[None]
                    >= torch.tensor(counts, device=native.device)[:, None])
        require(torch.equal(padding, expected) and bool(torch.isfinite(native).all())
                and not bool(native[padding].any()), "MNLM_STREAM_VALUES",
                "native prefix/mask mismatch, nonfinite value or nonzero padding")
        positions = block.positions
        require((positioner is None) == (positions is None), "MNLM_STREAM_POSITION",
                "temporal coordinates and their learned projection must travel together")
        if positioner is not None:
            require(positions.shape == (*padding.shape, positioner.in_features)
                    and positions.dtype == torch.float32 and not positions.requires_grad
                    and positions.device == native.device and bool(torch.isfinite(positions).all())
                    and not bool(positions[padding].any())
                    and positioner.out_features == width
                    and positioner.weight.device == query.device,
                    "MNLM_STREAM_POSITION", "invalid temporal coordinates or projection")
        if torch.is_grad_enabled():
            value, mass, active = checkpoint(partial, q, native, padding, positions,
                use_reentrant=False, preserve_rng_state=False)
        else:
            value, mass, active = partial(q, native, padding, positions)
        both = (seen & active)[:, None, None]
        weight = torch.where(seen[:, None, None],
            torch.where(active[:, None, None], torch.sigmoid(log_mass-mass), 1.), 0.)
        combined = weight[..., None]*combined + (1-weight[..., None])*value
        log_mass = torch.where(both, torch.logaddexp(log_mass, mass),
                               torch.where(active[:, None, None], mass, log_mass))
        seen = seen | active
        cursor += tokens
    require(cursor == max(counts) and bool(seen.all()), "MNLM_STREAM_COVERAGE",
            "incomplete native stream; no partial observation is returned")
    result = attention.out(combined.transpose(1, 2).contiguous().reshape(b, length, width))
    require(bool(torch.isfinite(result).all()), "MNLM_STREAM_NONFINITE",
            "reject nonfinite derived model state")
    return result
