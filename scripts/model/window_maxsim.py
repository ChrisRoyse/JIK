"""Exact per-reference MaxSim reduction with bounded score tiles (PC-16).

Every query/reference window participates. Checkpointing recomputes the score
tile during backward; it never stores the quadratic score volume across tiles.
The reference axis is reduced in one amax so tied maxima retain its gradient.
"""
import torch
from torch.utils.checkpoint import checkpoint

from senses.contract import require


def tile(query, keys, padding):
    scores = torch.einsum("bqh,rwh->bqrw", query, keys)*10.
    return scores.masked_fill(padding[None, None], -torch.inf).amax(-1)


def maxsim(query, keys, padding, config):
    b, q, h = query.shape
    r, w, depth = keys.shape
    require(h == depth and padding.shape == (r, w) and padding.dtype == torch.bool
            and query.device == keys.device == padding.device
            and b*q*r <= config.max_attention_elements and not bool(padding.all(1).any()),
            "MNLM_WINDOW_MAXSIM_SHAPE", "aligned nonempty reference regions required")
    step = min(256, config.max_attention_elements//(b*r*w))
    require(step > 0, "MNLM_WINDOW_MAXSIM_BUDGET",
            "reference roster exceeds fixed tile budget; choose a smaller proof corpus")
    if b*q*r*w <= config.max_attention_elements:
        return tile(query, keys, padding)
    result = []
    for start in range(0, q, step):
        part = query[:, start:start+step]
        if torch.is_grad_enabled() and (query.requires_grad or keys.requires_grad):
            value = checkpoint(tile, part, keys, padding, use_reentrant=False)
        else:
            value = tile(part, keys, padding)
        result.append(value)
    return torch.cat(result, dim=1)
