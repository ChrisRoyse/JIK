"""Explicit attention arithmetic, invariant to parameter freeze flags (#25).

The installed CUDA MHA path changed results when unchanged parameters were
frozen. Use one declared projection/matmul path for both regimes; no silent
backend switch. Bound the attention matrix before allocation (PC-16/38).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from senses.contract import require


class Linear(nn.Linear):
    def forward(self, x):
        # Separate bias addition and explicit 2-D layout avoid an epilogue or
        # layout choice depending on whether a parameter requires gradients.
        y = F.linear(x.contiguous().reshape(-1, self.in_features), self.weight, None)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], self.out_features)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width, self.budget = config.heads, config.width, config.max_attention_elements
        self.q = Linear(self.width, self.width)
        self.k = Linear(self.width, self.width)
        self.v = Linear(self.width, self.width)
        self.out = Linear(self.width, self.width)

    def forward(self, query, memory, *, padding=None, causal=False):
        b, length, _ = query.shape
        tokens, depth = memory.shape[1], self.width // self.heads
        require(b * self.heads * length * tokens <= self.budget, "MNLM_MODEL_ATTENTION_BUDGET",
                "attention matrix exceeds fixed budget; reduce batch/prefix, never truncate native observations")
        q = self.q(query).reshape(b, length, self.heads, depth).transpose(1, 2)
        k = self.k(memory).reshape(b, tokens, self.heads, depth).transpose(1, 2)
        v = self.v(memory).reshape(b, tokens, self.heads, depth).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(depth)
        if padding is not None:
            scores = scores.masked_fill(padding[:, None, None, :], -torch.inf)
        if causal:
            require(length == tokens, "MNLM_MODEL_ATTENTION", "causal self attention requires equal query/key length")
            scores = scores.masked_fill(torch.ones((length, tokens), device=query.device, dtype=torch.bool).triu(1), -torch.inf)
        result = torch.matmul(scores.softmax(-1), v).transpose(1, 2).contiguous().reshape(b, length, self.width)
        return self.out(result)
