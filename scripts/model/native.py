"""Preserve Dense/Sparse/Multi/Absent, with explicit row identity and resource bounds."""
from dataclasses import dataclass

import numpy as np
import torch

from senses.contract import require
from slot_layout import validate


@dataclass(frozen=True)
class NativeInput:
    identity: str
    rows: tuple[str, ...]
    values: torch.Tensor                 # B x native tokens x native dimensions
    padding: torch.Tensor                # B x native tokens; True means no token
    absent: torch.Tensor                 # B; absence is not a zero measurement
    indices: torch.Tensor | None = None  # sparse vocabulary coordinates only

    def to(self, device):
        return NativeInput(self.identity, self.rows, self.values.to(device), self.padding.to(device),
                           self.absent.to(device), None if self.indices is None else self.indices.to(device))

    def check(self, spec, config):
        b, v, p, a = len(self.rows), self.values, self.padding, self.absent
        require(self.identity == spec.identity, "MNLM_MODEL_IDENTITY", f"{spec.name}: instrument version mismatch")
        require(b > 0 and len(set(self.rows)) == b and all(isinstance(x, str) and x for x in self.rows),
                "MNLM_MODEL_ROWS", "unique explicit observation identities required")
        d = 1 if spec.layout == "sparse" else spec.dims
        require(v.ndim == 3 and v.shape[0] == b and v.shape[2] == d and 0 < v.shape[1] <= config.max_native_tokens
                and v.numel() <= config.max_native_elements and v.dtype == torch.float32,
                "MNLM_MODEL_SHAPE", f"{spec.name}: wrong shape/dtype or native budget exceeded")
        require(p.shape == v.shape[:2] and p.dtype == torch.bool and a.shape == (b,) and a.dtype == torch.bool
                and p.device == v.device == a.device, "MNLM_MODEL_MASK", "mask shape/dtype/device mismatch")
        require(torch.equal(a, p.all(1)) and not bool((p[:, :-1] & ~p[:, 1:]).any()),
                "MNLM_MODEL_MASK", "absence must agree with a contiguous native token prefix")
        require(not v.requires_grad and bool(torch.isfinite(v).all()) and not bool(v[p].any()),
                "MNLM_MODEL_VALUES", "frozen finite native values and zero-only padding required")
        if spec.layout == "dense":
            require(v.shape[1] == 1, "MNLM_MODEL_SHAPE", "Dense is exactly one native vector")
        if spec.layout == "sparse":
            ix = self.indices
            require(ix is not None and ix.shape == p.shape and ix.dtype == torch.int64 and ix.device == v.device
                    and bool(((ix >= 0) & (ix < spec.dims)).all()) and not bool(ix[p].any()),
                    "MNLM_MODEL_SPARSE", "retain sparse coordinates; no densification or invalid indices")
        else:
            require(self.indices is None, "MNLM_MODEL_LAYOUT", "dense/Multi cannot carry sparse coordinates")


def from_arrays(spec, arrays, rows, config):
    shape = validate(arrays, spec.layout)
    require(shape["n"] == len(rows) and shape["dims"] == spec.dims,
            "MNLM_MODEL_SHAPE", "native storage differs from instrument contract")
    n, d = len(rows), 1 if spec.layout == "sparse" else spec.dims
    if spec.layout == "dense":
        lengths = (~arrays["absent"]).astype(np.int64)
        offsets = None
    else:
        offsets = arrays["indptr" if spec.layout == "sparse" else "offsets"]
        lengths = np.diff(offsets)
    width = max(1, int(lengths.max()))
    require(width <= config.max_native_tokens and n * width * d <= config.max_native_elements,
            "MNLM_MODEL_INPUT_BUDGET", "native input exceeds declared budget; choose a smaller batch, never truncate")
    values = np.zeros((n, width, d), np.float32)
    padding = np.ones((n, width), bool)
    indices = np.zeros((n, width), np.int64) if spec.layout == "sparse" else None
    for i, length in enumerate(lengths):
        length = int(length)
        padding[i, :length] = False
        if not length:
            continue
        if spec.layout == "dense":
            values[i, 0] = arrays["X"][i]
        else:
            lo, hi = offsets[i:i + 2]
            if spec.layout == "sparse":
                indices[i, :length] = arrays["indices"][lo:hi]
                values[i, :length, 0] = arrays["data"][lo:hi]
            else:
                values[i, :length] = arrays["tokens"][lo:hi]
    result = NativeInput(spec.identity, tuple(rows), torch.from_numpy(values), torch.from_numpy(padding),
                         torch.from_numpy(arrays["absent"].copy()), None if indices is None else torch.from_numpy(indices))
    result.check(spec, config)
    return result
