"""Exact symmetric MaxSim, tiled on both token axes (PC-04/13).

Work O(Ta*Tb*d); device storage O(chunk**2 + chunk*d + Tb). CPU is only an
explicit reference diagnostic, never a production fallback. Source slots stay Multi.
"""
from __future__ import annotations

import numpy as np


def validate_multi(tokens, offsets, absent):
    if (tokens.ndim != 2 or absent.ndim != 1 or absent.dtype != np.bool_
            or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) != len(absent) + 1 or offsets[0] != 0
            or offsets[-1] != len(tokens) or np.any(offsets[1:] < offsets[:-1])
            or not np.array_equal(np.diff(offsets) == 0, absent)):
        raise ValueError("MNLM_SLOT_LAYOUT: Multi offsets/absence do not match tokens")


def maxsim_pair(a, b, *, device="cuda", chunk=1024):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b) or chunk < 1:
        raise ValueError("MNLM_SLOT_LAYOUT: MaxSim requires nonempty compatible token matrices")
    if device == "cpu":  # deliberately requested reference diagnostic
        bm = np.full(len(b), -np.inf, np.float32)
        total = 0.0
        for i in range(0, len(a), chunk):
            aa = np.asarray(a[i:i + chunk], np.float32)
            am = np.full(len(aa), -np.inf, np.float32)
            for j in range(0, len(b), chunk):
                s = aa @ np.asarray(b[j:j + chunk], np.float32).T
                if not np.isfinite(s).all():
                    raise ValueError("MNLM_SLOT_NONFINITE: nonfinite MaxSim score")
                am = np.maximum(am, s.max(1))
                bm[j:j + s.shape[1]] = np.maximum(bm[j:j + s.shape[1]], s.max(0))
            total += am.sum(dtype=np.float64)
        return float(0.5 * (total / len(a) + bm.mean(dtype=np.float64)))
    if device != "cuda":
        raise ValueError("MNLM_ASSAY_NO_GPU: unsupported MaxSim runtime")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MNLM_ASSAY_NO_GPU: MaxSim requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    bm = torch.full((len(b),), -torch.inf, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for i in range(0, len(a), chunk):
            aa = torch.tensor(np.asarray(a[i:i + chunk]), dtype=torch.float32, device=device)
            am = torch.full((len(aa),), -torch.inf, device=device)
            for j in range(0, len(b), chunk):
                bb = torch.tensor(np.asarray(b[j:j + chunk]), dtype=torch.float32, device=device)
                s = aa @ bb.T
                if not torch.isfinite(s).all().item():
                    raise ValueError("MNLM_SLOT_NONFINITE: nonfinite MaxSim score")
                am = torch.maximum(am, s.max(1).values)
                bm[j:j + len(bb)] = torch.maximum(bm[j:j + len(bb)], s.max(0).values)
            total += am.sum(dtype=torch.float64)
        return float((0.5 * (total / len(a) + bm.double().mean())).item())
