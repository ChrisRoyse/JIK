"""Small dual projections of parameter displacements; no native vectors are pooled."""
import numpy as np
import torch
from scipy.optimize import nnls

from claims.codata_source import check


def project(gradients, displacement):
    check(gradients.ndim == 2 and displacement.ndim == 1
          and gradients.shape[1] == len(displacement)
          and bool(torch.isfinite(gradients).all()) and bool(torch.isfinite(displacement).all()),
          "RETENTION_PROJECTION_INPUT")
    g, d = gradients.double(), displacement.double()
    norm = torch.linalg.vector_norm(g, dim=1)
    keep = norm > 0
    g = g[keep]/norm[keep, None]
    gram = (g@g.T).cpu().numpy()
    dot = (g@d).cpu().numpy()
    report = dict(rows=len(gradients), active_rows=int(keep.sum()), gram=gram.tolist(),
                  dot=dot.tolist(), proposal_norm=float(torch.linalg.vector_norm(d)))
    if not len(dot) or np.all(dot <= 0):
        return d, dict(report, status="already_descent", dual=[], scale=1.)
    bound = -np.abs(dot)
    rhs = dot-bound
    eigen, vectors = np.linalg.eigh(gram)
    rank = eigen > np.finfo(np.float64).eps*len(dot)*max(float(eigen.max()), 1.)
    basis = vectors[:, rank]
    residual = rhs-basis@(basis.T@rhs)
    tol = 64*np.finfo(np.float64).eps*max(1., float(np.linalg.norm(rhs)))
    report.update(bound=bound.tolist(), tolerance=tol, rank=int(rank.sum()))
    if not rank.any() or np.linalg.norm(residual) > tol:
        return None, dict(report, status="infeasible_rank", residual=float(np.linalg.norm(residual)))
    factors = np.sqrt(eigen[rank])[:, None]*basis.T
    target = (basis.T@rhs)/np.sqrt(eigen[rank])
    try:
        dual, _ = nnls(factors, target, maxiter=3*len(dot))
    except RuntimeError:
        return None, dict(report, status="solver_iteration_limit")
    after = dot-gram@dual
    tol = 64*np.finfo(np.float64).eps*max(
        1., float(np.linalg.norm(gram, np.inf)*np.linalg.norm(dual, np.inf)+np.linalg.norm(dot, np.inf)))
    report["tolerance"] = tol
    # These numerical checks validate the solve, not the native retention gate.
    if np.max(after-bound) > tol or np.max(np.abs(dual*(after-bound))) > tol:
        return None, dict(report, status="infeasible_dual", dual=dual.tolist(), projected_dot=after.tolist())
    projected = d-g.T@torch.from_numpy(dual).to(d.device)
    length = float(torch.linalg.vector_norm(projected))
    scale = min(1., report["proposal_norm"]/length) if length else 1.
    return projected*scale, dict(report, status="projected", dual=dual.tolist(),
                                projected_dot=after.tolist(), projected_norm=length, scale=scale)
