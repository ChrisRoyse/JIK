"""The Stage 3 assay estimator, ported from calyx-assay (C:\\code\\Calyx @ fe1d796a, crates/calyx-assay/src/ksg/mixed_ci.rs,
projection.rs, subsample.rs; issue #12 records the port decision). Mutual information between a continuous slot and a
discrete anchor by the Ross mixed continuous-discrete KSG estimator with Chebyshev radii (spec Def. 13.2): per sample,
the radius to the k-th same-class neighbour, the same-class count within it (ties included), the count of all samples
within it, and bits = mean[ψ(n) + ψ(same) − ψ(class) − ψ(all)] / ln 2, clamped at 0. The interval is the m-out-of-n
no-replacement subsampling root interval of calyx-assay: m is the smallest in [max(⌈n^(2/3)⌉, k+2), n/2] whose
hypergeometric class-support failure probability is ≤ 0.01, 999 draws (seed 0, numpy PCG64 here; ChaCha8 in Rust) each
rejected up to 8 times until every class keeps more than k members, roots √(m/(1−m/n))·(sub − point), ci = point −
q(0.975)/√n .. point − q(0.025)/√n, hulled around the point and clamped at 0. High-dimensional slots are first reduced by
the deterministic BLAKE3 sign projection to min(d, 2⌈log₂ n⌉) dimensions and each projected dimension is z-scored (the
Chebyshev norm is scale-sensitive; the standardisation is part of the method and the FSV repeats it). Distances run on
the GPU with torch when available; the arithmetic is identical on CPU and the device is recorded."""
from __future__ import annotations

import math

import blake3
import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln

K = 3
RESAMPLES = 999
CI_SEED = 0
MAX_SUPPORT_FAILURE = 0.01
MAX_DRAW_ATTEMPTS = 8
MIN_SAMPLES = 50


class AssayError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ---------------------------------------------------------------- projection
def target_dim(rows: int, input_dim: int) -> int:
    if rows <= 1:
        return min(input_dim, 1)
    return min(input_dim, max(1, 2 * math.ceil(math.log2(rows))))


def sign_matrix(seed: int, input_dim: int, output_dim: int) -> np.ndarray:
    S = np.empty((input_dim, output_dim), np.float32)
    for i in range(input_dim):
        for o in range(output_dim):
            h = blake3.blake3(seed.to_bytes(8, "big") + i.to_bytes(8, "big") + o.to_bytes(8, "big")).digest()
            S[i, o] = 1.0 if h[0] & 1 == 0 else -1.0
    return S


_SIGN_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def project(X, seed: int, rows: int | None = None) -> np.ndarray:
    """X dense [n,d] or CSR -> [n, target] float32, then z-scored per dimension (a constant dimension stays 0)."""
    n, d = X.shape
    out_dim = target_dim(rows or n, d)
    key = (seed, d, out_dim)
    if key not in _SIGN_CACHE:
        _SIGN_CACHE[key] = sign_matrix(seed, d, out_dim)
    S = _SIGN_CACHE[key]
    P = (X @ S) if not sp.issparse(X) else np.asarray(X @ S)
    P = np.asarray(P, np.float64) / math.sqrt(out_dim)
    mu, sd = P.mean(0), P.std(0)
    P = np.where(sd > 0, (P - mu) / np.where(sd > 0, sd, 1), 0.0)
    return P.astype(np.float32)


# ---------------------------------------------------------------- the estimator
def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "numpy"


DEVICE = _device()


def _cheb(X: np.ndarray):
    """Pairwise Chebyshev distances [n,n] as a torch tensor on DEVICE (or numpy)."""
    if DEVICE == "numpy":
        return np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
    import torch

    T = torch.as_tensor(X, device=DEVICE, dtype=torch.float32)
    return torch.cdist(T, T, p=float("inf"))


def mixed_bits(X: np.ndarray, labels: np.ndarray, k: int = K, D=None) -> float:
    """Ross mixed KSG point estimate in bits (clamped at 0). D: precomputed Chebyshev matrix."""
    n = len(labels)
    if D is None:
        D = _cheb(X)
    if DEVICE == "numpy":
        return _mixed_numpy(D, labels, k)
    import torch

    L = torch.as_tensor(labels, device=D.device)
    same = L[:, None] == L[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=D.device)
    Ds = torch.where(same & ~eye, D, torch.full_like(D, float("inf")))
    radius = torch.kthvalue(Ds, k, dim=1).values
    within = D <= radius[:, None]
    same_count = (within & same & ~eye).sum(1)
    full_count = (within & ~eye).sum(1)
    class_count = torch.bincount(L)[L]
    nf = torch.tensor(float(n), device=D.device)
    local = torch.special.digamma(nf) + torch.special.digamma(same_count.float()) - torch.special.digamma(class_count.float()) - torch.special.digamma(full_count.float())
    return max(0.0, float(local.mean().item() / math.log(2)))


def _mixed_numpy(D: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = len(labels)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    Ds = np.where(same, D, np.inf)
    radius = np.partition(Ds, k - 1, axis=1)[:, k - 1]
    within = D <= radius[:, None]
    np.fill_diagonal(within, False)
    same_count = (within & same).sum(1)
    full_count = within.sum(1)
    class_count = np.bincount(labels)[labels]
    local = digamma(n) + digamma(same_count) - digamma(class_count) - digamma(full_count)
    return max(0.0, float(local.mean() / math.log(2)))


def degenerate(D, labels: np.ndarray, k: int = K) -> int:
    """Number of samples whose k-th same-class radius is zero (exact same-class duplicates >= k); > 0 refuses."""
    if DEVICE == "numpy":
        same = labels[:, None] == labels[None, :]
        np.fill_diagonal(same, False)
        return int(((D == 0) & same).sum(1).__ge__(k).sum())
    import torch

    L = torch.as_tensor(labels, device=D.device)
    same = (L[:, None] == L[None, :]) & ~torch.eye(len(labels), dtype=torch.bool, device=D.device)
    return int((((D == 0) & same).sum(1) >= k).sum().item())


# ---------------------------------------------------------------- the interval
def _ln_binom(n: int, k: int) -> float:
    return -math.inf if k > n else gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def hypergeom_cdf_at_most(population: int, successes: int, draws: int, threshold: int) -> float:
    first, last = max(0, draws - (population - successes)), min(threshold, successes, draws)
    if first > last:
        return 0.0
    den = _ln_binom(population, draws)
    terms = [_ln_binom(successes, s) + _ln_binom(population - successes, draws - s) - den for s in range(first, last + 1)]
    mx = max(terms)
    return min(1.0, max(0.0, math.exp(mx) * sum(math.exp(t - mx) for t in terms)))


def subsample_plan(n: int, class_counts: dict[int, int], k: int = K) -> tuple[int, float]:
    m_min, m_max = max(math.ceil(n ** (2 / 3)), k + 2), n // 2
    if m_min > m_max:
        raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"no half-sample range: n={n} m_min={m_min} m_max={m_max}")
    for m in range(m_min, m_max + 1):
        fail = min(1.0, sum(hypergeom_cdf_at_most(n, c, m, k) for c in class_counts.values()))
        if fail <= MAX_SUPPORT_FAILURE:
            return m, fail
    raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"no m in {m_min}..{m_max} with support failure <= {MAX_SUPPORT_FAILURE}: n={n} classes={class_counts}")


def percentile_index(length: int, p: float) -> int:
    return min(length - 1, max(0, int(math.floor(p * (length - 1) + 0.5))))


def estimate(X: np.ndarray, labels: np.ndarray, k: int = K, resamples: int = RESAMPLES, seed: int = CI_SEED, groups: np.ndarray | None = None) -> dict:
    """Point estimate with the subsampling interval. Refuses typed: insufficient samples, a class with <= k members,
    degenerate same-class duplicates. With `groups` (the row's entity/event cluster, doctrine #1: entity-clustered) a draw takes
    whole clusters in random order until it holds at least m rows, and each root is scaled by its own draw size."""
    n = len(labels)
    labels = np.asarray(labels, np.int64)
    if n < MIN_SAMPLES or len(X) != n:
        raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"n={n} (need >= {MIN_SAMPLES} paired samples)")
    counts = {int(c): int(v) for c, v in zip(*np.unique(labels, return_counts=True))}
    if any(v <= k for v in counts.values()):
        raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"a class has <= k={k} members: {counts}")
    D = _cheb(X)
    dg = degenerate(D, labels, k)
    if dg:
        raise AssayError("MNLM_ASSAY_DEGENERATE", f"{dg} samples have >= {k} exact same-class duplicates")
    point = mixed_bits(X, labels, k, D)
    m, fail = subsample_plan(n, counts, k)
    rng = np.random.default_rng(seed)
    clustered = groups is not None and len(np.unique(groups)) < n
    if clustered:
        _, ginv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(ginv == c) for c in range(ginv.max() + 1)]
    roots, rejected = [], 0
    for _ in range(resamples):
        for attempt in range(MAX_DRAW_ATTEMPTS):
            if clustered:
                take, tot = [], 0
                for c in rng.permutation(len(members)):
                    take.append(members[c])
                    tot += len(members[c])
                    if tot >= m:
                        break
                idx = np.concatenate(take)
            else:
                idx = rng.permutation(n)[:m]
            sub = labels[idx]
            c = np.bincount(sub, minlength=labels.max() + 1)
            if len(np.unique(sub)) == len(counts) and all(c[cl] > k for cl in counts):
                break
            rejected += 1
        else:
            raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"class-support rejection exhausted {MAX_DRAW_ATTEMPTS} draws at m={m}")
        Dsub = D[idx][:, idx] if DEVICE == "numpy" else D[idx][:, idx]
        mi = len(idx)
        if mi >= n:
            raise AssayError("MNLM_ASSAY_INSUFFICIENT_SAMPLES", f"a cluster draw of {mi} rows covers the population (n={n}, m={m}); clusters too large for subsampling")
        roots.append(math.sqrt(mi / (1 - mi / n)) * (mixed_bits(X[idx], sub, k, Dsub) - point))
    roots.sort()
    lo_root, hi_root = roots[percentile_index(len(roots), 0.025)], roots[percentile_index(len(roots), 0.975)]
    ci_low, ci_high = point - hi_root / math.sqrt(n), point - lo_root / math.sqrt(n)
    return {"bits": point, "ci_low": max(0.0, min(ci_low, point)), "ci_high": max(ci_high, point, 0.0), "n": n, "k": k, "m": m, "support_failure_upper": fail,
            "resamples": resamples, "rejected_draws": rejected, "device": DEVICE, "classes": counts, "clustered": bool(clustered)}


def point_only(X: np.ndarray, labels: np.ndarray, k: int = K) -> float:
    """Point estimate without the interval (the greedy conditional step); refuses the same degeneracies."""
    labels = np.asarray(labels, np.int64)
    D = _cheb(X)
    if degenerate(D, labels, k):
        raise AssayError("MNLM_ASSAY_DEGENERATE", "exact same-class duplicates >= k in the joint space")
    return mixed_bits(X, labels, k, D)
