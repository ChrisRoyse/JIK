"""Exact indexed Calyx/Ross counts; no admission or new information estimator.

The reference uses float32 coordinates AND float32 distance subtraction. Trees
use float64 arithmetic. The radius boundary below preserves the reference's
round-to-nearest-even distance ties instead of silently changing its metric.
An index may pack typed coordinate blocks solely to compute their max metric;
no native slot, profile or model input is averaged, projected or overwritten.
PC-16: O(n*d) stored coordinates/counts, no n-by-n distance allocation. Exact
tree queries can still approach O(n^2*d) work in high dimensions.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

K = 3


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_CALYX_INDEX_" + code)


def rounding_boundary(radius):
    """Largest double threshold that rounds to <= radius in float32."""
    r = np.asarray(radius, dtype=np.float32)
    following = np.nextafter(r, np.float32(np.inf))
    need(np.isfinite(r).all() and np.isfinite(following).all() and (r > 0).all(),
         "RADIUS")
    midpoint = (r.astype(np.float64) + following.astype(np.float64)) / 2
    even = (r.view(np.uint32) & 1) == 0
    return np.where(even, midpoint, np.nextafter(midpoint, -np.inf))


def indexed_counts(blocks, labels):
    """Exact same-class, class-size and all-class counts, excluding self."""
    y = np.asarray(labels)
    need(y.ndim == 1 and y.dtype.kind in "iu" and len(y) > K, "LABELS")
    need(isinstance(blocks, (list, tuple)) and len(blocks) > 0, "BLOCKS")
    parts = []
    for block in blocks:
        x = np.asarray(block)
        need(x.ndim == 2 and len(x) == len(y) and x.shape[1] > 0
             and np.isfinite(x).all(), "INPUT")
        with np.errstate(over="ignore"):
            x = x.astype(np.float32)
        need(np.isfinite(x).all(), "FLOAT32_RANGE")
        parts.append(x)
    coordinates = np.concatenate(parts, axis=1).astype(np.float64)
    groups, sizes = np.unique(y, return_counts=True)
    need(len(groups) > 1 and (sizes > K).all(), "CLASS_SUPPORT")
    all_tree = cKDTree(coordinates)
    counts = np.empty((len(y), 3), dtype=np.int64)
    radii = np.empty(len(y), dtype=np.float32)
    for label, size in zip(groups, sizes):
        rows = np.flatnonzero(y == label)
        local = coordinates[rows]
        tree = cKDTree(local)
        distance, _ = tree.query(local, k=[K+1], p=np.inf, eps=0, workers=1)
        r = distance[:, 0].astype(np.float32)
        need((r > 0).all(), "DEGENERATE")
        boundary = rounding_boundary(r)
        counts[rows, 0] = tree.query_ball_point(
            local, boundary, p=np.inf, eps=0, workers=1, return_length=True) - 1
        counts[rows, 1] = size
        counts[rows, 2] = all_tree.query_ball_point(
            local, boundary, p=np.inf, eps=0, workers=1, return_length=True) - 1
        radii[rows] = r
    need((counts[:, 0] >= K).all() and
         (counts[:, 2] >= counts[:, 0]).all(), "NEIGHBOR_COUNTS")
    return counts, radii


def raw_bits(counts):
    """Unfloored reference formula, so differences retain negative estimates."""
    c = np.asarray(counts)
    need(c.ndim == 2 and c.shape[1] == 3 and c.dtype.kind in "iu"
         and (c[:, 0] >= K).all() and (c[:, 1] > c[:, 0]).all()
         and (c[:, 2] >= c[:, 0]).all() and (c[:, 2] < len(c)).all(), "COUNTS")
    return float(np.mean(digamma(len(c)) + digamma(c[:, 0])
                         - digamma(c[:, 1]) - digamma(c[:, 2])) / np.log(2))
