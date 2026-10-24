"""Differentiation D1 as the spec defines it (§19.3 step 3, Poly 4,096 pairs): for every pair of lenses, the
|Spearman| between their cosine readings over a FIXED seeded sample of atom pairs. CPU, reads slot files only.
Writes d1.json beside the slots: the matrix, the pairs' seed, and which lens pairs exceed 0.6.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PAIRS, SEED, MAX_CORR = 4096, 20261018, 0.6
slots, lenses = Path(sys.argv[1]), sys.argv[2].split(",")
centre = len(sys.argv) > 3 and sys.argv[3] == "--centre"
CHUNK = 262144  # rows per pass when centring: one lens memory-mapped at a time, O(CHUNK*dims) RAM at any N
X = {k: np.load(slots / f"{k}.npy", mmap_mode="r") for k in lenses}
sizes = {len(M) for M in X.values()}
if len(sizes) != 1:
    sys.exit(f"MNLM_D1_ROWS: lenses disagree on atom count {sorted(sizes)}")
n = sizes.pop()
rng = np.random.default_rng(SEED)
a, b = rng.integers(0, n, PAIRS), rng.integers(0, n, PAIRS)
keep = a != b; a, b = a[keep], b[keep]
read = {}
for k, M in X.items():
    A, B = M[a].astype(np.float64), M[b].astype(np.float64)
    if centre:
        mu = sum(M[i:i + CHUNK].astype(np.float64).sum(0) for i in range(0, n, CHUNK)) / n
        A, B = A - mu, B - mu
    A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
    read[k] = (A * B).sum(1)
mat = {f"{p}|{q}": float(abs(spearmanr(read[p], read[q]).statistic)) for p, q in itertools.combinations(lenses, 2)}
over = {k: v for k, v in mat.items() if v > MAX_CORR}
out = {"schema": "d1-cosine-pairs-v1", "pairs": int(keep.sum()), "seed": SEED, "centred": centre, "max_corr": MAX_CORR,
       "matrix": mat, "over_gate": over, "max": max(mat.values()), "min": min(mat.values())}
name = "d1-centred.json" if centre else "d1.json"
(slots / name).write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps({"max": out["max"], "min": out["min"], "over_gate": len(over), "of": len(mat)}))
for k, v in sorted(mat.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {k}: {v:.3f}")
