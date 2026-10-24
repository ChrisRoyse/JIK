"""Copy baseline for next-atom retrieval (CPU): query = the CURRENT atom's centred slot, candidates = the true
next slots in the same evaluation batches the predictor used (constellation_lm.evaluate: dev split, rng seed 0,
batch 64, window 24). Reports rank-1 per sense, to compare with the predictor's rank-1.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.constellation_lm import Data

atoms, slots, lenses = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
batch, window = 64, 24
data = Data(atoms, slots, lenses, window)
wins = data.windows["dev"]
order = np.random.default_rng(0).permutation(len(wins))[: min(len(wins), 64 * batch)]
acc = {n: 0.0 for n in lenses}; nb = 0
for j in range(0, len(order), batch):
    rows = [wins[i] for i in order[j:j + batch]]
    if len(rows) < 8:
        break
    s = data.batch("dev", rows, "cpu")
    for k, n in enumerate(lenses):
        c = s[k][:, :-1].reshape(-1, s[k].shape[-1]); y = s[k][:, 1:].reshape(-1, s[k].shape[-1])
        sim = c @ y.T
        # candidate i-1 (the previous position's "next") IS the current atom: exclude the query's own atom from
        # its candidates, which favours the baseline (the predictor was scored without this exclusion)
        T1 = s[k].shape[1] - 1
        i = torch.arange(len(c)); own = (i % T1) != 0
        sim[i[own], i[own] - 1] = -2
        acc[n] += float((sim.argmax(1) == torch.arange(len(c))).float().mean())
    nb += 1
print(json.dumps({"batches": nb, "copy_rank1": {n: round(v / nb, 4) for n, v in acc.items()}}, indent=1))
