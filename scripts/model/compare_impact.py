"""Whole-model impact of the larger sense panel (#12/#25; rules predeclared on #12).

Predictors A (12 pilot senses) and B (A plus the whole-panel senses) trained with identical config on
text-pilot-v2, three seeds each. Compared on the 12 SHARED senses only, on the same dev windows (rng 0):
per-sense rank1 and NCE, B minus A, paired by seed. B "improves" iff the mean shared rank1 gain is > 0 for
every seed pair. Also reported: per extra sense, the mean NCE increase on the shared senses when that sense
is ablated in B (does the model use it).
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = sys.argv[3].split(",")
    rep = {(k, s): json.loads((root / f"{k}-s{s}" / "report.json").read_bytes()) for k in "AB" for s in seeds}
    shared = rep[("A", seeds[0])]["lenses"]
    extra = [n for n in rep[("B", seeds[0])]["lenses"] if n not in shared]
    pairs = []
    for s in seeds:
        a, b = rep[("A", s)]["eval"]["per_sense"], rep[("B", s)]["eval"]["per_sense"]
        d_rank = {n: b[n]["rank1"] - a[n]["rank1"] for n in shared}
        d_nce = {n: b[n]["nce"] - a[n]["nce"] for n in shared}
        abl = rep[("B", s)]["eval"]["ablation_nce_increase"]
        pairs.append({"seed": s, "mean_rank1_A": float(np.mean([a[n]["rank1"] for n in shared])),
                      "mean_rank1_B": float(np.mean([b[n]["rank1"] for n in shared])),
                      "mean_rank1_gain": float(np.mean(list(d_rank.values()))), "mean_nce_change": float(np.mean(list(d_nce.values()))),
                      "rank1_gain_per_sense": d_rank, "nce_change_per_sense": d_nce,
                      "extra_sense_ablation_on_shared": {e: float(np.mean([abl[e][n] for n in shared])) for e in extra}})
    gains = [p["mean_rank1_gain"] for p in pairs]
    res = {"schema": "impact-compare-v1", "shared_senses": shared, "extra_senses": extra, "seeds": seeds, "pairs": pairs,
           "mean_rank1_gain": float(np.mean(gains)), "min_rank1_gain": float(min(gains)), "improves": min(gains) > 0,
           "seed_sd_rank1_A": float(np.std([p["mean_rank1_A"] for p in pairs], ddof=1)) if len(pairs) > 1 else None}
    publish(out, res)
    print(json.dumps({k: res[k] for k in ("extra_senses", "mean_rank1_gain", "min_rank1_gain", "improves", "seed_sd_rank1_A")}))


if __name__ == "__main__":
    main()
