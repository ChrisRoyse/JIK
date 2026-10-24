"""SUPERSEDED METHOD (2026-10-19, #12): do not use for admission. Pools each whole description into one vector,
projects to 128 dims and probes linearly; this under-reads senses. Use scripts/text/chunk_anchor.py + calyx_bits.py
(the Calyx reference method, spec §19.3 step 2). Kept only to reproduce the recorded v3/v3b results.

Sense admission against the A2 record anchor (a2-wikidata-record-v1): does each frozen embedder add bits?

Anchor: per entity, 30 Wikidata property labels (33-way each, ABSENT explicit). Reading: held-out bits
summed over properties, from a factorised multinomial logistic probe (one softmax head per property),
L2 strength chosen from a predeclared grid by inner 3-fold CV on the training fold only (5 outer folds).
Blocks: each embedder's slot projected to 128 dims by lens/vinfo.block (fixed sign projection).
Admission (unchanged contract): greedy over candidates by screened conditional gain; confirm on five
fold seeds; admitted iff the minimum lower bound (bootstrap over entities) of the conditional gain is
>= 0.05 bits and D1 (|Spearman| of cosine readings on 4,096 fixed atom pairs) is <= 0.6 against every
admitted lens (D1 = the spec's pair-cosine form from d1_pairs.py; contribution correlation is a diagnostic). Nulls: the best lens with rows permuted, the records permuted, and a random-vector lens.
Held-out V-information: a usable-information lower bound, not certified Shannon CMI. GPU probe; one lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens import vinfo
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish

FLOOR, MAX_CORR, SEEDS, FOLDS = 0.05, 0.6, 5, 5
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)  # predeclared; chosen inside each training fold only
LN2 = float(np.log(2))


def fit_ce(Xtr, Ytr, Xte, Yte, l2, classes):
    """Per-row held-out CE in bits summed over properties."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    P = Ytr.shape[1]
    W = torch.zeros(A.shape[1], P * classes, device=A.device, requires_grad=True)
    b = torch.zeros(P * classes, device=A.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = (A @ W + b).view(len(A), P, classes)
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, classes), Ytr.reshape(-1)) * P + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        z = (B @ W + b).view(len(B), P, classes)
        ce = torch.nn.functional.cross_entropy(z.reshape(-1, classes), Yte.reshape(-1), reduction="none").view(len(B), P)
    return ce.sum(1) / LN2


def prior_ce(Ytr, Yte, classes):
    out = torch.zeros(len(Yte), device=Yte.device)
    for j in range(Ytr.shape[1]):
        f = torch.bincount(Ytr[:, j], minlength=classes).float() + 0.5
        out += -torch.log2(f / f.sum())[Yte[:, j]]
    return out


def cv(X, Y, seed, classes):
    n = len(Y); rng = np.random.default_rng(seed); perm = rng.permutation(n)
    folds = np.array_split(perm, FOLDS)
    prior, probe = torch.zeros(n, device=Y.device), torch.zeros(n, device=Y.device)
    for f in range(FOLDS):
        te = torch.as_tensor(folds[f], device=Y.device)
        tr_np = np.concatenate([folds[g] for g in range(FOLDS) if g != f]); tr = torch.as_tensor(tr_np, device=Y.device)
        prior[te] = prior_ce(Y[tr], Y[te], classes)
        if X is None:
            probe[te] = prior[te]; continue
        inner = np.array_split(rng.permutation(len(tr_np)), 3)
        scores = []
        for l2 in L2_GRID:
            s = 0.0
            for i in range(3):
                ite = torch.as_tensor(tr_np[inner[i]], device=Y.device)
                itr = torch.as_tensor(np.concatenate([tr_np[inner[k]] for k in range(3) if k != i]), device=Y.device)
                s += float(fit_ce(X[itr], Y[itr], X[ite], Y[ite], l2, classes).sum())
            scores.append(s)
        probe[te] = fit_ce(X[tr], Y[tr], X[te], Y[te], L2_GRID[int(np.argmin(scores))], classes)
    return prior, probe


def design(blocks, names):
    return torch.cat([blocks[n] for n in names], 1) if names else None


def gain(blocks, Y, panel, cand, seed, classes):
    pr, base = cv(design(blocks, panel), Y, seed, classes)
    _, with_c = cv(design(blocks, panel + [cand]), Y, seed, classes)
    d = (base - with_c).cpu().numpy()
    lo, hi = vinfo.paired_interval(d, None, seed=seed)
    return float(d.mean()), lo, hi, d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, required=True, help="anchors/a2-wikidata-record-v1 (rows.jsonl, summary.json)")
    p.add_argument("--slots", type=Path, required=True, help="measure_pilot output over the anchor texts")
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    take_lease("12", "record-assay")
    started = time.perf_counter()
    summ = json.loads((a.anchor / "summary.json").read_bytes())
    rows = [json.loads(l) for l in (a.anchor / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert hashlib.sha256((a.anchor / "rows.jsonl").read_bytes()).hexdigest() == summ["rows_sha256"], "MNLM_ANCHOR_HASH"
    props = summ["properties"]; classes = summ["rules"]["n_values"] + 2
    dev = "cuda"
    Y = torch.tensor([[r["y"][q] for q in props] for r in rows], device=dev)
    names = a.lenses.split(",")
    blocks = {n: torch.tensor(np.asarray(vinfo.block(np.load(a.slots / f"{n}.npy").astype(np.float64)), np.float32), device=dev) for n in names}
    assert all(len(b) == len(Y) for b in blocks.values()), "MNLM_ANCHOR_ROWS"
    H = float(prior_ce(Y, Y, classes).mean())
    R, standalone = {}, {}
    for n in names:
        g, lo, hi, d = gain(blocks, Y, [], n, 0, classes)
        standalone[n] = {"gain": g, "lo": lo, "hi": hi}; R[n] = d
        print(f"standalone {n}: {g:.4f} bits [{lo:.4f}, {hi:.4f}]", flush=True)
    contribution_corr = lambda p_, q_: float(abs(spearmanr(R[p_], R[q_]).statistic))  # diagnostic only
    d1 = json.loads((a.slots / "d1-centred.json").read_bytes())  # spec D1: pair-cosine |Spearman| (d1_pairs.py)
    corr = lambda p_, q_: d1["matrix"].get(f"{p_}|{q_}", d1["matrix"].get(f"{q_}|{p_}"))
    assert all(corr(p_, q_) is not None for i, p_ in enumerate(names) for q_ in names[i + 1:]), "MNLM_D1_MISSING"
    panel, steps = [], []
    while True:
        left = [n for n in names if n not in panel]
        if not left:
            break
        screened = sorted(((gain(blocks, Y, panel, c, 0, classes)[0], c) for c in left), reverse=True)
        chosen = None
        for g, c in screened:
            if g < FLOOR:
                break
            cors = {q: corr(c, q) for q in panel}
            if any(v > MAX_CORR for v in cors.values()):
                steps.append({"candidate": c, "screen_gain": g, "rejected": "correlation", "corr": cors}); continue
            conf = [gain(blocks, Y, panel, c, s, classes)[:3] for s in range(SEEDS)]
            lo = min(x[1] for x in conf); ok = lo >= FLOOR
            steps.append({"candidate": c, "screen_gain": g, "corr": cors, "seeds": conf, "lcb_min": lo, "admitted": ok})
            print(f"step {len(panel)} {c}: screen={g:.4f} lcb={lo:.4f} admitted={ok}", flush=True)
            if ok:
                chosen = c; break
        if chosen is None:
            break
        panel.append(chosen)
    best = max(standalone, key=lambda n: standalone[n]["gain"])
    rng = np.random.default_rng(20261018)
    blocks["ctl_permuted"] = blocks[best][torch.as_tensor(rng.permutation(len(Y)), device=dev)]
    blocks["ctl_random"] = torch.randn(len(Y), 128, generator=torch.Generator().manual_seed(7), dtype=torch.float32).to(dev)
    controls = {}
    for c in ("ctl_permuted", "ctl_random"):
        g, lo, hi, _ = gain(blocks, Y, [], c, 0, classes); controls[c] = {"gain": g, "lo": lo, "hi": hi}
    g, lo, hi, _ = gain(blocks, Y[torch.as_tensor(rng.permutation(len(Y)), device=dev)], [], best, 0, classes)
    controls["records_shuffled"] = {"gain": g, "lo": lo, "hi": hi}
    controls["all_below_floor"] = all(v["gain"] < FLOOR for v in controls.values() if isinstance(v, dict))
    print("controls", json.dumps(controls), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    publish(a.output / "assay-record-d1spec.json", {"schema": "record-assay-v1", "anchor": summ["schema"], "anchor_rows_sha256": summ["rows_sha256"],
        "n": len(rows), "properties": len(props), "prior_bits_per_entity": H, "l2_grid": L2_GRID, "floor_bits": FLOOR, "max_corr": MAX_CORR,
        "estimator": "held-out V-information, factorised L2 multinomial logistic, nested", "shannon_conditional_certified": False,
        "standalone": standalone, "panel": panel, "meaning_rule_met": len(panel) >= 5, "steps": steps, "controls": controls,
        "d1": d1, "contribution_corr_diagnostic": {f"{p_}|{q_}": contribution_corr(p_, q_) for i, p_ in enumerate(names) for q_ in names[i + 1:]},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started})
    print(json.dumps({"panel": panel, "meaning_rule_met": len(panel) >= 5}))


if __name__ == "__main__":
    run_cli(main)
