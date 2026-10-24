"""SUPERSEDED METHOD (2026-10-19, #12): do not use for admission. Pools each whole description into one vector,
projects to 128 dims and probes linearly; this under-reads senses. Use scripts/text/chunk_anchor.py + calyx_bits.py
(the Calyx reference method, spec §19.3 step 2). Kept only to reproduce the recorded v3/v3b results.

Whole-panel sense assay (#12; operator 2026-10-18 asked to test noise and impact as a whole). A COUNTERFACTUAL
beside the gate of record (record_assay.py, D1 <= 0.6); the contract is unchanged until the operator says so.

Same anchor, blocks, probe, folds, seeds and floor as record_assay.py (imported, not copied). Rules, predeclared
on #12 before any result:
  forward    greedy by screened conditional gain at seed 0; a candidate is admitted iff the minimum over the five
             fold seeds of the lower bound of its conditional gain is >= FLOOR (0.05 bits). No correlation gate.
  planted    after step 0 two copies of the first admitted lens join the candidates: null-rot (its 128-d block
  copies     times a seeded random orthogonal matrix: identical information) and null-noisy (block plus Gaussian
             noise, per-dimension sd 0.1 x the block's sd: strictly less information). A sound redundancy test
             admits neither.
  backward   leave-one-out over the forward panel: each member's gain given ALL other members, five seeds; while
             any member's minimum lower bound is < FLOOR, drop the lowest and recompute.
  nulls      PASS iff no planted copy is in the final panel and every control (rows permuted, random vectors,
             records shuffled) is below FLOOR. A family that fails its nulls is not counted.
  report     D1 (pair-cosine |Spearman|, d1_pairs.py) for every admitted pair, as a diagnostic only.
Cost: per forward step one screen over the remaining candidates (the panel-only baseline is cached per panel
and seed: it does not depend on the candidate), then five-seed confirmation; backward O(m^2) fits for m members.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens import vinfo
from lens.lenscommon import run_cli, take_lease
from senses.trajectory_population import publish
from text.record_assay import FLOOR, SEEDS, cv, design, prior_ce

NULLS = ("null-rot", "null-noisy")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--slots", type=Path, required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    take_lease("12", "panel-assay")
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
    d1 = json.loads((a.slots / "d1-centred.json").read_bytes())["matrix"]
    corr = lambda p_, q_: d1.get(f"{p_}|{q_}", d1.get(f"{q_}|{p_}"))
    base_cache = {}

    def base(panel, seed):
        k = (tuple(sorted(panel)), seed)
        if k not in base_cache:
            base_cache[k] = cv(design(blocks, sorted(panel)), Y, seed, classes)[1]
        return base_cache[k]

    def gain(panel, cand, seed, Yv=None):
        if Yv is not None:  # records-shuffled control: no caching
            _, b0 = cv(None, Yv, seed, classes); _, w = cv(design(blocks, [cand]), Yv, seed, classes)
        else:
            b0 = base(panel, seed); _, w = cv(design(blocks, sorted(panel) + [cand]), Y, seed, classes)
        d = (b0 - w).cpu().numpy()
        lo, hi = vinfo.paired_interval(d, None, seed=seed)
        return float(d.mean()), lo, hi

    def confirm(panel, cand):
        conf = [gain(panel, cand, s) for s in range(SEEDS)]
        return conf, min(c[1] for c in conf)

    standalone = {n: dict(zip(("gain", "lo", "hi"), gain([], n, 0))) for n in names}
    for n in sorted(standalone, key=lambda x: -standalone[x]["gain"]):
        print(f"standalone {n}: {standalone[n]['gain']:.4f}", flush=True)
    pool = list(names); panel, forward = [], []
    while True:
        left = [n for n in pool if n not in panel]
        screened = sorted(((gain(panel, c, 0)[0], c) for c in left), reverse=True)
        chosen = None
        for g, c in screened:
            if g < FLOOR:
                break
            conf, lo = confirm(panel, c)
            forward.append({"step": len(panel), "candidate": c, "screen_gain": g, "seeds": conf, "lcb_min": lo, "admitted": lo >= FLOOR,
                            "d1_vs_panel": {q: corr(c, q) for q in panel if c not in NULLS and q not in NULLS}})
            print(f"forward {len(panel)} {c}: screen={g:.4f} lcb={lo:.4f} admitted={lo >= FLOOR}", flush=True)
            if lo >= FLOOR:
                chosen = c; break
        if chosen is None:
            break
        panel.append(chosen)
        if len(panel) == 1:  # plant the copies of the first admitted lens
            first = blocks[chosen]
            q, _ = torch.linalg.qr(torch.randn(first.shape[1], first.shape[1], generator=torch.Generator().manual_seed(20261018)))
            blocks["null-rot"] = first @ q.to(dev)
            noise = torch.randn(first.shape, generator=torch.Generator().manual_seed(20261019)).to(dev)
            blocks["null-noisy"] = first + 0.1 * first.std(0, keepdim=True) * noise
            pool += list(NULLS)
    backward, final = [], list(panel)
    while len(final) > 1:
        loo = {}
        for m in final:
            conf, lo = confirm([x for x in final if x != m], m)
            loo[m] = {"seeds": conf, "lcb_min": lo}
        weakest = min(loo, key=lambda m: loo[m]["lcb_min"])
        backward.append({"panel": list(final), "leave_one_out": loo, "dropped": weakest if loo[weakest]["lcb_min"] < FLOOR else None})
        print("backward", json.dumps({m: round(v["lcb_min"], 4) for m, v in loo.items()}), flush=True)
        if loo[weakest]["lcb_min"] >= FLOOR:
            break
        final.remove(weakest)
    best = max(standalone, key=lambda n: standalone[n]["gain"])
    rng = np.random.default_rng(20261018)
    blocks["ctl_permuted"] = blocks[best][torch.as_tensor(rng.permutation(len(Y)), device=dev)]
    blocks["ctl_random"] = torch.randn(len(Y), 128, generator=torch.Generator().manual_seed(7), dtype=torch.float32).to(dev)
    controls = {c: dict(zip(("gain", "lo", "hi"), gain([], c, 0))) for c in ("ctl_permuted", "ctl_random")}
    controls["records_shuffled"] = dict(zip(("gain", "lo", "hi"), gain([], best, 0, Yv=Y[torch.as_tensor(rng.permutation(len(Y)), device=dev)])))
    planted_in_panel = [n for n in final if n in NULLS]
    nulls_pass = not planted_in_panel and all(v["gain"] < FLOOR for v in controls.values())
    senses = [n for n in final if n not in NULLS]
    out = {"schema": "panel-assay-v1", "counterfactual": True, "gate_of_record": "record_assay.py (D1 <= 0.6)",
           "anchor_rows_sha256": summ["rows_sha256"], "n": len(rows), "properties": len(props),
           "prior_bits_per_entity": float(prior_ce(Y, Y, classes).mean()), "floor_bits": FLOOR, "seeds": SEEDS,
           "standalone": standalone, "forward_panel": panel, "forward": forward, "backward": backward, "final_panel": senses,
           "planted_in_final": planted_in_panel, "planted_screened": {n: next((f for f in forward if f["candidate"] == n), None) for n in NULLS},
           "controls": controls, "nulls_pass": nulls_pass, "meaning_rule_met": nulls_pass and len(senses) >= 5,
           "d1_admitted_pairs": {f"{x}|{y}": corr(x, y) for i, x in enumerate(senses) for y in senses[i + 1:]},
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started}
    a.output.mkdir(parents=True, exist_ok=True)
    publish(a.output / "panel-assay.json", out)
    print(json.dumps({"final_panel": senses, "nulls_pass": nulls_pass, "meaning_rule_met": out["meaning_rule_met"]}))


if __name__ == "__main__":
    run_cli(main)
