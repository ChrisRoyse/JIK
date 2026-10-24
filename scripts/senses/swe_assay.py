"""Measure the SWE senses against the grounded outcome anchor (#12 senses plan; doctrine #1 D4).

Anchor: the independently replayed outcome of each pre-evaluation submission (labels.jsonl).
Folds keep every row of one task on the same side (StratifiedGroupKFold by task), so recognising a
task never counts. Greedy forward selection over instruments: at each step the candidate with the
largest screened conditional gain is confirmed over five fold seeds (usable and nonlinear-control
lower bounds >= 0.05 bits) and must correlate at most 0.6 (|Spearman| of held-out readings) with every
selected instrument. Each admitted instrument counts toward the five-sense rule; perspectives are reported.
Negative controls: a task-identity instrument and a row-permuted copy of the best instrument.
The estimator is held-out V-information (vinfo.SHANNON_CONDITIONAL_CERTIFIED is False): a diagnostic
of usable information, reported as such and never as certified conditional Shannon information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens import vinfo
from senses.trajectory_population import publish

FLOOR, MAX_CORR, MIN_SENSES = 0.05, 0.6, 5
PERSPECTIVE = {"surface": "surface", "structure": "structure", "graph": "graph", "behaviour": "behaviour",
               "behaviour_embed": "behaviour", "relations": "relations", "relations_embed": "relations",
               "diagnostic": "diagnostic", "provenance": "provenance"}


def reading(X, y, groups):
    """Held-out standalone log-odds: what the instrument alone says about the anchor, row by row."""
    prior, probe = vinfo.cv_cross_entropy(vinfo.block(X), y, groups)
    p = np.where(y == 1, 2.0 ** -probe, 1 - 2.0 ** -probe)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)), float((prior - probe).mean())


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    started = time.perf_counter()
    ev = args.root
    labels = [json.loads(l) for l in (ev / "observations/labels.jsonl").read_text().splitlines()]
    y = np.array([l["outcome"] for l in labels], np.int64)
    tasks = np.array([l["task"] for l in labels])
    _, groups = np.unique(tasks, return_inverse=True)
    n = len(y)
    blocks, rows = {}, {}
    for d in ("explicit", "embedded"):
        for name, spec in json.loads((ev / d / "manifest.json").read_bytes())["senses"].items():
            M = np.load(ev / d / spec["path"])
            assert M.shape[0] == n and hashlib.sha256((ev / d / spec["path"]).read_bytes()).hexdigest() == spec["sha256"]
            blocks[name], rows[name] = vinfo.block(M), np.arange(n)
    gm = json.loads((ev / "graph/manifest.json").read_bytes())
    G = np.load(ev / "graph/graph.npy"); assert hashlib.sha256((ev / "graph/graph.npy").read_bytes()).hexdigest() == gm["graph_sha256"]
    grows = np.load(ev / "graph/graph_rows.npy")
    blocks["graph"], rows["graph"] = vinfo.block(G[grows]), grows
    # readings for the correlation gate, on all rows each instrument covers
    R, standalone = {}, {}
    for name in blocks:
        r, bits = reading(blocks[name], y[rows[name]], groups[rows[name]])
        full = np.full(n, np.nan); full[rows[name]] = r
        R[name], standalone[name] = full, bits
        print(f"standalone {name}: {bits:.4f} bits (n={len(rows[name])})", flush=True)

    def corr(a_, b_):
        ok = np.isfinite(R[a_]) & np.isfinite(R[b_])
        return float(abs(spearmanr(R[a_][ok], R[b_][ok]).statistic))
    panel, steps, perspectives = [], [], []
    candidates = set(blocks)
    while candidates:
        screened = []
        for c in sorted(candidates):
            g = vinfo.conditional_gain(blocks, rows, y, panel, c, None, groups)
            screened.append((g["gain"], c, g["gain_ci_low"]))
        screened.sort(reverse=True)
        chosen = None
        for gain, c, lo in screened:
            if gain < FLOOR:
                break
            cors = {p: corr(c, p) for p in panel}
            if any(v > MAX_CORR for v in cors.values()):
                steps.append({"candidate": c, "screen_gain": gain, "rejected": "correlation", "corr": cors}); continue
            conf = vinfo.confirm(blocks, rows, y, panel, c, groups)
            ok = vinfo.admissible(conf, FLOOR, not panel)
            steps.append({"candidate": c, "screen_gain": gain, "screen_lo": lo, "corr": cors,
                          "confirm": {k: v for k, v in conf.items() if k not in ("common", "diffs")}, "admitted": ok})
            print(f"step {len(panel)}: {c} screen={gain:.4f} lcb_min={conf['lcb_min']:.4f} "
                  f"nonlin={conf.get('nonlinear_lcb_min', float('nan')):.4f} admitted={ok}", flush=True)
            if ok:
                chosen = c; break
        if chosen is None:
            break
        panel.append(chosen); candidates.discard(chosen)
        if PERSPECTIVE[chosen] not in perspectives:
            perspectives.append(PERSPECTIVE[chosen])
    # negative controls
    task_onehot = np.zeros((n, int(groups.max()) + 1), np.float32); task_onehot[np.arange(n), groups] = 1
    ctrl_blocks = {**blocks, "task_identity": vinfo.block(task_onehot)}; ctrl_rows = {**rows, "task_identity": np.arange(n)}
    ti = vinfo.conditional_gain(ctrl_blocks, ctrl_rows, y, [], "task_identity", None, groups)
    best = max(standalone, key=standalone.get)
    perm = np.random.default_rng(20261018).permutation(len(rows[best]))
    ctrl_blocks["permuted_best"] = blocks[best][perm]; ctrl_rows["permuted_best"] = rows[best]
    pb = vinfo.conditional_gain(ctrl_blocks, ctrl_rows, y, [], "permuted_best", None, groups)
    report = {"schema": "swe-sense-assay-v1", "anchor": "replayed outcome of a pre-evaluation submission", "n": n,
              "tasks": int(groups.max()) + 1, "positives": int(y.sum()), "estimator": vinfo.INSTRUMENT,
              "shannon_conditional_certified": vinfo.SHANNON_CONDITIONAL_CERTIFIED, "floor_bits": FLOOR, "max_corr": MAX_CORR,
              "standalone_bits": standalone, "panel": panel, "perspectives": perspectives,
              "meaning_rule_met": len(panel) >= MIN_SENSES,
              "steps": steps, "controls": {"task_identity_gain": ti["gain"], "task_identity_ci": [ti["gain_ci_low"], ti["gain_ci_high"]],
                                           "permuted_best": best, "permuted_gain": pb["gain"], "permuted_ci": [pb["gain_ci_low"], pb["gain_ci_high"]]},
              "pairwise_corr": {f"{p}|{q}": corr(p, q) for i, p in enumerate(sorted(blocks)) for q in sorted(blocks)[i + 1:]},
              "wall_seconds": time.perf_counter() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"panel": panel, "perspectives": perspectives, "meaning_rule_met": report["meaning_rule_met"],
                      "controls": report["controls"]}))
    publish(args.output / "assay.json", report)


if __name__ == "__main__":
    main()
