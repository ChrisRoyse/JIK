"""Second SWE sense assay (#12 decision 6243560817): within-task conditioning, nested probe capacity.

Modes
  within  rows: tasks with both outcomes; the task (one-hot) is conditioning context, not a sense;
          folds group identical normalised patch content within a task (duplicates never straddle).
          Measures usable information about the outcome given the task and the panel.
  across  rows: all; folds group by task (generalisation to unseen tasks); no context.
Probe: L2 logistic; C chosen from a fixed predeclared grid by inner grouped 3-fold CV on the training
fold only. Admission per sense: the minimum five-seed cluster-bootstrap lower bound of the conditional
gain and of the nonlinear-control gain is >= 0.05 bits; |Spearman| of held-out contribution readings
<= 0.6 against every admitted sense; each admitted sense counts (perspectives are reported). Held-out V-information:
a usable-information diagnostic, not certified conditional Shannon information.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lens import vinfo
from senses.trajectory_population import publish

FLOOR, MAX_CORR, MIN_SENSES, FOLDS, SEEDS = 0.05, 0.6, 5, 5, 5
C_GRID = (0.001, 0.01, 0.1, 1.0)  # predeclared; chosen inside each training fold only
PERSPECTIVE = {"surface": "surface", "structure": "structure", "graph": "graph", "behaviour": "behaviour",
               "behaviour_embed": "behaviour", "relations": "relations", "relations_embed": "relations",
               "diagnostic": "diagnostic", "provenance": "provenance", "consensus": "consensus", "judge": "judge"}
STATE = {}


def take(X, idx):
    return (X[0][idx], X[1][idx]) if isinstance(X, tuple) else X[idx]


def width(X):
    return X[0].shape[1] + X[1].shape[1] if isinstance(X, tuple) else X.shape[1]


def fit_ce(Xtr, ytr, Xte, yte, C):
    if isinstance(Xtr, tuple):  # (sparse one-hot context, dense senses): scale the dense part on the training fold
        sc = StandardScaler()
        dense = lambda D, fit: (sc.fit_transform(D) if fit else sc.transform(D)) if D.shape[1] else D
        tr = sp.hstack([Xtr[0], sp.csr_matrix(dense(Xtr[1], True))]).tocsr()
        te = sp.hstack([Xte[0], sp.csr_matrix(dense(Xte[1], False))]).tocsr()
    else:
        sc = StandardScaler(); tr = sc.fit_transform(Xtr); te = sc.transform(Xte)
    clf = LogisticRegression(C=C, max_iter=5000)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(tr, ytr)
    p = clf.predict_proba(te)[np.arange(len(yte)), np.searchsorted(clf.classes_, yte)]
    return -np.log2(np.clip(p, 1e-6, 1))


def cv_ce(X, y, g, seed):
    """Per-row held-out CE (bits) of the class prior and of the nested-capacity probe."""
    prior, probe = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g):
        freq = np.clip(np.array([(y[tr] == c).mean() for c in (0, 1)]), 1e-6, 1)
        prior[te] = -np.log2(freq[y[te]])
        if width(X) == 0:
            probe[te] = prior[te]; continue
        inner = GroupKFold(n_splits=3)
        scores = []
        for C in C_GRID:
            s = 0.0
            Xtr = take(X, tr)
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
                if len(np.unique(y[tr][itr])) < 2:
                    s = np.inf; break
                s += fit_ce(take(Xtr, itr), y[tr][itr], take(Xtr, ite), y[tr][ite], C).sum()
            scores.append(s)
        probe[te] = fit_ce(take(X, tr), y[tr], take(X, te), y[te], C_GRID[int(np.argmin(scores))])
    return prior, probe


def design(names, rows_idx, extra_rff=None):
    B, ctx = STATE["blocks"], STATE["context"]
    senses = [B[n][STATE["pos"][n][rows_idx]] for n in names]
    if extra_rff is not None:
        senses.append(vinfo.expansion(np.concatenate([B[n][STATE["pos"][n][rows_idx]] for n in extra_rff], 1)))
    if sp.issparse(ctx):
        return ctx[rows_idx], (np.concatenate(senses, 1) if senses else np.zeros((len(rows_idx), 0)))
    parts = ([ctx[rows_idx]] if ctx is not None else []) + senses
    return np.concatenate(parts, 1) if parts else np.zeros((len(rows_idx), 0))


def common(names):
    ok = STATE["base_rows"]
    for n in names:
        ok = ok[np.isin(ok, STATE["rows"][n])]
    return ok


def gain_task(args):
    panel, cand, seeds, control = args
    idx = common(panel + [cand]); y, g, bg = STATE["y"][idx], STATE["g"][idx], STATE["bg"][idx]
    out = []
    for s in seeds:
        pr, base = cv_ce(design(panel, idx), y, g, s)
        if not panel and STATE["context"] is None:
            base = pr
        _, with_c = cv_ce(design(panel + [cand], idx), y, g, s)
        d = base - with_c
        lo, hi = vinfo.paired_interval(d, bg, seed=s)  # clusters = tasks (attempts of a task are correlated)
        e = {"seed": s, "gain": float(d.mean()), "lo": lo, "hi": hi}
        if control and panel:
            _, pe = cv_ce(design(panel, idx, extra_rff=panel), y, g, s)
            _, ce = cv_ce(design(panel + [cand], idx, extra_rff=panel), y, g, s)
            dl, dh = vinfo.paired_interval(pe - ce, bg, seed=s)
            e.update(nonlinear_gain=float((pe - ce).mean()), nonlinear_lo=dl, nonlinear_hi=dh)
        out.append(e)
    return cand, out, idx, (base - with_c) if len(seeds) == 1 else None


def init(state):
    STATE.update(state)


def run(mode, ev, workers):
    labels = [json.loads(l) for l in (ev / "observations/labels.jsonl").read_text().splitlines()]
    recs = [json.loads(l) for l in (ev / "observations/observations.jsonl").read_text(encoding="utf-8").splitlines()]
    y = np.array([l["outcome"] for l in labels]); tasks = np.array([l["task"] for l in labels])
    blocks, rows = {}, {}
    for d in ("explicit", "embedded", "consensus") + (("judge",) if (ev / "judge/manifest.json").exists() else ()):
        for name, spec in json.loads((ev / d / "manifest.json").read_bytes())["senses"].items():
            assert hashlib.sha256((ev / d / spec["path"]).read_bytes()).hexdigest() == spec["sha256"]
            blocks[name], rows[name] = np.asarray(vinfo.block(np.load(ev / d / spec["path"])), np.float64), np.arange(len(y))
    gm = json.loads((ev / "graph/manifest.json").read_bytes())
    assert hashlib.sha256((ev / "graph/graph.npy").read_bytes()).hexdigest() == gm["graph_sha256"]
    grows = np.load(ev / "graph/graph_rows.npy")
    blocks["graph"], rows["graph"] = np.load(ev / "graph/graph.npy")[grows].astype(np.float64), grows
    pos = {n: np.full(len(y), -1) for n in blocks}
    for n in blocks:
        pos[n][rows[n]] = np.arange(len(rows[n]))
    if mode == "within":
        mixed = {t for t in set(tasks) if 0 < y[tasks == t].mean() < 1}
        base_rows = np.array([i for i in range(len(y)) if tasks[i] in mixed])
        ut = sorted(mixed); ti = {t: i for i, t in enumerate(ut)}
        context = np.zeros((len(y), len(ut)))
        for i in base_rows:
            context[i, ti[tasks[i]]] = 1.0
        norm = [re.sub(r"\s+", " ", r["patch"]) for r in recs]
        g = np.unique([hashlib.sha256((tasks[i] + "|" + norm[i]).encode()).hexdigest() for i in range(len(y))], return_inverse=True)[1]
    elif mode == "tests":
        # anchor tests-v1: rows are (observation, required test) pairs whose test varies within its task;
        # the (task, test) identity is context; all tests of one change share a fold; clusters are tasks.
        norm = [re.sub(r"\s+", " ", r["patch"]) for r in recs]
        at = {l["uid"]: i for i, l in enumerate(labels)}
        pairs = [json.loads(l) for l in (ev / "observations/test_labels.jsonl").read_text().splitlines()]
        pairs = [q for q in pairs if q["varies"]]
        obs = np.array([at[q["uid"]] for q in pairs])
        y = np.array([q["passed"] for q in pairs])
        key = np.unique([q["task"] + "|" + q["test"] for q in pairs], return_inverse=True)[1]
        context = sp.csr_matrix((np.ones(len(pairs)), (np.arange(len(pairs)), key)), shape=(len(pairs), key.max() + 1))
        g = np.unique([hashlib.sha256((tasks[i] + "|" + norm[i]).encode()).hexdigest() for i in obs], return_inverse=True)[1]
        rows = {n: np.flatnonzero(np.isin(obs, r)) for n, r in rows.items()}
        pos = {n: pos[n][obs] for n in pos}
        tasks = tasks[obs]; base_rows = np.arange(len(y))
        bg = np.unique(tasks, return_inverse=True)[1]
        return {"y": y, "g": g, "bg": bg, "shuffle_key": key, "blocks": blocks, "rows": rows, "pos": pos, "context": context,
                "base_rows": base_rows, "tasks_in_scope": int(len(set(tasks))), "positives": int(y.sum()),
                "observations_in_scope": int(len(set(obs)))}
    else:
        base_rows = np.arange(len(y)); context = None
        g = np.unique(tasks, return_inverse=True)[1]
    bg = np.unique(tasks, return_inverse=True)[1]
    return {"y": y, "g": g, "bg": bg, "shuffle_key": bg, "blocks": blocks, "rows": rows, "pos": pos, "context": context,
            "base_rows": base_rows, "tasks_in_scope": int(len(set(tasks[base_rows]))), "positives": int(y[base_rows].sum())}


def negative_controls(state, standalone):
    """Three nulls, each read like a sense (seed 0, task-clustered interval); each must stay below the floor.

    permuted_best  the best standalone sense with its rows permuted (destroys row/label pairing);
    label_shuffle_global       the best sense against globally permuted outcomes;
    label_shuffle_within_task  outcomes permuted within each task (keeps task base rates): a null in
                               within mode; in across mode it reads the best sense's task-level share;
    task_identity  a one-hot task instrument (across: tasks unseen in test folds; within: already context).
    """
    rng = np.random.default_rng(20261018)
    best = max(standalone, key=standalone.get)
    n = len(state["y"])
    B = state["blocks"]
    B["ctl_permuted"] = B[best][rng.permutation(len(B[best]))]
    state["rows"]["ctl_permuted"] = state["rows"][best]; state["pos"]["ctl_permuted"] = state["pos"][best]
    ut, ti = np.unique(state["bg"], return_inverse=True)
    B["ctl_task"] = np.eye(len(ut))[ti]
    state["rows"]["ctl_task"] = np.arange(n); state["pos"]["ctl_task"] = np.arange(n)
    init(state)
    out = {"best_sense": best}
    for name in ("ctl_permuted", "ctl_task"):
        _, e, _, _ = gain_task(([], name, [0], False))
        out[name] = {k: e[0][k] for k in ("gain", "lo", "hi")}
    y0 = state["y"].copy()
    within = y0.copy()
    for t in np.unique(state["shuffle_key"]):  # the task (within, across) or the (task, test) context (tests)
        m = np.flatnonzero(state["shuffle_key"] == t); within[m] = y0[rng.permutation(m)]
    for name, ys in (("label_shuffle_global", y0[rng.permutation(n)]), ("label_shuffle_within_task", within)):
        STATE["y"] = ys
        try:
            _, e, _, _ = gain_task(([], best, [0], False))
            out[name] = {k: e[0][k] for k in ("gain", "lo", "hi")}
        finally:
            STATE["y"] = y0
    for name in ("ctl_permuted", "ctl_task"):
        for d in ("blocks", "rows", "pos"):
            state[d].pop(name)
    # within mode: the within-task shuffle is a null. across mode: it keeps task base rates, so what the
    # best sense still reads is its task-level (which-task-is-solvable) share, reported, not a null.
    nulls = ["ctl_permuted", "ctl_task", "label_shuffle_global"] + (["label_shuffle_within_task"] if state["context"] is not None else [])
    out["nulls"] = nulls
    out["all_nulls_below_floor"] = all(out[k]["gain"] < FLOOR for k in nulls)
    if state["context"] is None:
        out["best_task_level_share"] = out["label_shuffle_within_task"]["gain"] / max(standalone[best], 1e-9)
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--mode", choices=("within", "across", "tests"), required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--workers", type=int, default=12)
    args = a.parse_args()
    started = time.perf_counter()
    state = run(args.mode, args.root, args.workers)
    init(state)
    names = sorted(state["blocks"])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(state,)) as pool:
        # standalone (given context) contribution readings for the correlation gate
        R, standalone = {}, {}
        for cand, out, idx, diff in pool.map(gain_task, [([], n, [0], False) for n in names]):
            standalone[cand] = out[0]["gain"]
            full = np.full(len(state["y"]), np.nan); full[idx] = diff; R[cand] = full
            print(f"[{args.mode}] standalone {cand}: {out[0]['gain']:.4f} bits [{out[0]['lo']:.4f}, {out[0]['hi']:.4f}] n={len(idx)}", flush=True)

        def corr(p, q):
            ok = np.isfinite(R[p]) & np.isfinite(R[q])
            return float(abs(spearmanr(R[p][ok], R[q][ok]).statistic)) if ok.sum() > 50 else 1.0
        panel, perspectives, steps = [], [], []
        while True:
            left = [n for n in names if n not in panel]
            if not left:
                break
            screened = sorted(((out[0]["gain"], c) for c, out, _, _ in pool.map(gain_task, [(panel, c, [0], False) for c in left])), reverse=True)
            chosen = None
            for gain, c in screened:
                if gain < FLOOR:
                    break
                cors = {p: corr(c, p) for p in panel}
                if any(v > MAX_CORR for v in cors.values()):
                    steps.append({"candidate": c, "screen_gain": gain, "rejected": "correlation", "corr": cors}); continue
                _, conf, _, _ = gain_task((panel, c, list(range(SEEDS)), True))
                lo = min(e["lo"] for e in conf); nl = min((e.get("nonlinear_lo", lo) for e in conf), default=lo)
                ok = lo >= FLOOR and nl >= FLOOR
                steps.append({"candidate": c, "screen_gain": gain, "corr": cors, "seeds": conf, "lcb_min": lo, "nonlinear_lcb_min": nl, "admitted": ok})
                print(f"[{args.mode}] step {len(panel)} {c}: screen={gain:.4f} lcb={lo:.4f} nonlin={nl:.4f} admitted={ok}", flush=True)
                if ok:
                    chosen = c; break
            if chosen is None:
                break
            panel.append(chosen)
            if PERSPECTIVE[chosen] not in perspectives:
                perspectives.append(PERSPECTIVE[chosen])
    controls = negative_controls(state, standalone)
    print(f"[{args.mode}] controls: {json.dumps(controls)}", flush=True)
    report = {"schema": "swe-sense-assay-v2", "mode": args.mode, "controls": controls, "n": int(len(state["base_rows"])), "tasks": state["tasks_in_scope"],
              "positives": state["positives"], "c_grid": C_GRID, "floor_bits": FLOOR, "max_corr": MAX_CORR,
              "estimator": "held-out V-information, nested L2 logistic", "shannon_conditional_certified": False,
              "standalone_bits_given_context": standalone, "panel": panel, "perspectives": perspectives,
              "meaning_rule_met": len(panel) >= MIN_SENSES, "steps": steps,
              "pairwise_corr": {f"{p}|{q}": (lambda p, q: (lambda ok: float(abs(spearmanr(R[p][ok], R[q][ok]).statistic)) if ok.sum() > 50 else None)(np.isfinite(R[p]) & np.isfinite(R[q])))(p, q)
                                for i, p in enumerate(names) for q in names[i + 1:]},
              "wall_seconds": time.perf_counter() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    publish(args.output / f"assay-{args.mode}.json", report)
    print(json.dumps({k: report[k] for k in ("mode", "n", "tasks", "positives", "panel", "perspectives", "meaning_rule_met")}))


if __name__ == "__main__":
    main()
