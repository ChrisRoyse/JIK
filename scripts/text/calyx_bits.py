"""Calyx reference sense assay on chunked anchors (calyx-bits-port-v2, #12). A port of the measurement that
built the Leapable Law roster (calyx-cli cmd/intelligence/bits_categorical.rs + calyx-assay ksg/mixed_ci.rs +
nmi.rs + calyx-core slot_similarity; lawsearch 50-51), so MNLM senses are measured the way the reference systems
measure them. Predeclared on #12 before any result.

Per anchor (chunk_anchor.py: categorical, balanced, ~192 rows), per lens:
  representation  chunk_multi (of record): each entity is the ordered set of its chunk readings; its feature
                  vector is its MaxSim profile against LANDMARKS landmark entities taken at a fixed stride over
                  the sample, MaxSim(A,B) = min(mean_a max_b cos, mean_b max_a cos); L2-normalised.
                  pooled (comparison): the whole-description vector (v3b slot), L2-normalised.
  bits            Ross mixed continuous-discrete KSG, Chebyshev metric, k=3:
                  mean_i[psi(n) + psi(m_same_i) - psi(n_class_i) - psi(m_all_i)] / ln 2 (calyx raw_bits);
                  interval: 999 no-replacement subsamples of size m (calyx subsample_plan: smallest m in
                  [ceil(n^(2/3)), n/2] whose class-support failure <= 0.01), roots sqrt(m/(1-m/n))(b_m - b),
                  [b - r_.975/sqrt(n), b - r_.025/sqrt(n)], hull with the point, floored at 0.
                  (calyx draws with ChaCha8 seed 0; this port draws with numpy seed 0: same design, different
                  stream.)
  redundancy      NMI of the two lenses' cosine-to-own-centroid scores, equal-width bins
                  clamp(round(sqrt(n)), 4, 32), NMI = MI / sqrt(Hx Hy) (calyx partitioned_histogram_nmi);
                  a pair's NMI = its maximum over anchors.
Historical reference roster (diagnostic): mean bits x (1 - max NMI to kept) >= 0.05, NMI <= 0.6.
This is NOT conditional information or scientific admission. calyx_screen screens each anchor using
lower bounds and individual controls; D1/D2/D3/D4 certification remains explicitly unresolved.
The joint reference-roster KSG and entropy ratio remain diagnostics. CPU only (no GPU lease).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil, log, sqrt
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.special import digamma, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.calyx_screen import FLOOR, MAX_NMI, screen

K, RESAMPLES, LANDMARKS, MAX_SUPPORT_FAILURE = 3, 999, 24, 0.01
LN2 = log(2.0)


class Refusal(Exception):
    """A lens the reference method cannot measure (named code). Calyx refuses such a slot; here the lens is
    recorded as refused with its code and anchor and the other lenses go on; controls and the joint panel
    still abort on any refusal."""


def plain(v):
    """numpy scalars -> Python numbers for the canonical JSON writer (type only; values unchanged)."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if (n <= 0).any() or not np.isfinite(n).all():
        raise Refusal(f"MNLM_CALYX_ZERO_NORM: {int((~(n > 0) | ~np.isfinite(n)).sum())} assay vectors have zero or non-finite norm")
    return X / n


def chebyshev_matrix(X):
    X = X.astype(np.float32)
    D = np.zeros((len(X), len(X)), np.float32)
    for i in range(len(X)):
        D[i] = np.abs(X - X[i]).max(1)
    return D


def raw_bits(D, y):
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    csize = dict(zip(classes, counts))
    total = 0.0
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf
        same = d[y == y[i]]
        radius = np.partition(same, K - 1)[K - 1]
        m_same = int((same <= radius).sum())
        m_all = int((d <= radius).sum())
        total += digamma(n) + digamma(m_same) - digamma(csize[y[i]]) - digamma(m_all)
    return total / n / LN2


def hyper_cdf_at_most(N, Ks, n, t):
    lo, hi = max(0, n - (N - Ks)), min(t, Ks, n)
    if lo > hi:
        return 0.0
    lb = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    terms = [lb(Ks, s) + lb(N - Ks, n - s) - lb(N, n) for s in range(lo, hi + 1)]
    mx = max(terms)
    return float(np.exp(mx) * sum(np.exp(t_ - mx) for t_ in terms))


def subsample_m(n, counts):
    for m in range(max(ceil(n ** (2 / 3)), K + 2), n // 2 + 1):
        if min(1.0, sum(hyper_cdf_at_most(n, c, m, K) for c in counts)) <= MAX_SUPPORT_FAILURE:
            return m
    raise SystemExit(f"MNLM_CALYX_SUBSAMPLE: no valid m for n={n} counts={counts}")


def ksg(X, y):
    D = chebyshev_matrix(X)
    for i in range(len(y)):
        if ((D[i] == 0) & (y == y[i])).sum() - 1 >= K:
            raise Refusal("MNLM_CALYX_DEGENERATE: kth same-class radius is zero (>= k exact ties within a class)")
    b = raw_bits(D, y)
    counts = np.unique(y, return_counts=True)[1]
    n = len(y); m = subsample_m(n, counts)
    rng = np.random.default_rng(0)
    scale = sqrt(m / (1 - m / n)); roots = []
    nclass = len(counts)
    for _ in range(RESAMPLES):
        for _attempt in range(8):
            idx = rng.choice(n, m, replace=False)
            c = np.unique(y[idx], return_counts=True)[1]
            if len(c) == nclass and (c > K).all():
                break
        else:
            raise SystemExit("MNLM_CALYX_SUPPORT: class-support rejection exhausted")
        roots.append(scale * (raw_bits(D[np.ix_(idx, idx)], y[idx]) - b))
    roots = np.sort(roots); pi = lambda p: min(len(roots) - 1, int(round((len(roots) - 1) * p)))
    lo, hi = b - roots[pi(0.975)] / sqrt(n), b - roots[pi(0.025)] / sqrt(n)
    return {"bits": max(b, 0.0), "ci_low": max(min(lo, b), 0.0), "ci_high": max(max(hi, b), 0.0), "raw": b, "n": n, "m": m}


def centroid_scores(V):
    c = V.mean(0); c = c / np.linalg.norm(c)
    return V @ c


def nmi(a, b):
    bins = int(np.clip(round(sqrt(len(a))), 4, 32))
    def binv(v):
        w = max(v.max() - v.min(), np.finfo(np.float32).eps)
        return np.minimum(((v - v.min()) / w * bins).astype(int), bins - 1)
    xa, xb = binv(a), binv(b)
    H = lambda z: -sum((c / len(z)) * log(c / len(z), 2) for c in np.unique(z, return_counts=True)[1])
    hx, hy, hxy = H(xa), H(xb), H(xa * bins + xb)
    mi = max(hx + hy - hxy, 0.0)
    return mi / sqrt(hx * hy) if hx * hy > 0 else 0.0


def maxsim_profiles(chunk_vecs, rows):
    """chunk_vecs: list (per sampled entity, in sample order) of unit [c_i, d] arrays."""
    lm = [k * len(rows) // LANDMARKS for k in range(min(LANDMARKS, len(rows)))]
    prof = np.zeros((len(rows), len(lm)))
    for j, l in enumerate(lm):
        B = chunk_vecs[l]
        for i in range(len(rows)):
            S = chunk_vecs[i] @ B.T
            prof[i, j] = min(S.max(1).mean(), S.max(0).mean())
    return unit(prof)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunked", type=Path, required=True, help="chunk_anchor.py output")
    p.add_argument("--chunk-slots", type=Path, required=True, help="slots measured over chunked/chunks")
    p.add_argument("--pooled-slots", type=Path, required=True, help="whole-description slots (v3b)")
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    anchors = json.loads((a.chunked / "anchors.json").read_bytes())
    ch = pq.read_table(a.chunked / "chunks" / "atoms.parquet", columns=["row", "chunk"])
    crow = ch.column("row").to_numpy()
    names = a.lenses.split(",")
    by_row = {}
    for idx, r in enumerate(crow):
        by_row.setdefault(int(r), []).append(idx)
    res = {n: {} for n in names}; scores = {n: {} for n in names}; refused = {}
    for n in names:
        C = np.load(a.chunk_slots / f"{n}.npy").astype(np.float64)
        if len(C) != len(crow):
            raise SystemExit(f"MNLM_CALYX_ROWS: {n} chunk slots {len(C)} != chunks {len(crow)}")
        if C.shape[1] < 2:  # cosine of 1-d readings is the sign alone: not measurable by the reference method
            refused[n] = {"code": "MNLM_CALYX_SCALAR_LENS", "dims": int(C.shape[1]), "anchor": None}
            print(n, "REFUSED", json.dumps(refused[n]), flush=True); continue
        P = np.load(a.pooled_slots / f"{n}.npy").astype(np.float64)
        prop = None
        try:
            C = unit(C)
            for prop, an in anchors.items():
                rows, y = an["rows"], np.array(an["labels"])
                prof = maxsim_profiles([C[by_row[r]] for r in rows], rows)
                res[n][prop] = {"chunk_multi": ksg(prof, y)}
                scores[n][prop] = centroid_scores(prof)
                try:  # the pooled whole-text reading is a comparison only; its refusal does not refuse the lens
                    res[n][prop]["pooled"] = ksg(unit(P[rows]), y)
                except Refusal as e:
                    res[n][prop]["pooled"] = {"bits": None, "refused": str(e)}
        except Refusal as e:
            refused[n] = {"code": str(e).split(":")[0], "detail": str(e), "dims": int(C.shape[1]), "anchor": prop}
            res.pop(n, None); scores.pop(n, None)
            print(n, "REFUSED", json.dumps(refused[n]), flush=True); continue
        pb = [res[n][q]["pooled"]["bits"] for q in anchors if res[n][q]["pooled"]["bits"] is not None]
        print(n, json.dumps({"chunk_multi": round(float(np.mean([res[n][q]["chunk_multi"]["bits"] for q in anchors])), 4),
                             "pooled": round(float(np.mean(pb)), 4) if pb else None}), flush=True)
    names = [n for n in names if n not in refused]
    if not names:
        raise SystemExit("MNLM_CALYX_NO_MEASURABLE_LENSES: all requested lenses refused; inspect lens diagnostics")
    pair = {}
    for i, x in enumerate(names):
        for y_ in names[i + 1:]:
            pair[f"{x}|{y_}"] = max(nmi(scores[x][q], scores[y_][q]) for q in anchors)
    pn = lambda x, y_: pair.get(f"{x}|{y_}", pair.get(f"{y_}|{x}"))
    mean_bits = {n: float(np.mean([res[n][q]["chunk_multi"]["bits"] for q in anchors])) for n in names}
    kept, steps = [], []
    while True:
        cands = []
        for n in names:
            if n in kept:
                continue
            mx = max((pn(n, k) for k in kept), default=0.0)
            cands.append((mean_bits[n] * (1 - mx), mx, n))
        cands.sort(reverse=True)
        pick = next(((m, mx, n) for m, mx, n in cands if m >= FLOOR and mx <= MAX_NMI), None)
        if pick is None:
            break
        kept.append(pick[2]); steps.append({"lane": pick[2], "redundancy_adjusted_standalone_bits": pick[0], "max_nmi_to_kept": pick[1], "mean_bits": mean_bits[pick[2]]})
        print(f"reference roster {pick[2]}: score={pick[0]:.4f} max_nmi={pick[1]:.3f}", flush=True)
    panel = {}
    if kept:
        C_all = {n: unit(np.load(a.chunk_slots / f"{n}.npy").astype(np.float64)) for n in kept}
        for prop, an in anchors.items():
            rows, y = an["rows"], np.array(an["labels"])
            blocks = [maxsim_profiles([C_all[n][by_row[r]] for r in rows], rows) for n in kept]
            J = np.concatenate(blocks, 1)
            est = ksg(J, y)
            _, cnt = np.unique(y, return_counts=True); pr = cnt / cnt.sum()
            H = float(-(pr * np.log2(pr)).sum())
            panel[prop] = {**est, "sample_entropy": H, "sufficiency": min(1.0, est["bits"] / H)}
    # controls (must fail): a random-vector lens through the same chunk path; the best lens with labels permuted
    best = max(mean_bits, key=mean_bits.get)
    Cb = unit(np.load(a.chunk_slots / f"{best}.npy").astype(np.float64))
    Cr = unit(np.random.default_rng(7).normal(size=(len(crow), 768)))
    perm_rng = np.random.default_rng(20261019)
    controls = {"random_lens": {}, "labels_permuted": {}}
    for prop, an in anchors.items():
        rows, y = an["rows"], np.array(an["labels"])
        controls["random_lens"][prop] = ksg(maxsim_profiles([Cr[by_row[r]] for r in rows], rows), y)["bits"]
        controls["labels_permuted"][prop] = ksg(maxsim_profiles([Cb[by_row[r]] for r in rows], rows), perm_rng.permutation(y))["bits"]
    controls = {k: {"per_anchor": v, "mean_bits": float(np.mean(list(v.values())))} for k, v in controls.items()}
    controls_pass = all(v < FLOOR for c in controls.values() for v in c["per_anchor"].values())
    print("controls", json.dumps({k: round(v["mean_bits"], 4) for k, v in controls.items()}), flush=True)
    out = {"schema": "calyx-bits-port-v2", "controls": controls, "controls_pass": controls_pass, "representation_of_record": "chunk_multi (maxsim landmark profile)",
           "estimator": "ross_mixed_continuous_discrete_ksg_k3_chebyshev_subsample_root_999 (numpy seed 0)",
           "pair_estimator": "partitioned_histogram_nmi on cosine-to-own-centroid, max over anchors",
           "anchors": {q: {"classes": an["classes"], "n": len(an["rows"])} for q, an in anchors.items()},
           "per_lens": res, "mean_bits": mean_bits,
           "mean_bits_pooled": {n: (float(np.mean(v)) if (v := [res[n][q]["pooled"]["bits"] for q in anchors if res[n][q]["pooled"]["bits"] is not None]) else None) for n in names},
           "refused_lenses": refused,
           "standalone_ci_pass_counts": {n: int(sum(res[n][q]["chunk_multi"]["ci_low"] >= FLOOR for q in anchors)) for n in names},
           "pair_nmi": pair, "reference_roster": kept, "reference_roster_steps": steps,
           "meaning_rule_met": False, "served": False, "shannon_conditional_certified": False,
           "scientifically_admitted_lenses": [], "joint_panel_kind": "reference-roster diagnostic; not certified sufficiency",
           "joint_panel": panel, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "wall_seconds": time.perf_counter() - started}
    out["screening"] = screen(plain(out))
    out["screening_code_sha256"] = hashlib.sha256(Path(__file__).with_name("calyx_screen.py").read_bytes()).hexdigest()
    a.output.mkdir(parents=True, exist_ok=True)
    publish(a.output / "calyx-bits.json", plain(out))
    print(json.dumps({"reference_roster": kept, "controls_pass": controls_pass, "meaning_rule_met": False}))


if __name__ == "__main__":
    main()
