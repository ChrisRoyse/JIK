"""Calibration of the admission instrument with known-information controls (issue #12; operator review 2026-10-17: "calibrate
with known-information controls, then rerun admission on real data"). On real questions and a real one-lens panel, the
instrument (vinfo v3 diagnostic: grouped folds, cluster bootstrap, five fold seeds, gain against the panel's random-Fourier
expansion) is run on candidates whose truth is known:

- known-info high / low: Ŷ = Y kept with probability 1 − ε, else a uniformly drawn other class (seeded), one-hot. Its mutual
  information I(Y; Ŷ) is computed exactly from the empirical class distribution and ε. The standalone usable information
  must not exceed it beyond the interval, the high control (≈ 0.3–0.5 bits) must be admitted from an empty panel, the low
  control (≈ 0.02 bits) must not.
- noise: seeded Gaussian columns; must not be admitted.
- computation: a random-Fourier expansion of the panel under a different seed (a function of Φ only, zero Shannon information
  beyond Φ). Its usable gain is reported (it may be positive: the probe gains by computation); its nonlinear-control gain must stay
  below the floor for this control. Passing one computation control does not certify conditional information;
  the developer audit includes a high-frequency deterministic counterexample that passes both probe gates.
- shuffled: the panel lens's own block with its rows permuted; must not be admitted.

All controls passing is diagnostic only: SHANNON_CONDITIONAL_CERTIFIED stays false. Thresholds are not moved.
Writes <evidence>/calib-vinfo-<stamp>/calib.json. Cost: per question
O(controls × SEEDS × 4 × FOLDS × fit(n, width)), CPU.
    python -I scripts/lens/calib_vinfo.py --root R --out O --version p0-v1 --questions a2-wikidata-p54,a4-scripture-genre-keyfree,a3-gdelt-dyad-7d
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vinfo  # noqa: E402
from assay import MIN_SIGNAL_BITS, load_slot  # noqa: E402
from lenscommon import Paths, lens_root, read_family, run_stamp, sealed_registry, write_json  # noqa: E402


def noisy_label(y: np.ndarray, eps: float, seed: int) -> tuple[np.ndarray, float]:
    """(Ŷ, exact I(Y;Ŷ) in bits) for the symmetric channel keeping Y with probability 1 − eps."""
    classes, counts = np.unique(y, return_counts=True)
    K, p = len(classes), counts / counts.sum()
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y)) < eps
    other = (np.searchsorted(classes, y) + rng.integers(1, K, len(y))) % K
    yh = np.where(flip, classes[other], y)
    # P(Ŷ=j | Y=i) = 1-eps if i == j else eps/(K-1)
    T = np.full((K, K), eps / (K - 1))
    np.fill_diagonal(T, 1 - eps)
    J = p[:, None] * T
    q = J.sum(0)
    mi = float(np.nansum(J * np.log2(J / (p[:, None] * q[None, :]))))
    onehot = np.zeros((len(y), K))
    onehot[np.arange(len(y)), np.searchsorted(classes, yh)] = 1.0
    return onehot, mi


def eps_for(y: np.ndarray, target: float) -> float:
    lo, hi = 0.0, 1 - 1 / len(np.unique(y))
    for _ in range(60):
        mid = (lo + hi) / 2
        if noisy_label(y, mid, 0)[1] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_question(root: Path, fam: str, registry: dict) -> dict:
    rows_cal, summary = read_family(root / "cal", fam)
    y = np.asarray([r["y"] for r in rows_cal], np.int64)
    groups = np.unique([str(r["cluster"]) for r in rows_cal], return_inverse=True)[1].astype(np.int64)
    rep = json.loads((root / "assay" / f"{fam}.json").read_text(encoding="utf-8"))
    ranked = sorted(((s["bits_mean"] or -1, nm) for nm, s in rep["lenses"].items() if s.get("coverage", 0) >= 0.99 and s.get("kind") != "multi" and not registry["lenses"].get(nm, {}).get("error")), reverse=True)
    base = ranked[0][1]
    X, absent, _ = load_slot(root / "slots" / fam / f"{base}.npz", fam)
    keep = np.flatnonzero(~absent)
    blocks, rows = {base: vinfo.block(X[keep])}, {base: keep}
    n = len(y)
    allrows = np.arange(n)
    out = {"question": fam, "n": n, "clusters": int(groups.max() + 1), "H_Y_bits": summary["H_Y_bits"], "panel": [base], "controls": {}}
    hi_eps, lo_eps = eps_for(y, 0.4), eps_for(y, 0.02)
    ctl = {}
    for name, eps in (("known_info_high", hi_eps), ("known_info_low", lo_eps)):
        oh, mi = noisy_label(y, eps, 11)
        ctl[name] = (oh, {"eps": eps, "true_mi_bits": mi})
    ctl["noise"] = (np.random.default_rng(12).normal(size=(n, 64)), {})
    E = np.zeros((n, vinfo.RFF_DIMS))
    saved = vinfo.RFF_SEED
    try:
        vinfo.RFF_SEED = saved + 99
        E[keep] = vinfo.expansion(blocks[base])
    finally:
        vinfo.RFF_SEED = saved
    ctl["computation"] = (E, {"note": "random Fourier features of the panel lens, seed RFF_SEED+99"})
    perm = np.random.default_rng(13).permutation(len(keep))
    Bs = blocks[base][perm]
    blocks["known_info_high"], rows["known_info_high"] = ctl["known_info_high"][0], allrows
    blocks["known_info_low"], rows["known_info_low"] = ctl["known_info_low"][0], allrows
    blocks["noise"], rows["noise"] = vinfo.block(ctl["noise"][0]), allrows
    blocks["computation"], rows["computation"] = vinfo.block(E[keep]), keep
    blocks["shuffled"], rows["shuffled"] = Bs, keep
    ctl["shuffled"] = (None, {"note": "the panel lens block with rows permuted"})
    for name, (_Z, meta) in ctl.items():
        alone = vinfo.confirm(blocks, rows, y, [], name, groups) if name.startswith("known") else None
        cond = vinfo.confirm(blocks, rows, y, [base], name, groups)
        rec = {**meta, "conditional": {k: v for k, v in cond.items() if k not in ("common", "diffs")}, "admitted_conditional": vinfo.admissible(cond, MIN_SIGNAL_BITS, False)}
        if alone is not None:
            rec["standalone"] = {k: v for k, v in alone.items() if k not in ("common", "diffs")}
            rec["admitted_standalone"] = vinfo.admissible(alone, MIN_SIGNAL_BITS, True)
            rec["standalone_not_above_truth"] = bool(min(s["lo"] for s in alone["seeds"]) <= meta["true_mi_bits"] + 1e-9)
            rec["conditional_not_above_truth"] = bool(min(s["lo"] for s in cond["seeds"]) <= meta["true_mi_bits"] + 1e-9)
        out["controls"][name] = rec
    c = out["controls"]
    out["pass"] = {
        "known_info_high_admitted_alone": c["known_info_high"]["admitted_standalone"],
        "known_info_bounds_respected": all(c[k]["standalone_not_above_truth"] and c[k]["conditional_not_above_truth"] for k in ("known_info_high", "known_info_low")),
        "known_info_low_refused": not c["known_info_low"]["admitted_standalone"] and not c["known_info_low"]["admitted_conditional"],
        "noise_refused": not c["noise"]["admitted_conditional"],
        "computation_refused": not c["computation"]["admitted_conditional"],
        "shuffled_refused": not c["shuffled"]["admitted_conditional"],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--issue", default="12")
    ap.add_argument("--questions", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    root = lens_root(paths)
    registry = sealed_registry(root)
    res = [run_question(root, q, registry) for q in args.questions.split(",")]
    ok = all(all(r["pass"].values()) for r in res)
    rdir = paths.evidence(args.issue, f"calib-vinfo-{run_stamp()}")
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "calib.json", {"instrument": vinfo.INSTRUMENT, "floor": MIN_SIGNAL_BITS, "questions": res,
                                    "all_pass": ok, "shannon_conditional_certified": False, "acceptance": "diagnostic only"})
    for r in res:
        print(r["question"], json.dumps(r["pass"]), {k: round(v.get("conditional", {}).get("gain_mean", math.nan), 4) for k, v in r["controls"].items()},
              {k: round(v.get("conditional", {}).get("nonlinear_gain_mean", math.nan), 4) for k, v in r["controls"].items()}, flush=True)
    print(f"all_pass={ok} report {rdir / 'calib.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
