"""Rejected CMI commissioning experiment (#12, PC-04/07/35); CPU reference only.

Implements the published eqs. 18-20 of arXiv:1912.03387 independently from the
mathematics. No GPL source is imported. Unlike the authors' repository revision,
eq19 uses log at tied neighborhoods. For empty Z, n_Z=n (paper's MI convention).
No projection, learned probe, fitted scaling, jitter or pooled conditioning.

This is NOT an admission instrument. Asymptotic consistency is not a finite-
sample bias bound. The deterministic-function controls have true CMI=0 but
nonzero estimates; confidence intervals around the estimate would not cure that.
Controls are artificial boundary inputs, never production data. Native retained
slots are read only for a separately labelled real-input redundancy control.
Cost O(n^2*d) for distances, O(n^2) scratch; no coordinate-by-n-by-n tensor.
Run -I with --input <developer-audit input> --output <new evidence directory>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from slotio import slot_digest

K = 7  # paper's experimental value; fixed before the controls, never tuned
FLOOR = 0.05
SEEDS = tuple(range(42, 47))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def counts(dx, dy, dz):
    """All counts exclude self; exact kth neighbor includes boundary ties."""
    joint = np.maximum(np.maximum(dx, dy), dz)
    radius = np.partition(joint, K, axis=1)[:, K]
    r = radius[:, None]
    kt = np.count_nonzero(joint <= r, axis=1) - 1
    nxz = np.count_nonzero(np.maximum(dx, dz) <= r, axis=1) - 1
    nyz = np.count_nonzero(np.maximum(dy, dz) <= r, axis=1) - 1
    nz = np.count_nonzero(dz <= r, axis=1) - 1
    return np.stack((kt, nxz, nyz, nz), axis=1), radius


def equation(c):
    kt, nxz, nyz, nz = c.T
    if np.any(c < K):
        raise ValueError("MNLM_CMI_COUNT: neighborhood counts below k")
    continuous = digamma(kt) - digamma(nxz) - digamma(nyz) + digamma(nz)
    tied = np.log(kt) - np.log(nxz) - np.log(nyz) + np.log(nz)
    return np.where(kt == K, continuous, tied) / math.log(2)


def reference(x, y, z, rows):
    """Separate scalar count implementation; no cdist, partition or equation()."""
    out, vals = [], []
    for i in rows:
        neighbors = []
        for j in range(len(y)):
            if i == j:
                continue
            a = max(abs(float(v) - float(w)) for v, w in zip(x[i], x[j]))
            b = float(y[i] != y[j])
            c = max(abs(float(v) - float(w)) for v, w in zip(z[i], z[j]))
            neighbors.append((a, b, c))
        r = sorted(max(a, b, c) for a, b, c in neighbors)[K - 1]
        kn = sum(max(a, b, c) <= r for a, b, c in neighbors)
        xz = sum(max(a, c) <= r for a, b, c in neighbors)
        yz = sum(max(b, c) <= r for a, b, c in neighbors)
        zn = sum(c <= r for a, b, c in neighbors)
        out.append((kn, xz, yz, zn))
        # psi(m) = H_(m-1) - gamma; constants cancel in this four-term sum.
        f = (lambda m: math.fsum(1.0 / j for j in range(1, m))) if kn == K else math.log
        vals.append((f(kn) - f(xz) - f(yz) + f(zn)) / math.log(2))
    return np.asarray(out), np.asarray(vals)


def case(root, name, x, y, z, truth, kind, extra=None):
    start = time.monotonic()
    x, y, z = np.asarray(x, float), np.asarray(y), np.asarray(z, float)
    if len(y) <= K or any(not np.isfinite(a).all() for a in (x, y, z)):
        raise ValueError("MNLM_CMI_INPUT: finite aligned observations required")
    if x.ndim != 2 or z.ndim != 2 or y.shape != (len(x),) or len(z) != len(y):
        raise ValueError("MNLM_CMI_INPUT: shape mismatch")
    p = root / (name + ".npz")
    with p.open("xb") as f:
        np.savez(f, x=x, y=y, z=z)
    # Read the actual persisted inputs; array return values are not acceptance.
    with np.load(p, allow_pickle=False) as q:
        x, y, z = q["x"], q["y"], q["z"]
    dz = cdist(z, z, "chebyshev")
    dx = dz if np.array_equal(x, z) else cdist(x, x, "chebyshev")
    dy = np.not_equal(y[:, None], y[None, :]).astype(float)
    c, radius = counts(dx, dy, dz)
    local = equation(c)
    with (root / (name + ".counts.npz")).open("xb") as f:
        np.savez(f, counts=c, radius=radius, local_bits=local)
    rows = np.unique(np.linspace(0, len(y) - 1, 8, dtype=int))
    ref, values = reference(x, y, z, rows)
    with np.load(root / (name + ".counts.npz"), allow_pickle=False) as saved:
        count_ok = bool(np.array_equal(saved["counts"][rows], ref))
        local_error = float(np.max(np.abs(saved["local_bits"][rows] - values)))
        replay = float(equation(saved["counts"]).mean())
    bits = float(local.mean())
    return {"name": name, "kind": kind, "n": len(y), "x_dims": x.shape[1], "z_dims": z.shape[1],
            "k": K, "true_cmi_bits": truth, "estimate_bits": bits, "positive_part_bits": max(0, bits),
            "point_exceeds_floor": bits >= FLOOR, "admitted": False,
            "independent_rows": rows.tolist(), "independent_counts_equal": count_ok,
            "independent_local_error": local_error, "cold_replay_equal": replay == bits,
            "input_sha256": sha(p), "counts_sha256": sha(root / (name + ".counts.npz")),
            "wall_seconds_measured": time.monotonic() - start, **(extra or {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, source = Path(args.output), Path(args.input)
    root.mkdir(parents=True, exist_ok=False)
    results, start = [], time.monotonic()
    for n in (600, 2000):
        for seed in SEEDS:
            z = np.random.default_rng(seed).uniform(-1, 1, (n, 1))
            for frequency in (80, 512):
                y = (np.sin(frequency * z[:, 0]) > 0).astype(int)
                results.append(case(root, f"computed-{n}-{seed}-{frequency}", y[:, None], y, z, 0,
                                    "artificial_zero_cmi_boundary", {"seed": seed, "frequency": frequency}))
        print(f"persisted and cold-checked controls n={n}", flush=True)
    rng = np.random.default_rng(52)
    z = rng.uniform(-1, 1, (2000, 1))
    y = rng.integers(0, 2, 2000)
    results.append(case(root, "independent", rng.normal(size=(2000, 1)), y, z, 0, "artificial_zero_cmi_boundary"))
    results.append(case(root, "one-bit-positive", y[:, None], y, z, 1, "artificial_positive_boundary"))
    # Balanced finite categorical XOR: exact empirical CMI = 1 bit. This
    # exercises the log/ties branch and dependence only visible conditionally.
    bits = np.tile(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]), (150, 1))
    results.append(case(root, "xor", bits[:, :1], bits[:, 0] ^ bits[:, 1], bits[:, 1:], 1,
                        "artificial_positive_boundary"))
    # Removing a conditioning coordinate changes the estimand. This control
    # makes that failure observable without calling projection a valid bound.
    full_z = np.column_stack((z, y))
    results.append(case(root, "full-conditioning", y[:, None], y, full_z, 0,
                        "artificial_zero_cmi_boundary"))
    results.append(case(root, "lossy-conditioning", y[:, None], y, z, 1,
                        "changed_estimand_boundary", {"note": "Y coordinate intentionally removed from Z; different CMI"}))
    slot = source / "record-vector.npz"
    labels = source / "a2-wikidata-p54.jsonl"
    before = {str(p): sha(p) for p in (slot, labels)}
    layout = slot_digest(slot)
    with np.load(slot, allow_pickle=False) as q:
        keep, x = ~q["absent"], q["X"]
    records = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    if len(records) != len(x) or any(r["source"] != "wikidata5m" for r in records):
        raise ValueError("MNLM_CMI_SOURCE: expected paired retained Wikidata observations")
    y = np.asarray([r["y"] for r in records])[keep]
    x = x[keep]
    results.append(case(root, "real-slot-duplicate", x, y, x, 0, "real_input_redundancy_boundary",
                        {"note": "native full coordinates, no projection; duplicate of conditioning slot",
                         "source_layout": layout, "independent_clusters": len({r["cluster"] for r in records})}))
    after = {str(p): sha(p) for p in (slot, labels)}
    correct = all(r["independent_counts_equal"] and r["independent_local_error"] < 1e-12
                  and r["cold_replay_equal"] for r in results) and before == after
    failed = [r["name"] for r in results if r["true_cmi_bits"] == 0 and r["point_exceeds_floor"]]
    named = {r["name"]: r for r in results}
    diagnostics = {"duplicate_zero_within_roundoff": abs(named["real-slot-duplicate"]["estimate_bits"]) < 1e-12,
                   "independence_below_floor": named["independent"]["estimate_bits"] < FLOOR,
                   "positive_signal_detected": named["one-bit-positive"]["estimate_bits"] >= FLOOR,
                   "categorical_xor_near_one": abs(named["xor"]["estimate_bits"] - 1) < 0.01,
                   "conditioning_loss_changes_estimand": named["full-conditioning"]["estimate_bits"] == 0
                       and named["lossy-conditioning"]["estimate_bits"] >= FLOOR}
    correct = correct and all(diagnostics.values())
    report = {"schema": "cmi-commissioning-audit-v1", "source": "https://arxiv.org/abs/1912.03387",
              "implementation_sha256": sha(__file__), "author_checked": correct, "independently_accepted": False,
              "shannon_conditional_certified": False, "served": False, "commissioned": False,
              "status": "MNLM_CMI_FINITE_SAMPLE_BIAS", "false_point_signals": failed,
              "diagnostic_checks": diagnostics,
              "floor_bits_unchanged": FLOOR, "source_before": before, "source_after": after,
              "rows": results, "wall_seconds_measured": time.monotonic() - start,
              "scope": "Numerical experiment; neither point estimates nor sampling variability certify admission."}
    write(root / "report.json", report)
    # Cold read report and every explicitly created file; no recursive discovery.
    cold = json.loads((root / "report.json").read_text())
    files = [root / "report.json"] + [root / (r["name"] + suffix) for r in cold["rows"]
                                    for suffix in (".npz", ".counts.npz")]
    write(root / "manifest.json", {"files": {p.name: sha(p) for p in files},
                                    "retention_owner": "ChrisRoyse/MNLM#12"})
    print(json.dumps({"checks_correct": correct, "cases": len(results), "false_point_signals": len(failed),
                      "commissioned": False, "report": str(root / "report.json")}))
    return 0 if correct else 1  # successful audit can (and does) reject commissioning


if __name__ == "__main__":
    raise SystemExit(main())
