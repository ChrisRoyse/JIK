"""The Stage 3 usable-information diagnostic, instrument v3 (issue #12; operator review of
2026-10-17). What it measures: held-out *usable* conditional information for a declared probe family (V-information, Xu et
al. 2020), I_V(φ → Y | Φ) = CE_cv(Y | Φ) − CE_cv(Y | Φ ∪ {φ}) in bits, where CE_cv is the per-sample held-out cross-entropy
of an L2 multinomial logistic probe (C fixed) over the lenses' probe blocks. What it does not measure: a lower bound on the
Shannon conditional mutual information. Each held-out cross-entropy upper-bounds its own conditional entropy, and the
difference of two upper bounds bounds nothing; usable information can also rise through computation alone (a nonlinear
function of Φ makes a linear probe better while adding zero information about Y beyond Φ). v1's docstring claimed the bound;
that claim is withdrawn. The finite nonlinear control also cannot certify Shannon information:

- `confirm` measures gain against the panel and against its fixed random-Fourier expansion. Both are usable-information
  diagnostics. A deterministic high-frequency function of the panel can pass both with zero new Shannon information.
  Neither licenses an admitted sense, a grounded meaning, or a sufficiency certificate. Publication stays unserved.
- The doctrine's confidence contract (#1: lower confidence bound at 0.05 bits, five estimator seeds, entity-clustered
  bootstrap). Folds are StratifiedGroupKFold by the row's entity/event cluster, intervals are cluster bootstraps, and
  `confirm` repeats the measurement over five fold seeds; the admission rule (assay.py) reads the minimum lower bound.

Blocks: dense slots are reduced by the deterministic BLAKE3 sign projection to at most PROBE_DIMS columns; scaling and
nonlinear-control normalization are fitted only on training folds. A Multi slot enters through its late-interaction view
(slotio.maxsim_landmarks). Cost per greedy step: O(candidates × FOLDS × fit(n, width)) for the screen, plus
O(SEEDS × 4 × FOLDS × fit) for each candidate that passes the screen; CPU, parallel over candidates."""
from __future__ import annotations

import math

import blake3
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from threadpoolctl import threadpool_limits

PROBE_DIMS = 128
PROBE_SEED = 7000
FOLDS = 5
C_L2 = 1.0
CV_SEED = 0
SEEDS = 5  # estimator seeds of the doctrine's confidence contract (#1): fold assignments CV_SEED .. CV_SEED + SEEDS - 1
BOOT = 1000
MAX_ITER = 3000
RFF_DIMS = 256
RFF_SEED = 7100
INSTRUMENT = "vinfo-v3-diagnostic"
SHANNON_CONDITIONAL_CERTIFIED = False
_SIGN: dict[tuple[int, int, int], np.ndarray] = {}


def sign_matrix(d: int, dims: int, seed: int = PROBE_SEED) -> np.ndarray:
    key = (d, dims, seed)
    if key not in _SIGN:
        S = np.empty((d, dims), np.float32)
        for i in range(d):
            h = blake3.blake3(seed.to_bytes(8, "big") + i.to_bytes(8, "big")).digest(length=dims)
            S[i] = [1.0 if (b & 1) == 0 else -1.0 for b in h]
        _SIGN[key] = S
    return _SIGN[key]


def block(X, dims: int = PROBE_DIMS):
    """One lens's diagnostic block: dense -> fixed projection if wider than dims; sparse -> CSR. No fitted scaling here."""
    if sp.issparse(X):
        return sp.csr_matrix(X, dtype=np.float64)
    X = np.asarray(X, np.float64)
    if X.shape[1] > dims:
        X = (X @ sign_matrix(X.shape[1], dims)) / math.sqrt(dims)
    return X  # scaling is fitted on the training fold, never on held-out rows


def hstack(blocks: list):
    if any(sp.issparse(b) for b in blocks):
        return sp.hstack([sp.csr_matrix(b) for b in blocks], format="csr")
    return np.concatenate(blocks, axis=1)


def expansion(X, fit_rows=None) -> np.ndarray:
    """Random Fourier features of the panel's own probe matrix (computation control): the panel projected to PROBE_DIMS dense
    columns, z-scored, then sqrt(2/D) cos(Z W + b) with W ~ N(0, 1/PROBE_DIMS), b ~ U(0, 2π), seeded. A function of Φ only."""
    Z = X @ sign_matrix(X.shape[1], PROBE_DIMS) / math.sqrt(PROBE_DIMS) if X.shape[1] > PROBE_DIMS else X
    Z = np.asarray(Z.todense() if sp.issparse(Z) else Z, np.float64)
    fit = Z if fit_rows is None else Z[fit_rows]
    sd = fit.std(0)
    Z = np.where(sd > 0, (Z - fit.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
    if not Z.shape[1]:
        raise ValueError("MNLM_PROBE_NO_FEATURES: empty nonlinear control")
    rng = np.random.default_rng(RFF_SEED)
    W = rng.normal(0.0, 1.0 / math.sqrt(Z.shape[1]), (Z.shape[1], RFF_DIMS))
    b = rng.uniform(0.0, 2 * math.pi, RFF_DIMS)
    return math.sqrt(2.0 / RFF_DIMS) * np.cos(Z @ W + b)


def cv_cross_entropy(X, y: np.ndarray, groups: np.ndarray | None = None, folds: int = FOLDS, C: float = C_L2, seed: int = CV_SEED, expansion_inputs=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample held-out cross-entropy in bits of (the class prior of the training fold, the logistic probe); folds keep
    every row of one cluster on the same side (StratifiedGroupKFold), so a pair or entity seen in training is not scored."""
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(y) < 50 or len(groups) != len(y) or X.shape[0] != len(y) or len(np.unique(groups)) < folds:
        raise ValueError("MNLM_PROBE_INSUFFICIENT_SUPPORT: need paired rows and at least one group per fold")
    if not np.isfinite(X.data if sp.issparse(X) else X).all():
        raise ValueError("MNLM_PROBE_NONFINITE: nonfinite input")
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    classes = np.unique(y)
    prior, probe = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        if not len(te) or len(classes) < 2 or not np.array_equal(np.unique(y[tr]), classes):
            raise ValueError("MNLM_PROBE_CLASS_SUPPORT: every training fold must contain every outcome class")
        freq = np.clip(np.array([(y[tr] == c).mean() for c in classes]), 1e-6, 1)
        prior[te] = -np.log2(freq[np.searchsorted(classes, y[te])])
        design = X if expansion_inputs is None else hstack([X, expansion(expansion_inputs, tr)])
        scaler = StandardScaler(with_mean=not sp.issparse(design))
        train = scaler.fit_transform(design[tr])
        test = scaler.transform(design[te])
        clf = LogisticRegression(C=C, max_iter=MAX_ITER)
        with threadpool_limits(limits=1):  # one BLAS thread: the lbfgs path, hence the probe, is identical in a joblib worker and in a serial replay
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    clf.fit(train, y[tr])
                except ConvergenceWarning as exc:
                    raise ValueError("MNLM_PROBE_NOT_CONVERGED: probe did not converge") from exc
        p = clf.predict_proba(test)
        probe[te] = -np.log2(np.clip(p[np.arange(len(te)), np.searchsorted(clf.classes_, y[te])], 1e-6, 1))
    return prior, probe


def paired_interval(diff: np.ndarray, groups: np.ndarray | None = None, resamples: int = BOOT, seed: int = CV_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Cluster percentile bootstrap of the mean paired difference at level 1 - alpha: clusters are drawn with replacement and
    the statistic is the row mean over the drawn clusters (alpha below 0.05 is the Bonferroni level of the override)."""
    groups = np.arange(len(diff)) if groups is None else np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=diff), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    g = len(sums)
    draws = rng.integers(0, g, (resamples, g))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def _common(rows: dict, names: list[str]) -> np.ndarray:
    common = rows[names[0]]
    for nm in names[1:]:
        common = np.intersect1d(common, rows[nm])
    return common


def joint(blocks: dict, rows: dict, y_all: np.ndarray, names: list[str], groups_all: np.ndarray | None = None, seed: int = CV_SEED, expand: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(common rows, prior CE, probe CE) of the probe over the concatenated blocks of `names` on the rows all of them cover;
    `expand` adds the random-Fourier expansion of those blocks (the computation control) to the design."""
    common = _common(rows, names)
    parts = [blocks[q][np.searchsorted(rows[q], common)] for q in names]
    extra = hstack([blocks[q][np.searchsorted(rows[q], common)] for q in expand]) if expand else None
    g = None if groups_all is None else groups_all[common]
    prior, probe = cv_cross_entropy(hstack(parts), y_all[common], g, seed=seed, expansion_inputs=extra)
    return common, prior, probe


def conditional_gain(blocks: dict, rows: dict, y_all: np.ndarray, panel: list[str], cand: str, cur: tuple | None, groups_all: np.ndarray | None = None) -> dict:
    """Screen (fold seed CV_SEED): held-out gain of adding `cand` to `panel`; `cur` is (common rows, probe CE) of the panel or
    None for the empty panel. The interval is the cluster bootstrap."""
    common, prior, probe = joint(blocks, rows, y_all, panel + [cand], groups_all)
    # Missingness changes population and fold assignments. The baseline is invariant
    # across candidates only when their common rows equal the cached panel rows.
    if not panel:
        base = prior
    elif cur is not None and np.array_equal(common, cur[0]):
        base = cur[1]
    else:
        base = _restricted(blocks, rows, y_all, panel, common, groups_all, CV_SEED)
    diff = base - probe
    lo, hi = paired_interval(diff, None if groups_all is None else groups_all[common])
    return {"joint_usable_bits": float((prior - probe).mean()), "gain": float(diff.mean()), "gain_ci_low": lo, "gain_ci_high": hi, "n": int(len(common)), "common": common, "probe": probe, "prior": prior, "base": base}


def confirm(blocks: dict, rows: dict, y_all: np.ndarray, panel: list[str], cand: str, groups_all: np.ndarray | None = None, alpha: float = 0.05) -> dict:
    """Diagnostic over SEEDS fold seeds, on rows panel and candidate share: usable gain
    CE(Φ) − CE(Φ ∪ φ) and nonlinear-control gain CE(Φ ∪ rff(Φ)) − CE(Φ ∪ rff(Φ) ∪ φ), each with its cluster
    bootstrap lower bound. Returns per-seed values, the per-row differences (for stratum reads) and the minima."""
    common = _common(rows, panel + [cand])
    g = None if groups_all is None else groups_all[common]
    per, diffs = [], []
    for s in range(CV_SEED, CV_SEED + SEEDS):
        _, prior, with_c = joint(blocks, rows, y_all, panel + [cand], groups_all, seed=s)
        base = prior if not panel else _restricted(blocks, rows, y_all, panel, common, groups_all, s)
        d = base - with_c
        lo, hi = paired_interval(d, g, seed=s, alpha=alpha)
        e = {"seed": s, "gain": float(d.mean()), "lo": lo, "hi": hi}
        if panel:
            pe = _restricted(blocks, rows, y_all, panel, common, groups_all, s, expand=True)
            ce = _restricted(blocks, rows, y_all, panel + [cand], common, groups_all, s, expand=True, expand_names=panel)
            de = pe - ce
            elo, ehi = paired_interval(de, g, seed=s, alpha=alpha)
            e.update({"nonlinear_gain": float(de.mean()), "nonlinear_lo": elo, "nonlinear_hi": ehi})
            diffs.append((d, de))
        else:
            diffs.append((d, d))
        per.append(e)
    out = {"seeds": per, "n": int(len(common)), "gain_mean": float(np.mean([p["gain"] for p in per])), "lcb_min": float(min(p["lo"] for p in per))}
    if panel:
        out.update({"nonlinear_gain_mean": float(np.mean([p["nonlinear_gain"] for p in per])), "nonlinear_lcb_min": float(min(p["nonlinear_lo"] for p in per))})
    out["common"], out["diffs"] = common, diffs
    return out


def _restricted(blocks, rows, y_all, names, common, groups_all, seed, expand=False, expand_names=None):
    """Probe CE of `names` on exactly the rows `common` (a subset of the rows every name covers)."""
    parts = [blocks[q][np.searchsorted(rows[q], common)] for q in names]
    extra = None
    if expand:
        en = expand_names or names
        extra = hstack([blocks[q][np.searchsorted(rows[q], common)] for q in en])
    g = None if groups_all is None else groups_all[common]
    return cv_cross_entropy(hstack(parts), y_all[common], g, seed=seed, expansion_inputs=extra)[1]


def admissible(conf: dict, floor: float, panel_empty: bool) -> bool:
    """Diagnostic candidate selection only: five finite usable/nonlinear lower bounds above the floor.
    This predicate cannot certify the doctrine's conditional Shannon-information admission contract."""
    seeds = conf.get("seeds", [])
    if [s.get("seed") for s in seeds] != list(range(CV_SEED, CV_SEED + SEEDS)):
        return False
    values = [s.get("lo", float("nan")) for s in seeds]
    if not panel_empty:
        values += [s.get("nonlinear_lo", float("nan")) for s in seeds]
    return all(np.isfinite(v) and v >= floor for v in values)


def gains_parallel(blocks: dict, rows: dict, y_all: np.ndarray, panel: list[str], cands: list[str], cur: tuple | None, jobs: int, groups_all: np.ndarray | None = None) -> dict[str, dict]:
    def one(nm):
        try:
            return nm, conditional_gain(blocks, rows, y_all, panel, nm, cur, groups_all)
        except ValueError as exc:
            return nm, {"error": "MNLM_PROBE_REFUSED", "message": str(exc), "remediation": "inspect class/group support, finite values and probe convergence"}

    return dict(Parallel(n_jobs=jobs)(delayed(one)(nm) for nm in cands))
