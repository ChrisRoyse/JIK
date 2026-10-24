"""Stage 3 measurement and diagnostic candidate selection (#12).

Standalone KSG measurements use five seeds. A KSG degeneracy is recorded explicitly;
usable-information probes provide a separate diagnostic, never a Shannon substitute.
Greedy candidates must pass coverage, pairwise correlation and five grouped-fold
usable/nonlinear-control gain intervals. These intervals do not certify conditional
Shannon information. Joint probe information is never floored at a different estimator's
standalone value. Sufficiency remains unmeasured and every family stays unserved.

Multi slots retain token structure; their diagnostic view is tiled symmetric MaxSim to
label-free landmarks. Source slots and calibration bytes are checked against the sealed
registry before use. Reports bind registry, calibration and instrument digests.
Cost: standalone O(L * seeds * resamples * n^2) on GPU, L approximately 70 and n
approximately 2,000 in this calibration; conditional diagnostics O(steps * L * folds *
fit(n, 128*panel_size)) on CPU. The baseline is cached only for identical rows;
changed coverage requires a refit on the candidate's exact population and folds.
    python -I scripts/lens/assay.py --root R --out O --version p0-v1 [--families a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import blake3
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ksg  # noqa: E402
import slotio  # noqa: E402
import vinfo  # noqa: E402
import admission_state  # noqa: E402
from lenscommon import HOST, sealed_registry, STRATA_FIELDS, LensError, Paths, families_present, hash_order, ledger_append, lens_root, now_iso, read_family, require_gpu, spec_family, take_lease, write_json  # noqa: E402

MIN_SIGNAL_BITS = 0.05
MAX_PAIRWISE_CORR = 0.6
MIN_LENSES_PER_MEANING = 5
MIN_COVERAGE = 0.9
MIN_STRATUM = 50  # MIN_ASSAY_SAMPLES: the smallest stratum the override may read
MIN_PAIRS_VALID = 500
N_PAIRS = 4096
BASE_PROJECTION_SEED = 1000
BASELINE_LENS = "record-vector"


def load_slot(path: Path, family: str = "") -> tuple[object, np.ndarray, str]:
    """(assay view X [n,d] dense or CSR, absent mask, kind) from a slot file. A Multi slot enters through its native
    late-interaction view (slotio.maxsim_landmarks: symmetric MaxSim to label-free landmark atoms), not a mean token vector
    (instrument v3 diagnostic: averaging discards the token-level structure the lens exists to keep)."""
    z = np.load(path)
    absent = z["absent"].astype(bool)
    if "tokens" in z.files:
        X, _lm = slotio.maxsim_landmarks(path, absent, family)
        return X, absent, "multi"
    if "data" in z.files:
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), absent, "sparse"
    return z["X"], absent, "dense"


def fixed_pairs(family: str, n: int, seed: int) -> np.ndarray:
    pairs = []
    t = 0
    while len(pairs) < N_PAIRS:
        i, j = hash_order(seed, "pairs", family, t, "i") % n, hash_order(seed, "pairs", family, t, "j") % n
        t += 1
        if i != j:
            pairs.append((i, j))
    return np.asarray(pairs, np.int64)


def readings(path: Path, X, absent: np.ndarray, kind: str, sim: str, pairs: np.ndarray) -> np.ndarray:
    """Similarity reading per fixed pair; NaN where either atom is absent."""
    i, j = pairs[:, 0], pairs[:, 1]
    out = np.full(len(pairs), np.nan)
    ok = ~absent[i] & ~absent[j]
    if kind == "multi":
        from multi_view import maxsim_pair, validate_multi
        toks, offs = slotio.npz_member(path, "tokens"), np.asarray(slotio.npz_member(path, "offsets"))
        validate_multi(toks, offs, absent)
        for t in np.flatnonzero(ok):
            a, b = toks[offs[i[t]]:offs[i[t] + 1]], toks[offs[j[t]]:offs[j[t] + 1]]
            out[t] = maxsim_pair(a, b)
        return out
    if sim == "neg_cheb":
        D = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X, np.float64)
        sd = D.std(0)
        Z = np.where(sd > 0, (D - D.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
        out[ok] = -np.abs(Z[i[ok]] - Z[j[ok]]).max(1) if Z.shape[1] else 0.0
        return out
    if sp.issparse(X):
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        dots = np.asarray(X[i[ok]].multiply(X[j[ok]]).sum(1)).ravel()
        den = nrm[i[ok]] * nrm[j[ok]]
        out[ok] = np.where(den > 0, dots / np.where(den > 0, den, 1), 0.0)
        return out
    Xn = np.asarray(X, np.float64)
    nrm = np.linalg.norm(Xn, axis=1)
    den = nrm[i[ok]] * nrm[j[ok]]
    out[ok] = np.where(den > 0, (Xn[i[ok]] * Xn[j[ok]]).sum(1) / np.where(den > 0, den, 1), 0.0)
    return out


def standalone(X, absent: np.ndarray, y: np.ndarray, seeds: int, resamples: int, groups: np.ndarray | None = None) -> tuple[dict, np.ndarray | None, np.ndarray]:
    rows = np.flatnonzero(~absent)
    Xu = X[rows]
    per_seed, P0 = [], None
    for s in range(seeds):
        P = ksg.project(Xu, BASE_PROJECTION_SEED + s)  # the projection is defined for every slot; only the estimate can refuse
        try:
            e = ksg.estimate(P, y[rows], resamples=resamples, seed=s, groups=None if groups is None else groups[rows])
            per_seed.append({"seed": s, "projection_seed": BASE_PROJECTION_SEED + s, **{k: e[k] for k in ("bits", "ci_low", "ci_high", "m", "rejected_draws")}, "dim": int(P.shape[1])})
        except ksg.AssayError as exc:
            per_seed.append({"seed": s, "projection_seed": BASE_PROJECTION_SEED + s, "error": exc.code, "detail": str(exc)[:200]})
        if s == 0:
            P0 = P
    ok = [p for p in per_seed if "error" not in p]
    summary = {"seeds": per_seed, "n_used": int(len(rows)), "coverage": float(len(rows) / len(absent)),
               "bits_mean": float(np.mean([p["bits"] for p in ok])) if ok else None, "lcb_min": float(min(p["ci_low"] for p in ok)) if ok else None,
               "ucb_max": float(max(p["ci_high"] for p in ok)) if ok else None, "seeds_ok": len(ok),
               "error": None if len(ok) == seeds else (per_seed[0].get("error") or next(p["error"] for p in per_seed if "error" in p))}
    return summary, P0, rows


def spearman_matrix(R: np.ndarray) -> np.ndarray:
    L = R.shape[0]
    C = np.full((L, L), np.nan)
    for a in range(L):
        C[a, a] = 1.0
        for b in range(a + 1, L):
            ok = np.isfinite(R[a]) & np.isfinite(R[b])
            if ok.sum() >= MIN_PAIRS_VALID and R[a][ok].std() > 0 and R[b][ok].std() > 0:
                C[a, b] = C[b, a] = float(spearmanr(R[a][ok], R[b][ok]).statistic)
    return C


def strata_of(fam: str, rows_cal: list[dict]) -> dict[str, np.ndarray]:
    """Declared strata (lenscommon.STRATA_FIELDS): one value of one categorical field, holding between MIN_STRATUM and n/2 rows."""
    out: dict[str, np.ndarray] = {}
    n = len(rows_cal)
    for fld in STRATA_FIELDS.get(spec_family(fam), []):
        vals = [str(r["fields"].get(fld)) for r in rows_cal]
        for v in sorted(set(vals)):
            idx = np.flatnonzero([x == v for x in vals])
            if v not in ("None", "") and MIN_STRATUM <= len(idx) <= n // 2:
                out[f"{fld}={v}"] = idx
    return out


def stratified_override(res: dict, todo: list, cur: tuple | None, strata: dict[str, np.ndarray], blocks: dict, rows: dict, y: np.ndarray, panel: list[str], groups: np.ndarray | None) -> dict:
    """spec Def. 13.3: "a stratified override admits a globally weak lens that is the sole carrier of a rare but critical
    stratum, with bits never multiplied by raw frequency". For each candidate that cleared the correlation gate, the per-row
    held-out gain already computed on the full panel is read inside each declared stratum: the stratum's cluster-bootstrap lower
    bound must be >= 0.05 bits (never scaled by the stratum's frequency) at the Bonferroni level over the strata tested, and the
    same must hold on every fold seed for the usable and the nonlinear-control gain (vinfo.confirm). The best stratum per candidate is
    returned; the caller admits the largest."""
    out = {}
    tested = [nm for nm, _ in todo if res.get(nm) is not None and not res[nm].get("error")]
    alpha = 0.05 / max(1, len(strata) * len(tested))  # every (candidate, stratum) pair the step examines is a test (readback H2b)
    for nm, _mc in todo:
        r = res.get(nm)
        if r is None or r.get("error"):
            continue
        base = r["base"]
        diff = base - r["probe"]
        g_all = None if groups is None else groups[r["common"]]
        best = None
        for sname, sidx in strata.items():
            m = np.isin(r["common"], sidx)
            if int(m.sum()) < MIN_STRATUM:
                continue
            g = float(diff[m].mean())
            lo, hi = vinfo.paired_interval(diff[m], None if g_all is None else g_all[m], alpha=alpha)
            if lo < MIN_SIGNAL_BITS:  # the contract: the lower bound, not the point, at the floor (instrument v3 diagnostic)
                continue
            if best is None or g > best["gain"]:
                best = {"stratum": sname, "n": int(m.sum()), "gain": g, "gain_ci_low": lo, "gain_ci_high": hi, "alpha": alpha, "tests": len(strata) * len(tested), "global_gain": r["gain"]}
        if best is None:
            continue
        conf = vinfo.confirm(blocks, rows, y, panel, nm, groups, alpha=alpha)  # five fold seeds, nonlinear-control gain, inside the stratum
        m = np.isin(conf["common"], strata[best["stratum"]])
        gc = None if groups is None else groups[conf["common"]][m]
        seeds = []
        for (d, de), sd in zip(conf["diffs"], conf["seeds"]):
            lo_u = vinfo.paired_interval(d[m], gc, seed=sd["seed"], alpha=alpha)[0]
            lo_e = vinfo.paired_interval(de[m], gc, seed=sd["seed"], alpha=alpha)[0]
            seeds.append({"seed": sd["seed"], "gain": float(d[m].mean()), "lo": lo_u, "nonlinear_gain": float(de[m].mean()), "nonlinear_lo": lo_e})
        best["confirm"] = {"seeds": seeds, "lcb_min": min(x["lo"] for x in seeds), "nonlinear_lcb_min": min(x["nonlinear_lo"] for x in seeds)}
        if best["confirm"]["lcb_min"] >= MIN_SIGNAL_BITS and (not panel or best["confirm"]["nonlinear_lcb_min"] >= MIN_SIGNAL_BITS):
            out[nm] = best
    return out


def admit(cands: dict[str, dict], blocks: dict, rows: dict, y: np.ndarray, C: np.ndarray, order: list[str], jobs: int, strata: dict[str, np.ndarray] | None = None,
          groups: np.ndarray | None = None) -> tuple[list[str], list[dict], dict, tuple | None]:
    """Greedy forward selection by held-out usable conditional information (vinfo.py, instrument v3 diagnostic): a candidate must clear
    the standalone diagnostic gate, the 0.6 correlation gate against every selected lens, and the cluster-
    bootstrap lower bound of its gain >= 0.05 bits on each of five fold seeds, for the usable gain and (with a panel) for the
    nonlinear-control gain against the panel plus its own random-Fourier expansion. A one-seed screen (lower bound >= 0.05 on fold seed
    0) orders the candidates; confirmation decides; the screen admits nothing by itself. When no candidate is confirmed, the
    stratified override is tried inside a stratum. This remains candidate selection, not certified D2 admission."""
    idx = {nm: i for i, nm in enumerate(order)}
    # fail closed: every KSG seed must have estimated (or every seed refused as degenerate and the held-out bound stands in)
    eligible, ineligible = [], {}
    for nm, s in cands.items():
        why = ("standalone_unreadable" if s["lcb_min"] is None else "standalone_below_gate" if s["lcb_min"] < MIN_SIGNAL_BITS else "coverage_below_gate" if s["coverage"] < MIN_COVERAGE
               else "no_block" if nm not in blocks else "ksg_seeds_refused" if not (s.get("standalone_estimator") == "vinfo" or s.get("seeds_ok") == len(s.get("seeds", []))) else None)
        if why is None:
            eligible.append(nm)
        elif s["lcb_min"] is not None and s["lcb_min"] >= MIN_SIGNAL_BITS:
            ineligible[nm] = {"reason": why, "lcb_min": s["lcb_min"], "coverage": s["coverage"], "seeds_ok": s.get("seeds_ok")}  # readback G2: a dropped carrier is recorded, not silent
    panel, trace, J, cur = [], [], 0.0, None
    while True:
        step = {"panel": list(panel), "joint_before": J, "candidates": {}}
        todo = []
        for nm in eligible:
            if nm in panel:
                continue
            corr = [float(abs(C[idx[nm], idx[p]])) if np.isfinite(C[idx[nm], idx[p]]) else None for p in panel]
            if any(c is None for c in corr):  # fail closed: a correlation that cannot be read is not a correlation below 0.6
                step["candidates"][nm] = {"skipped": "correlation_undefined"}
                continue
            if any(c > MAX_PAIRWISE_CORR for c in corr):
                step["candidates"][nm] = {"skipped": "correlation", "max_abs_corr_with_panel": max(corr)}
                continue
            todo.append((nm, max(corr, default=None)))
        res = vinfo.gains_parallel(blocks, rows, y, panel, [nm for nm, _ in todo], cur, jobs, groups)
        best, best_gain, best_res = None, -1.0, None
        screened = []
        for nm, mc in todo:
            r = res[nm]
            if r is None or r.get("error"):
                step["candidates"][nm] = {"skipped": "probe_fit", **(r or {})}
                continue
            step["candidates"][nm] = {k: r[k] for k in ("joint_usable_bits", "gain", "gain_ci_low", "gain_ci_high", "n")} | {"max_abs_corr_with_panel": mc, "screen": r["gain_ci_low"] >= MIN_SIGNAL_BITS, "admissible": False}
            if r["gain_ci_low"] >= MIN_SIGNAL_BITS:
                screened.append(nm)
        for nm in sorted(screened, key=lambda c: -res[c]["gain"]):  # confirm in screen order; the first confirmed candidate is the step's maximum
            conf = vinfo.confirm(blocks, rows, y, panel, nm, groups)
            step["candidates"][nm]["confirm"] = {k: v for k, v in conf.items() if k not in ("common", "diffs")}
            step["candidates"][nm]["admissible"] = vinfo.admissible(conf, MIN_SIGNAL_BITS, not panel)
            if step["candidates"][nm]["admissible"]:
                best, best_gain, best_res = nm, res[nm]["gain"], res[nm]
                break
        if best is None and strata:
            over = stratified_override(res, todo, cur, strata, blocks, rows, y, panel, groups)
            step["override_candidates"] = over
            if over:
                best = max(over, key=lambda k: over[k]["gain"])
                best_res = res[best]
                step["admitted_by"] = {"rule": "stratified_override", **over[best]}
                step["decision"] = (f"admit {best} by stratified override: stratum {over[best]['stratum']} (n {over[best]['n']}) +{over[best]['gain']:.4f} bits, "
                                    f"interval [{over[best]['gain_ci_low']:.4f}, {over[best]['gain_ci_high']:.4f}] at alpha {over[best]['alpha']:.4f}; global gain {best_res['gain']:.4f}")
        if best is None:
            top = max((v["gain"] for v in step["candidates"].values() if "gain" in v), default=None)
            step["decision"] = f"stop: best screen gain {top:.4f}; no candidate's lower bound >= {MIN_SIGNAL_BITS} on every seed (usable and nonlinear-control)" if top is not None else "stop: no candidate"
            trace.append(step)
            break
        if "admitted_by" not in step:
            step["admitted_by"] = {"rule": "conditional_gain"}
            step["decision"] = f"admit {best} (+{best_gain:.4f} bits, interval [{best_res['gain_ci_low']:.4f}, {best_res['gain_ci_high']:.4f}])"
        trace.append(step)
        panel.append(best)
        J = best_res["joint_usable_bits"]
        cur = (best_res["common"], best_res["probe"], best_res["prior"])
    return panel, trace, {"eligible": eligible, "ineligible_carriers": ineligible, "joint_point": J}, cur


def effective_rank(C: np.ndarray, panel: list[str], order: list[str]) -> float | None:
    if not panel:
        return None
    ii = [order.index(p) for p in panel]
    G = np.abs(np.nan_to_num(C[np.ix_(ii, ii)], nan=0.0))
    np.fill_diagonal(G, 1.0)
    return float(np.trace(G) ** 2 / (G ** 2).sum())


def assay_family(root: Path, fam: str, registry: dict, seeds: int, resamples: int, seed: int, jobs: int) -> dict:
    if seeds != vinfo.SEEDS:
        raise LensError("MNLM_ASSAY_SEED_CONTRACT", f"requires exactly {vinfo.SEEDS} estimator seeds")
    rows_cal, summary = read_family(root / "cal", fam)
    y = np.asarray([r["y"] for r in rows_cal], np.int64)
    n = len(y)
    groups = np.unique([str(r["cluster"]) for r in rows_cal], return_inverse=True)[1].astype(np.int64)  # the entity / event cluster: folds and bootstraps resample it
    pairs = fixed_pairs(fam, n, seed)
    lenses = {nm: e for nm, e in registry["lenses"].items() if not e.get("error") and not e.get("spec", {}).get("retired") and fam in e.get("families", {})}
    order = sorted(lenses)
    R = np.full((len(order), N_PAIRS), np.nan)
    cands, P, used, blocks = {}, {}, {}, {}
    (root / "proj" / fam).mkdir(parents=True, exist_ok=True)
    for li, nm in enumerate(order):
        t0 = time.time()
        path = Path(lenses[nm]["families"][fam]["slot"])
        if lenses[nm]["families"][fam].get("calibration_blake3") != summary["file_blake3"]:
            raise LensError("MNLM_SLOT_STALE", f"{fam}/{nm}: measurement is not bound to these calibration bytes; remeasure")
        measured = slotio.slot_digest(path, norm=lenses[nm]["contract"].get("norm"))
        from slot_layout import check_seal
        check_seal(measured, lenses[nm]["families"][fam], f"{fam}/{nm}")
        if measured["digest"] != lenses[nm]["families"][fam]["digest"] or measured["n"] != n:
            raise LensError("MNLM_SLOT_STALE", f"{fam}/{nm}: slot differs from registered bytes or calibration length")
        X, absent, kind = load_slot(path, fam)
        sim = lenses[nm]["contract"].get("sim") or ("maxsim" if kind == "multi" else "cosine")
        R[li] = readings(path, X, absent, kind, sim, pairs)
        if (~absent).sum() < ksg.MIN_SAMPLES or (X.shape[1] if hasattr(X, "shape") else 0) == 0:
            cands[nm] = {"n_used": int((~absent).sum()), "coverage": float((~absent).mean()), "bits_mean": None, "lcb_min": None, "ucb_max": None, "seeds_ok": 0,
                         "error": "MNLM_ASSAY_NO_COVERAGE", "seeds": []}
        else:
            cands[nm], P0, rows = standalone(X, absent, y, seeds, resamples, groups)
            if P0 is not None:
                P[nm], used[nm] = P0, rows
                blk = vinfo.block(X[rows])
                if blk.shape[1] > 0:
                    blocks[nm] = blk
                cands[nm]["standalone_estimator"] = "ksg"
                if cands[nm]["error"] == "MNLM_ASSAY_DEGENERATE" and all(sp_.get("error") == "MNLM_ASSAY_DEGENERATE" for sp_ in cands[nm]["seeds"]) and nm in blocks:
                    # A slot with exact same-class duplicates (count and histogram encoders) has no k-th-neighbour radius, so the
                    # KSG refuses it; the held-out usable-information probe is a separate diagnostic for this slot.
                    # The KSG refusal stays recorded per seed. This does not repair the missing Shannon estimate.
                    try:
                        prior, probe = vinfo.cv_cross_entropy(blocks[nm], y[rows], groups[rows])
                        lo, hi = vinfo.paired_interval(prior - probe, groups[rows])
                        cands[nm].update({"bits_mean": float((prior - probe).mean()), "lcb_min": lo, "ucb_max": hi, "error": None, "standalone_estimator": "vinfo",
                                          "ksg_error": "MNLM_ASSAY_DEGENERATE", "vinfo": {"usable_bits": float((prior - probe).mean()), "ci_low": lo, "ci_high": hi, "n": int(len(rows)), "dim": int(blocks[nm].shape[1])}})
                    except ValueError as exc:
                        cands[nm]["detail"] = f"vinfo: {str(exc)[:120]}"
                with (root / "proj" / fam / f"{nm}.npz").open("wb") as fh:
                    np.savez(fh, P=P0, rows=rows, y=y[rows])
        cands[nm].update({"kind": kind, "sim": sim, "assay_view": f"maxsim to {slotio.LANDMARKS} label-free landmarks" if kind == "multi" else "slot", "secs": round(time.time() - t0, 1)})
        print(f"  {fam} {nm}: bits={cands[nm]['bits_mean']} lcb={cands[nm]['lcb_min']} cov={cands[nm]['coverage']:.2f} {cands[nm].get('error') or ''}", flush=True)
    C = spearman_matrix(R)
    with (root / "readings" / f"{fam}.npz").open("wb") as fh:
        np.savez(fh, pairs=pairs, R=R, C=C, order=np.asarray(order))
    baseline = cands.get(BASELINE_LENS, {}).get("bits_mean")
    for nm, s in cands.items():
        inputs = lenses[nm]["contract"].get("inputs")
        s["baseline_bits"] = baseline
        s["vs_baseline"] = (s["bits_mean"] - baseline) if (baseline is not None and s["bits_mean"] is not None) else None
        # None = not tested (the baseline carried no estimate in this family); False = tested and clear; True = suspect
        s["fingerprint_suspect"] = None if (baseline is None or inputs != "fields" or nm == BASELINE_LENS or s["vs_baseline"] is None) else bool(s["vs_baseline"] <= 0.01)
    strata = strata_of(fam, rows_cal)
    panel, trace, meta, cur = admit(cands, blocks, used, y, C, order, jobs, strata, groups)
    meta["strata"] = {k: int(len(v)) for k, v in strata.items()}
    meta["admitted_by"] = {nm: trace[i]["admitted_by"] for i, nm in enumerate(panel)}
    joint = None
    if panel:
        common, probe, prior = cur
        lb = float((prior - probe).mean())
        lo, hi = vinfo.paired_interval(prior - probe, groups[common])
        joint = {"vinfo": {"usable_bits": lb, "ci_low": lo, "ci_high": hi, "n": int(len(common))}, "n": int(len(common)), "bits": lb, "ci_low": lo,
                 "quantity": "held-out usable information; not a Shannon sufficiency certificate", "floored": False}
        with (root / "assay" / f"{fam}.vinfo.npz").open("wb") as fh:
            np.savez(fh, panel=np.asarray(panel), common=common, prior=prior, probe=probe)
    H = summary["H_Y_bits"]
    I = joint["bits"] if joint and "bits" in joint else 0.0
    deficit = max(0.0, H - I)
    w = {nm: 1 / (cands[nm]["bits_mean"] + 0.01) for nm in panel}
    tot = sum(w.values()) or 1.0
    suff = {"H_Y_bits": H, "I_panel_bits": I, "I_panel_lcb": joint.get("ci_low") if joint else None, "sufficient": False, "deficit_bits": deficit,
            "status": "MNLM_SHANNON_SUFFICIENCY_UNMEASURED", "deficit_kind": "probe diagnostic, not a certified Shannon deficit",
            "deficit_split": {nm: deficit * w[nm] / tot for nm in panel},
            "suggested_action": "propose a lens" if panel and n >= 500 else ("gather more samples" if n < 500 else "add an outcome anchor or propose a lens")}
    return {"family": fam, "n": n, "classes": summary["classes"], "H_Y_bits": H, "seeds": seeds, "resamples": resamples, "k": ksg.K, "device": ksg.DEVICE, "pairs": N_PAIRS,
            "conditional_estimator": {"name": "held-out usable conditional information (V-information) of an L2 logistic probe (vinfo.py); not a bound on Shannon conditional MI", "instrument": vinfo.INSTRUMENT,
                                      "probe_dims": vinfo.PROBE_DIMS, "folds": vinfo.FOLDS, "fold_split": "StratifiedGroupKFold by cluster", "C": vinfo.C_L2, "cv_seed": vinfo.CV_SEED, "seeds": vinfo.SEEDS,
                                      "projection_seed": vinfo.PROBE_SEED, "bootstrap": f"cluster bootstrap x{vinfo.BOOT}", "computation_control": f"panel + random Fourier expansion ({vinfo.RFF_DIMS} dims, seed {vinfo.RFF_SEED})",
                                      "rule": "screen: lower bound >= 0.05 on fold seed 0; admit: lower bound >= 0.05 on each of five fold seeds for the usable gain and, with a panel, the nonlinear-control gain",
                                      "stratified_override": f"when no candidate is confirmed: inside one declared stratum ({MIN_STRATUM} to n/2 rows of a categorical field) the cluster-bootstrap lower bound of the per-row gain >= 0.05 at alpha 0.05/(strata x candidates tested) on every fold seed, usable and nonlinear-control gain; never scaled by frequency"},
            "inputs": admission_state.bindings(root, fam, registry), "shannon_conditional_certified": False,
            "lenses": cands, "correlation": {"order": order, "matrix": np.where(np.isfinite(C), C, None).tolist()}, "admission": {"panel": panel, "trace": trace, "status": "usable_candidates_only", **meta},
            "served": False, "n_eff": effective_rank(C, panel, order), "joint": joint, "sufficiency": suff, "baseline_lens": BASELINE_LENS, "ts": now_iso()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--families", default="")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--resamples", type=int, default=ksg.RESAMPLES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--issue", default="12")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    require_gpu(ksg.DEVICE)
    take_lease(args.issue, "lens assay")
    paths = Paths(args.root, args.out, args.version)
    root = lens_root(paths)
    registry = sealed_registry(root)
    admission_state.validate_ledger(paths.ledger, registry)
    (root / "assay").mkdir(parents=True, exist_ok=True)
    (root / "readings").mkdir(parents=True, exist_ok=True)
    digests, overview = {}, {}
    prior = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8")) if args.families and (root / "assay" / "admission.json").exists() else None
    if prior:  # a partial run merges into the record: every family not assayed now must still match its recorded digest (fail closed)
        admission_state.validate_ledger(paths.ledger, registry, prior)
        admission_state.load_reports(root, registry, prior)
        for fam, dg in prior["digests"].items():
            if fam in args.families.split(","):
                continue
            now = blake3.blake3((root / "assay" / f"{fam}.json").read_bytes()).hexdigest() if (root / "assay" / f"{fam}.json").exists() else None
            if now != dg:
                raise LensError("MNLM_ASSAY_RECORD_TORN", f"{fam}: assay file digest {now} differs from the admission record {dg}")
            digests[fam], overview[fam] = dg, prior["families"][fam]
    for fam in (args.families.split(",") if args.families else families_present(root)):
        from training_scope import require_training_question
        require_training_question(fam)
        t0 = time.time()
        rep = assay_family(root, fam, registry, args.seeds, args.resamples, args.seed, args.jobs)
        rep["secs"] = round(time.time() - t0, 1)
        write_json(root / "assay" / f"{fam}.json", rep)
        digests[fam] = blake3.blake3((root / "assay" / f"{fam}.json").read_bytes()).hexdigest()
        overview[fam] = {"admitted": rep["admission"]["panel"], "served": rep["served"], "n_eff": rep["n_eff"], "sufficient": rep["sufficiency"]["sufficient"],
                         "I_panel_bits": rep["sufficiency"]["I_panel_bits"], "H_Y_bits": rep["H_Y_bits"], "secs": rep["secs"]}
        print(f"{fam}: admitted={rep['admission']['panel']} served={rep['served']} n_eff={rep['n_eff']} I={rep['sufficiency']['I_panel_bits']:.3f} H={rep['H_Y_bits']:.3f} {rep['secs']} s", flush=True)
    top = {"version": args.version, "families": overview, "digests": digests, "registry_blake3": registry.get("registry_blake3_before_ledger"), "host": HOST, "device": ksg.DEVICE, "ts": now_iso(),
           "assayed_now": args.families.split(",") if args.families else sorted(overview), "merged_from_seq": prior.get("ledger_seq") if prior else None}
    e = ledger_append(paths, f"assay@{HOST}", "lenses.admission", f"stage3/{args.version}", top)
    top["ledger_seq"], top["ledger_head"] = e["seq"], e["hash"]
    write_json(root / "assay" / "admission.json", top)
    print(json.dumps({"ledger_seq": e["seq"], "served": [f for f, o in overview.items() if o["served"]]}), flush=True)
    return 0


if __name__ == "__main__":
    from lenscommon import run_cli
    sys.exit(run_cli(main))
