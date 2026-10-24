"""Stage 3 FSV, admission re-derivation under instrument v3 diagnostic (issue #12; operator review 2026-10-17). Split from
fsv_stage3.py for the 500-line gate. From the stored assay record and the slots it re-derives, per question: the decision
rules (standalone and correlation gates; the one-seed screen orders, five-seed confirmation decides; the first confirmed
candidate in screen order is admitted; no confirmed candidate is left behind at the stop; the stratified override only when
nothing is confirmed, under the same contract inside its stratum) and the numbers (panel held-out cross-entropy, every
admitted step's screen gain and cluster-bootstrap interval, every admitted step's five-seed confirmation including the
nonlinear-control gain against the panel's random-Fourier expansion, the best rejected candidate of the final step). Folds and
bootstraps resample the row's cluster, as the doctrine requires. A Multi slot is read through its native view
(slotio.maxsim_landmarks), the view the assay measured. Cost: one screen fit and one five-seed confirmation per admitted
lens, plus the best rejected candidate; CPU, plus the GPU MaxSim view for Multi slots."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import slotio
import vinfo

TOL = 1e-6


def view(root: Path, fam: str, nm: str, load_slot):
    """(probe view X, absent, kind, members) of a slot as the assay read it."""
    path = root / "slots" / fam / f"{nm}.npz"
    X, absent, kind, z = load_slot(path)
    if kind == "multi":
        X, _ = slotio.maxsim_landmarks(path, absent, fam)
    return X, absent, kind, z


def rules(rep: dict, C: np.ndarray, order: list[str], min_bits: float, max_corr: float, min_cov: float) -> list[str]:
    panel, trace = rep["admission"]["panel"], rep["admission"]["trace"]
    idx = {nm: i for i, nm in enumerate(order)}
    viol = []
    if rep.get("conditional_estimator", {}).get("instrument") != vinfo.INSTRUMENT:
        viol.append("old or unknown assay instrument")
    if rep.get("served") or rep.get("shannon_conditional_certified") or rep.get("sufficiency", {}).get("sufficient"):
        viol.append("usable probe result presented as certified Shannon information")
    if len(trace) != len(panel) + 1:
        return viol + ["trace must contain one step per candidate and a final stop"]
    for step_i, nm in enumerate(panel):
        s, step = rep["lenses"][nm], trace[step_i]
        cands = step["candidates"]
        if not (s["lcb_min"] is not None and s["lcb_min"] >= min_bits and s["coverage"] >= min_cov):
            viol.append(f"{nm}: standalone gate")
        if step.get("panel") != panel[:step_i]:
            viol.append(f"{nm}: trace prefix differs from panel")
        if any(not np.isfinite(C[idx[nm], idx[p]]) or abs(C[idx[nm], idx[p]]) > max_corr for p in panel[:step_i]):
            viol.append(f"{nm}: correlation gate")
        screened = sorted((c for c, v in cands.items() if v.get("gain_ci_low") is not None and v["gain_ci_low"] >= min_bits), key=lambda c: -cands[c]["gain"])
        if any(bool(v.get("screen")) != (c in screened) for c, v in cands.items() if "gain" in v):
            viol.append(f"step {step_i}: screen flags do not match lower bound >= {min_bits}")
        by = (step.get("admitted_by") or {}).get("rule", "conditional_gain")
        if by == "stratified_override":
            over = step.get("override_candidates") or {}
            o = over.get(nm) or {}
            conf = o.get("confirm") or {}
            if any(v.get("admissible") for v in cands.values()) or not o or o["gain"] < max(v["gain"] for v in over.values()) - 1e-12 or o["gain_ci_low"] < min_bits \
                    or conf.get("lcb_min", -1) < min_bits or (step_i > 0 and conf.get("nonlinear_lcb_min", -1) < min_bits):
                viol.append(f"{nm}: stratified override not under the contract, or a global candidate was confirmed")
            continue
        conf = cands.get(nm, {}).get("confirm") or {}
        if not cands.get(nm, {}).get("admissible") or nm not in screened or not vinfo.admissible(conf, min_bits, step_i == 0):
            viol.append(f"{nm}: not confirmed under the contract")
            continue
        for c in screened[:screened.index(nm)]:
            if cands[c].get("admissible") or "confirm" not in cands[c]:
                viol.append(f"{nm}: screened candidate {c} ahead of it was not confirmed-and-refused")
    if trace:
        last = trace[-1]["candidates"]
        if any(v.get("admissible") for v in last.values()):
            viol.append("stopped with a confirmed candidate remaining")
        if any(v.get("screen") and "confirm" not in v for v in last.values()):
            viol.append("stopped with a screened candidate never confirmed")
    return viol


def numbers(root: Path, fam: str, rep: dict, rows_cal: list[dict], y: np.ndarray, load_slot, strata_of, min_bits: float) -> tuple[dict, float, list[str]]:
    """Replays the decisive numbers; returns (gains replayed, worst |difference|, violations)."""
    groups = np.unique([str(r["cluster"]) for r in rows_cal], return_inverse=True)[1].astype(np.int64)
    panel, trace = rep["admission"]["panel"], rep["admission"]["trace"]
    viol, gains, worst = [], {}, 0.0
    blocks, used = {}, {}
    for nm in panel:
        X, absent, _k, _z = view(root, fam, nm, load_slot)
        used[nm] = np.flatnonzero(~absent)
        blocks[nm] = vinfo.block(X[used[nm]])
    vz = np.load(root / "assay" / f"{fam}.vinfo.npz")
    common, prior, probe = vinfo.joint(blocks, used, y, panel, groups)
    if not (np.array_equal(common, vz["common"]) and float(np.abs(prior - vz["prior"]).max()) < 1e-9 and float(np.abs(probe - vz["probe"]).max()) < TOL):
        viol.append("panel held-out cross-entropy not replayed")
    strata = strata_of(fam, rows_cal)
    cur = None
    for i, nm in enumerate(panel):
        rec = trace[i]["candidates"][nm]
        g = vinfo.conditional_gain(blocks, used, y, panel[:i], nm, cur, groups)
        gains[nm] = g["gain"]
        worst = max(worst, abs(g["gain"] - rec["gain"]), abs(g["gain_ci_low"] - rec["gain_ci_low"]))
        ov = trace[i].get("admitted_by") or {}
        if ov.get("rule") == "stratified_override":
            o = trace[i]["override_candidates"][nm]
            conf = vinfo.confirm(blocks, used, y, panel[:i], nm, groups, alpha=o["alpha"])
            m = np.isin(conf["common"], strata.get(o["stratum"], np.zeros(0, np.int64)))
            gc = groups[conf["common"]][m]
            los = [vinfo.paired_interval(d[m], gc, seed=sd["seed"], alpha=o["alpha"])[0] for (d, _de), sd in zip(conf["diffs"], conf["seeds"])]
            elos = [vinfo.paired_interval(de[m], gc, seed=sd["seed"], alpha=o["alpha"])[0] for (_d, de), sd in zip(conf["diffs"], conf["seeds"])]
            worst = max(worst, abs(min(los) - o["confirm"]["lcb_min"])) if m.any() else float("inf")
            worst = max(worst, abs(min(elos) - o["confirm"]["nonlinear_lcb_min"])) if m.any() else float("inf")
            if not m.any() or min(los) < min_bits or (i > 0 and min(elos) < min_bits):
                viol.append(f"{nm}: stratified usable/nonlinear-control confirmation fails on replay")
            gains[f"{nm}@{o['stratum']}"] = float(np.mean([float(d[m].mean()) for d, _ in conf["diffs"]])) if m.any() else float("nan")
        else:
            conf = vinfo.confirm(blocks, used, y, panel[:i], nm, groups)
            rc = rec["confirm"]
            worst = max(worst, abs(conf["lcb_min"] - rc["lcb_min"]), abs(conf.get("nonlinear_lcb_min", 0.0) - rc.get("nonlinear_lcb_min", 0.0)))
            if not vinfo.admissible(conf, min_bits, i == 0):
                viol.append(f"{nm}: replayed confirmation not admissible")
        cur = (g["common"], g["probe"])
    last = trace[-1]["candidates"] if trace else {}
    best_rej = max(((v["gain"], c) for c, v in last.items() if "gain" in v), default=None)
    if best_rej is not None:
        c = best_rej[1]
        Xc, ac, _k, _z = view(root, fam, c, load_slot)
        used[c] = np.flatnonzero(~ac)
        blocks[c] = vinfo.block(Xc[used[c]])
        g = vinfo.conditional_gain(blocks, used, y, panel, c, cur, groups)
        gains[f"rejected:{c}"] = g["gain"]
        worst = max(worst, abs(g["gain"] - best_rej[0]))
        if g["gain_ci_low"] >= min_bits and vinfo.admissible(vinfo.confirm(blocks, used, y, panel, c, groups), min_bits, not panel):
            viol.append(f"rejected {c} is admissible on replay")
    return gains, worst, viol
