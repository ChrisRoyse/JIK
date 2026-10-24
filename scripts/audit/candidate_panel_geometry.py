"""Bounded ten/fourteen-instrument D1 feasibility audit, never admission or D2.

All 1,024 or 16,384 subsets are inspected for the declared diagnostic roster.
A D1-only maximum bounds possible panel size;
the optional standalone filter is exploratory, since synergy can make a sense
conditionally useful despite weak standalone information.
"""
import argparse
import copy
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import ROSTER, sha, check, equal
from senses.trajectory_population import publish

SCHEMA = "ten-candidate-d1-feasibility-v1"
EXTRA = ("cs-natural-history-v1", "cs-patent-v1", "cs-regulatory-v1", "cs-science-v1")
REMAINING = ("cs-legal-v1", "cs-biomedical-v1", "cs-finance-v1", "cs-software-v1")
NAMES = ROSTER+list(EXTRA)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path, pin=None):
    raw = path.read_bytes()
    check(pin is None or sha(raw) == pin, "PANEL_INPUT_HASH")
    return json.loads(raw), sha(raw)


def cache(path, pin):
    raw = path.read_bytes(); check(sha(raw) == pin, "PANEL_CACHE_HASH")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def inputs(manifest):
    candidates = set(manifest["candidates"])
    check(candidates == set(EXTRA) or candidates == set(EXTRA+REMAINING), "PANEL_ROSTER")
    extra = EXTRA if candidates == set(EXTRA) else EXTRA+REMAINING
    sources = {}; roots = {}; reports = {}
    for name in ("public", "paired")+extra:
        item = manifest[name] if name in ("public", "paired") else manifest["candidates"][name]
        root = Path(item["root"]); report, pin = read_json(root/"report.json", item["report_sha256"])
        proof, proof_pin = read_json(Path(item["fsv"]))
        check(proof["passed"] == proof["total"], "PANEL_FSV")
        if name in ("public", "paired"):
            check(proof["cold_readback"]["report_sha256"] == pin, "PANEL_FSV_BINDING")
        else:
            check(proof["source_report_sha256"] == pin, "PANEL_FSV_BINDING")
        check(report["complete"] and not report["errors"] and report["scientific_admission"] is False, "PANEL_SOURCE_CLAIM")
        sources[name] = dict(report_sha256=pin, fsv_sha256=proof_pin); roots[name] = root; reports[name] = report
    check(reports["paired"]["source_report_sha256"] == sources["public"]["report_sha256"], "PANEL_PAIRED_BINDING")
    questions = set(reports["public"]["findings"])
    check(all(set(r["findings"]) == questions for r in reports.values()), "PANEL_QUESTIONS")
    values = {}
    for q in sorted(questions):
        pub = reports["public"]["findings"][q]; base = cache(roots["public"]/(q+".npz"), pub["artifact_sha256"])
        readings = [base[s+"__pair_readings"] for s in ROSTER]
        lower = [reports["paired"]["findings"][q]["cases"]["authentic"]["statistics"]["standalone:"+s]["outer_low"] for s in ROSTER]
        for name in extra:
            report = reports[name]; rec = report["findings"][q]
            check(report["public_diagnostic_sha256"] == sources["public"]["report_sha256"]
                  and rec["source_artifact_sha256"] == pub["artifact_sha256"], "PANEL_CANDIDATE_BINDING")
            candidate = cache(roots[name]/(q+".npz"), rec["artifact_sha256"])
            equal(candidate["pairs"], base["pairs"], "PANEL_PAIR_IDENTITY", 0)
            equal(candidate["labels"], base["labels"], "PANEL_LABEL_IDENTITY", 0)
            readings.append(candidate["pair_readings"]); lower.append(rec["standalone"]["outer_low"])
        values[q] = (np.asarray(readings), np.asarray(lower))
    return sources, values, ROSTER+list(extra)


def maxima(matrix, eligible, names=NAMES):
    n = len(names)
    check(n in (10, 14) and len(set(names)) == n and matrix.shape == (n, n)
          and np.isfinite(matrix).all() and len(eligible) == n, "PANEL_BOUNDED_ROSTER")
    best = 0; winners = []
    for mask in range(1, 1 << n):
        members = [i for i in range(n) if mask & (1 << i)]
        if len(members) < best or not all(eligible[i] for i in members): continue
        if not all(matrix[i, j] <= .6 for i, j in combinations(members, 2)): continue
        if len(members) > best: best = len(members); winners = []
        winners.append([names[i] for i in members])
    return dict(size=best, all_maximizers=sorted(winners))


def independent_maxima(matrix, eligible, names=NAMES):
    available = [i for i in range(len(names)) if eligible[i]]
    for width in range(len(available), 0, -1):
        found = [[names[i] for i in group] for group in combinations(available, width)
                 if all(matrix[i, j] <= .6 for i in group for j in group if i < j)]
        if found: return dict(size=width, all_maximizers=sorted(found))
    return dict(size=0, all_maximizers=[])


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inv]


def validate_result(claim, saved, values, names=NAMES):
    check(claim["scientific_admission"] is False and claim["d2_certified"] is False
          and claim["served"] is False, "PANEL_CLAIM")
    equal(saved["readings"], values[0], "PANEL_READINGS", 0)
    equal(saved["standalone_lower"], values[1], "PANEL_LOWER", 0)
    ranked = np.asarray([ranks(x) for x in values[0]]); ranked -= ranked.mean(1, keepdims=True)
    matrix = ranked@ranked.T / np.outer(np.linalg.norm(ranked, axis=1), np.linalg.norm(ranked, axis=1))
    equal(saved["correlations"], matrix, "PANEL_CORRELATIONS")
    n = len(names)
    check(claim["d1_only"] == independent_maxima(matrix, np.ones(n, dtype=bool), names)
          and claim["standalone_filtered_exploratory"] == independent_maxima(matrix, values[1] >= .05, names)
          and claim["public_only_d1"] == independent_maxima(matrix, np.arange(n) < 6, names)
          and claim["public_only_standalone_filtered"] == independent_maxima(matrix, (values[1] >= .05) & (np.arange(n) < 6), names), "PANEL_MAXIMIZERS")


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args(); begin = time.perf_counter()
    check(not a.output.exists(), "PANEL_EXISTS"); manifest, pin = read_json(a.manifest); sources, values, names = inputs(manifest)
    a.output.mkdir(parents=True); loaded = time.perf_counter(); claims = {}; checks = []; n = len(names)
    phases = dict(pinned_inputs=loaded-begin); tick = loaded
    def ok(name, value): checks.append(dict(name=name, passed=bool(value)))
    for q, (readings, lower) in values.items():
        check(readings.shape == (n, 4096) and np.isfinite(readings).all() and np.isfinite(lower).all(), "PANEL_VALUES")
        matrix = spearmanr(readings, axis=1).statistic
        claim = dict(d1_only=maxima(matrix, np.ones(n, dtype=bool), names), standalone_filtered_exploratory=maxima(matrix, lower >= .05, names),
                     public_only_d1=maxima(matrix, np.arange(n) < 6, names),
                     public_only_standalone_filtered=maxima(matrix, (lower >= .05) & (np.arange(n) < 6), names),
                     scientific_admission=False, d2_certified=False, served=False)
        np.savez(a.output/(q+".npz"), readings=readings, standalone_lower=lower, correlations=matrix)
        claim["artifact_sha256"] = digest(a.output/(q+".npz")); claims[q] = claim
        saved = cache(a.output/(q+".npz"), claim["artifact_sha256"]); validate_result(claim, saved, (readings, lower), names)
        ok(q+": independent ranks and exhaustive subsets replay", True)
    phases["three_question_ranks_and_exhaustive_subsets"] = time.perf_counter()-tick; tick = time.perf_counter()
    # Persisted structural controls: all incompatible, all compatible, and a
    # chain whose largest clique is a pair (never a transitive component).
    for name, matrix, expected in (("all-copies", np.ones((n, n)), 1),
                                   ("all-distinct", np.eye(n), n)):
        np.save(a.output/(name+".npy"), matrix)
        ok(name, maxima(matrix, np.ones(n, bool), names)["size"] == independent_maxima(matrix, np.ones(n, bool), names)["size"] == expected)
    chain = np.ones((n, n))
    for i in range(n-1): chain[i, i+1] = chain[i+1, i] = 0
    np.save(a.output/"chain.npy", chain)
    ok("chain is not a full-roster clique", maxima(chain, np.ones(n, bool), names)["size"] == independent_maxima(chain, np.ones(n, bool), names)["size"] == 2)
    # Orthogonal rotation leaves pair cosine readings unchanged.
    rng = np.random.default_rng(25); vectors = rng.normal(size=(16, 3)); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3))); rotated = vectors@rotation
    np.savez(a.output/"rotated-copy.npz", vectors=vectors, rotated=rotated, rotation=rotation)
    equal(vectors@vectors.T, rotated@rotated.T, "PANEL_ROTATED_COPY")
    tri = np.triu_indices(16, 1); rho = float(spearmanr((vectors@vectors.T)[tri], (rotated@rotated.T)[tri]).statistic)
    ok("rotated duplicate exceeds D1 ceiling", rho > .6)
    q = next(iter(claims)); changed = copy.deepcopy(claims[q]); changed["scientific_admission"] = True
    publish(a.output/"false-admission.json", changed)
    try: validate_result(changed, cache(a.output/(q+".npz"), changed["artifact_sha256"]), values[q], names); refused = False
    except ValueError as exc: refused = "PANEL_CLAIM" in str(exc)
    ok("false admission refuses", refused)
    incomplete = copy.deepcopy(manifest); incomplete["candidates"].pop(EXTRA[0])
    publish(a.output/"incomplete-roster.json", incomplete)
    try: inputs(incomplete); refused = False
    except ValueError as exc: refused = "PANEL_ROSTER" in str(exc)
    ok("incomplete diagnostic roster refuses", refused)
    phases["persisted_structural_and_claim_controls"] = time.perf_counter()-tick; tick = time.perf_counter()
    after, _, after_names = inputs(manifest)
    ok("B1/B3 source pins unchanged after action", after == sources and names == after_names and digest(a.manifest) == pin)
    code_pin = digest(Path(__file__))
    phases["source_and_code_drift_checks"] = time.perf_counter()-tick; wall = time.perf_counter()-begin
    result = dict(schema=SCHEMA if n == 10 else "fourteen-candidate-d1-feasibility-v1", names=names, sources=sources, manifest_sha256=pin, findings=claims,
                  complete=all(c["passed"] for c in checks), scientific_admission=False, d2_certified=False, served=False, checks=checks,
                  passed=sum(c["passed"] for c in checks), total=len(checks), acceptance="author-checked",
                  caveat="D1-only maximum is an upper bound; standalone-filtered maximum is exploratory and can exclude synergistic senses",
                  timing=dict(load_seconds=loaded-begin, wall_seconds=wall, phases_seconds=phases,
                              phase_coverage=sum(phases.values())/wall), code_sha256=code_pin)
    publish(a.output/"report.json", result)
    print(json.dumps(dict(passed=result["passed"], total=result["total"], findings={q: {k: c[k]["size"] for k in ("d1_only", "standalone_filtered_exploratory")} for q, c in claims.items()})))
    if result["passed"] != result["total"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
