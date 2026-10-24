"""Paired Calyx-root confidence diagnostics; never scientific admission (#12)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import ROSTER, SEEDS, ross, sha, check
from text.calyx_bits import subsample_m, RESAMPLES, K
from senses.trajectory_population import publish

SCHEMA = "paired-calyx-root-diagnostic-v1"


def matrices(cache):
    """Typed distance products; deduplicate identical matrices before resampling."""
    unique, aliases, definitions = {}, {}, {}
    def add(name, value):
        key = sha(value.tobytes())
        unique.setdefault(key, value)
        aliases[name] = key
        return key
    singles = {s: add("single:"+s, cache[s+"__distance"]) for s in ROSTER}
    joint, previous = None, None
    for s in ROSTER:
        d = cache[s+"__distance"]
        joint = d if joint is None else np.maximum(joint, d)
        key = add("prefix:"+s, joint)
        definitions["standalone:"+s] = [singles[s], None]
        definitions["increment:"+s] = [key, previous]
        previous = key
    for s in ROSTER:
        without = np.maximum.reduce([cache[t+"__distance"] for t in ROSTER if t != s])
        definitions["drop:"+s] = [previous, add("without:"+s, without)]
    definitions["duplicate"] = [add("duplicate", np.maximum(joint, cache[ROSTER[0]+"__distance"])), previous]
    return unique, aliases, definitions


def plans(y):
    n = len(y); counts = np.unique(y, return_counts=True)[1]
    m = subsample_m(n, counts)
    indices = np.zeros((len(SEEDS), RESAMPLES, m), dtype=np.int32)
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        for b in range(RESAMPLES):
            for _ in range(8):
                ix = rng.choice(n, m, replace=False)
                c = np.unique(y[ix], return_counts=True)[1]
                if len(c) == len(counts) and (c > K).all():
                    indices[si,b] = ix
                    break
            else:
                raise ValueError("MNLM_PAIRED_SUPPORT")
    return indices


def interval(point, samples, n, m):
    root = np.sort(np.sqrt(m/(1-m/n))*(samples-point))
    at = lambda p: min(len(root)-1, int(round((len(root)-1)*p)))
    return {"raw_point": float(point), "low": float(min(point, point-root[at(.975)]/np.sqrt(n))),
            "high": float(max(point, point-root[at(.025)]/np.sqrt(n)))}


def evaluate(y, unique, definitions):
    ix = plans(y); n,m = len(y), ix.shape[2]
    estimates = {key: np.empty(ix.shape[:2]) for key in unique}
    points = {key: ross(d,y) for key,d in unique.items()}
    for si in range(len(SEEDS)):
        for b in range(RESAMPLES):
            selected = ix[si,b]; sublabels = y[selected]
            subset = np.ix_(selected, selected)
            for key,d in unique.items():
                estimates[key][si,b] = ross(d[subset], sublabels)
    statistics = {}
    for name,(a,b) in definitions.items():
        point = points[a] - (points[b] if b else 0)
        samples = estimates[a] - (estimates[b] if b else 0)
        cis = [interval(point, samples[i], n, m) for i in range(len(SEEDS))]
        lo,hi = min(c["low"] for c in cis), max(c["high"] for c in cis)
        statistics[name] = {"raw_point": float(point), "per_seed": dict(zip(map(str,SEEDS),cis)),
                            "outer_low": lo, "outer_high": hi, "lower_bound_at_least_point05": bool(lo >= .05)}
    check(statistics["duplicate"]["outer_low"] == statistics["duplicate"]["outer_high"] == 0, "DUPLICATE")
    return ix, estimates, statistics


def run(source, source_hash, fsv, output):
    begin = time.perf_counter()
    check(not output.exists(), "EXISTS")
    raw = (source/"report.json").read_bytes(); report = json.loads(raw)
    proofraw = fsv.read_bytes(); proof = json.loads(proofraw)
    check(sha(raw) == source_hash and proof["passed"] == proof["total"] == 15 and
          proof["cold_readback"]["report_sha256"] == source_hash, "BINDING")
    check(report["complete"] and not report["errors"] and not report["scientific_admission"], "PUBLICATION")
    output.mkdir(parents=True)
    findings, errors = {}, []
    for question, finding in report["findings"].items():
        start = time.perf_counter()
        try:
            native = (source/(question+".npz")).read_bytes()
            check(sha(native) == finding["artifact_sha256"], "ARTIFACT_HASH")
            with np.load(io.BytesIO(native), allow_pickle=False) as z:
                cache = {k:z[k].copy() for k in z.files}
            unique, aliases, definitions = matrices(cache)
            cases, artifacts = {}, {}
            for case in range(6):
                label = "authentic" if case == 0 else "label_null_"+str(20261018+case)
                y = cache["labels"] if case == 0 else cache["permuted_labels"][case-1]
                ix, estimates, statistics = evaluate(y, unique, definitions)
                artifacts[label+"__indices"] = ix
                artifacts[label+"__labels"] = y
                for key, values in estimates.items():
                    artifacts[label+"__samples__"+key] = values
                cases[label] = {"statistics": statistics, "n": len(y), "m": ix.shape[2]}
                print(json.dumps({"question": question, "case": label, "standalone_lcb_passes": sum(
                      s["lower_bound_at_least_point05"] for n,s in statistics.items() if n.startswith("standalone:")),
                      "increment_lcb_passes": sum(s["lower_bound_at_least_point05"] for n,s in statistics.items()
                                                 if n.startswith("increment:"))}), flush=True)
            dest = output/(question+".npz"); np.savez(dest, **artifacts)
            result = {"cases": cases, "unique_matrices": len(unique), "aliases": aliases, "definitions": definitions,
                      "artifact_sha256": sha(dest.read_bytes()), "wall_seconds": time.perf_counter()-start}
            findings[question] = result; publish(output/(question+".json"), result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({"question": question, "error": str(exc)})
    result = {"schema": SCHEMA, "source_report_sha256": source_hash, "fsv_sha256": sha(proofraw),
              "complete": not errors, "errors": errors, "findings": findings,
              "scientific_admission": False, "conditional_information_certified": False, "served": False,
              "rules": {"seeds": list(SEEDS), "resamples": RESAMPLES, "null_permutations": 5,
                        "resampling": "reference no-replacement support plan; paired entities; not ordinary bootstrap",
                        "bounds": "signed raw 95% roots; hull with point; outer envelope across five seeds"},
              "limitations": ["no IID or near-duplicate certificate", "fixed calibration-derived landmarks",
                              "MI differences inherit finite-sample metric/estimator behavior",
                              "full-label nulls do not establish conditional-null calibration"],
              "code_sha256": sha(Path(__file__).read_bytes()), "wall_seconds": time.perf_counter()-begin}
    publish(output/"report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("diagnostic", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--diagnostic-sha256", required=True)
    a = p.parse_args()
    result = run(a.diagnostic, a.diagnostic_sha256, a.fsv, a.output)
    print(json.dumps({"complete": result["complete"], "errors": result["errors"], "wall_seconds": result["wall_seconds"]}))
    if not result["complete"]:
        raise SystemExit(2)
