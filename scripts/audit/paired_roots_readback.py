"""Cold-check every resample plan/bound and fixed raw samples via original Calyx."""
import argparse
import io
import json
from math import sqrt
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import ROSTER, SEEDS, sha, check, equal
from text.calyx_bits import raw_bits, subsample_m

CHECK_DRAWS = (0, 249, 499, 749, 998)


def reference_geometry(cache):
    # Rebuild from typed persisted slots; do not call the interval producer.
    d = {s:cache[s+"__distance"] for s in ROSTER}
    full = np.maximum.reduce(list(d.values()))
    definitions, aliases, unique = {}, {}, {}
    def record(name, matrix):
        key = sha(matrix.tobytes()); unique[key] = matrix; aliases[name] = key
        return key
    for s in ROSTER:
        record("single:"+s, d[s])
    for i,s in enumerate(ROSTER):
        part = np.maximum.reduce([d[k] for k in ROSTER[:i+1]])
        key = record("prefix:"+s, part)
        definitions["standalone:"+s] = [aliases["single:"+s], None]
        definitions["increment:"+s] = [key, aliases["prefix:"+ROSTER[i-1]] if i else None]
    full_key = sha(full.tobytes())
    for s in ROSTER:
        without = record("without:"+s, np.maximum.reduce([v for k,v in d.items() if k != s]))
        definitions["drop:"+s] = [full_key, without]
    definitions["duplicate"] = [record("duplicate", np.maximum(full,d[ROSTER[0]])), full_key]
    return unique, aliases, definitions


def indices(y, m):
    counts = len(np.unique(y)); n = len(y)
    result = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); accepted = []
        for _ in range(999):
            for _attempt in range(8):
                chosen = rng.choice(n,m,replace=False)
                c = np.unique(y[chosen], return_counts=True)[1]
                if len(c) == counts and min(c) > 3:
                    accepted.append(chosen); break
            else:
                raise ValueError("MNLM_PAIRED_READBACK_SUPPORT")
        result.append(accepted)
    return np.asarray(result, dtype=np.int32)


def bounds(point, values, n, m):
    # Same stated root construction, independently use list order statistics.
    ordered = sorted(float(v-point)*sqrt(m/(1-m/n)) for v in values)
    lower = point-ordered[round((len(ordered)-1)*.975)]/sqrt(n)
    upper = point-ordered[round((len(ordered)-1)*.025)]/sqrt(n)
    return min(point,lower), max(point,upper)


def readback(publication, diagnostic, source_hash):
    begin = time.perf_counter()
    sr = (diagnostic/"report.json").read_bytes(); source = json.loads(sr)
    raw = (publication/"report.json").read_bytes(); report = json.loads(raw)
    check(sha(sr) == source_hash == report["source_report_sha256"], "PAIRED_BINDING")
    check(report["schema"] == "paired-calyx-root-diagnostic-v1" and report["complete"] and not report["errors"], "PAIRED_COMPLETE")
    check(all(report[k] is False for k in ("scientific_admission", "conditional_information_certified", "served")), "PAIRED_CLAIM")
    check(report["rules"] == {"seeds": list(SEEDS), "resamples": 999, "null_permutations": 5,
          "resampling": "reference no-replacement support plan; paired entities; not ordinary bootstrap",
          "bounds": "signed raw 95% roots; hull with point; outer envelope across five seeds"}, "PAIRED_RULES")
    check(set(report["findings"]) == set(source["findings"]), "PAIRED_QUESTIONS")
    summary = {}; sampled_values = 0
    for question, finding in report["findings"].items():
        raw_cache = (diagnostic/(question+".npz")).read_bytes()
        check(sha(raw_cache) == source["findings"][question]["artifact_sha256"], "PAIRED_SOURCE_CACHE")
        with np.load(io.BytesIO(raw_cache), allow_pickle=False) as z:
            cache = {k:z[k].copy() for k in z.files}
        check(json.loads((publication/(question+".json")).read_bytes()) == finding, "PAIRED_REPORT_COPY")
        raw_samples = (publication/(question+".npz")).read_bytes()
        check(sha(raw_samples) == finding["artifact_sha256"], "PAIRED_SAMPLES_HASH")
        with np.load(io.BytesIO(raw_samples), allow_pickle=False) as z:
            saved = {k:z[k].copy() for k in z.files}
        unique, aliases, definitions = reference_geometry(cache)
        check(finding["aliases"] == aliases and finding["definitions"] == definitions
              and finding["unique_matrices"] == len(unique), "PAIRED_METRICS")
        expected_cases = ["authentic"]+["label_null_"+str(20261019+i) for i in range(5)]
        check(set(finding["cases"]) == set(expected_cases), "PAIRED_CASES")
        checked = {}
        for ci,case in enumerate(expected_cases):
            value = finding["cases"][case]
            y = cache["labels"] if ci == 0 else cache["permuted_labels"][ci-1]
            n = len(y); m = subsample_m(n, np.unique(y,return_counts=True)[1])
            equal(saved[case+"__labels"], y, "PAIRED_LABELS", 0)
            check(value["n"] == n and value["m"] == m, "PAIRED_SUPPORT")
            ix = indices(y,m)
            equal(saved[case+"__indices"], ix, "PAIRED_PLAN", 0)
            points, estimates = {}, {}
            for key,d in unique.items():
                points[key] = raw_bits(d,y)
                samples = saved[case+"__samples__"+key]
                check(samples.shape == (5,999) and np.isfinite(samples).all(), "PAIRED_SAMPLES")
                estimates[key] = samples
                for si in range(5):
                    for draw in CHECK_DRAWS:
                        chosen = ix[si,draw]
                        equal(samples[si,draw], raw_bits(d[np.ix_(chosen,chosen)], y[chosen]), "PAIRED_RAW_SAMPLE")
                        sampled_values += 1
            check(set(value["statistics"]) == set(definitions), "PAIRED_STATISTICS")
            for name,(a,b) in definitions.items():
                point = points[a]-(points[b] if b else 0)
                samples = estimates[a]-(estimates[b] if b else 0)
                claim = value["statistics"][name]
                equal(claim["raw_point"], point, "PAIRED_POINT")
                check(set(claim["per_seed"]) == set(map(str,SEEDS)), "PAIRED_SEEDS")
                lows, highs = [], []
                for si,seed in enumerate(SEEDS):
                    lo,hi = bounds(point,samples[si],n,m); lows.append(lo); highs.append(hi)
                    c = claim["per_seed"][str(seed)]
                    equal([c["raw_point"],c["low"],c["high"]], [point,lo,hi], "PAIRED_BOUNDS")
                equal([claim["outer_low"],claim["outer_high"]], [min(lows),max(highs)], "PAIRED_ENVELOPE")
                check(claim["lower_bound_at_least_point05"] == (min(lows) >= .05), "PAIRED_GATE")
            checked[case] = {"standalone_lower_bound_crossings": sum(v["lower_bound_at_least_point05"]
                           for k,v in value["statistics"].items() if k.startswith("standalone:")),
                           "increment_lower_bound_crossings": sum(v["lower_bound_at_least_point05"]
                           for k,v in value["statistics"].items() if k.startswith("increment:"))}
        summary[question] = checked
    return {"report_sha256": sha(raw), "source_report_sha256": source_hash, "questions": summary,
            "every_index_plan_and_interval_recounted": True, "raw_sample_values_replayed_original_calyx": sampled_values,
            "raw_sample_positions_per_seed": CHECK_DRAWS, "all_raw_samples_replayed": False,
            "scientific_admission": False, "acceptance": "author-checked", "wall_seconds": time.perf_counter()-begin}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "diagnostic", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--diagnostic-sha256", required=True)
    a = p.parse_args()
    result = readback(a.publication,a.diagnostic,a.diagnostic_sha256)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2)
    print(json.dumps(result))
