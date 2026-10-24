"""Independent candidate geometry/count replay and persisted corrupt controls."""
import copy
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.commissioned_candidate import parser, sources, RULES, SCHEMA
from audit.calyx_diagnostic_readback import geometry, similarity, normalized, ROSTER, SEEDS, sha, check, equal
from audit.paired_roots_readback import indices, bounds
from text.calyx_bits import raw_bits, subsample_m
from audit.commissioned_window_readback import digest
from senses.trajectory_population import publish


def read_cache(path, pin):
    raw = path.read_bytes(); check(sha(raw) == pin, "CANDIDATE_ARTIFACT_HASH")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def audit_question(record, saved, sets, y, public):
    n = len(y); upper = np.stack(np.triu_indices(n, 1), axis=1)
    pairs = upper[np.random.default_rng(20261019).choice(len(upper), 4096, replace=False)]
    equal(saved["pairs"], pairs, "CANDIDATE_PAIRS", 0); equal(saved["labels"], y, "CANDIDATE_LABELS", 0)
    landmarks = [n*i//24 for i in range(24)]
    profile = normalized(np.asarray([[similarity(a, sets[j]) for j in landmarks] for a in sets]))
    equal(saved["profile"], profile, "CANDIDATE_PROFILE")
    distances, readings = geometry(sets, pairs)
    equal(saved["distance"], distances, "CANDIDATE_DISTANCE", 0)
    equal(saved["pair_readings"], readings, "CANDIDATE_PAIR_VALUES")
    correlations = {s: float(np.corrcoef(rankdata(readings), rankdata(public[s+"__pair_readings"]))[0, 1]) for s in ROSTER}
    for sense in ROSTER:
        equal(record["d1_spearman"][sense], correlations[sense], "CANDIDATE_D1")
    check(record["n"] == n and record["below_d1_ceiling"] == [s for s in ROSTER if correlations[s] <= .6]
          and record["above_d1_ceiling"] == [s for s in ROSTER if correlations[s] > .6], "CANDIDATE_D1_CLAIM")
    m = subsample_m(n, np.unique(y, return_counts=True)[1]); ix = indices(y, m)
    equal(saved["indices"], ix, "CANDIDATE_PLAN", 0)
    point = raw_bits(distances, y); equal(record["standalone"]["raw_point"], point, "CANDIDATE_POINT")
    check(saved["samples"].shape == (5, 999), "CANDIDATE_SAMPLES")
    lows, highs = [], []; replayed = 0
    for si, seed in enumerate(SEEDS):
        values = np.asarray([raw_bits(distances[np.ix_(rows, rows)], y[rows]) for rows in ix[si]])
        equal(saved["samples"][si], values, "CANDIDATE_RAW_SAMPLES"); replayed += len(values)
        lo, hi = bounds(point, values, n, m); lows.append(lo); highs.append(hi)
        c = record["standalone"]["per_seed"][str(seed)]
        equal([c["raw_point"], c["low"], c["high"]], [point, lo, hi], "CANDIDATE_INTERVAL")
    equal([record["standalone"]["outer_low"], record["standalone"]["outer_high"]], [min(lows), max(highs)], "CANDIDATE_ENVELOPE")
    check(record["standalone"]["lower_bound_at_least_point05"] == (min(lows) >= .05), "CANDIDATE_STANDALONE_CLAIM")
    check(record["duplicate_control"]["raw_point"] == record["duplicate_control"]["outer_low"]
          == record["duplicate_control"]["outer_high"] == 0, "CANDIDATE_DUPLICATE")
    labels = np.asarray([np.random.default_rng(20261019+i).permutation(y) for i in range(199)])
    equal(saved["permuted_labels"], labels, "CANDIDATE_NULL_LABELS", 0)
    nulls = np.asarray([raw_bits(distances, q) for q in labels]); equal(saved["null_scores"], nulls, "CANDIDATE_NULL_VALUES")
    c = record["label_nulls"]
    equal([c["n"], c["max_raw_bits"], c["point_floor_crossings"], c["tail_probability"]],
          [199, nulls.max(), np.sum(nulls >= .05), (1+np.sum(nulls >= point))/200], "CANDIDATE_NULL_CLAIMS")
    return dict(ordinary_pairs_replayed=4096, all_standalone_samples_replayed=replayed,
                all_label_nulls_replayed=len(nulls), below_d1_ceiling=record["below_d1_ceiling"])


def audit_header(report, cold, public_pin):
    check(report["schema"] == SCHEMA and report["complete"] is True and not report["errors"]
          and report["rules"] == RULES, "CANDIDATE_HEADER")
    check(all(report[k] is False for k in ("scientific_admission", "conditional_information_certified", "served")), "CANDIDATE_CLAIM")
    check(report["candidate_report_sha256"] == cold["report_sha256"] and report["candidate_lens_id"] == cold["lens_id"]
          and report["public_diagnostic_sha256"] == public_pin, "CANDIDATE_HEADER_BINDING")


def main():
    p = parser(); p.add_argument("--candidate-diagnostic", type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.perf_counter(); checks = []
    cold, inputs, public, candidate, offsets = sources(a)
    raw = (a.candidate_diagnostic/"report.json").read_bytes(); report = json.loads(raw)
    before = {str(a.candidate_diagnostic/n): digest(a.candidate_diagnostic/n)
              for n in ["report.json"]+[q+s for q in report["findings"] for s in (".json", ".npz")]}
    publish(a.output/"before.json", before); audit_header(report, cold, a.public_diagnostic_sha256)

    def ok(name, value):
        checks.append(dict(name=name, passed=bool(value)))

    def refuses(name, code, fn):
        observed = None
        try: fn()
        except ValueError as exc: observed = str(exc)
        checks.append(dict(name=name, passed=observed is not None and code in observed, observed=observed))

    expected = {f+"-"+p for f, questions in inputs["anchors"].items() for p in questions}
    check(set(report["findings"]) == expected and report["gpu_replay_sha256"] == digest(a.gpu_replay), "CANDIDATE_QUESTIONS")
    ok("B1 pinned native inputs and GPU replay bind exact questions", True); results = {}; retained = None
    for name, record in report["findings"].items():
        family, prop = name.split("-"); anchor = inputs["anchors"][family][prop]; y = np.asarray(anchor["labels"])
        saved = read_cache(a.candidate_diagnostic/(name+".npz"), record["artifact_sha256"])
        pin = public["findings"][name]["artifact_sha256"]
        check(record["source_artifact_sha256"] == pin, "CANDIDATE_PUBLIC_CACHE")
        source_cache = read_cache(a.public_diagnostic/(name+".npz"), pin)
        sets = [np.vstack([candidate[offsets[c]:offsets[c+1]] for c in inputs["rows"][(family, row)]]) for row in anchor["rows"]]
        results[name] = audit_question(record, saved, sets, y, source_cache)
        check(json.loads((a.candidate_diagnostic/(name+".json")).read_bytes()) == record, "CANDIDATE_RECORD")
        if retained is None: retained = record, saved, sets, y, source_cache
    ok("B2 all distances, pairs, 14985 subsamples and 597 label nulls replay", sum(v["all_standalone_samples_replayed"] for v in results.values()) == 14985)
    record, saved, sets, y, source_cache = retained
    for field, index, value, code in (("distance", (0, 1), 99, "CANDIDATE_DISTANCE"),
                                     ("pairs", (0, 0), 999, "CANDIDATE_PAIRS"),
                                     ("indices", (0, 0, 0), -1, "CANDIDATE_PLAN"),
                                     ("samples", (0, 0), 99, "CANDIDATE_RAW_SAMPLES")):
        bad = {k: v.copy() for k, v in saved.items()}; bad[field][index] = value
        path = a.output/(field+"-corrupt.npz"); np.savez(path, **bad)
        refuses("B2 rehashed corrupt "+field+" refuses", code,
                lambda: audit_question(record, read_cache(path, digest(path)), sets, y, source_cache))
    for field in ("scientific_admission", "conditional_information_certified", "served"):
        bad = {**report, field: True}; publish(a.output/(field+".json"), bad)
        refuses("B3 false "+field+" refuses", "CANDIDATE_CLAIM", lambda: audit_header(bad, cold, a.public_diagnostic_sha256))
    after = {path: digest(Path(path)) for path in before}; publish(a.output/"after.json", after)
    ok("B1/B3 positive publication unchanged", before == after)
    result = dict(acceptance="author-checked", source_report_sha256=sha(raw), checks=checks,
                  passed=sum(c["passed"] for c in checks), total=len(checks), questions=results,
                  scientific_admission=False, conditional_information_certified=False, wall_seconds=time.perf_counter()-started)
    publish(a.output/"fsv.json", result); print(json.dumps({k: result[k] for k in ("passed", "total", "wall_seconds")}))
    if result["passed"] != result["total"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
