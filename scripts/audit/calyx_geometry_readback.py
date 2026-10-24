"""Independent normalization/geometry replay plus original Calyx sample checks."""
import argparse
import io
import json
from math import fsum, sqrt
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import load_inputs, ROSTER, SEEDS, sha, check, equal, similarity, normalized
from audit.paired_roots_readback import reference_geometry, indices, bounds, CHECK_DRAWS
from text.calyx_bits import raw_bits, subsample_m

RULES = dict(normalization="per-coordinate population mean/std; constant coordinates zero; float32 reference distances",
             fit="all calibration profiles, label-blind; frozen across paired draws", data_seed=25,
             seeds=list(SEEDS), resamples=999, k=3, label_nulls=5, floor=.05)


def read_npz(path, pin):
    raw = path.read_bytes(); check(sha(raw) == pin, "GEOMETRY_ARTIFACT_HASH")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def normalize(value):
    # Scalar two-pass moments, independent of the producer's numpy axis std.
    mean = np.asarray([fsum(map(float, col))/len(col) for col in value.T])
    constant = np.asarray([len(set(map(float, col))) == 1 for col in value.T])
    sd = np.asarray([0 if constant[i] else sqrt(fsum((float(v)-mean[i])**2 for v in col)/len(col))
                     for i, col in enumerate(value.T)])
    result = np.asarray([[0 if constant[j] else (float(v)-mean[j])/sd[j] for j, v in enumerate(row)] for row in value])
    return result, mean, sd, constant


def distance(value):
    p = value.astype(np.float32)
    return cdist(p, p, "chebyshev").astype(np.float32)


def audit_scale(saved, prefix, profile):
    value, mean, std, constant = normalize(profile)
    equal(saved[prefix+"__mean"], mean, "GEOMETRY_MEAN")
    equal(saved[prefix+"__std"], std, "GEOMETRY_STD")
    equal(saved[prefix+"__constant"], constant, "GEOMETRY_CONSTANT", 0)
    return distance(value)


def audit_evaluation(label, record, saved, y, unique, definitions):
    n = len(y); m = subsample_m(n, np.unique(y, return_counts=True)[1]); ix = indices(y, m)
    check(record["n"] == n and record["m"] == m and record["unique_matrices"] == len(unique)
          and record["definitions"] == definitions, "GEOMETRY_DEFINITIONS")
    equal(saved[label+"__labels"], y, "GEOMETRY_LABELS", 0)
    equal(saved[label+"__indices"], ix, "GEOMETRY_PLAN", 0)
    points = {}; samples = {}; replayed = 0
    for key, d in unique.items():
        points[key] = raw_bits(d, y); values = saved[label+"__samples__"+key]
        check(values.shape == (5, 999) and np.isfinite(values).all(), "GEOMETRY_SAMPLES")
        samples[key] = values
        for seed in range(5):
            for draw in CHECK_DRAWS:
                rows = ix[seed, draw]
                equal(values[seed, draw], raw_bits(d[np.ix_(rows, rows)], y[rows]), "GEOMETRY_RAW_SAMPLE")
                replayed += 1
    check(set(record["statistics"]) == set(definitions), "GEOMETRY_STATISTICS")
    for name, (left, right) in definitions.items():
        point = points[left]-(points[right] if right else 0)
        values = samples[left]-(samples[right] if right else 0); item = record["statistics"][name]
        equal(item["raw_point"], point, "GEOMETRY_POINT")
        check(set(item["per_seed"]) == set(map(str, SEEDS)), "GEOMETRY_SEEDS")
        lows, highs = [], []
        for i, seed in enumerate(SEEDS):
            lo, hi = bounds(point, values[i], n, m); lows.append(lo); highs.append(hi)
            c = item["per_seed"][str(seed)]
            equal([c["raw_point"], c["low"], c["high"]], [point, lo, hi], "GEOMETRY_INTERVAL")
        equal([item["outer_low"], item["outer_high"]], [min(lows), max(highs)], "GEOMETRY_ENVELOPE")
        check(item["lower_bound_at_least_point05"] == (min(lows) >= .05), "GEOMETRY_GATE")
    return replayed


def boundary_inputs():
    rng = np.random.Generator(np.random.PCG64(25)); panel = rng.uniform(-1, 1, (504, 2))
    labels = np.asarray(panel[:, 0]+.25*panel[:, 1] > 0, dtype=np.int64)
    theta = np.pi/4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    values = {"duplicate": (panel.copy(), labels, 0), "scaled_copy": (panel*8, labels, 0),
              "rotated_copy": (np.matmul(panel, rotation), labels, 0),
              "smooth_function": (np.column_stack([panel[:, 0]**3, np.exp(panel[:, 1])]), labels, 0)}
    for f in (80, 512):
        signal = np.sin(f*panel[:, 0]); values["frequency_"+str(f)] = (np.column_stack([signal, panel[:, 1]]), (signal > 0).astype(np.int64), 0)
    values["noisy_copy"] = (panel+rng.normal(size=(504, 2))*.5, labels, 0)
    residual = rng.uniform(-1, 1, (504, 1)); values["positive_residual"] = (residual, (residual[:, 0] > 0).astype(np.int64), 1)
    return panel, values


def audit_control_inputs(saved, candidate, panel, y):
    # libm/BLAS evaluation of sin, exp and rotation is platform-specific in
    # float64. Require roundoff-only differences AND exact estimator float32
    # inputs; all derived distance matrices remain checked by exact hashes.
    x = saved["candidate"]
    check(x.shape == candidate.shape and np.isfinite(x).all(), "GEOMETRY_CONTROL_INPUT")
    error = np.abs(x-candidate)
    limit = 8*np.finfo(np.float64).eps*np.maximum(1, np.abs(candidate))
    check(np.all(error <= limit), "GEOMETRY_CONTROL_INPUT")
    equal(x.astype(np.float32), candidate.astype(np.float32), "GEOMETRY_CONTROL_INPUT32", 0)
    equal(saved["panel"], panel, "GEOMETRY_CONTROL_INPUT", 0)
    equal(saved["labels"], y, "GEOMETRY_CONTROL_LABELS", 0)
    return dict(float64_exact=bool(np.array_equal(x, candidate)), max_absolute_error=float(error.max()),
                estimator_float32_inputs_exact=True, labels_exact=True)


def audit_controls(root):
    raw = (root/"report.json").read_bytes(); report = json.loads(raw); panel, expected = boundary_inputs()
    check(report["schema"] == "calyx-geometry-diagnostic-v1-controls" and report["rules"] == RULES
          and report["complete"] is True and set(report["records"]) == set(expected), "GEOMETRY_CONTROLS")
    check(report["scientific_admission"] is False and report["conditional_information_certified"] is False, "GEOMETRY_CLAIM")
    failed = {m: [] for m in ("reference", "standardized")}; total = 0; arithmetic = {}
    for name, (candidate, y, truth) in expected.items():
        record = report["records"][name]; saved = read_npz(root/(name+".npz"), record["artifact_sha256"])
        check(json.loads((root/(name+".json")).read_bytes()) == record and record["true_cmi_bits"] == truth
              and record["kind"] == "artificial analytic boundary, never a production observation", "GEOMETRY_TRUTH")
        arithmetic[name] = audit_control_inputs(saved, candidate, panel, y)
        check(set(record["cases"]) == {"reference", "standardized"}, "GEOMETRY_METHODS")
        for method, value in record["cases"].items():
            xx, zz = candidate, panel
            if method == "standardized":
                for label, arr in (("candidate", xx), ("panel", zz)):
                    _, mean, sd, constant = normalize(arr)
                    equal(saved[label+"_mean"], mean, "GEOMETRY_CONTROL_SCALE")
                    equal(saved[label+"_std"], sd, "GEOMETRY_CONTROL_SCALE")
                    equal(saved[label+"_constant"], constant, "GEOMETRY_CONTROL_SCALE", 0)
                xx = normalize(candidate)[0]; zz = normalize(panel)[0]
            dx, dz = distance(xx), distance(zz); joint = np.maximum(dx, dz)
            a, b = sha(dz.tobytes()), sha(joint.tobytes()); unique = {a: dz, b: joint}
            definitions = dict(panel=[a, None], joint=[b, None], gain=[b, a], duplicate=[b, b])
            total += audit_evaluation(method, value, saved, y, unique, definitions)
            if truth == 0 and value["statistics"]["gain"]["lower_bound_at_least_point05"]:
                failed[method].append(name)
    check({k: sorted(v) for k, v in failed.items()} == {k: sorted(v) for k, v in report["false_conditional_signals"].items()}, "GEOMETRY_FALSE_SIGNALS")
    return dict(report_sha256=sha(raw), controls=len(expected), false_conditional_signals=failed,
                raw_samples_replayed=total, source_arithmetic=arithmetic)


def audit_dominance(claim, distances):
    n = len(next(iter(distances.values()))); previous = np.zeros((n, n), np.float32)
    full = np.maximum.reduce(list(distances.values())); mask = np.triu(np.ones((n, n), bool), 1)
    expected = {}
    for name in ROSTER:
        new = np.maximum(previous, distances[name]); rest = np.maximum.reduce([d for s, d in distances.items() if s != name])
        expected[name] = dict(prefix_changed_pairs=int(((new != previous) & mask).sum()),
                              unique_full_max_pairs=int(((distances[name] > rest) & mask).sum()),
                              joint_equals_this_matrix=bool(np.array_equal(full, distances[name])), pair_count=int(mask.sum()))
        previous = new
    check(claim == expected, "GEOMETRY_DOMINANCE")


def audit_question(root, name, record, source, inputs):
    saved = read_npz(root/(name+".npz"), record["artifact_sha256"])
    source_record = source["report"]["findings"][name]
    native = read_npz(source["root"]/(name+".npz"), source_record["artifact_sha256"])
    check(record["source_artifact_sha256"] == source_record["artifact_sha256"], "GEOMETRY_SOURCE_CACHE")
    family, prop = name.split("-"); anchor = inputs["anchors"][family][prop]; n = len(anchor["labels"])
    distances = {}; original = {}; expected_slots = {}
    for sense in ROSTER:
        v, offsets = inputs["matrices"][sense], inputs["offsets"][sense]
        sets = [np.vstack([v[offsets[c]:offsets[c+1]] for c in inputs["rows"][(family, row)]]) for row in anchor["rows"]]
        landmarks = [len(sets)*i//24 for i in range(24)]
        profile = normalized(np.asarray([[similarity(s, sets[i]) for i in landmarks] for s in sets]))
        equal(saved[sense+"__profile"], profile, "GEOMETRY_PROFILE")
        equal(distance(profile), native[sense+"__distance"], "GEOMETRY_REFERENCE", 0)
        d = audit_scale(saved, sense, profile); equal(d, saved[sense+"__distance"], "GEOMETRY_DISTANCE", 0)
        distances[sense] = d; original[sense] = native[sense+"__distance"]
        constant = saved[sense+"__constant"]
        expected_slots[sense] = dict(constant_coordinates=int(constant.sum()), nonconstant_coordinates=int((~constant).sum()))
    check(record["slots"] == expected_slots and record["n"] == n, "GEOMETRY_SLOTS")
    audit_dominance(record["reference_dominance"], original); audit_dominance(record["standardized_dominance"], distances)
    check(record["unchanged_d1_spearman"] == source_record["d1_spearman"], "GEOMETRY_D1")
    unique, aliases, defs = reference_geometry({s+"__distance": d for s, d in distances.items()})
    check(record["aliases"] == aliases and record["definitions"] == defs, "GEOMETRY_ALIAS")
    expected_cases = ["authentic"]+["label_null_"+str(20261019+i) for i in range(5)]
    check(set(record["cases"]) == set(expected_cases), "GEOMETRY_CASES")
    total = 0
    for i, case in enumerate(expected_cases):
        y = native["labels"] if i == 0 else native["permuted_labels"][i-1]
        total += audit_evaluation(case, record["cases"][case], saved, y, unique, defs)
    return total


def readback(root, diagnostic, slots, calibration, fsv):
    start = time.perf_counter(); raw = (root/"report.json").read_bytes(); report = json.loads(raw)
    sr = (diagnostic/"report.json").read_bytes(); source = dict(root=diagnostic, report=json.loads(sr))
    check(report["schema"] == "calyx-geometry-diagnostic-v1" and report["complete"] is True and not report["errors"], "GEOMETRY_COMPLETE")
    check(report["source_report_sha256"] == sha(sr) and report["rules"] == RULES, "GEOMETRY_BINDING")
    check(all(report[k] is False for k in ("scientific_admission", "conditional_information_certified", "served")), "GEOMETRY_CLAIM")
    inputs = load_inputs(slots, calibration, fsv)
    check(report["slots_report_sha256"] == inputs["report_sha256"] == source["report"]["source_report_sha256"], "GEOMETRY_SLOTS_PIN")
    controls = audit_controls(root/"controls")
    check(controls["report_sha256"] == report["controls_report_sha256"] and
          {k: sorted(v) for k, v in controls["false_conditional_signals"].items()} ==
          {k: sorted(v) for k, v in report["false_conditional_signals"].items()}, "GEOMETRY_CONTROLS_BINDING")
    check(set(report["findings"]) == set(source["report"]["findings"]), "GEOMETRY_QUESTIONS")
    summaries = {}; total = controls["raw_samples_replayed"]
    for name, record in report["findings"].items():
        check(json.loads((root/(name+".json")).read_bytes()) == record, "GEOMETRY_RECORD_COPY")
        total += audit_question(root, name, record, source, inputs)
        summaries[name] = dict(standalone_crossings=sum(s["lower_bound_at_least_point05"] for n, s in record["cases"]["authentic"]["statistics"].items() if n.startswith("standalone:")),
                              prefix_increment_crossings=sum(s["lower_bound_at_least_point05"] for n, s in record["cases"]["authentic"]["statistics"].items() if n.startswith("increment:")),
                              disappearing_views=[s for s, d in record["standardized_dominance"].items() if d["prefix_changed_pairs"] == 0])
    return dict(acceptance="author-checked", report_sha256=sha(raw), controls=controls, questions=summaries,
                all_geometry_indices_and_intervals_equal=True, raw_samples_replayed_original_calyx=total,
                sample_positions_per_seed=CHECK_DRAWS, all_raw_samples_replayed=False,
                scientific_admission=False, wall_seconds=time.perf_counter()-start)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "diagnostic", "slots", "calibration", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); r = readback(a.publication, a.diagnostic, a.slots, a.calibration, a.fsv)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(r, f, sort_keys=True)
    print(json.dumps(r))
