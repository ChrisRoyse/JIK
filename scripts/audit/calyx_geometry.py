"""Fixed label-blind geometry comparator and conditional-null audit; no admission.

Calyx/Ross count and paired-root formulas are unchanged. Standardization is an
explicit experimental preprocessing step, not an assertion about upstream Rust.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import load_inputs, ROSTER, SEEDS, sha, check
from audit.calyx_paired_roots import evaluate, matrices
from text.calyx_bits import maxsim_profiles, chebyshev_matrix, plain, RESAMPLES
from senses.trajectory_population import publish

SCHEMA = "calyx-geometry-diagnostic-v1"
RULES = dict(normalization="per-coordinate population mean/std; constant coordinates zero; float32 reference distances",
             fit="all calibration profiles, label-blind; frozen across paired draws", data_seed=25,
             seeds=list(SEEDS), resamples=RESAMPLES, k=3, label_nulls=5, floor=.05)


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    check(x.ndim == 2 and len(x) >= 50 and np.isfinite(x).all(), "GEOMETRY_INPUT")
    mean = x.mean(axis=0); sd = x.std(axis=0)
    constant = np.all(x == x[0], axis=0)
    check(np.all(sd[~constant] > 0), "GEOMETRY_SCALE")
    sd[constant] = 0
    value = np.where(constant, 0, (x-mean)/np.where(constant, 1, sd))
    return value, mean, sd, constant


def controls():
    rng = np.random.default_rng(25); z = rng.uniform(-1, 1, (504, 2))
    y = (z[:, 0]+.25*z[:, 1] > 0).astype(np.int64)
    angle = np.pi/4; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = dict(duplicate=(z.copy(), y, z, 0), scaled_copy=(8*z, y, z, 0),
                  rotated_copy=(z@rotation, y, z, 0), smooth_function=(np.column_stack((z[:, 0]**3, np.exp(z[:, 1]))), y, z, 0))
    for frequency in (80, 512):
        value = np.sin(frequency*z[:, 0])
        result["frequency_"+str(frequency)] = (np.column_stack((value, z[:, 1])), (value > 0).astype(np.int64), z, 0)
    result["noisy_copy"] = (z+.5*rng.normal(size=z.shape), y, z, 0)
    residual = rng.uniform(-1, 1, (504, 1))
    result["positive_residual"] = (residual, (residual[:, 0] > 0).astype(np.int64), z, 1)
    return result


def two_slot_matrices(panel, candidate):
    joint = np.maximum(panel, candidate)
    unique = {sha(d.tobytes()): d for d in (panel, joint)}
    a, b = sha(panel.tobytes()), sha(joint.tobytes())
    definitions = dict(panel=[a, None], joint=[b, None], gain=[b, a], duplicate=[b, b])
    return unique, definitions


def evaluate_save(label, y, unique, definitions, arrays):
    ix, estimates, statistics = evaluate(y, unique, definitions)
    arrays[label+"__indices"] = ix; arrays[label+"__labels"] = y
    for key, values in estimates.items():
        arrays[label+"__samples__"+key] = values
    return dict(statistics=statistics, definitions=definitions, n=len(y), m=ix.shape[2], unique_matrices=len(unique))


def control_run(output):
    output.mkdir(parents=True); records = {}; started = time.perf_counter()
    for name, (x, y, z, truth) in controls().items():
        begin = time.perf_counter(); arrays = dict(candidate=x, panel=z, labels=y)
        cases = {}
        for method in ("reference", "standardized"):
            xx, zz = x, z
            if method == "standardized":
                xx, xm, xs, xc = standardize(x); zz, zm, zs, zc = standardize(z)
                arrays.update(candidate_mean=xm, candidate_std=xs, candidate_constant=xc,
                              panel_mean=zm, panel_std=zs, panel_constant=zc)
            dx, dz = chebyshev_matrix(xx), chebyshev_matrix(zz)
            unique, defs = two_slot_matrices(dz, dx)
            cases[method] = evaluate_save(method, y, unique, defs, arrays)
        dest = output/(name+".npz"); np.savez(dest, **arrays)
        record = dict(kind="artificial analytic boundary, never a production observation", true_cmi_bits=truth,
                      cases=cases, artifact_sha256=sha(dest.read_bytes()), wall_seconds=time.perf_counter()-begin)
        records[name] = record; publish(output/(name+".json"), record)
        print(json.dumps(dict(control=name, truth=truth, standardized_gain=cases["standardized"]["statistics"]["gain"])), flush=True)
    failed = {method: [n for n, r in records.items() if r["true_cmi_bits"] == 0 and
                      r["cases"][method]["statistics"]["gain"]["lower_bound_at_least_point05"]]
              for method in ("reference", "standardized")}
    result = dict(schema=SCHEMA+"-controls", rules=RULES, records=records, false_conditional_signals=failed,
                  complete=True, scientific_admission=False, conditional_information_certified=False,
                  wall_seconds=time.perf_counter()-started)
    publish(output/"report.json", result)
    return result


def dominance(distances):
    n = len(next(iter(distances.values()))); ix = np.triu_indices(n, 1)
    joint = np.maximum.reduce(list(distances.values()))
    previous = np.zeros_like(joint); rows = {}
    for name in ROSTER:
        d = distances[name]; updated = np.maximum(previous, d)
        without = np.maximum.reduce([v for k, v in distances.items() if k != name])
        rows[name] = dict(prefix_changed_pairs=int(np.count_nonzero(updated[ix] != previous[ix])),
                          unique_full_max_pairs=int(np.count_nonzero(d[ix] > without[ix])),
                          joint_equals_this_matrix=bool(np.array_equal(joint, d)), pair_count=len(ix[0]))
        previous = updated
    return rows


def question(inputs, family, prop, source, output):
    started = time.perf_counter(); anchor = inputs["anchors"][family][prop]
    key = family+"-"+prop; spec = source["report"]["findings"][key]
    raw = (source["root"]/(key+".npz")).read_bytes(); check(sha(raw) == spec["artifact_sha256"], "GEOMETRY_SOURCE_CACHE")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        cache = {k: z[k].copy() for k in z.files}
    y = np.asarray(anchor["labels"]); check(np.array_equal(y, cache["labels"]), "GEOMETRY_LABELS")
    arrays = dict(labels=y); normalized = {}; original = {}; slots = {}
    for name in ROSTER:
        vector = inputs["matrices"][name]; offset = inputs["offsets"][name]
        sets = [np.concatenate([vector[offset[c]:offset[c+1]] for c in inputs["rows"][(family, r)]]) for r in anchor["rows"]]
        profile = maxsim_profiles(sets, anchor["rows"]); reference = chebyshev_matrix(profile)
        check(np.array_equal(reference, cache[name+"__distance"]), "GEOMETRY_REFERENCE_REPLAY")
        transformed, mean, std, constant = standardize(profile)
        d = chebyshev_matrix(transformed); normalized[name] = d; original[name] = reference
        arrays.update({name+"__profile": profile, name+"__mean": mean, name+"__std": std,
                       name+"__constant": constant, name+"__distance": d})
        slots[name] = dict(constant_coordinates=int(constant.sum()), nonconstant_coordinates=int((~constant).sum()))
    geometry_seconds = time.perf_counter()-started
    unique, aliases, definitions = matrices({n+"__distance": d for n, d in normalized.items()})
    cases = {}; t = time.perf_counter()
    for number in range(6):
        label = "authentic" if number == 0 else "label_null_"+str(20261018+number)
        labels = y if number == 0 else cache["permuted_labels"][number-1]
        cases[label] = evaluate_save(label, labels, unique, definitions, arrays)
        print(json.dumps(dict(question=key, case=label, unique_matrices=len(unique))), flush=True)
    estimate_seconds = time.perf_counter()-t; t = time.perf_counter()
    dest = output/(key+".npz"); np.savez(dest, **arrays)
    record = dict(n=len(y), cases=cases, slots=slots, aliases=aliases, definitions=definitions,
                  reference_dominance=dominance(original), standardized_dominance=dominance(normalized),
                  unchanged_d1_spearman=spec["d1_spearman"], source_artifact_sha256=sha(raw), artifact_sha256=sha(dest.read_bytes()),
                  timing=dict(geometry_seconds=geometry_seconds, estimates_seconds=estimate_seconds,
                              write_seconds=time.perf_counter()-t, wall_seconds=time.perf_counter()-started))
    publish(output/(key+".json"), plain(record))
    return record


def run(slots, calibration, fsv, diagnostic, diagnostic_hash, output):
    started = time.perf_counter(); check(not output.exists(), "GEOMETRY_EXISTS")
    raw = (diagnostic/"report.json").read_bytes(); report = json.loads(raw)
    check(sha(raw) == diagnostic_hash and report["complete"] and not report["errors"], "GEOMETRY_SOURCE_PIN")
    inputs = load_inputs(slots, calibration, fsv)
    check(report["source_report_sha256"] == inputs["report_sha256"], "GEOMETRY_BINDING")
    output.mkdir(parents=True); timing = dict(input_seconds=time.perf_counter()-started)
    t = time.perf_counter(); boundary = control_run(output/"controls"); timing["controls_seconds"] = time.perf_counter()-t
    source = dict(root=diagnostic, report=report); records = {}; errors = []
    # The previously failing real question is the fixed minimum source case.
    order = [("science", "P123"), ("biomedical", "P1057"), ("biomedical", "P1995")]
    for family, prop in order:
        try:
            records[family+"-"+prop] = question(inputs, family, prop, source, output)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(question=family+"-"+prop, error=str(exc)))
            if not records:
                break
    timing["questions_seconds"] = sum(r["timing"]["wall_seconds"] for r in records.values())
    wall = time.perf_counter()-started; timing.update(other_seconds=wall-sum(timing.values()), wall_seconds=wall)
    result = dict(schema=SCHEMA, source_report_sha256=diagnostic_hash, slots_report_sha256=inputs["report_sha256"],
                  controls_report_sha256=sha((output/"controls/report.json").read_bytes()), rules=RULES, findings=records,
                  errors=errors, complete=not errors and len(records) == len(order), scientific_admission=False,
                  conditional_information_certified=False, served=False, false_conditional_signals=boundary["false_conditional_signals"],
                  code_sha256=sha(Path(__file__).read_bytes()), reference_code_sha256=sha(Path(__file__).parents[1].joinpath("text/calyx_bits.py").read_bytes()),
                  paired_code_sha256=sha(Path(__file__).with_name("calyx_paired_roots.py").read_bytes()), timing=timing)
    publish(output/"report.json", plain(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("slots", "calibration", "fsv", "diagnostic", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--diagnostic-sha256", required=True)
    a = p.parse_args(); r = run(a.slots, a.calibration, a.fsv, a.diagnostic, a.diagnostic_sha256, a.output)
    print(json.dumps({k: r[k] for k in ("complete", "errors", "false_conditional_signals", "timing")}))
    if not r["complete"]:
        raise SystemExit(2)
