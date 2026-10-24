"""Disjoint-design conditional-geometry experiment; no admission (#12)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_geometry import controls
from audit.calyx_resolution import SEEDS, ROSTER, SOURCE_PIN, sha, write, save, need, reference
from text.calyx_indexed import indexed_counts, raw_bits
from text.conditional_geometry import fit, predict


def case(output, name, x, y, blocks, seed, truth=None, source=None, fixed_split=None):
    started = time.perf_counter()
    if fixed_split is None:
        order = np.random.default_rng(seed).permutation(len(y))
        design, evaluation = order[:len(y)//2], order[len(y)//2:]
    else:
        design, evaluation = (np.asarray(v, dtype=np.int64) for v in fixed_split)
        need(not set(design) & set(evaluation) and
             sorted(np.r_[design, evaluation]) == list(range(len(y))), 'FIXED_SPLIT')
    z = np.concatenate(blocks, axis=1)
    tick = time.perf_counter()
    model = fit(z[design], x[design])
    fit_seconds = time.perf_counter()-tick
    model_path = output/(name+".model.json"); write(model_path, model)
    frozen = json.loads(model_path.read_bytes())
    prediction = predict(frozen, z[evaluation])
    need(np.isfinite(prediction).all(), "PREDICTION")
    arrays = dict(labels=y, design=design, evaluation=evaluation, candidate=x,
                  predictions=prediction, **{f"block_{i}": b for i, b in enumerate(blocks)})
    selected = [b[evaluation] for b in blocks]
    yy, xx = y[evaluation], x[evaluation]
    variants = dict(panel=selected, joint=selected+[xx],
                    closed_panel=selected+[prediction], closed_joint=selected+[prediction, xx])
    estimates = {}; tick = time.perf_counter()
    for key, values in variants.items():
        counts, radius = indexed_counts(values, yy)
        independent, ir = reference(values, yy, np.arange(len(yy)))
        need(np.array_equal(counts, independent) and np.array_equal(radius, ir), "CLOSURE_COUNTS")
        arrays[key+"_counts"], arrays[key+"_radii"] = counts, radius
        estimates[key] = raw_bits(counts)
    count_seconds = time.perf_counter()-tick
    estimates.update(raw_gain=estimates["joint"]-estimates["panel"],
                     closure_gain=estimates["closed_joint"]-estimates["closed_panel"])
    sse = np.sum((xx-prediction)**2, axis=0); base = np.sum((xx-x[design].mean(0))**2, axis=0)
    result = dict(name=name, n=len(y), design_rows=len(design), evaluation_rows=len(evaluation),
        blocks=[b.shape[1] for b in blocks], candidate_dimensions=x.shape[1], seed=seed,
        model_file=model_path.name, model_sha256=sha(model_path), source=source,
        array_file=name+".npz", array_sha256=save(output/(name+".npz"), arrays), bits=estimates,
        reconstruction_r2=(1-sse/np.maximum(base, 1e-30)).tolist(), true_cmi_bits=truth,
        scope="artificial_analytic_boundary" if truth is not None else "real_profile_diagnostic",
        timing=dict(fit_seconds=fit_seconds, count_seconds=count_seconds, wall_seconds=time.perf_counter()-started))
    write(output/(name+".json"), result)
    print(json.dumps(dict(name=name, raw_gain=estimates["raw_gain"], closure_gain=estimates["closure_gain"],
                         r2_min=min(result["reconstruction_r2"]), seconds=result["timing"]["wall_seconds"])), flush=True)
    return result


def compatible_inputs(manifest_path, evidence, profile_source):
    """D1-only candidate panels, with every fixed-order prefix retained."""
    from audit.candidate_panel_geometry import inputs, cache, validate_result
    manifest_pin = sha(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    sources, values, names = inputs(manifest)
    report = json.loads((evidence/"report.json").read_bytes())
    need(report["complete"] and report["manifest_sha256"] == manifest_pin
         and report["sources"] == sources and report["names"] == names, "D1_SOURCE")
    public = Path(manifest["public"]["root"])
    public_report = json.loads((public/"report.json").read_bytes())
    need(sha(profile_source/"report.json") == SOURCE_PIN, "PROFILE_SOURCE")
    geometry = json.loads((profile_source/"report.json").read_bytes())
    need(geometry["source_report_sha256"] == sources["public"]["report_sha256"], "PROFILE_BINDING")
    cases, selections = [], {}
    for question, value in sorted(values.items()):
        claim = report["findings"][question]
        saved = cache(evidence/(question+".npz"), claim["artifact_sha256"])
        validate_result(claim, saved, value, names)
        base_path = public/(question+".npz")
        base = cache(base_path, public_report["findings"][question]["artifact_sha256"])
        geometry_record = geometry["findings"][question]
        need(geometry_record["source_artifact_sha256"] == sha(base_path), "PROFILE_ARRAY_BINDING")
        profile_path = profile_source/(question+".npz")
        data = cache(profile_path, geometry_record["artifact_sha256"])
        need(np.array_equal(data["labels"], base["labels"]), "PROFILE_LABELS")
        profiles = {name: data[name+"__profile"] for name in ROSTER}
        pins = {"public": dict(file=str(base_path), sha256=sha(base_path))}
        pins["profiles"] = dict(file=str(profile_path), sha256=sha(profile_path))
        from text.calyx_bits import chebyshev_matrix
        for name in ROSTER:
            need(np.array_equal(chebyshev_matrix(profiles[name]), base[name+"__distance"]),
                 "PROFILE_DISTANCE_REPLAY")
        for name, item in manifest["candidates"].items():
            root = Path(item["root"])
            candidate_report = json.loads((root/"report.json").read_bytes())
            path = root/(question+".npz")
            data = cache(path, candidate_report["findings"][question]["artifact_sha256"])
            need(np.array_equal(base["labels"], data["labels"]) and
                 np.array_equal(base["pairs"], data["pairs"]), "CANDIDATE_ALIGNMENT")
            profiles[name] = data["profile"]
            pins[name] = dict(file=str(path), sha256=sha(path))
        prefixes = sorted({tuple(panel[:i+1]) for panel in claim["d1_only"]["all_maximizers"]
                           for i in range(1, len(panel))})
        selections[question] = dict(panels=claim["d1_only"], prefixes=prefixes)
        for prefix in prefixes:
            need(list(prefix) == sorted(prefix, key=names.index), "FIXED_REGISTRY_ORDER")
            label = question+"-"+hashlib.sha256("|".join(prefix).encode()).hexdigest()[:16]
            cases.append(dict(name=label, x=profiles[prefix[-1]], y=base["labels"],
                              blocks=[profiles[n] for n in prefix[:-1]],
                              source=dict(pins=pins, candidate=prefix[-1], conditioning=prefix[:-1])))
    return cases, dict(manifest_sha256=manifest_pin, d1_report_sha256=sha(evidence/"report.json"),
                       sources=sources, selection=selections, unique_prefixes=len(cases))


def run(source, output, panel_manifest=None, panel_evidence=None):
    started = time.perf_counter()
    if panel_manifest is not None:
        cases, provenance = compatible_inputs(panel_manifest, panel_evidence, source)
        need(len(cases) == 35, "PREFLIGHT_CASE_COUNT")
    else:
        need(sha(source/"report.json") == SOURCE_PIN, "SOURCE")
        report = json.loads((source/"report.json").read_bytes())
        provenance = dict(source_sha256=SOURCE_PIN)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"intent.json", dict(provenance=provenance, seeds=SEEDS, design_fraction=.5,
        method="full_panel_plus_disjoint_outcome_blind_reconstruction", scientific_admission=False))
    rows = []
    artificial = controls()
    rng = np.random.default_rng(25)
    artificial["independent"] = (rng.normal(size=(504, 1)), rng.integers(0,2,504), rng.normal(size=(504,2)), 0)
    # BLAS is single-threaded: this first implementation measures its CPU cost.
    with threadpool_limits(limits=1):
        for name, (x,y,z,truth) in (artificial.items() if panel_manifest is None else ()):
            for seed in SEEDS:
                rows.append(case(output, f"control-{name}-{seed}", x, y, [z], seed, truth))
        for question in (("science-P123", "biomedical-P1057", "biomedical-P1995") if panel_manifest is None else ()):
            path = source/(question+".npz")
            need(sha(path) == report["findings"][question]["artifact_sha256"], "SOURCE_ARRAYS")
            with np.load(path, allow_pickle=False) as data:
                y = data["labels"].copy(); profiles = [data[n+"__profile"].copy() for n in ROSTER]
            for i in range(1,len(ROSTER)):
                for seed in SEEDS:
                    rows.append(case(output, f"{question}-{ROSTER[i]}-{seed}", profiles[i], y, profiles[:i],
                                     seed, source=dict(file=str(path), sha256=sha(path))))
        if panel_manifest is not None:
            for item in cases:
                for seed in SEEDS:
                    rows.append(case(output, item["name"]+f"-{seed}", item["x"], item["y"],
                                     item["blocks"], seed, source=item["source"]))
    result = dict(schema="calyx-reconstruction-closure-v1", rows=rows, complete=True,
        provenance=provenance, scientific_admission=False, conditional_information_certified=False,
        served=False, wall_seconds=time.perf_counter()-started,
        code_sha256={str(p.relative_to(Path(__file__).parents[1])): sha(p) for p in
            (Path(__file__), Path(__file__).parents[1]/"text/conditional_geometry.py")})
    write(output/"report.json", result)
    write(output/"manifest.json", dict(report_sha256=sha(output/"report.json"), intent_sha256=sha(output/"intent.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--panel-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if bool(a.panel_manifest) != bool(a.panel_evidence):
        parser.error("--panel-manifest and --panel-evidence must be supplied together")
    run(a.source, a.output, a.panel_manifest, a.panel_evidence)
