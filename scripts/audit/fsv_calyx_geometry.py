"""Persisted corruption controls across geometry, estimates and claim boundaries."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_geometry import run
from audit.calyx_geometry_readback import (readback, audit_scale, audit_evaluation, audit_dominance,
                                         read_npz, boundary_inputs, audit_control_inputs)
from audit.paired_roots_readback import reference_geometry
from audit.calyx_diagnostic_readback import sha, equal, ROSTER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "diagnostic", "slots", "calibration", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); checks = []

    def put(name, obj):
        with (a.output/(name+".json")).open("x", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, allow_nan=False)

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except ValueError as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, expected=code, observed=observed)

    report = json.loads((a.publication/"report.json").read_bytes())
    source = json.loads((a.diagnostic/"report.json").read_bytes())
    files = [a.publication/"report.json", a.diagnostic/"report.json", a.slots/"report.json", a.calibration/"report.json", a.fsv]
    files += [a.publication/(q+s) for q in report["findings"] for s in (".npz", ".json")]
    files += [a.diagnostic/(q+".npz") for q in source["findings"]]
    before = {str(p): sha(p.read_bytes()) for p in files}; put("before", before)
    cold = readback(a.publication, a.diagnostic, a.slots, a.calibration, a.fsv); put("cold-readback", cold)
    check("B1 native profiles and fixed normalization cold-reconstruct", cold["all_geometry_indices_and_intervals_equal"])
    check("B2 original Calyx point/subsample counts replay", cold["raw_samples_replayed_original_calyx"] > 0,
          replayed=cold["raw_samples_replayed_original_calyx"], all_raw_samples_replayed=False)
    check("B3 known conditional-null failures remain explicit", cold["controls"]["false_conditional_signals"] ==
          {"reference": ["frequency_80", "frequency_512"], "standardized": ["frequency_80", "frequency_512"]})
    panel, controls = boundary_inputs(); candidate, y, _ = controls["frequency_512"]
    changed = candidate.copy(); changed[0, 0] += 1e-4
    np.savez(a.output/"changed-control-input.npz", candidate=changed, panel=panel, labels=y)
    refuses("B1 material analytic input changes refuse", "GEOMETRY_CONTROL_INPUT",
            lambda: audit_control_inputs(dict(candidate=changed, panel=panel, labels=y), candidate, panel, y))
    name = "science-P123"; record = report["findings"][name]; saved = read_npz(a.publication/(name+".npz"), record["artifact_sha256"])
    source_cache = read_npz(a.diagnostic/(name+".npz"), source["findings"][name]["artifact_sha256"])
    sense = ROSTER[0]; profile = saved[sense+"__profile"]

    def changed_arrays(label, change):
        bad = {k: v.copy() for k, v in saved.items()}; change(bad)
        path = a.output/(label+".npz"); np.savez(path, **bad)
        pin = sha(path.read_bytes()); put(label+"-binding", dict(source_sha256=record["artifact_sha256"], changed_sha256=pin))
        return read_npz(path, pin)

    bad = changed_arrays("changed-scale", lambda b: b[sense+"__std"].__setitem__(0, b[sense+"__std"][0]*1.1))
    refuses("B1 rehashed changed scale refuses", "GEOMETRY_STD", lambda: audit_scale(bad, sense, profile))
    bad = changed_arrays("changed-distance", lambda b: b[sense+"__distance"].__setitem__((0, 1), 999))
    refuses("B1 rehashed changed normalized distance refuses", "GEOMETRY_DISTANCE",
            lambda: equal(audit_scale(bad, sense, profile), bad[sense+"__distance"], "GEOMETRY_DISTANCE", 0))
    unique, _, definitions = reference_geometry({s+"__distance": saved[s+"__distance"] for s in ROSTER})
    label = "authentic"; y = source_cache["labels"]; value = record["cases"][label]
    bad = changed_arrays("replacement-index", lambda b: b[label+"__indices"].__setitem__((0, 0, 0), b[label+"__indices"][0, 0, 1]))
    refuses("B2 sampling replacement hidden in rehashed indices refuses", "GEOMETRY_PLAN",
            lambda: audit_evaluation(label, value, bad, y, unique, definitions))
    key = next(iter(unique))
    bad = changed_arrays("altered-raw-sample", lambda b: b[label+"__samples__"+key].__setitem__((0, 0), b[label+"__samples__"+key][0, 0]+.1))
    refuses("B2 changed saved reference sample refuses", "GEOMETRY_RAW_SAMPLE",
            lambda: audit_evaluation(label, value, bad, y, unique, definitions))
    bad_value = copy.deepcopy(value); bad_value["statistics"]["increment:"+ROSTER[1]]["outer_low"] += .1
    put("inflated-envelope", bad_value)
    refuses("B2 inflated confidence envelope refuses", "GEOMETRY_ENVELOPE",
            lambda: audit_evaluation(label, bad_value, saved, y, unique, definitions))
    bad_dominance = copy.deepcopy(record["standardized_dominance"]); bad_dominance[ROSTER[0]]["unique_full_max_pairs"] += 1
    put("inflated-contribution", bad_dominance)
    refuses("B2 false neighborhood contribution refuses", "GEOMETRY_DOMINANCE",
            lambda: audit_dominance(bad_dominance, {s: saved[s+"__distance"] for s in ROSTER}))
    for flag in ("scientific_admission", "conditional_information_certified"):
        folder = a.output/flag; folder.mkdir(); bad_report = {**report, flag: True}
        (folder/"report.json").write_text(json.dumps(bad_report, sort_keys=True), encoding="utf-8")
        refuses("B3 false "+flag+" refuses", "GEOMETRY_CLAIM",
                lambda: readback(folder, a.diagnostic, a.slots, a.calibration, a.fsv))
    refuses("B3 immutable publication refuses overwrite", "GEOMETRY_EXISTS",
            lambda: run(a.slots, a.calibration, a.fsv, a.diagnostic, sha((a.diagnostic/"report.json").read_bytes()), a.publication))
    after = {str(p): sha(p.read_bytes()) for p in files}; put("after", after)
    check("B1/B3 all source and positive-publication files unchanged", before == after)
    result = dict(acceptance="author-checked", checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  cold_readback=cold, scientific_admission=False)
    put("fsv", result); print(json.dumps({k: result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
