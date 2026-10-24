"""Source/measurement/calibration/verdict mutations on real persisted guard data."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.guard_score_readback import read, sha, lines
from audit.guard_measure_readback import read as measure_read
from senses.contract import canonical, digest


def main():
    ap = argparse.ArgumentParser()
    for name in ("root", "measurements", "plan", "renders", "population", "comparison", "conditioning", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    before = sha(a.root / "report.json"); a.report_sha256 = before
    cold = read(a)
    (a.out / "cold-readback.json").write_bytes(canonical(cold))
    original = json.loads((a.root / "report.json").read_bytes())
    measure_pin = sha(a.measurements / "report.json")
    checks = [{"name": "B1 all source bytes, identities, vectors, cosines and quantiles cold read", "passed": True}]

    def fault(name, mutate, code, file=None, measurement=False):
        shadow = a.out / name
        source = a.measurements if measurement else a.root
        shutil.copytree(source, shadow)
        rep = json.loads((shadow / "report.json").read_bytes())
        if file:
            path = shadow / file
            if file.endswith(".jsonl"):
                value = lines(path); mutate(value)
                path.write_bytes(b"".join(canonical(v) + b"\n" for v in value))
            elif file.endswith(".npy"):
                value = np.load(path, allow_pickle=False); mutate(value)
                np.save(path, value, allow_pickle=False)
            else:
                value = json.loads(path.read_bytes()); mutate(value)
                path.write_bytes(canonical(value))
            rep["files"][file] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        else:
            mutate(rep)
        (shadow / "report.json").write_bytes(canonical(rep))
        try:
            if measurement:
                measure_read(shadow, sha(shadow / "report.json"), a.plan, a.renders)
            else:
                args = copy.copy(a); args.root = shadow; args.report_sha256 = sha(shadow / "report.json")
                read(args)
            observed = "accepted"
        except ValueError as exc:
            observed = str(exc)
        checks.append({"name": name, "before_report_sha256": sha(source / "report.json"),
                       "mutant_report_sha256": sha(shadow / "report.json"), "expected": "MNLM_RP_READ_" + code,
                       "observed": observed, "passed": observed == "MNLM_RP_READ_" + code})

    first = original["lenses"][0]
    fault("B1_wrong_measurement_pin", lambda r: r.update(measurements_sha256="0" * 64), "GUARD_MEASURE_PIN")

    def changed_identity(r):
        r["identities"][first]["weights_sha256"] = "0" * 64
        r["readings"][first]["identity_sha256"] = digest(r["identities"][first])

    fault("B1_changed_frozen_weight", changed_identity, "GUARD_MEASURE_IDENTITY", measurement=True)
    fault("B1_invalid_vector", lambda r: r.__setitem__((0, 0), np.nan), "GUARD_MEASURE_VALUES", first + ".npy", True)
    fault("B1_zero_vector", lambda r: r.__setitem__(0, 0), "GUARD_MEASURE_VALUES", first + ".npy", True)
    fault("B1_missing_native_lane", lambda r: r["readings"].pop(first), "GUARD_MEASURE_LANES", measurement=True)
    fault("B2_evaluation_as_calibration", lambda r: r["policy"].update(split="evaluation"), "GUARD_SCORE_CALIBRATION", "calibration.json")
    fault("B2_changed_threshold", lambda r: r["thresholds"].__setitem__(first, r["thresholds"][first] - .1), "GUARD_SCORE_THRESHOLD", "calibration.json")
    fault("B2_changed_calibration_row", lambda r: r["rows"].__setitem__(0, -1), "GUARD_SCORE_CALIBRATION", "calibration.json")
    fault("B3_changed_cosine", lambda r: r[0]["slots"][first].update(cosine=1.), "GUARD_SCORE_VALUE", "traces.jsonl")
    fault("B3_missing_scored_lane", lambda r: r[0]["slots"].pop(first), "GUARD_SCORE_MISSING_LANE", "traces.jsonl")
    fault("B3_false_assertion", lambda r: r[0].update(asserted=True), "GUARD_SCORE_ASSERTION", "traces.jsonl")

    def fake_completion(rows):
        case = next(t for t in rows if t["surface_status"] == "MNLM_SURFACE_NO_EOS")
        case["surface_status"] = "complete_unasserted"

    fault("B3_NO_EOS_not_a_success", fake_completion, "GUARD_SCORE_CASE", "traces.jsonl")
    fault("B3_invented_grounding", lambda r: r.update(independent_anchors_evaluated=True), "GUARD_SCORE_CLAIM")
    fault("B3_false_semantic_negative", lambda r: r[-1].update(semantic_negative_adjudicated=True), "GUARD_SCORE_CONTROL_SOURCE", "controls.jsonl")
    after = read(a)
    checks.append({"name": "B3 original reports unchanged and whole replay survives", "passed":
                   sha(a.root / "report.json") == before and sha(a.measurements / "report.json") == measure_pin
                   and {k: v for k, v in cold.items() if k != "wall_seconds"} == {k: v for k, v in after.items() if k != "wall_seconds"}})
    report = {"author_checked": True, "passed": all(c["passed"] for c in checks), "checks": checks,
              "before_report_sha256": before, "after_report_sha256": sha(a.root / "report.json"),
              "scientific_admission": False, "independent_anchors_evaluated": False}
    (a.out / "fsv.json").write_bytes(canonical(report))
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
