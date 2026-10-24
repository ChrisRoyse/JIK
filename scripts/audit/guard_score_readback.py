"""Reconstruct all diagnostic cosines and thresholds from pinned native arrays.

The scoring producer is not imported. Quantiles are sorted/interpolated here;
dot products are recomputed one pair at a time from the persisted source arrays.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import sha, check
from audit.guard_render_readback import read as read_renders, lines
from audit.guard_measure_readback import read as read_measurements
from audit.renderer_comparison_readback import inspect
from senses.trajectory_population import publish

POLICY = {"arm": "true", "route": "true", "split": "calibration", "quantile": .1,
          "method": "linear", "minimum_complete": 50, "semantic_far_certified": False}
LIMITS = ["MNLM_PANEL_NOT_ADMITTED", "MNLM_ANCHOR_UNAVAILABLE", "MNLM_GUARD_IDENTITY_UNCERTIFIED"]


def close(actual, expected, code):
    check(actual is None and expected is None or actual is not None and expected is not None
          and math.isfinite(actual) and abs(actual - expected) < 2e-12, code)


def read(a):
    started = time.perf_counter()
    check(sha(a.root / "report.json") == a.report_sha256, "GUARD_SCORE_PIN")
    rep = json.loads((a.root / "report.json").read_bytes())
    check(rep["schema"] == "strict-guard-scores-v1" and rep["geometric_diagnostic_evaluated"] is True
          and all(rep[k] is False for k in ("independent_anchors_evaluated", "scientific_admission", "composition_success",
                "serving_ready", "final_test", "semantic_far_certified", "historical_identity_equivalence_certified")), "GUARD_SCORE_CLAIM")
    read_measurements(a.measurements, rep["measurements_sha256"], a.plan, a.renders)
    measured = json.loads((a.measurements / "report.json").read_bytes())
    read_renders(a.renders, rep["renders_sha256"], a.population, a.comparison)
    renders = json.loads((a.renders / "report.json").read_bytes())
    inspect(a.comparison, a.population, a.conditioning, rep["comparison_sha256"])
    check(all(rep[k] == renders[k] for k in ("comparison_sha256", "population_sha256", "conditioning_sha256"))
          and rep["plan_sha256"] == measured["plan_sha256"]
          and rep["runtime_fork_declared"] == measured["runtime_fork_declared"], "GUARD_SCORE_SOURCES")
    names = rep["lenses"]
    check(names == measured["lenses"] and rep["calibration_policy"] == renders["calibration_policy"] == POLICY, "GUARD_SCORE_POLICY")
    check(set(rep["files"]) == {"traces.jsonl", "controls.jsonl", "calibration.json"}, "GUARD_SCORE_FILES")
    for name, meta in rep["files"].items():
        check(sha(a.root / name) == meta["sha256"] and (a.root / name).stat().st_size == meta["bytes"], "GUARD_SCORE_FILE_PIN")
    targets, cases, pool = [lines(a.renders / n) for n in ("targets.jsonl", "candidates.jsonl", "texts.jsonl")]
    index = {r["id"]: i for i, r in enumerate(pool)}
    means = np.load(a.population / "centres.npz", allow_pickle=False)
    raw, valid, truth, pred = {}, {}, {}, {}
    for n in names:
        raw[n] = np.load(a.measurements / (n + ".npy"), allow_pickle=False).astype(np.float64) - means[n].astype(np.float64)
        valid[n] = np.load(a.measurements / (n + ".valid.npy"), allow_pickle=False)
        truth[n] = np.load(a.population / (n + ".npy"), mmap_mode="r", allow_pickle=False)[:, -1]
        pred[n] = np.load(a.conditioning / (n + ".npy"), mmap_mode="r", allow_pickle=False)

    def score(key, name, row, route):
        if key is None or not valid[name][index[key]]:
            return None
        x = raw[name][index[key]]
        y = np.asarray((truth if route == "true" else pred)[name][row], dtype=np.float64)
        nx, ny = math.sqrt(float(np.dot(x, x))), math.sqrt(float(np.dot(y, y)))
        if not nx or not ny or not math.isfinite(nx + ny):
            return None
        return max(-1., min(1., float(np.dot(x, y)) / nx / ny))

    rows, scores = [], {n: [] for n in names}
    for c in cases:
        if c["split"] != "calibration":
            continue
        values = {n: score(c["text_id"], n, c["row"], "true") for n in names}
        if all(v is not None for v in values.values()):
            rows.append(c["row"])
            for n in names:
                scores[n].append(values[n])
    cal = json.loads((a.root / "calibration.json").read_bytes())
    check(len(rows) >= 50 and cal["policy"] == POLICY and cal["rows"] == rows
          and set(cal["scores"]) == set(cal["thresholds"]) == set(names), "GUARD_SCORE_CALIBRATION")
    thresholds = {}
    for n in names:
        check(len(cal["scores"][n]) == len(scores[n]), "GUARD_SCORE_CALIBRATION")
        for actual, expected in zip(cal["scores"][n], scores[n]):
            close(actual, expected, "GUARD_SCORE_CALIBRATION_VALUE")
        ordered = sorted(scores[n]); location = (len(ordered) - 1) / 10
        lo, hi = math.floor(location), math.ceil(location)
        thresholds[n] = ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)
        close(cal["thresholds"][n], thresholds[n], "GUARD_SCORE_THRESHOLD")

    def audit_slots(item, key, row, route, surface="complete_unasserted"):
        check(set(item) == set(names), "GUARD_SCORE_MISSING_LANE")
        failed = []
        for n in names:
            s, v = item[n], score(key, n, row, route)
            close(s["cosine"], v, "GUARD_SCORE_VALUE")
            close(s["threshold"], thresholds[n], "GUARD_SCORE_THRESHOLD")
            # Use stored, independently checked values for exact boundary ties.
            passed = s["cosine"] is not None and s["cosine"] >= cal["thresholds"][n]
            check(s["passed"] is passed and s["status"] == ("measured" if v is not None else surface if key is None
                  else "MNLM_GUARD_SLOT_UNAVAILABLE"), "GUARD_SCORE_SLOT_VERDICT")
            if not passed:
                failed.append(n)
        return failed

    traces = lines(a.root / "traces.jsonl")
    check(len(traces) == len(cases), "GUARD_SCORE_ROWS")
    grouped = {}
    for t, c in zip(traces, cases):
        check(all(t[k] == c[k] for k in ("case", "arm", "split", "route", "row", "doc", "text_id"))
              and t["surface_status"] == c["status"], "GUARD_SCORE_CASE")
        failed = audit_slots(t["slots"], c["text_id"], c["row"], c["route"], c["status"])
        reasons = ([] if not failed else [c["status"] if c["text_id"] is None else "MNLM_GUARD_SLOT_BELOW_OR_UNAVAILABLE"]) + LIMITS
        check(t["failing_slots"] == failed and t["geometric_pass"] is (not failed)
              and t["asserted"] is False and t["verdict"] == "refuse" and t["reasons"] == reasons, "GUARD_SCORE_ASSERTION")
        grouped.setdefault("/".join((c["arm"], c["split"], c["route"])), []).append(t)
    check(set(rep["groups"]) == set(grouped), "GUARD_SCORE_GROUPS")
    for group, part in grouped.items():
        actual = rep["groups"][group]; k, count = sum(t["geometric_pass"] for t in part), len(part)
        check(actual["rows"] == count and actual["complete"] == sum(t["surface_status"] == "complete_unasserted" for t in part)
              and actual["geometric_pass"] == k and actual["geometric_rate"] == k / count and actual["asserted"] == 0
              and actual["slot_pass"] == {n: sum(t["slots"][n]["passed"] for t in part) for n in names}
              and actual["surface_statuses"] == dict(Counter(t["surface_status"] for t in part)), "GUARD_SCORE_COUNTS")
        z, p = 1.959963984540054, k / count
        d = 1 + z ** 2 / count
        centre = (p + z ** 2 / (2 * count)) / d
        half = z * math.sqrt(p * (1 - p) / count + z ** 2 / (4 * count ** 2)) / d
        for v, expected in zip(actual["descriptive_wilson_95"], (max(0, centre - half), min(1, centre + half))):
            close(v, expected, "GUARD_SCORE_INTERVAL")
    controls = lines(a.root / "controls.jsonl")
    evaluation = [t for t in targets if t["split"] == "evaluation"]
    expected_controls = [("source_replay", t, t) for t in targets] + [
        ("wrong_document", t, evaluation[(j + 1) % len(evaluation)]) for j, t in enumerate(evaluation)]
    check(len(controls) == len(expected_controls), "GUARD_SCORE_CONTROLS")
    for c, (kind, target, donor) in zip(controls, expected_controls):
        eligible = kind == "source_replay" or (target["doc"] != donor["doc"] and target["text_id"] != donor["text_id"])
        check(c["kind"] == kind and c["row"] == target["row"] and c["donor_row"] == donor["row"]
              and c["split"] == target["split"] and c["text_id"] == donor["text_id"] and c["eligible"] is eligible
              and c["semantic_negative_adjudicated"] is False and c["asserted"] is False, "GUARD_SCORE_CONTROL_SOURCE")
        failed = audit_slots(c["slots"], donor["text_id"], target["row"], "true")
        check(c["geometric_pass"] is (eligible and not failed), "GUARD_SCORE_CONTROL_VERDICT")
    counts = {kind: {"rows": sum(c["kind"] == kind for c in controls),
                     "eligible": sum(c["eligible"] for c in controls if c["kind"] == kind),
                     "geometric_pass": sum(c["geometric_pass"] for c in controls if c["kind"] == kind)}
              for kind in ("source_replay", "wrong_document")}
    check(counts == rep["controls"], "GUARD_SCORE_CONTROL_COUNTS")
    for n in names:
        values = [c["slots"][n]["cosine"] for c in controls if c["kind"] == "source_replay" and c["slots"][n]["cosine"] is not None]
        close(rep["source_replay"][n]["minimum_cosine"], min(values) if values else None, "GUARD_SCORE_SOURCE_REPLAY")
        close(rep["source_replay"][n]["mean_cosine"], math.fsum(values) / len(values) if values else None, "GUARD_SCORE_SOURCE_REPLAY")
    check(.9 <= rep["phase_coverage"] <= 1.001 and abs(sum(rep["phases_seconds"].values()) / rep["wall_seconds"]
          - rep["phase_coverage"]) < 1e-10, "GUARD_SCORE_TIMING")
    return {"author_checked": True, "report_sha256": a.report_sha256, "cases": len(traces), "controls": counts,
            "calibration_complete": len(rows), "all_slot_cosines_recomputed": True, "all_thresholds_independently_recomputed": True,
            "groups": rep["groups"], "source_replay": rep["source_replay"], "asserted": 0,
            "historical_identity_equivalence_certified": False, "scientific_admission": False,
            "wall_seconds": time.perf_counter() - started}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("root", "measurements", "plan", "renders", "population", "comparison", "conditioning", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--report-sha256", required=True)
    a = ap.parse_args()
    result = read(a)
    publish(a.output, result)
    print(json.dumps(result))
