"""Fixed-calibration per-slot geometry; absent anchors/admission always refuse.

No lens forward, threshold search, pooled score or semantic correctness inference.
All development cases are retained, including incomplete surface refusals.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import check, sha
from audit.guard_render_readback import read as read_renders, lines
from audit.guard_measure_readback import read as read_measurements
from audit.renderer_comparison_readback import inspect
from senses.contract import canonical
from senses.trajectory_population import publish

POLICY = {"arm": "true", "route": "true", "split": "calibration", "quantile": .1,
          "method": "linear", "minimum_complete": 50, "semantic_far_certified": False}
LIMITS = ["MNLM_PANEL_NOT_ADMITTED", "MNLM_ANCHOR_UNAVAILABLE", "MNLM_GUARD_IDENTITY_UNCERTIFIED"]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    valid = np.isfinite(x).all(-1) & (norms[..., 0] > 0)
    return np.divide(x, norms, out=np.zeros_like(x), where=valid[..., None]), valid


def inputs(a):
    read_measurements(a.measurements, a.measurements_sha256, a.plan, a.renders)
    m = json.loads((a.measurements / "report.json").read_bytes())
    read_renders(a.renders, m["renders_sha256"], a.population, a.comparison)
    r = json.loads((a.renders / "report.json").read_bytes())
    inspect(a.comparison, a.population, a.conditioning, r["comparison_sha256"])
    check(r["calibration_policy"] == POLICY, "GUARD_SCORE_POLICY")
    pool, cases, targets = [lines(a.renders / n) for n in ("texts.jsonl", "candidates.jsonl", "targets.jsonl")]
    index = {x["id"]: i for i, x in enumerate(pool)}
    centres = np.load(a.population / "centres.npz", allow_pickle=False)
    measured, available, truth, predicted = {}, {}, {}, {}
    for name in m["lenses"]:
        measured[name], good = unit(np.load(a.measurements / (name + ".npy"), allow_pickle=False) - centres[name].astype(np.float64))
        available[name] = np.load(a.measurements / (name + ".valid.npy"), allow_pickle=False) & good
        truth[name], ok = unit(np.load(a.population / (name + ".npy"), mmap_mode="r", allow_pickle=False)[:, -1])
        predicted[name], pk = unit(np.load(a.conditioning / (name + ".npy"), allow_pickle=False))
        check(ok.all() and pk.all(), "GUARD_SCORE_CONDITIONING")
    return m, r, cases, targets, index, measured, available, truth, predicted


def cosine(measured, available, index, text_id, name, target):
    if text_id is None:
        return None
    i = index[text_id]
    return float(np.clip(np.dot(measured[name][i], target), -1, 1)) if available[name][i] else None


def intervals(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [float(max(0, mid - half)), float(min(1, mid + half))]


def summarize(traces, names):
    groups = {}
    for trace in traces:
        group = "/".join((trace["arm"], trace["split"], trace["route"]))
        groups.setdefault(group, []).append(trace)
    result = {}
    for group, rows in groups.items():
        passed = sum(t["geometric_pass"] for t in rows)
        result[group] = {"rows": len(rows), "complete": sum(t["surface_status"] == "complete_unasserted" for t in rows),
                         "geometric_pass": passed, "geometric_rate": passed / len(rows),
                         "descriptive_wilson_95": intervals(passed, len(rows)), "asserted": 0,
                         "slot_pass": {name: sum(t["slots"][name]["passed"] for t in rows) for name in names},
                         "surface_statuses": dict(Counter(t["surface_status"] for t in rows))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("measurements", "plan", "renders", "population", "comparison", "conditioning", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--measurements-sha256", required=True)
    a = ap.parse_args()
    check(not a.output.exists(), "GUARD_SCORE_OUTPUT_EXISTS")
    start = time.perf_counter(); phases = {}
    m, r, cases, targets, index, measured, available, truth, predicted = inputs(a)
    names = m["lenses"]
    phases["input_cold_read_and_normalization"] = time.perf_counter() - start
    tick = time.perf_counter()
    calibration = [c for c in cases if c["split"] == "calibration"]
    rows, scores = [], {n: [] for n in names}
    for c in calibration:
        values = {n: cosine(measured, available, index, c["text_id"], n, truth[n][c["row"]]) for n in names}
        if all(v is not None for v in values.values()):
            rows.append(c["row"])
            for n in names:
                scores[n].append(values[n])
    check(len(rows) >= POLICY["minimum_complete"], "GUARD_SCORE_CALIBRATION_INSUFFICIENT")
    thresholds = {n: float(np.quantile(scores[n], .1, method="linear")) for n in names}
    phases["fixed_calibration"] = time.perf_counter() - tick
    tick = time.perf_counter()

    def slots_for(text_id, row, route, surface_status="complete_unasserted"):
        wanted = truth if route == "true" else predicted
        result = {}
        for name in names:
            value = cosine(measured, available, index, text_id, name, wanted[name][row])
            result[name] = {"cosine": value, "threshold": thresholds[name],
                            "passed": value is not None and value >= thresholds[name],
                            "status": "measured" if value is not None else surface_status if text_id is None else "MNLM_GUARD_SLOT_UNAVAILABLE"}
        return result

    traces = []
    for c in cases:
        slots = slots_for(c["text_id"], c["row"], c["route"], c["status"])
        failing = [name for name in names if not slots[name]["passed"]]
        reasons = ([] if not failing else [c["status"] if c["text_id"] is None else "MNLM_GUARD_SLOT_BELOW_OR_UNAVAILABLE"]) + LIMITS
        traces.append({**{k: c[k] for k in ("case", "arm", "split", "route", "row", "doc", "text_id")},
                       "surface_status": c["status"], "slots": slots, "failing_slots": failing,
                       "geometric_pass": not failing, "asserted": False, "verdict": "refuse", "reasons": reasons})
    controls = []
    evaluation = [t for t in targets if t["split"] == "evaluation"]
    for kind, selected in (("source_replay", targets), ("wrong_document", evaluation)):
        for j, target in enumerate(selected):
            donor = target if kind == "source_replay" else selected[(j + 1) % len(selected)]
            eligible = kind == "source_replay" or (target["doc"] != donor["doc"] and target["text_id"] != donor["text_id"])
            slots = slots_for(donor["text_id"], target["row"], "true")
            controls.append({"kind": kind, "row": target["row"], "donor_row": donor["row"], "text_id": donor["text_id"],
                             "split": target["split"], "eligible": eligible, "slots": slots,
                             "geometric_pass": eligible and all(s["passed"] for s in slots.values()),
                             "semantic_negative_adjudicated": False, "asserted": False})
    groups = summarize(traces, names)
    control_counts = {kind: {"rows": sum(c["kind"] == kind for c in controls),
                            "eligible": sum(c["eligible"] for c in controls if c["kind"] == kind),
                            "geometric_pass": sum(c["geometric_pass"] for c in controls if c["kind"] == kind)}
                      for kind in ("source_replay", "wrong_document")}
    replay = {}
    for name in names:
        values = [c["slots"][name]["cosine"] for c in controls if c["kind"] == "source_replay"
                  and c["slots"][name]["cosine"] is not None]
        replay[name] = {"minimum_cosine": min(values) if values else None,
                        "mean_cosine": float(np.mean(values)) if values else None}
    phases["slot_scoring_and_controls"] = time.perf_counter() - tick
    tick = time.perf_counter()
    a.output.mkdir(parents=True)
    for name, values in (("traces.jsonl", traces), ("controls.jsonl", controls)):
        (a.output / name).write_bytes(b"".join(canonical(v) + b"\n" for v in values))
    calibration_record = {"policy": POLICY, "rows": rows, "scores": scores, "thresholds": thresholds}
    publish(a.output / "calibration.json", calibration_record)
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in a.output.iterdir()}
    phases["trace_retention"] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    report = {"schema": "strict-guard-scores-v1", "measurements_sha256": a.measurements_sha256,
              "plan_sha256": m["plan_sha256"], "renders_sha256": m["renders_sha256"],
              **{k: r[k] for k in ("comparison_sha256", "population_sha256", "conditioning_sha256")},
              "lenses": names, "calibration_policy": POLICY, "groups": groups, "controls": control_counts,
              "source_replay": replay, "files": files, "source_code_sha256": sha(Path(__file__)),
              "geometric_diagnostic_evaluated": True, "independent_anchors_evaluated": False,
              "scientific_admission": False, "composition_success": False, "serving_ready": False, "final_test": False,
              "semantic_far_certified": False, "historical_identity_equivalence_certified": False,
              "runtime_fork_declared": m["runtime_fork_declared"], "phases_seconds": phases,
              "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall}
    publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha(a.output / "report.json"), "groups": groups, "controls": control_counts,
                      "wall_seconds": wall, "phase_coverage": report["phase_coverage"]}))


if __name__ == "__main__":
    main()
