"""Fail-closed screening of persisted chunked Calyx measurements (#12).

Ross MI and histogram NMI remain reference diagnostics. Neither standalone MI
nor MI * (1 - NMI) establishes conditional information. This module can shortlist
candidates; it cannot admit a lens, certify sufficiency, or assert meaning.
Replay costs O(report bytes + anchors * lenses**2), with no model/slot reload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

FLOOR, MAX_NMI = 0.05, 0.6
SCHEMA = "calyx-screen-v1"


class ScreeningError(ValueError):
    pass


def require(condition, detail):
    if not condition:
        raise ScreeningError(f"MNLM_CALYX_SCREEN_INPUT: {detail}")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label} must be a finite number")
    return value


def validate(report):
    require(isinstance(report, dict), "report must be an object")
    require(report.get("schema") in ("calyx-bits-port-v1", "calyx-bits-port-v2"),
            "unsupported measurement schema")
    anchors = report.get("anchors")
    require(isinstance(anchors, dict) and bool(anchors), "anchors are missing")
    measurements = report.get("per_lens")
    require(isinstance(measurements, dict), "per_lens is missing")
    refused = report.get("refused_lenses", {})
    require(isinstance(refused, dict), "refused_lenses must be an object")
    names = sorted(set(measurements) - set(refused))
    require(bool(names), "no measurable lenses")
    for q, anchor in anchors.items():
        require(isinstance(anchor, dict) and type(anchor.get("n")) is int
                and anchor["n"] >= 50, f"{q}: fewer than 50 samples")
        classes = anchor.get("classes")
        require(isinstance(classes, list) and len(classes) >= 2
                and all(isinstance(c, str) and c for c in classes)
                and len(set(classes)) == len(classes), f"{q}: invalid classes")
    for name in names:
        require(isinstance(measurements[name], dict)
                and set(measurements[name]) == set(anchors),
                f"{name}: incomplete anchor measurements")
        for q in anchors:
            row = measurements[name][q]
            require(isinstance(row, dict) and isinstance(row.get("chunk_multi"), dict),
                    f"{name}/{q}: missing chunk_multi")
            est = row["chunk_multi"]
            values = [number(est.get(key), f"{name}/{q}/{key}")
                      for key in ("ci_low", "bits", "ci_high")]
            require(0 <= values[0] <= values[1] <= values[2],
                    f"{name}/{q}: invalid interval")
            require(type(est.get("n")) is int and est["n"] == anchors[q]["n"],
                    f"{name}/{q}: sample count mismatch")
    pairs = report.get("pair_nmi")
    require(isinstance(pairs, dict), "pair_nmi missing")
    adjacency = {name: {} for name in names}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            keys = [key for key in (f"{x}|{y}", f"{y}|{x}") if key in pairs]
            require(len(keys) == 1, f"{x}/{y}: missing or ambiguous pair NMI")
            value = number(pairs[keys[0]], keys[0])
            require(0 <= value <= 1, f"{keys[0]}: NMI outside [0,1]")
            adjacency[x][y] = adjacency[y][x] = value
    require(len(pairs) == len(names) * (len(names) - 1) // 2,
            "unexpected pair NMI entries")
    controls = report.get("controls")
    require(isinstance(controls, dict)
            and {"random_lens", "labels_permuted"} <= set(controls),
            "random_lens and labels_permuted controls are required")
    for name, control in controls.items():
        require(isinstance(control, dict) and isinstance(control.get("per_anchor"), dict)
                and set(control["per_anchor"]) == set(anchors),
                f"{name}: every anchor needs a control")
        for q, value in control["per_anchor"].items():
            require(number(value, f"{name}/{q}") >= 0, "negative floored control bits")
    return names, adjacency


def screen(report):
    names, adjacency = validate(report)
    per_anchor, failures = {}, []
    for q in sorted(report["anchors"]):
        failed = [{"control": name, "anchor": q, "bits": row["per_anchor"][q]}
                  for name, row in sorted(report["controls"].items())
                  if row["per_anchor"][q] >= FLOOR]
        failures.extend(failed)
        lower = {n: report["per_lens"][n][q]["chunk_multi"]["ci_low"] for n in names}
        remaining, maximum, steps = set(names), dict.fromkeys(names, 0.0), []
        # A failing anchor produces no shortlist. Never average failures away.
        while remaining and not failed:
            eligible = [n for n in remaining if maximum[n] <= MAX_NMI
                        and lower[n] * (1 - maximum[n]) >= FLOOR]
            if not eligible:
                break
            pick = min(eligible, key=lambda n: (-lower[n] * (1 - maximum[n]), n))
            steps.append({"lens": pick, "standalone_ci_low": lower[pick],
                          "redundancy_adjusted_lower_bound": lower[pick] * (1 - maximum[pick]),
                          "max_nmi_to_kept": maximum[pick]})
            remaining.remove(pick)
            for n in remaining:
                maximum[n] = max(maximum[n], adjacency[n][pick])
        per_anchor[q] = {"controls_pass": not failed, "failed_controls": failed,
                         "standalone_ci_pass_count": sum(v >= FLOOR for v in lower.values()),
                         "candidate_roster": [s["lens"] for s in steps], "steps": steps}
    blockers = [
        {"code": "MNLM_CALYX_CONDITIONAL_UNMEASURED",
         "remedy": "Measure I(Y; candidate | panel); standalone MI times (1 - NMI) is a heuristic."},
        {"code": "MNLM_CALYX_RELIABILITY_UNMEASURED",
         "remedy": "Supply five-seed, entity-clustered reliability evidence for the conditional gain."},
        {"code": "MNLM_CALYX_D1_UNMEASURED",
         "remedy": "Supply contract D1 pair evidence; centroid-score histogram NMI is a separate diagnostic."},
        {"code": "MNLM_CALYX_PROVENANCE_UNBOUND",
         "remedy": "Bind frozen identities, determinism probes, population, fingerprint controls and per-atom coverage."},
    ]
    if failures:
        blockers.append({"code": "MNLM_CALYX_CONTROLS_FAILED",
                         "remedy": "Resolve the listed per-anchor control failures before interpreting those screens."})
    return {"schema": SCHEMA, "purpose": "candidate screening only; no scientific admission",
            "rules": {"lower_bound_floor": FLOOR, "max_reference_nmi": MAX_NMI,
                      "control_rule": "every control on every anchor must be below the floor",
                      "selection_score": "standalone ci_low * (1 - max reference NMI), not conditional MI"},
            "controls_pass": not failures, "failed_controls": failures, "per_anchor": per_anchor,
            "scientifically_admitted_lenses": [], "shannon_conditional_certified": False,
            "meaning_rule_met": False, "served": False, "admission_blockers": blockers}


def replay(path):
    data = path.read_bytes()
    source = json.loads(data)
    screening = screen(source)
    return {"schema": "calyx-screen-replay-v1",
            "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data), "schema": source.get("schema")},
            "source_flags": {k: source.get(k) for k in ("controls_pass", "meaning_rule_met")},
            "screening": screening,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from senses.trajectory_population import publish

    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for path in args.report:
        try:
            result = replay(path)
            out = args.output / (result["source"]["sha256"] + ".json")
            publish(out, result)
            print(json.dumps({"source": str(path), "output": str(out),
                              "controls_pass": result["screening"]["controls_pass"],
                              "meaning_rule_met": False}), flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            failed = True
            print(json.dumps({"source": str(path), "error": str(error)}), file=sys.stderr, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
