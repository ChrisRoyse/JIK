"""Target-aware capacity audit of persisted forecasts; this is not forecasting."""
import argparse
import base64
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from audit.tool_loop_readback import readback
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.constellation import ordered_distance
from model.tool_temporal_input import ToolInputs


RADIUS = .1  # The existing ConstellationWorld placement architecture, not a gate.


def historical(inputs, record, references=None):
    request = record["request"]
    expected = tuple(inputs.population.lookup[request["task"], turn]
                     for turn in range(3, request["turn"], 2))
    check(references is None or canonical(references) == canonical(expected),
          "TOOL_REGION_HISTORY")
    check(all(str(r["turn"]) in record["support"] for r in expected),
          "TOOL_REGION_UNOBSERVED")
    return sorted(set(inputs.world_units) | {r["unit"] for r in expected})


def optimistic_cosines(cosines, radius):
    """An independently rotatable base window can move by at most asin(radius)."""
    check(0 <= radius < 1, "TOOL_REGION_RADIUS")
    return np.cos(np.maximum(0., np.arccos(cosines)-np.arcsin(radius)))


def audit(inputs, run):
    database = run/"experience.sqlite3"
    before = sha(database.read_bytes())
    snapshot = json.loads((run/"snapshot.json").read_bytes())
    prior_readback = readback(database, inputs, snapshot)
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as db:
        payloads = {pin: json.loads(raw) for pin, raw in db.execute("SELECT hash,data FROM payloads")}
        events = [json.loads(raw) for raw, in db.execute("SELECT body FROM events ORDER BY seq")]
    forecasts, predictions, eval_keys = {}, [], set()
    for event in events:
        payload = payloads.get(event.get("payload"), {})
        if "forecast" in payload:
            record = payload["forecast"]
            key = digest(record)
            forecasts[key] = record
            predictions.append((key, record, "live"))
        if event["kind"] == "candidate":
            eval_keys.update(event["evaluate"])
            for kind in ("baseline", "candidate"):
                predictions.extend((key, value, event["candidate"]+"/"+kind)
                                   for key, value in payload[kind].items())
    normalized, costs, checks = {}, {}, dict(radius_zero=0, placement=0, bound=0)

    def windows(name, unit):
        key = name, unit
        if key not in normalized:
            values, ptr = inputs.arrays[name]
            x = values[ptr[unit]:ptr[unit+1]].astype(np.float64)
            normalized[key] = x/np.linalg.norm(x, axis=1, keepdims=True)
        return normalized[key]

    def cost(name, target, unit):
        key = name, target, unit
        if key not in costs:
            a, b = windows(name, target), windows(name, unit)
            c = np.clip(a@b.T, -1, 1)
            exact = ordered_distance(c)
            check(abs(exact-ordered_distance(optimistic_cosines(c, 0))) < 1e-12,
                  "TOOL_REGION_ZERO_RADIUS")
            checks["radius_zero"] += 1
            costs[key] = dict(exact=exact, lower=ordered_distance(optimistic_cosines(c, RADIUS)),
                              length_lower=abs(len(a)-len(b))/max(len(a), len(b)))
            check(costs[key]["length_lower"] <= costs[key]["lower"]+1e-12 <= exact+1e-12,
                  "TOOL_REGION_BOUND_ORDER")
        return costs[key]

    results = {}
    for key, record in forecasts.items():
        req = record["request"]
        target = inputs.population.lookup[req["task"], req["turn"]+1]["unit"]
        past = historical(inputs, record)
        by_lane, matrices = {}, {}
        for spec in inputs.specs:
            name = spec.name
            baseline = cost(name, target, record["baseline_unit"])["exact"]
            matrices[name] = {str(u): cost(name, target, u) for u in past}
            fixed = [matrices[name][str(u)] for u in inputs.world_units]
            by_lane[name] = dict(persistence=baseline,
                fixed_min_lower=min(v["lower"] for v in fixed),
                fixed_min_exact=min(v["exact"] for v in fixed),
                historical_min_lower=min(v["lower"] for v in matrices[name].values()))
        feasible = lambda units: [u for u in units if all(
            matrices[s.name][str(u)]["lower"] < by_lane[s.name]["persistence"]
            for s in inputs.specs)]
        exact_better = lambda units: [u for u in units if all(
            matrices[s.name][str(u)]["exact"] < by_lane[s.name]["persistence"]
            for s in inputs.specs)]
        results[key] = dict(context=req["context"], task=req["task"], cutoff=req["turn"],
            withheld=key in eval_keys, target=target, persistence_unit=record["baseline_unit"],
            history_units=past, fixed_units=list(inputs.world_units), instruments=by_lane,
            fixed_optimistically_feasible=feasible(inputs.world_units),
            historical_optimistically_feasible=feasible(past),
            fixed_exact_better=exact_better(inputs.world_units),
            historical_exact_better=exact_better(past), costs=matrices)
    for key, item, label in predictions:
        target = results[key]["target"]
        unit = int(item["region"].split(":")[0])
        check(item["region"] == f"{unit}:{inputs.refs[unit]['text_sha256']}"
              and unit in inputs.world_units, "TOOL_REGION_PREDICTION_SOURCE")
        for spec in inputs.specs:
            value = item["predictions"][spec.name]
            p = np.frombuffer(base64.b64decode(value["base64"], validate=True),
                             dtype="<f4").astype(np.float64).reshape(value["shape"])
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            b, t = windows(spec.name, unit), windows(spec.name, target)
            check(p.shape == b.shape and np.all(np.sum(p*b, axis=1) >= np.sqrt(1-RADIUS**2)-1e-6),
                  "TOOL_REGION_PLACEMENT_RADIUS")
            checks["placement"] += 1
            actual = ordered_distance(np.clip(t@p.T, -1, 1))
            check(actual+1e-6 >= cost(spec.name, target, unit)["lower"], "TOOL_REGION_PREDICTION_BOUND")
            checks["bound"] += 1
    controls = []
    record = next(iter(forecasts.values()))
    task, cutoff = record["request"]["task"], record["request"]["turn"]
    real_history = tuple(inputs.population.lookup[task, turn] for turn in range(3, cutoff, 2))
    other = next(r["request"]["task"] for r in forecasts.values() if r["request"]["task"] != task)
    for label, reference in (("future", inputs.population.lookup[task, cutoff+1]),
                             ("cross_context", inputs.population.lookup[other, 3])):
        try:
            historical(inputs, record, real_history+(reference,))
        except ValueError as error:
            check(str(error) == "MNLM_CODATA_TOOL_REGION_HISTORY", "TOOL_REGION_CONTROL")
            controls.append(label)
        else:
            check(False, "TOOL_REGION_CONTROL_ACCEPTED")
    check(sha(database.read_bytes()) == before, "TOOL_REGION_JOURNAL_MUTATION")
    return dict(journal_sha256=before, journal_readback=prior_readback,
        historical_reconstruction="at_original_commit_before_later_control_withdrawal",
        results=results, checks=checks, negative_controls=controls,
        cached_native_messages=len(normalized), cached_distances=len(costs),
        summary=dict(forecasts=len(results), withheld=len(eval_keys),
            fixed_impossible_all_slots=sum(not v["fixed_optimistically_feasible"] for v in results.values()),
            withheld_fixed_impossible_all_slots=sum(v["withheld"] and not v["fixed_optimistically_feasible"]
                                                   for v in results.values()),
            historical_impossible_all_slots=sum(not v["historical_optimistically_feasible"] for v in results.values())))


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "run", "output", "population-pin", "native-pin"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = ToolSource(Path(args.raw))
    inputs = ToolInputs(source, args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    loaded = time.perf_counter()
    result = audit(inputs, Path(args.run))
    assessed = time.perf_counter()
    report = dict(schema="tool-region-support-audit-v1", complete=True, acceptance="author-checked",
        predictive_capability=False, scientific_admission=False, uses_targets=True,
        radius=RADIUS, source=source.pin, native=inputs.source_id, result=result,
        code=sha(Path(__file__).read_bytes()),
        timings=dict(load_seconds=loaded-started, assessment_seconds=assessed-loaded))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = canonical(report)
    (output/"report.json").write_bytes(raw)
    (output/"COMMITTED").write_text(sha(raw))
    ended = time.perf_counter()
    timing = dict(wall_seconds=ended-started, load_seconds=loaded-started,
                  assessment_seconds=assessed-loaded, publication_seconds=ended-assessed)
    (output/"timing.json").write_bytes(canonical(timing))
    print(json.dumps(dict(complete=True, pin=sha(raw), summary=result["summary"],
                          checks=result["checks"], timing=timing)))


if __name__ == "__main__":
    main()
