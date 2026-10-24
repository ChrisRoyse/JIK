"""Diagnose committed private-region predictions; targets are audit-only inputs."""
import argparse
import base64
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.tool_loop_readback import readback
from audit.tool_region_support import optimistic_cosines, RADIUS
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.constellation import ordered_distance
from model.tool_temporal_input import ToolInputs


def history(inputs, record, bank, support):
    req, desc = record["request"], record["region_context"]["definition"]
    check(record["support"] == {str(t): ep for t, ep in support.items()}
          and set(support) == set(range(1, req["turn"]+1)), "PRIVATE_SUPPORT_PREFIX")
    private = [dict(reference=inputs.population.lookup[req["task"], t], episode=support[t])
               for t in range(3, req["turn"], 2)]
    shared = [bank[i] for i in record["routing"]["allowed"]]
    view = list(dict.fromkeys(shared+[r["reference"]["unit"] for r in private]))
    expected = dict(schema="tool-private-regions-v1", source=inputs.source_id,
                    context=req["context"], task=req["task"], cutoff=req["turn"],
                    kernel=record["routing"]["kernel"], shared_world=record["world"],
                    shared=shared, private=private, units=view)
    check(desc == expected, "PRIVATE_SUPPORT_HISTORY")
    return dict(view=view, shared=list(bank),
                historical=list(dict.fromkeys(bank+[r["reference"]["unit"] for r in private])))


def selected(inputs, prediction, units):
    region = prediction["region"]
    unit = int(region.split(":")[0])
    check(unit in units and region == f"{unit}:{inputs.refs[unit]['text_sha256']}",
          "PRIVATE_SUPPORT_SELECTION")
    return unit


def audit(inputs, run, expected):
    run_raw = (run/"run.json").read_bytes()
    check(sha(run_raw) == expected, "PRIVATE_SUPPORT_RUN_PIN")
    run_record = json.loads(run_raw)
    database = run/"experience.sqlite3"
    before = sha(database.read_bytes())
    snapshot = json.loads((run/"snapshot.json").read_bytes())
    journal = readback(database, inputs, snapshot)
    check(journal["head"] == run_record["head"], "PRIVATE_SUPPORT_HEAD")
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as db:
        payloads = {k: json.loads(v) for k, v in db.execute("SELECT hash,data FROM payloads")}
        events = [json.loads(v) for v, in db.execute("SELECT body FROM events ORDER BY seq")]
    bank, supports, forecasts, predictions, observed, pending = list(inputs.world_units), {}, {}, [], {}, {}
    first_record = None
    for event in events:
        payload = payloads.get(event.get("payload"), {})
        if event["kind"] == "grow":
            additions = []
            for key in event["keys"]:
                r = forecasts[key]["record"]
                req = r["request"]
                response = inputs.population.lookup[req["task"], req["turn"]+1]
                check(req["reference"]["partition"] == "train" and key in observed
                      and observed[key] == event["supports"][str(response["unit"])]["response"],
                      "PRIVATE_SUPPORT_PUBLICATION")
                additions.append(response["unit"])
            check(event["units"] == bank+additions, "PRIVATE_SUPPORT_BANK")
            bank = event["units"]
        if event["kind"] == "advance":
            req = event["request"]
            context = req["context"]
            supports.setdefault(context, {})[req["turn"]] = event["episode"]
            if "forecast" in payload:
                r = payload["forecast"]
                pools = history(inputs, r, bank, supports[context])
                key = digest(r)
                forecasts[key] = dict(record=r, pools=pools)
                pending[context] = key
                predictions.append(dict(key=key, kind="live", value=r, teacher=None))
                if first_record is None:
                    first_record = r, list(bank), dict(supports[context])
            if "assessment" in payload:
                key = pending.pop(context)
                observed[key] = event["episode"]
                forecasts[key]["live_assessment"] = payload["assessment"]
        if event["kind"] == "candidate":
            for key, label in payload["labels"].items():
                check(payload["baseline"][key]["predictions"] == forecasts[key]["record"]["predictions"]
                      and payload["baseline"][key]["region"] == forecasts[key]["record"]["region"],
                      "PRIVATE_SUPPORT_PARENT_REPLAY")
                for kind in ("baseline", "candidate"):
                    predictions.append(dict(key=key, kind=event["candidate"]+"/"+kind,
                        value=payload[kind][key], teacher=label, evaluate=key in event["evaluate"],
                        parent=payload["baseline"][key]["assessment"]))
    normalized, distances = {}, {}
    def windows(name, unit):
        key = name, unit
        if key not in normalized:
            x, ptr = inputs.arrays[name]
            value = x[ptr[unit]:ptr[unit+1]].astype(np.float64)
            normalized[key] = value/np.linalg.norm(value, axis=1, keepdims=True)
        return normalized[key]
    def cost(name, target, unit):
        key = name, target, unit
        if key not in distances:
            a, b = windows(name, target), windows(name, unit)
            c = np.clip(a@b.T, -1, 1)
            exact = ordered_distance(c)
            lower = ordered_distance(optimistic_cosines(c, RADIUS))
            length = abs(len(a)-len(b))/max(len(a), len(b))
            check(abs(exact-ordered_distance(optimistic_cosines(c, 0))) < 1e-12
                  and length <= lower+1e-12 <= exact+1e-12, "PRIVATE_SUPPORT_BOUND")
            distances[key] = dict(exact=exact, lower=lower, length_lower=length,
                                  target_windows=len(a), region_windows=len(b))
        return distances[key]
    results = {}
    for key, value in forecasts.items():
        r, pools = value["record"], value["pools"]
        req = r["request"]
        target = inputs.population.lookup[req["task"], req["turn"]+1]["unit"]
        costs, persistence = {}, {}
        for spec in inputs.specs:
            costs[spec.name] = {str(u): cost(spec.name, target, u) for u in pools["historical"]}
            persistence[spec.name] = cost(spec.name, target, r["baseline_unit"])["exact"]
        pool_summary = {}
        for name, units in pools.items():
            pool_summary[name] = {}
            for label, field in (("exact_better", "exact"), ("optimistic_better", "lower")):
                pool_summary[name][label] = [u for u in units if all(
                    costs[s.name][str(u)][field] < persistence[s.name] for s in inputs.specs)]
        summed = {u: sum(costs[s.name][str(u)]["exact"] for s in inputs.specs) for u in pools["view"]}
        best = min(pools["view"], key=lambda u: summed[u])
        results[key] = dict(context=req["context"], task=req["task"], cutoff=req["turn"],
            partition=req["reference"]["partition"], target=target, persistence_unit=r["baseline_unit"],
            pools=pools, costs=costs, persistence=persistence, pool_summary=pool_summary,
            summed_label=best, summed_label_exact_better=best in pool_summary["view"]["exact_better"],
            summed_label_optimistic_better=best in pool_summary["view"]["optimistic_better"])
    assessed = []
    for row in predictions:
        key, item, kind = row["key"], row["value"], row["kind"]
        result = results[key]
        unit = selected(inputs, item, result["pools"]["view"])
        lanes, ceilings = {}, {}
        for spec in inputs.specs:
            v = item["predictions"][spec.name]
            raw = base64.b64decode(v["base64"], validate=True)
            check(sha(raw) == v["sha256"] and v["instrument"] == spec.identity, "PRIVATE_SUPPORT_BYTES")
            p = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(v["shape"])
            b, t = windows(spec.name, unit), windows(spec.name, result["target"])
            check(p.shape == b.shape, "PRIVATE_SUPPORT_WINDOWS")
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            check(np.all(np.sum(p*b, axis=1) >= np.sqrt(1-RADIUS**2)-1e-6), "PRIVATE_SUPPORT_RADIUS")
            actual = ordered_distance(np.clip(t@p.T, -1, 1))
            c = result["costs"][spec.name][str(unit)]
            check(actual+1e-6 >= c["lower"], "PRIVATE_SUPPORT_ACTUAL_BOUND")
            saved = item.get("assessment", forecasts[key]["live_assessment"])["instruments"][spec.name]
            check(abs(actual-saved["prediction_loss"]) < 1e-12, "PRIVATE_SUPPORT_SCORE")
            parent = row.get("parent", forecasts[key]["live_assessment"])["instruments"][spec.name]["prediction_loss"]
            ceiling = min(parent, result["persistence"][spec.name])
            ceilings[spec.name] = ceiling
            lanes[spec.name] = dict(actual=actual, parent=parent, persistence=result["persistence"][spec.name],
                exact=c["exact"], lower=c["lower"], length_lower=c["length_lower"],
                better_parent=actual < parent, better_persistence=actual < result["persistence"][spec.name],
                selected_optimistic_better=c["lower"] < ceiling)
        possible = lambda units: [u for u in units if all(
            result["costs"][s.name][str(u)]["lower"] < ceilings[s.name] for s in inputs.specs)]
        teacher = None if row["teacher"] is None else result["pools"]["view"][row["teacher"]]
        check(teacher is None or teacher == result["summed_label"], "PRIVATE_SUPPORT_TEACHER")
        assessed.append(dict(key=key, kind=kind, selected=unit, teacher=teacher, lanes=lanes,
            evaluate=row.get("evaluate", False), selected_optimistic_better=all(
                v["selected_optimistic_better"] for v in lanes.values()),
            view_optimistic_better=possible(result["pools"]["view"]),
            historical_optimistic_better=possible(result["pools"]["historical"])))
    controls = []
    r, original_bank, support = first_record
    for label, mutate in (
        ("future", lambda x: x["region_context"]["definition"]["private"].append(
            dict(reference=inputs.population.lookup[x["request"]["task"], x["request"]["turn"]+1], episode="0"*64))),
        ("foreign", lambda x: x["region_context"]["definition"].update(context="foreign")),
        ("view_order", lambda x: x["region_context"]["definition"]["units"].reverse())):
        changed = copy.deepcopy(r)
        mutate(changed)
        try:
            history(inputs, changed, original_bank, support)
        except ValueError as error:
            check(str(error) == "MNLM_CODATA_PRIVATE_SUPPORT_HISTORY", "PRIVATE_SUPPORT_CONTROL")
            controls.append(label)
        else:
            check(False, "PRIVATE_SUPPORT_CONTROL_ACCEPTED")
    changed = copy.deepcopy(r)
    outside = next(u for u in range(len(inputs.refs)) if u not in r["region_context"]["definition"]["units"])
    changed["region"] = f"{outside}:{inputs.refs[outside]['text_sha256']}"
    try:
        selected(inputs, changed, r["region_context"]["definition"]["units"])
    except ValueError as error:
        check(str(error) == "MNLM_CODATA_PRIVATE_SUPPORT_SELECTION", "PRIVATE_SUPPORT_CONTROL")
        controls.append("off_view_prediction")
    else:
        check(False, "PRIVATE_SUPPORT_CONTROL_ACCEPTED")
    check(before == sha(database.read_bytes()), "PRIVATE_SUPPORT_MUTATION")
    return dict(run=expected, journal_sha256=before, journal=journal, forecasts=results,
                predictions=assessed, controls=controls, cached_costs=len(distances),
                normalized_messages=len(normalized))


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "run", "output", "population-pin", "native-pin", "run-pin"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(Path(args.raw)), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    loaded = time.perf_counter()
    result = audit(inputs, Path(args.run), args.run_pin)
    assessed = time.perf_counter()
    report = dict(schema="tool-private-support-v1", complete=True, acceptance="author-checked",
                  uses_targets=True, predictive_capability=False, scientific_admission=False,
                  radius=RADIUS, source=inputs.source_id, result=result, code=sha(Path(__file__).read_bytes()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = canonical(report)
    (output/"report.json").write_bytes(raw)
    (output/"COMMITTED").write_text(sha(raw))
    finished = time.perf_counter()
    timing = dict(wall=finished-started, load=loaded-started, audit=assessed-loaded, publication=finished-assessed)
    (output/"timing.json").write_bytes(canonical(timing))
    print(json.dumps(dict(complete=True, pin=sha(raw), forecasts=len(result["forecasts"]),
                         predictions=len(result["predictions"]), costs=result["cached_costs"], timing=timing)))


if __name__ == "__main__":
    main()
