"""Independent SQL/source/table-DP and spherical-cap readback of private support."""
import argparse
import base64
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.tool_loop_readback import readback as read_journal
from audit.tool_region_support_readback import table_distance
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.tool_temporal_input import ToolInputs


def publication(root, pin):
    raw = (root/"report.json").read_bytes()
    check(sha(raw) == pin == (root/"COMMITTED").read_text(), "PRIVATE_READ_PIN")
    report = json.loads(raw)
    check(report["uses_targets"] and not report["predictive_capability"]
          and not report["scientific_admission"] and report["radius"] == .1, "PRIVATE_READ_SCOPE")
    return report


def readback(inputs, root, run, pin):
    report = publication(root, pin)
    result = report["result"]
    check(sha((run/"run.json").read_bytes()) == result["run"] and
          sha((run/"experience.sqlite3").read_bytes()) == result["journal_sha256"]
          and report["source"] == inputs.source_id, "PRIVATE_READ_SOURCE")
    journal = read_journal(run/"experience.sqlite3", inputs, json.loads((run/"snapshot.json").read_bytes()))
    check(journal == result["journal"], "PRIVATE_READ_JOURNAL")
    with sqlite3.connect(f"file:{(run/'experience.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        payloads = {h: json.loads(b) for h, b in db.execute("SELECT hash,data FROM payloads")}
        events = [json.loads(b) for b, in db.execute("SELECT body FROM events ORDER BY seq")]
    bank = list(inputs.world_units)
    records, predicted, labels = {}, {}, {}
    for event in events:
        if event["kind"] == "grow":
            bank = event["units"]
        p = payloads.get(event.get("payload"), {})
        if "forecast" in p:
            record = p["forecast"]
            key = digest(record)
            request = record["request"]
            own = [inputs.population.lookup[request["task"], t]["unit"]
                   for t in range(3, request["turn"], 2)]
            shared = [bank[i] for i in record["routing"]["allowed"]]
            expected = dict(shared=bank, historical=list(dict.fromkeys(bank+own)),
                            view=list(dict.fromkeys(shared+own)))
            saved = result["forecasts"][key]
            check(expected == saved["pools"] and
                  record["region_context"]["definition"]["units"] == expected["view"],
                  "PRIVATE_READ_POOLS")
            check(saved["target"] == inputs.population.lookup[request["task"], request["turn"]+1]["unit"]
                  and saved["persistence_unit"] == record["baseline_unit"], "PRIVATE_READ_TARGET")
            records[key] = record
            predicted[key, "live"] = record
        if event["kind"] == "candidate":
            for label in ("baseline", "candidate"):
                for key, value in p[label].items():
                    predicted[key, event["candidate"]+"/"+label] = value
                    labels[key, event["candidate"]+"/"+label] = \
                        records[key]["region_context"]["definition"]["units"][p["labels"][key]]
    check(set(result["forecasts"]) == set(records) and
          {(r["key"], r["kind"]) for r in result["predictions"]} == set(predicted), "PRIVATE_READ_ROSTER")
    normalized, comparisons, max_error, ties = {}, 0, 0., []
    def windows(lane, unit):
        address = lane, unit
        if address not in normalized:
            x, offsets = inputs.arrays[lane]
            a = x[offsets[unit]:offsets[unit+1]].astype(np.float64)
            normalized[address] = a/np.sqrt(np.einsum("ij,ij->i", a, a))[:, None]
        return normalized[address]
    def check_set(calculated, saved, margins, scope):
        for unit in set(calculated) ^ set(saved):
            maximum = max(margins(unit))
            check(abs(maximum) < 1e-12, "PRIVATE_READ_STRICT_SET")
            ties.append(dict(scope=scope, unit=unit, maximum=maximum))
    recomputed = {}
    for key, saved in result["forecasts"].items():
        costs = {}
        for spec in inputs.specs:
            name = spec.name
            a = windows(name, saved["target"])
            costs[name] = {}
            for unit in saved["pools"]["historical"]:
                b = windows(name, unit)
                cosines = np.clip(np.einsum("ik,jk->ij", a, b), -1, 1)
                boundary = np.sqrt(1-.1**2)
                best = np.where(cosines >= boundary, 1., cosines*boundary+np.sqrt(1-cosines*cosines)*.1)
                values = dict(exact=table_distance(1-cosines), lower=table_distance(1-best),
                    length_lower=abs(len(a)-len(b))/max(len(a), len(b)),
                    target_windows=len(a), region_windows=len(b))
                original = saved["costs"][name][str(unit)]
                delta = max(abs(values[k]-original[k]) for k in values)
                check(delta < 1e-12, "PRIVATE_READ_COST")
                comparisons += 1
                max_error = max(max_error, delta)
                costs[name][unit] = values
            persistence = costs[name][saved["persistence_unit"]]["exact"]
            check(abs(persistence-saved["persistence"][name]) < 1e-12, "PRIVATE_READ_PERSISTENCE")
        for pool, units in saved["pools"].items():
            for label, field in (("exact_better", "exact"), ("optimistic_better", "lower")):
                margins = lambda u: [costs[s.name][u][field]-saved["persistence"][s.name] for s in inputs.specs]
                calculated = [u for u in units if max(margins(u)) < 0]
                check_set(calculated, saved["pool_summary"][pool][label], margins, key+"/"+pool+"/"+label)
        sums = {u: sum(costs[s.name][u]["exact"] for s in inputs.specs) for u in saved["pools"]["view"]}
        check(abs(sums[saved["summed_label"]]-min(sums.values())) < 1e-12, "PRIVATE_READ_LABEL")
        recomputed[key] = costs
    actual_values = {}
    def prediction_loss(key, kind, name):
        address = key, kind, name
        if address not in actual_values:
            raw = predicted[key, kind]["predictions"][name]
            p = np.frombuffer(base64.b64decode(raw["base64"]), dtype="<f4").astype(np.float64).reshape(raw["shape"])
            p /= np.sqrt(np.einsum("ij,ij->i", p, p))[:, None]
            target = result["forecasts"][key]["target"]
            actual_values[address] = table_distance(1-np.clip(
                np.einsum("ik,jk->ij", windows(name, target), p), -1, 1))
        return actual_values[address]
    failures = []
    for saved in result["predictions"]:
        key, kind = saved["key"], saved["kind"]
        value = predicted[key, kind]
        unit = int(value["region"].split(":")[0])
        check(unit == saved["selected"] and unit in result["forecasts"][key]["pools"]["view"],
              "PRIVATE_READ_SELECTION")
        check(saved["teacher"] == labels.get((key, kind)), "PRIVATE_READ_TEACHER_SOURCE")
        for spec in inputs.specs:
            name, lane = spec.name, saved["lanes"][spec.name]
            actual = prediction_loss(key, kind, name)
            parent_kind = "live" if kind == "live" else kind.split("/")[0]+"/baseline"
            parent = prediction_loss(key, parent_kind, name)
            persistence = recomputed[key][name][result["forecasts"][key]["persistence_unit"]]["exact"]
            check(max(abs(actual-lane["actual"]), abs(parent-lane["parent"]),
                      abs(persistence-lane["persistence"])) < 1e-12, "PRIVATE_READ_PREDICTION")
            check(lane["better_parent"] == (lane["actual"] < lane["parent"]) and
                  lane["better_persistence"] == (lane["actual"] < lane["persistence"]) and
                  lane["selected_optimistic_better"] == (lane["lower"] < min(lane["parent"], lane["persistence"])),
                  "PRIVATE_READ_PREDICATE")
            check(all(abs(lane[n]-recomputed[key][name][unit][n]) < 1e-12
                      for n in ("exact", "lower", "length_lower")), "PRIVATE_READ_SELECTED_BOUND")
        check(saved["selected_optimistic_better"] == all(
            v["selected_optimistic_better"] for v in saved["lanes"].values()), "PRIVATE_READ_SELECTED_PREDICATE")
        for pool in ("view", "historical"):
            margins = lambda u: [recomputed[key][s.name][u]["lower"]-min(
                saved["lanes"][s.name]["parent"], saved["lanes"][s.name]["persistence"]) for s in inputs.specs]
            calculated = [u for u in result["forecasts"][key]["pools"][pool] if max(margins(u)) < 0]
            check_set(calculated, saved[pool+"_optimistic_better"], margins, key+"/"+kind+"/"+pool)
        if kind.endswith("/candidate") and saved["evaluate"]:
            check(saved["teacher"] == result["forecasts"][key]["summed_label"], "PRIVATE_READ_TEACHER")
            failures.append(dict(context=result["forecasts"][key]["context"], cutoff=result["forecasts"][key]["cutoff"],
                kind=kind, selected=unit, teacher=saved["teacher"],
                selected_blocked=not saved["selected_optimistic_better"],
                view_blocked=not saved["view_optimistic_better"],
                historical_blocked=not saved["historical_optimistic_better"],
                failed_instruments=[n for n, v in saved["lanes"].items() if not
                                    (v["better_parent"] and v["better_persistence"])]))
    return dict(complete=True, acceptance="author-checked", comparisons=comparisons, max_error=max_error,
                independently_scored_slots=len(predicted)*len(inputs.specs), near_zero_comparisons=ties,
                candidate_failures=failures, scientific_admission=False, predictive_capability=False,
                report=pin, source_journal=result["journal_sha256"])


def main():
    parser = argparse.ArgumentParser()
    for n in ("raw", "population", "native", "run", "output", "population-pin", "native-pin", "report-pin"):
        parser.add_argument("--"+n, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(Path(args.raw)), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    loaded = time.perf_counter()
    root = Path(args.output)
    result = readback(inputs, root, Path(args.run), args.report_pin)
    broken = root/"corrupt-publication-control"
    broken.mkdir(exist_ok=False)
    (broken/"report.json").write_bytes((root/"report.json").read_bytes()+b"\n")
    (broken/"COMMITTED").write_text(args.report_pin)
    try:
        publication(broken, args.report_pin)
    except ValueError as error:
        check(str(error) == "MNLM_CODATA_PRIVATE_READ_PIN", "PRIVATE_READ_CONTROL")
        result["corrupt_publication_refused"] = True
    else:
        check(False, "PRIVATE_READ_CONTROL_ACCEPTED")
    read = time.perf_counter()
    result["timing"] = dict(load=loaded-started, readback=read-loaded, wall=read-started)
    with (root/"readback.json").open("xb") as f:
        f.write(canonical(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
