"""Source-bound diagnosis of consecutive-learning region errors; no model run."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.tool_forecast_gradients import inspect as gradient, geometry
from audit.tool_growth_readback import checkpoint, manifest
from audit.tool_loop_readback import readback as journal_readback, distance
from audit.tool_native_recognition import scores
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.tool_temporal_input import ToolInputs


def read_json(path):
    return json.loads(path.read_bytes())


def pin_file(path, expected):
    raw = path.read_bytes()
    check(sha(raw) == expected, "FORECAST_DIAG_SOURCE_PIN")
    return json.loads(raw)


def diagnose(args):
    torch.set_num_threads(1)
    start = time.perf_counter()
    inputs = ToolInputs(ToolSource(args.raw), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    source_at = time.perf_counter()
    summary = pin_file(args.run/"summary.json", args.summary_pin)
    files = {name.replace("\\", "/"): value for name, value in summary["files"].items()}
    check(len(files) == len(summary["files"]), "FORECAST_DIAG_PATH_COLLISION")
    # Bind every artifact consumed here to the previous append-once evidence seal.
    consumed = ["training.json", "experience.sqlite3", "snapshot.json",
                "start-2/manifest.json", "mid-2/manifest.json",
                "candidates/increment-2/manifest.json"]
    for name in consumed:
        entry = files[name]
        raw = (args.run/name).read_bytes()
        check(sha(raw) == entry["sha256"] and len(raw) == entry["bytes"],
              "FORECAST_DIAG_INPUT_PIN")
    journal = journal_readback(args.run/"experience.sqlite3", inputs,
                              read_json(args.run/"snapshot.json"))
    check(journal["head"] == summary["head"], "FORECAST_DIAG_HEAD")
    training = read_json(args.run/"training.json")[1]
    decision = training["decision"]
    records, hidden, payloads = {}, {}, {}
    with sqlite3.connect(f"file:{(args.run/'experience.sqlite3').as_posix()}?mode=ro",
                         uri=True) as db:
        payloads = {p: json.loads(raw) for p, raw in db.execute("SELECT hash,data FROM payloads")}
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            event = json.loads(raw)
            p = payloads.get(event.get("payload"), {})
            if "forecast" in p:
                record = p["forecast"]
                key = digest(record)
                records[key], hidden[key] = record, event["hidden"]
    candidate = payloads[decision["payload"]]
    check(candidate["result"] == decision["result"], "FORECAST_DIAG_DECISION")
    paths = dict(start="start-2", midpoint="mid-2", final="candidates/increment-2")
    checkpoints = {}
    for stage, name in paths.items():
        pin = files[name+"/manifest.json"]["sha256"]
        m = manifest(args.run/name, pin)
        checkpoints[stage] = checkpoint(args.run/name/"model", m["model"])
    loaded_at = time.perf_counter()
    controls = []

    def refuse(name, code, action):
        try:
            action()
        except ValueError as error:
            check(code in str(error), "FORECAST_DIAG_WRONG_REFUSAL")
            controls.append(dict(name=name, code=code))
        else:
            raise ValueError("FORECAST_DIAG_ACCEPTED_CORRUPTION")

    refuse("summary_pin", "FORECAST_DIAG_SOURCE_PIN",
           lambda: pin_file(args.run/"summary.json", "0"*64))
    refuse("checkpoint_pin", "GROWTH_READ_PIN",
           lambda: checkpoint(args.run/"start-2/model", "0"*64))
    keys = decision["train"]+decision["evaluate"]
    check(len(keys) == 8 and len(set(keys)) == 8 and len(decision["train"]) == 4,
          "FORECAST_DIAG_ROSTER")
    damaged = copy.deepcopy(records[keys[0]])
    damaged["region_context"]["definition"]["history"].reverse()
    first = checkpoints["start"]
    refuse("reordered_prefix", "FORECAST_DIAG_HISTORY",
           lambda: gradient(inputs, damaged, first[2], first[3], hidden[keys[0]], 0))
    rows, excerpts, extents = [], [], []
    gradients = {"start": {}, "final": {}}
    gradient_errors = {}
    for key in keys:
        record = records[key]
        req = record["request"]
        units = record["region_context"]["definition"]["units"]
        target = inputs.population.lookup[req["task"], req["turn"]+1]
        label = candidate["labels"][key]
        check(req["reference"] == inputs.population.lookup[req["task"], req["turn"]],
              "FORECAST_DIAG_CURRENT_SOURCE")
        costs = np.zeros(len(units))
        sizes = {}
        for spec in inputs.specs:
            a, ptr = inputs.arrays[spec.name]
            get = lambda u: a[ptr[u]:ptr[u+1]].astype(np.float64)
            costs += np.array([distance(get(u), get(target["unit"])) for u in units])
            sizes[spec.name] = dict(history=sum(int(ptr[r["unit"]+1]-ptr[r["unit"]])
                for r in record["region_context"]["definition"]["history"]),
                current=len(get(req["reference"]["unit"])),
                region_max=max(len(get(u)) for u in units), target=len(get(target["unit"])))
        check(int(costs.argmin()) == label and target["unit"] not in units,
              "FORECAST_DIAG_TEACHER_CUTOFF")
        name = f'{req["context"]}/{req["turn"]}'
        row = dict(case=name, forecast=key, train=key in decision["train"],
                   teacher_unit=units[label], target=target, regions=len(units), stages={})
        for stage, values in checkpoints.items():
            lane_scores, selected = scores(inputs, record, values[2], values[3], hidden[key])
            total = sum(lane_scores.values())
            order = sorted(range(len(units)), key=lambda i: (-float(total[i]), i))
            other = max(float(total[i]) for i in range(len(units)) if i != label)
            logsum = float(np.log(np.exp(total-total.max()).sum())+total.max())
            ce = (logsum-float(total[label]))/len(inputs.specs)
            row["stages"][stage] = dict(selected=selected, teacher_rank=order.index(label)+1,
                margin=float(total[label])-other, classification_loss=ce,
                lane_scores={n: s.tolist() for n, s in lane_scores.items()})
            if row["train"] and stage in gradients:
                g = gradient(inputs, record, values[2], values[3], hidden[key], label)
                check(abs(g["loss"]-ce) < 1e-10, "FORECAST_DIAG_LOSS")
                gradients[stage][name] = g["gradients"]
                gradient_errors[stage+"/"+name] = g["errors"]
        check(row["stages"]["final"]["selected"] ==
              int(candidate["candidate"][key]["region"].split(":")[0]),
              "FORECAST_DIAG_FINAL_SELECTION")
        rows.append(row)
        extents.append(dict(case=name, sizes=sizes))
        texts = inputs.population.units
        excerpts.append(dict(case=name, source=req["reference"],
            history=[dict(reference=inputs.population.lookup[req["task"], t],
                          text=inputs.population.source.texts[req["task"], t])
                     for t in range(1, req["turn"]+1)],
            outcome=dict(reference=target, text=texts[target["unit"]]["text"]),
            teacher=dict(unit=units[label], text=texts[units[label]]["text"]),
            selected=dict(unit=row["stages"]["final"]["selected"],
                          text=texts[row["stages"]["final"]["selected"]]["text"])))
    scored_at = time.perf_counter()
    train_names = [r["case"] for r in rows if r["train"]]
    interference = {stage: geometry(g, train_names, checkpoints[stage][3],
                                    checkpoints["final"][3])
                    for stage, g in gradients.items()}
    # Prove the analysis has not changed any consumed evidence or checkpoint.
    for name in consumed:
        check(sha((args.run/name).read_bytes()) == files[name]["sha256"],
              "FORECAST_DIAG_INPUT_CHANGED")
    for stage, name in paths.items():
        current = checkpoint(args.run/name/"model", read_json(args.run/name/"manifest.json")["model"])
        check(all(np.array_equal(v, current[3][k]) for k, v in checkpoints[stage][3].items()),
              "FORECAST_DIAG_WEIGHTS_CHANGED")
    finished = time.perf_counter()
    phases = dict(source=source_at-start, artifacts=loaded_at-source_at,
                  diagnosis=scored_at-loaded_at, readback=finished-scored_at)
    result = dict(schema="tool-forecast-diagnosis-v1", acceptance="author-checked",
        summary_pin=args.summary_pin, source_pin=inputs.population.source.pin,
        native_pin=args.native_pin, journal=journal, cases=rows, extents=extents,
        gradient_errors=gradient_errors, interference=interference, controls=controls,
        phases_seconds=phases, wall_seconds=finished-start,
        timing_coverage=sum(phases.values())/(finished-start),
        code={p.name: sha(p.read_bytes()) for p in (
            Path(__file__), Path(__file__).with_name("tool_forecast_gradients.py"))},
        scientific_admission=False, useful_capability=False, optimizer_updates=0,
        interpretation="exposed_source_report_diagnostic_not_sealed_generalization")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("diagnosis.json", result), ("source-excerpts.json", excerpts)):
        with (args.output/name).open("xb") as f:
            f.write(canonical(data))
    print(json.dumps(dict(complete=True, cases=len(rows), controls=len(controls),
                          wall_seconds=result["wall_seconds"],
                          pin=sha((args.output/"diagnosis.json").read_bytes()))))


def main():
    p = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "run", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("population-pin", "native-pin", "summary-pin"):
        p.add_argument("--"+name, required=True)
    diagnose(p.parse_args())


if __name__ == "__main__":
    main()
