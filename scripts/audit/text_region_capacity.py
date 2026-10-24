"""Target-aware native capacity diagnosis, never a learned prediction or admission."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np

from audit.tool_region_support import optimistic_cosines, RADIUS
from model.constellation import ordered_distance, unpack
from model.contracts import ModelConfig
from model.text_sequence_data import SequenceData
from senses.contract import digest
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def history(data, left, supplied=None):
    group = data.by_index[left]
    expected = group["indices"][:group["indices"].index(left)+1]
    need(supplied is None or supplied == expected, "REGION_HISTORY_CUTOFF")
    return list(dict.fromkeys(data.refs+expected))


def run(a):
    started = time.perf_counter()
    need(file_sha(a.run/"report.json") == a.pin, "REGION_RUN_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    data = SequenceData(corpus, bank, ModelConfig())
    need(data.bind == report["data"], "REGION_DATA_BINDING")
    arrays = {s.name: [data.array(s.name, i) for i in range(len(data.sources))] for s in data.inputs.specs}
    phases = dict(source=time.perf_counter()-started)
    tick = time.perf_counter()
    results, checks = {}, dict(zero_radius=0, placement=0, lower_bound=0)
    for left, right in data.successor.items():
        group = data.by_index[left]
        if group["split"] != "development":
            continue
        past = history(data, left)
        costs, by_lane = {}, {}
        for spec in data.inputs.specs:
            values = arrays[spec.name]
            target = values[right]
            persistence = ordered_distance(np.clip(target @ values[left].T, -1, 1))
            costs[spec.name] = {}
            for index in past:
                cosines = np.clip(target @ values[index].T, -1, 1)
                exact = ordered_distance(cosines)
                need(abs(exact-ordered_distance(optimistic_cosines(cosines, 0))) < 1e-12, "REGION_ZERO_RADIUS")
                lower = ordered_distance(optimistic_cosines(cosines, RADIUS))
                need(lower <= exact+1e-12, "REGION_BOUND_ORDER")
                checks["zero_radius"] += 1
                costs[spec.name][index] = dict(exact=exact, lower=lower)
            fixed = [costs[spec.name][i] for i in data.refs]
            by_lane[spec.name] = dict(persistence=persistence, fixed_lower=min(v["lower"] for v in fixed),
                fixed_exact=min(v["exact"] for v in fixed), historical_lower=min(v["lower"] for v in costs[spec.name].values()))
        feasible = lambda indices: [i for i in indices if all(
            costs[s.name][i]["lower"] < by_lane[s.name]["persistence"] for s in data.inputs.specs)]
        results[str(left)] = dict(right=right, domain=group["domain"], fixed=data.refs, history=past,
            lanes=by_lane, fixed_feasible=feasible(data.refs), history_feasible=feasible(past), costs=costs)
    phases["target_aware_capacity"] = time.perf_counter()-tick
    tick = time.perf_counter()
    for name in ("baseline", "cycle-1", "cycle-2"):
        path = a.run/(name+".db")
        before = file_sha(path)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
                body = json.loads(raw)
                if body["kind"] != "source_prediction" or str(body["request"]["left"]) not in results:
                    continue
                payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
                need(digest(payload) == body["payload"], "REGION_PREDICTION_PIN")
                result = results[str(body["request"]["left"])]
                selected = next(i for i in data.refs if data.sources[i][0]["atom_id"] == payload["region"])
                for spec in data.inputs.specs:
                    pred = unpack(payload["predictions"][spec.name], spec, max_windows=data.config.max_native_tokens)
                    base, target = arrays[spec.name][selected], arrays[spec.name][result["right"]]
                    need(pred.shape == base.shape and np.all((pred*base).sum(1) >= np.sqrt(1-RADIUS**2)-1e-6),
                         "REGION_PLACEMENT_RADIUS")
                    loss = ordered_distance(np.clip(target @ pred.T, -1, 1))
                    need(loss+1e-6 >= result["costs"][spec.name][selected]["lower"], "REGION_PREDICTION_BOUND")
                    checks["placement"] += 1
                    checks["lower_bound"] += 1
        need(file_sha(path) == before, "REGION_MUTATED_JOURNAL")
    phases["prediction_bounds"] = time.perf_counter()-tick
    tick = time.perf_counter()
    left = int(next(iter(results)))
    group = data.by_index[left]
    prefix = group["indices"][:group["indices"].index(left)+1]
    controls = []
    for kind, bad in (("future", prefix+[data.successor[left]]),
                      ("other_document", prefix+[data.groups[0]["indices"][0]])):
        try:
            history(data, left, bad)
        except ValueError as exc:
            need(str(exc) == "MNLM_TEXT_CORPUS_REGION_HISTORY_CUTOFF", "REGION_CONTROL_CODE")
            controls.append(kind)
        else:
            raise ValueError("MNLM_TEXT_CORPUS_REGION_CONTROL_ACCEPTED")
    summaries = {}
    for domain in ("general", "legal", "science"):
        rows = [r for r in results.values() if r["domain"] == domain]
        summaries[domain] = dict(count=len(rows),
            fixed_without_feasible_region=sum(not r["fixed_feasible"] for r in rows),
            history_without_feasible_region=sum(not r["history_feasible"] for r in rows),
            lanes={s.name: {k: float(np.mean([r["lanes"][s.name][k] for r in rows]))
                           for k in ("fixed_lower", "fixed_exact", "historical_lower", "persistence")}
                   for s in data.inputs.specs})
    phases["controls_and_summary"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    need(.9 <= sum(phases.values())/wall <= 1.001, "REGION_PHASE_COVERAGE")
    publish(a.output, dict(schema="text-native-region-capacity-v1", run=a.pin, data=data.pin,
        radius=RADIUS, summaries=summaries, results=results, checks=checks, negative_controls=controls,
        phases_seconds=phases, main_function_wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        target_aware=True, forecast=False, scientific_admission=False, served=False,
        acceptance="author-checked only"))
    print(json.dumps(dict(report_sha256=file_sha(a.output), rows=len(results),
        fixed_infeasible=sum(not r["fixed_feasible"] for r in results.values()),
        history_infeasible=sum(not r["history_feasible"] for r in results.values()), seconds=wall)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
