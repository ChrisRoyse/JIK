"""Direct artifact/SQL reader for growth; no model, router, or growth execution."""
import argparse
import base64
import io
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_kernel_route import independent_score
from audit.tool_loop_readback import readback as read_journal
from audit.tool_private_regions import inspect as inspect_private
from audit.tool_learner_readback import inspect as inspect_learner
from audit.tool_native_recognition import scores as recognition_scores, context as world_context
from audit.tool_shared_objective import read_loss as read_shared_loss
from audit.tool_causal_history import read as read_history
from audit import tool_retained_update
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.tool_temporal_input import ToolInputs


def manifest(root, expected):
    raw = (root/"manifest.json").read_bytes()
    check(sha(raw) == expected == (root/"COMMITTED").read_text(), "GROWTH_READ_PIN")
    return json.loads(raw)


def fingerprint(array):
    a = np.ascontiguousarray(array)
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=sha(a.tobytes()))


def checkpoint(root, expected):
    m = manifest(root, expected)
    core = manifest(root/"core", m["core"])
    for name, pin in core["files"].items():
        raw = (root/"core"/name).read_bytes()
        check(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "GROWTH_READ_CORE")
    raw = (root/"world.pt").read_bytes()
    check(len(raw) == m["payload_bytes"] and sha(raw) == m["payload_sha256"], "GROWTH_READ_WORLD")
    weights = {k: v.numpy() for k, v in torch.load(root/"core"/"weights.pt",
               weights_only=True, map_location="cpu").items()}
    extra = {k: v.numpy() for k, v in torch.load(io.BytesIO(raw),
             weights_only=True, map_location="cpu").items()}
    states = np.load(root/"core"/"contexts.npy", allow_pickle=False)
    return m, core, weights, extra, states


def readback(inputs, run, parent, parent_pin, run_pin):
    check(sha((parent/"run.json").read_bytes()) == parent_pin and
          sha((run/"run.json").read_bytes()) == run_pin, "GROWTH_READ_RUN")
    start = json.loads((parent/"run.json").read_bytes())
    run_record = json.loads((run/"run.json").read_bytes())
    snapshot = json.loads((run/"snapshot.json").read_bytes())
    check(run_record["contract"] == snapshot["contract"], "GROWTH_READ_CONTRACT")
    journal = read_journal(run/"experience.sqlite3", inputs, snapshot)
    check(run_record["head"] == journal["head"], "GROWTH_READ_HEAD")
    bank = list(inputs.world_units)
    pairs = {str(r["unit"]): dict(response=r, action=inputs.population.lookup[r["task"], r["turn"]-1])
             for r in inputs.population.initial_bank()}
    old = checkpoint(parent/"initial", start["initial"])
    active_checkpoint, world = start["initial"], start["world"]
    kernel_pin = (run/"initial-kernel"/"COMMITTED").read_text()
    kernel = manifest(run/"initial-kernel", kernel_pin)
    check(kernel["pairs"] == pairs, "GROWTH_READ_INITIAL")
    episodes, promoted, receipts, hidden, observed, records = {}, set(), {}, {}, {}, {}
    counts = dict(growths=0, retained_native_windows=0, added_native_windows=0,
                  scored_forecast_pairs=0, scored_calibration_pairs=0, forecasts=0,
                  candidates=0, checked_candidate_regions=0, supervised_slots=0)
    versions = []
    training = {r["name"]: r for r in json.loads((run/"training.json").read_bytes())} \
        if (run/"training.json").is_file() else {}
    forecast_hidden = {}
    learner_history = {}
    context_support, private_views, objectives, histories, retained_updates = {}, [], [], [], []

    def windows(name, unit):
        a, p = inputs.arrays[name]
        return a[p[unit]:p[unit+1]]

    def calibrate(k):
        check(k["pairs"] == pairs and k["source"] == inputs.source_id and
              k["instruments"] == {s.name: s.identity for s in inputs.specs},
              "GROWTH_READ_KERNEL_SOURCE")
        for spec in inputs.specs:
            saved = k["calibration"][spec.name]
            recalls = []
            expected_refs = [inputs.population.lookup[e["task"], t]
                             for e in inputs.population.source.episodes
                             if e["status"] == "ready" and e["partition"] == "development"
                             for t in (6, 10)]
            check([r["reference"] for r in saved["rows"]] == expected_refs,
                  "GROWTH_READ_CALIBRATION_ROSTER")
            for r in saved["rows"]:
                scores = {key: independent_score(windows(spec.name, r["reference"]["unit"]),
                          windows(spec.name, pair["action"]["unit"])) for key, pair in pairs.items()}
                exact = sorted(scores, key=lambda key: (-scores[key], key))[:k["plan"]["top_k"]]
                hits = r["route"]["hits"]
                check(exact == r["exact"] and all(abs(scores[key]-v) < 2e-12 for key, v in hits),
                      "GROWTH_READ_CALIBRATION_SCORE")
                recall = len(set(exact) & {key for key, _ in hits})/len(exact)
                check(recall == r["recall"], "GROWTH_READ_CALIBRATION_RECALL")
                recalls.append(recall)
                counts["scored_calibration_pairs"] += len(scores)
            check(sum(recalls)/len(recalls) == saved["recall"] and saved["passes"]
                  and saved["recall"] >= .95, "GROWTH_READ_RECALL_GATE")
        versions.append(dict(pin=kernel_pin, regions=len(bank),
                             recall={s.name: k["calibration"][s.name]["recall"] for s in inputs.specs}))

    calibrate(kernel)
    with sqlite3.connect(f"file:{(run/'experience.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        payloads = {pin: json.loads(raw) for pin, raw in db.execute("SELECT hash,data FROM payloads")}
        for previous, raw in db.execute("SELECT previous,body FROM events ORDER BY seq"):
            b = json.loads(raw)
            for item in ([b["record"]] if b["kind"] == "memory" else b.get("memory", [])):
                event = item["event"]
                if event["op"] == "intake":
                    ep = event["episode"]
                    ref = ep["source"]
                    check(ref == inputs.population.source.reference(ref["task"], ref["turn"])
                          and ep["content"] == inputs.population.source.texts[ref["task"], ref["turn"]]
                          and digest({k: v for k, v in ep.items() if k != "id"}) == ep["id"],
                          "GROWTH_READ_EPISODE")
                    episodes[ep["id"]] = ep
                elif event["op"] == "check_source":
                    receipt = event["result"]
                    check(receipt["source"] == episodes[event["episode"]]["source"] and
                          receipt["scope"] == "pinned_artifact_contents_only", "GROWTH_READ_RECEIPT")
                    receipts[event["episode"]] = receipt["id"]
                elif event["op"] == "promote":
                    check(event["receipt"] == receipts[event["episode"]], "GROWTH_READ_PROMOTION")
                    promoted.add(event["episode"])
            if b["kind"] == "advance":
                req = b["request"]
                context_support.setdefault(req["context"], {})[req["turn"]] = b["episode"]
                hidden[req["context"]] = dict(sequence=req["sequence"]+1, last_event=digest(req),
                                              raw=base64.b64decode(b["hidden"], validate=True))
                p = payloads.get(b.get("payload"), {})
                if "forecast" in p:
                    r = p["forecast"]
                    records[digest(r)] = r
                    forecast_hidden[digest(r)] = b["hidden"]
                    route = r["routing"]
                    check(r["world"] == world and route["kernel"] == kernel_pin and
                          route["query"] == req["reference"], "GROWTH_READ_FORECAST_VERSION")
                    votes = {}
                    for spec in inputs.specs:
                        scores = {key: independent_score(windows(spec.name, route["query"]["unit"]),
                                  windows(spec.name, pair["action"]["unit"])) for key, pair in pairs.items()}
                        for rank, (key, v) in enumerate(route["traces"][spec.name]["hits"], 1):
                            check(abs(scores[key]-v) < 2e-12, "GROWTH_READ_ROUTE_SCORE")
                            votes[key] = votes.get(key, 0.)+1/(60+rank)
                        counts["scored_forecast_pairs"] += len(scores)
                    chosen = sorted(votes, key=lambda key: (-votes[key], key))[:kernel["plan"]["top_k"]]
                    allowed = [i for i, u in enumerate(bank) if str(u) in chosen]
                    check(chosen == route["responses"] and route["allowed"] == allowed,
                          "GROWTH_READ_ROUTE_MASK")
                    if "region_context" in r:
                        private_views.append(inspect_private(inputs, r, bank, episodes, promoted,
                            context_support[req["context"]], old[2], old[3], b["hidden"]))
                    else:
                        check(bank.index(int(r["region"].split(":")[0])) in allowed, "GROWTH_READ_ROUTE_MASK")
                    check(all(ep in promoted for ep in r["bank_support"].values()) and
                          all(str(u) in r["bank_support"] for u in bank), "GROWTH_READ_BANK_SUPPORT")
                    counts["forecasts"] += 1
                if "assessment" in p:
                    key = next(k for k, r in records.items() if r["request"]["context"] == req["context"]
                               and r["request"]["turn"]+1 == req["turn"])
                    observed[key] = b["episode"]
            if b["kind"] == "candidate":
                candidate_root = run/"candidates"/b["candidate"]
                cm = manifest(candidate_root, b["pin"])
                check(cm["parent"] == b["parent"] == active_checkpoint
                      and cm["binding"]["world"] == world and cm["step"] == 8
                      and cm["binding"]["plan"]["placement_conditioning"] ==
                          "checked_region_label_with_historical_mask"
                      and cm["binding"]["plan"]["region_objective"] ==
                          "shared_constellation_ce_per_instrument_v1"
                      and cm["binding"]["plan"]["context_interpretation"] ==
                          "ordered_native_query_recurrence_v1", "GROWTH_READ_CANDIDATE_BINDING")
                raw_optimizer = (candidate_root/"training.pt").read_bytes()
                check(len(raw_optimizer) == cm["bytes"] and sha(raw_optimizer) == cm["sha256"],
                      "GROWTH_READ_OPTIMIZER_BYTES")
                trained = checkpoint(candidate_root/"model", cm["model"])
                check(trained[0]["world"] == world and trained[0]["standard"] == start["standard"]
                      and trained[2].keys() == old[2].keys()
                      and all(np.array_equal(a, old[2][k]) for k, a in trained[2].items())
                      and all(np.array_equal(a, old[3][k]) for k, a in trained[3].items()
                              if k.startswith("world_")), "GROWTH_READ_CANDIDATE_RETENTION")
                payload = payloads[b["payload"]]
                for key, label in payload["labels"].items():
                    record = records[key]
                    units = record.get("region_context", {}).get("definition", {}).get("units", bank)
                    allowed = list(range(len(units))) if "region_context" in record else record["routing"]["allowed"]
                    check(label in allowed, "GROWTH_READ_CANDIDATE_LABEL")
                    for kind in ("baseline", "candidate"):
                        selected = units.index(int(payload[kind][key]["region"].split(":")[0]))
                        check(selected in allowed, "GROWTH_READ_CANDIDATE_MASK")
                        weights = old if kind == "baseline" else trained
                        _, recognized = recognition_scores(inputs, record, weights[2], weights[3],
                                                            forecast_hidden[key])
                        check(recognized == units[selected], "GROWTH_READ_CANDIDATE_RECOGNITION")
                        counts["checked_candidate_regions"] += 1
                proof = training[b["candidate"]]
                check(cm["binding"]["plan"].get("region_risk") == proof.get("region_risk_objective"),
                      "RISK_READ_PLAN")
                check(proof["decision"] == b and proof["parent"] == active_checkpoint
                      and proof["exact_optimizer_recovery"], "GROWTH_READ_TRAINING_PROOF")
                learner_start = inspect_learner(run, b, proof, old, learner_history, checkpoint, manifest)
                previous = learner_history[b["continuation"]][0] if "continuation" in b else None
                previous_payload = payloads[previous["payload"]] if previous else None
                retained_update = tool_retained_update.read(b, cm, proof, payload, previous, previous_payload,
                                                           learner_start, trained)
                retained_update["controls"] = tool_retained_update.negative_readback(
                    b, cm, proof, payload, previous, previous_payload, learner_start, trained) \
                    if "training_retention" in cm["binding"] else []
                retained_updates.append(retained_update)
                objectives.append(read_shared_loss(inputs, records, payload["labels"], forecast_hidden,
                                                    learner_start, proof))
                histories.extend(read_history(inputs, records, forecast_hidden, learner_start, trained, proof))
                for sample in proof["recognition"]:
                    key = sample["forecast"]
                    check(sample["fixed_state"] == forecast_hidden[key], "GROWTH_READ_RECOGNITION_STATE")
                    current, selected = recognition_scores(inputs, records[key], learner_start[2],
                                                            learner_start[3], sample["fixed_state"])
                    ref = records[key]["request"]
                    earlier = inputs.population.lookup[ref["task"], ref["turn"]-2]["unit"]
                    altered, other = recognition_scores(inputs, records[key], learner_start[2],
                        learner_start[3], sample["fixed_state"], unit=earlier)
                    deltas = {n: float(np.max(np.abs(v-altered[n]))) for n, v in current.items()}
                    check(sample["earlier_action"] == earlier and sample["selected"] == selected
                          and sample["earlier_action_selected"] == other
                          and deltas == sample["direct_native_deltas"], "GROWTH_READ_NATIVE_SENSITIVITY")
                for sample in proof["supervision"]:
                    key, unit = sample["forecast"], sample["label"]
                    units = records[key].get("region_context", {}).get("definition", {}).get("units", bank)
                    check(sample["hidden"] == forecast_hidden[key]
                          and unit == units[payload["labels"][key]] and key in b["train"],
                          "GROWTH_READ_SUPERVISION_SOURCE")
                    h = world_context(inputs, records[key], learner_start[2], learner_start[3], sample["hidden"])
                    for i, spec in enumerate(inputs.specs):
                        book = windows(spec.name, unit).astype(np.float64)
                        name = spec.name
                        project = book@old[2][f"projectors.{name}.weight"].astype(np.float64).T
                        project += old[2][f"projectors.{name}.bias"]
                        residual = np.tanh((project+h[i])@learner_start[3][f"place.{name}.weight"].astype(np.float64).T
                                          + learner_start[3][f"place.{name}.bias"])
                        residual = .1*residual/np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1.)
                        expected = book/np.linalg.norm(book, axis=1, keepdims=True)+residual
                        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                        packed = sample["predictions"][name]
                        raw = base64.b64decode(packed["base64"], validate=True)
                        check(packed["instrument"] == spec.identity and sha(raw) == packed["sha256"],
                              "GROWTH_READ_SUPERVISION_BYTES")
                        actual = np.frombuffer(raw, dtype="<f4").reshape(packed["shape"])
                        check(actual.shape == expected.shape and np.max(np.abs(actual-expected)) < 1e-6,
                              "GROWTH_READ_SUPERVISION_VALUES")
                        counts["supervised_slots"] += 1
                counts["candidates"] += 1
            if b["kind"] != "grow":
                continue
            check(previous == b["prior_head"] and b["old_checkpoint"] == active_checkpoint
                  and b["old_world"] == world and b["old_kernel"] == kernel_pin
                  and all(s["sequence"] % 5 == 0 for s in hidden.values()), "GROWTH_READ_TRANSITION")
            additions = []
            for key in b["keys"]:
                r = records[key]
                req = r["request"]
                response = inputs.population.lookup[req["task"], req["turn"]+1]
                unit = response["unit"]
                support = b["supports"][str(unit)]
                check(key in observed and req["reference"]["partition"] == "train" and
                      support == dict(action=r["support"][str(req["turn"])], response=observed[key],
                                      forecast=key) and unit not in bank and
                      support["action"] in promoted and support["response"] in promoted,
                      "GROWTH_READ_ELIGIBILITY")
                check(episodes[support["response"]]["source"] ==
                      inputs.population.source.reference(req["task"], req["turn"]+1), "GROWTH_READ_RESPONSE")
                pairs[str(unit)] = dict(action=req["reference"], response=response)
                additions.append(unit)
            check(b["units"] == bank+additions, "GROWTH_READ_APPEND")
            root = run/"growth"/b["name"]
            new = checkpoint(root/"model", b["checkpoint"])
            m, core, weights, extra, states = new
            check(m["world"] == b["world"] and m["standard"] == start["standard"] == b["standard"]
                  and weights.keys() == old[2].keys() and
                  all(np.array_equal(a, old[2][k]) for k, a in weights.items()), "GROWTH_READ_CORE_RETENTION")
            for key in extra:
                if not key.startswith("world_"):
                    check(np.array_equal(extra[key], old[3][key]), "GROWTH_READ_READOUT_RETENTION")
            parameters = {**{"core."+k: fingerprint(a) for k, a in weights.items() if not k.startswith("ref_")},
                          **{k: fingerprint(a) for k, a in extra.items() if not k.startswith("world_")}}
            check(digest(parameters) == b["parameters"], "GROWTH_READ_PARAMETERS")
            for i, spec in enumerate(inputs.specs):
                rows = [f"{u}:{inputs.refs[u]['text_sha256']}" for u in b["units"]]
                check(m["world_rows"][spec.name] == rows, "GROWTH_READ_ROWS")
                for j, unit in enumerate(b["units"]):
                    a = extra[f"world_values_{i}"][j, ~extra[f"world_padding_{i}"][j]]
                    check(np.array_equal(a, windows(spec.name, unit)), "GROWTH_READ_NATIVE_SOURCE")
                    if j < len(bank):
                        prior = old[3][f"world_values_{i}"][j, ~old[3][f"world_padding_{i}"][j]]
                        check(np.array_equal(a, prior), "GROWTH_READ_WINDOW_RETENTION")
                        counts["retained_native_windows"] += len(a)
                    else:
                        counts["added_native_windows"] += len(a)
            context_fingerprints = {}
            check({s["context"] for s in core["contexts"]} == set(hidden), "GROWTH_READ_CONTEXT_ROSTER")
            for i, s in enumerate(core["contexts"]):
                saved = hidden[s["context"]]
                check(s == {k: v for k, v in saved.items() if k != "raw"} | {"context": s["context"]}
                      and states[i].tobytes() == saved["raw"], "GROWTH_READ_STATE_RETENTION")
                context_fingerprints[s["context"]] = dict(sequence=s["sequence"], last_event=s["last_event"],
                                                          hidden=fingerprint(states[i]))
            check(digest(context_fingerprints) == b["context_state"], "GROWTH_READ_STATE_PIN")
            old, bank, active_checkpoint, world = new, b["units"], b["checkpoint"], b["world"]
            kernel_pin = b["kernel"]
            kernel = manifest(root/"kernel", kernel_pin)
            calibrate(kernel)
            counts["growths"] += 1
    check(bank == snapshot["region_units"] and active_checkpoint == snapshot["active_model"] and
          world == snapshot["world"] and kernel_pin == snapshot["kernel"], "GROWTH_READ_FINAL")
    return dict(complete=True, acceptance="author-checked", counts=counts, versions=versions, journal=journal,
                private_views=private_views, shared_objectives=objectives, learned_contexts=histories,
                retained_updates=retained_updates,
                scientific_admission=False, predictive_gain=False, no_model_execution=True)


def main():
    p = argparse.ArgumentParser()
    for n in ("raw", "population", "native", "run", "parent", "population-pin", "native-pin", "parent-pin", "run-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--baseline-run", type=Path)
    p.add_argument("--baseline-pin")
    args = p.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(Path(args.raw)), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    result = readback(inputs, Path(args.run), Path(args.parent), args.parent_pin, args.run_pin)
    if args.baseline_run:
        check(sha((args.baseline_run/"run.json").read_bytes()) == args.baseline_pin,
              "GROWTH_READ_BASELINE_PIN")
        old_snapshot = json.loads((args.baseline_run/"snapshot.json").read_bytes())
        read_journal(args.baseline_run/"experience.sqlite3", inputs, old_snapshot)
        def forecasts(path):
            with sqlite3.connect(f"file:{(path/'experience.sqlite3').as_posix()}?mode=ro", uri=True) as db:
                result = {}
                for raw, in db.execute("SELECT data FROM payloads"):
                    value = json.loads(raw)
                    if "forecast" in value:
                        r = value["forecast"]
                        result[r["request"]["context"], r["request"]["turn"]] = (r["region"], r["predictions"])
                return result
        baseline, current = forecasts(args.baseline_run), forecasts(Path(args.run))
        check(baseline == current, "GROWTH_READ_TARGET_FREE_INFERENCE_DRIFT")
        result["target_free_forecasts_equal_prior_implementation"] = len(current)
    result["wall_seconds"] = time.perf_counter()-started
    with (Path(args.run)/"growth-readback.json").open("xb") as f:
        f.write(canonical(result))
    print(json.dumps(dict(complete=True, **result["counts"], wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
