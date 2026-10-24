"""Read the SQLite bytes directly; independently score committed native windows."""
import base64
import json
import sqlite3

import numpy as np

from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from audit.tool_retention_readback import inspect as read_retention


def distance(a, b):
    a = a/np.linalg.norm(a, axis=1, keepdims=True)
    b = b/np.linalg.norm(b, axis=1, keepdims=True)
    cosines = np.clip(a@b.T, -1, 1)
    table = np.empty((len(a)+1, len(b)+1))
    table[:, 0], table[0, :] = np.arange(len(a)+1), np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+1-cosines[i-1, j-1])
    return float(table[-1, -1]/max(len(a), len(b)))


def readback(path, inputs, snapshot, execution_inputs=None):
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        check(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "TOOL_READBACK_SQLITE")
        contract = json.loads(db.execute("SELECT contract FROM metadata").fetchone()[0])
        check(digest(contract) == snapshot["contract"], "TOOL_READBACK_CONTRACT")
        retention_required = "model/tool_retention.py" in contract["code"]
        retention_history, retained_checks = [], []
        payloads = {}
        for pin, raw in db.execute("SELECT hash,data FROM payloads"):
            check(sha(raw) == pin and canonical(json.loads(raw)) == raw, "TOOL_READBACK_PAYLOAD")
            payloads[pin] = json.loads(raw)
        head, count, transitions, scored, decisions, candidate_scored = ZERO, 0, 0, 0, 0, 0
        pending, sequences, forecasts, lineage = {}, {}, {}, []
        observed_targets = {}
        observed_echoes = {}
        conditions = {}
        enrollment = None
        enrollments, enrolled = [], set()
        def assess_saved(record, predictions, assessment):
            task, cutoff = record["request"]["task"], record["request"]["turn"]
            target_unit = inputs.population.lookup[task, cutoff+1]["unit"]
            target_inputs = inputs
            if digest(record) in observed_targets:
                target_inputs, target_unit = observed_targets[digest(record)]
            old_unit = record["baseline_unit"]
            check(set(predictions) == {s.name for s in inputs.specs}, "TOOL_READBACK_ROSTER")
            for spec in inputs.specs:
                v = predictions[spec.name]
                raw_pred = base64.b64decode(v["base64"], validate=True)
                check(sha(raw_pred) == v["sha256"] and v["instrument"] == spec.identity,
                      "TOOL_READBACK_PREDICTION")
                pred = np.frombuffer(raw_pred, dtype="<f4").astype(np.float64).reshape(v["shape"])
                check(pred.shape[1] == spec.dims and
                      np.all(np.abs(np.linalg.norm(pred, axis=1)-1) < 1e-4), "TOOL_READBACK_NORM")
                values, ptr = inputs.arrays[spec.name]
                targets, target_ptr = target_inputs.arrays[spec.name]
                target = targets[target_ptr[target_unit]:target_ptr[target_unit+1]].astype(np.float64)
                old = values[ptr[old_unit]:ptr[old_unit+1]].astype(np.float64)
                saved = assessment["instruments"][spec.name]
                check(abs(distance(target, pred)-saved["prediction_loss"]) < 1e-12 and
                      abs(distance(target, old)-saved["persistence_loss"]) < 1e-12,
                      "TOOL_READBACK_SCORE")
        for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            check(seq == count and previous == head and canonical(body) == raw and
                  pin == digest(dict(seq=seq, previous=previous, body=body)), "TOOL_READBACK_CHAIN")
            payload = payloads[body["payload"]] if body.get("payload") else {}
            kind = body["kind"]
            if kind == "execution_enrollment":
                check("process_observations" not in contract
                      and body["prior_head"] == head and execution_inputs is not None
                      and not enrolled.intersection(body["links"])
                      and all(execution_inputs.for_observation(o).identity == body["native"]
                              for o in body["links"].values())
                      and body["native"]["observations"] == body["spec"]["observation_pin"]
                      and all(k in pending.values() for k in body["links"]), "EXECUTION_READBACK_ENROLLMENT")
                enrollment = body
                enrollments.append(body)
                enrolled.update(body["links"])
            elif kind == "execution_conditions":
                context, action = body["context"], body["action"]
                binding = body["conditions"]
                check(body["prior_head"] == head and body["sequence"] == sequences.get(context, 0)
                      and body["sequence"] % 5 == 1 and action == 2+2*(body["sequence"]//5)
                      and (context,action) not in conditions, "CONDITION_READBACK_ORDER")
                declared = contract["condition_inputs"]
                providers = declared["providers"] if declared.get("schema") == "condition-native-catalog-v1" else [declared]
                check(sum(all(binding["native"].get(k) == v for k,v in provider.items())
                          for provider in providers) == 1,
                      "CONDITION_READBACK_NATIVE")
                conditions[context,action] = binding
            elif kind == "advance":
                req = body["request"]
                c, turn = req["context"], req["turn"]
                check(req["reference"] == inputs.population.lookup[req["task"], turn]
                      and req["sequence"] == sequences.get(c, 0)
                      and req["phase"] == req["sequence"] % 5, "TOOL_READBACK_SOURCE")
                check(sha(base64.b64decode(body["hidden"], validate=True)) == body["hidden_sha256"],
                      "TOOL_READBACK_HIDDEN")
                sequences[c] = req["sequence"]+1
                if req["phase"] in (1,2):
                    check(req.get("execution_conditions") == conditions.get((c,turn)), "CONDITION_READBACK_REQUEST")
                if req["phase"] == 4:
                    check(req.get("execution_context") == observed_echoes.get((c, turn)),
                          "TOOL_READBACK_CORRECTION")
                if "forecast" in payload:
                    record = payload["forecast"]
                    check(req["phase"] == 2 and c not in pending and record["request"] == req,
                          "TOOL_READBACK_COMMIT")
                    key = digest(record)
                    pending[c], forecasts[key] = key, record
                if "assessment" in payload:
                    key = pending.pop(c)
                    record = forecasts[key]
                    check(req["phase"] == 3 and turn == record["request"]["turn"]+1
                          and req["receipt"] == inputs.population.source.receipt(req["task"], turn),
                          "TOOL_READBACK_ECHO")
                    if "execution" in payload:
                        binding = payload["execution"]["native"]
                        check(execution_inputs is not None and req["execution"] == binding
                              and execution_inputs.checked(binding["observation"]) == binding
                              and execution_inputs.observations.entries[binding["observation"]]["target"] ==
                                  inputs.population.source.reference(req["task"], turn),
                              "TOOL_READBACK_EXECUTION")
                        provider = execution_inputs.for_observation(binding["observation"])
                        observed_targets[key] = provider, binding["unit"]
                        observed_echoes[c, turn] = binding
                    else:
                        check("execution" not in req, "TOOL_READBACK_EXECUTION_MISSING")
                    assess_saved(record, record["predictions"], payload["assessment"])
                    scored += len(inputs.specs)
                transitions += 1
            elif kind == "repair_bank":
                for c, key in body["canceled"].items():
                    check(pending.pop(c) == key, "TOOL_READBACK_CANCEL")
            elif kind == "candidate":
                check(body["result"] == payload["result"] and body["result"]["selected"] is False,
                      "TOOL_READBACK_DECISION")
                for values in (payload["baseline"], payload["candidate"]):
                    check(set(values) == set(body["train"]+body["evaluate"]), "TOOL_READBACK_EVAL_KEYS")
                    for key, result in values.items():
                        assess_saved(forecasts[key], result["predictions"], result["assessment"])
                        candidate_scored += len(inputs.specs)
                for key in body["evaluate"]:
                    for spec in inputs.specs:
                        baseline = payload["baseline"][key]["assessment"]["instruments"][spec.name]
                        candidate = payload["candidate"][key]["assessment"]["instruments"][spec.name]
                        expected = dict(candidate=candidate["prediction_loss"], parent=baseline["prediction_loss"],
                            persistence=candidate["persistence_loss"],
                            better_than_parent=candidate["prediction_loss"] < baseline["prediction_loss"],
                            better_than_persistence=candidate["prediction_loss"] < candidate["persistence_loss"])
                        check(body["result"]["comparisons"][key][spec.name] == expected,
                              "TOOL_READBACK_COMPARISON")
                if retention_required:
                    retained_checks.append(read_retention(body, payload, retention_history, payloads,
                                           {s.name: s.identity for s in inputs.specs}))
                retention_history.append(body)
                lineage.append(dict(candidate=body["candidate"], pin=body["pin"], parent=body["parent"]))
                decisions += 1
            head, count = pin, seq+1
        check(head == snapshot["head"] and count == snapshot["events"] and not pending
              and set(forecasts) == set(snapshot["forecasts"]), "TOOL_READBACK_FINAL")
        check([dict(context=c, action=t, **v) for (c,t),v in sorted(conditions.items())] ==
              snapshot.get("execution_conditions", []), "CONDITION_READBACK_FINAL")
        check(enrollment == snapshot.get("execution_enrollment"), "EXECUTION_READBACK_FINAL")
        check((enrollments if len(enrollments) > 1 else None) == snapshot.get("execution_enrollments"),
              "EXECUTION_READBACK_BATCHES")
        return dict(head=head, events=count, transitions=transitions, independently_scored_slots=scored,
                    independently_scored_candidate_slots=candidate_scored,
                    forecasts=len(forecasts), decisions=decisions, lineage=lineage,
                    retention_required=retention_required, retained_checks=retained_checks,
                    payloads=len(payloads), sqlite_integrity=True, world_truth_established=False)
    finally:
        db.close()
