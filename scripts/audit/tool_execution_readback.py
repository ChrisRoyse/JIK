"""Direct SQL and captured-byte readback for attached process observations."""
import base64
from datetime import datetime
import json
import sqlite3

from audit.tool_loop_readback import readback as forecast_readback
from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, digest


def readback(path, inputs, snapshot, root, execution_inputs=None):
    forecasts = forecast_readback(path, inputs, snapshot, execution_inputs)
    roots = root if isinstance(root, (list, tuple)) else [root]
    batches, providers = [], {}
    for location in roots:
        raw = (location/"manifest.json").read_bytes()
        manifest, pin = json.loads(raw), sha(raw)
        check(pin == (location/"COMMITTED").read_text(), "EXEC_READ_COMMITTED")
        check(not providers.keys() & manifest["observations"].keys(), "EXEC_READ_COLLISION")
        check(all(k == digest(v) for k,v in manifest["observations"].items()), "EXEC_READ_IDS")
        batches.append((location, manifest, pin))
        providers.update((k, batches[-1]) for k in manifest["observations"])
        for name, expected in manifest["files"].items():
            raw = (location/name).read_bytes()
            check(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "EXEC_READ_FILE")
    set_pin = batches[0][2] if len(batches) == 1 else digest(dict(
        schema="process-observation-batches-v1", batches=[b[2] for b in batches]))
    check(set_pin == snapshot["process_observations"]["pin"], "EXEC_READ_SET")
    condition_sets = {}
    for root, manifest, pin in batches:
        if "condition_commitment" not in manifest:
            continue
        raw = (root/"condition-commitment.json").read_bytes()
        check(sha(raw) == manifest["condition_commitment"] == (root/"conditions-COMMITTED").read_text(),
              "EXEC_READ_CONDITION_COMMITMENT")
        commitment = json.loads(raw)
        raw = (root/"condition-inputs.json").read_bytes()
        check(sha(raw) == commitment["inputs"], "EXEC_READ_CONDITION_INPUT")
        conditions = json.loads(raw)["inputs"]
        check(all(k == digest(v) for k,v in conditions.items()), "EXEC_READ_CONDITION_IDS")
        condition_sets[manifest["condition_commitment"]] = (commitment, conditions)
    memory_head, memory_count, states, episodes, children, attached = ZERO, 0, {}, {}, {}, {}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        payloads = {pin:json.loads(raw) for pin,raw in db.execute("SELECT hash,data FROM payloads")}
        pending = {}
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            body = json.loads(raw)
            payload = payloads[body["payload"]] if body.get("payload") else {}
            if "forecast" in payload:
                pending[body["request"]["context"]] = digest(payload["forecast"])
            if body["kind"] == "repair_bank":
                for c, key in body["canceled"].items():
                    check(pending.pop(c) == key, "EXEC_READ_CANCEL")
            records = ([body["record"]] if body["kind"] in ("memory", "invalidate")
                       else body.get("memory", []))
            for record in records:
                check(record["previous"] == memory_head and record["sequence"] == memory_count
                      and record["sha256"] == digest({k:v for k,v in record.items() if k != "sha256"}),
                      "EXEC_READ_MEMORY_CHAIN")
                event = record["event"]
                if event["op"] == "intake":
                    ep = event["episode"]
                    key = ep["id"]
                    check(key == digest({k:v for k,v in ep.items() if k != "id"})
                          and key not in episodes and all(k in episodes for k in ep["evidence"]),
                          "EXEC_READ_EPISODE")
                    if ep["claim_kind"] == "source_statement":
                        ref = ep["source"]
                        check(ref == inputs.population.source.reference(ref["task"], ref["turn"])
                              and ep["content"] == inputs.population.source.texts[ref["task"], ref["turn"]],
                              "EXEC_READ_SOURCE")
                    elif ep["claim_kind"] == "execution_precondition":
                        key_condition = ep["source"]["condition"]
                        check(ep["source"]["commitment"] in condition_sets, "EXEC_READ_CONDITION_SET")
                        commitment, conditions = condition_sets[ep["source"]["commitment"]]
                        check(key_condition in conditions
                              and ep["content"] == conditions[key_condition]
                              and ep["event_time"] == commitment["committed_at"]
                              and ep["when"] == dict(clock=commitment["clock"], at=commitment["committed_at"])
                              and ep["where"] == dict(image=conditions[key_condition]["image"])
                              and ep["independent_occurrence"] is False
                              and ep["native_measured"] is False
                              and ep["model_consumption_established"] is False,
                              "EXEC_READ_PRECONDITION")
                    else:
                        check(ep["claim_kind"] == "event_occurrence", "EXEC_READ_KIND")
                        root, manifest, pin = providers[ep["source"]["observation"]]
                        entry = manifest["observations"][ep["source"]["observation"]]
                        captured = json.loads((root/entry["execution"]/"execution.json").read_bytes())
                        result = captured["result"]
                        direct = {n:base64.b64decode(result[n]["base64"], validate=True).decode()
                                  for n in ("stdout", "stderr")}
                        after = json.loads((root/entry["execution"]/"after.json").read_bytes())
                        check(ep["content"] == direct and ep["what"]["returncode"] == result["returncode"]
                              and ep["what"]["program_sha256"] == entry["program_sha256"]
                              and ep["where"] == dict(image=captured["image"], container=captured["container_id"])
                              and ep["event_time"] == after["State"]["FinishedAt"]
                              and ep["when"] == dict(clock="container_runtime_utc",
                                  started=after["State"]["StartedAt"], finished=after["State"]["FinishedAt"])
                              and ep["independent_occurrence"] is True and ep["native_measured"] is False
                              and ep["source"]["observation_set"] == pin
                              and ep["source"]["reexecution_of"] == entry["target"],
                              "EXEC_READ_CAPTURE")
                        if "condition_commitment" in manifest:
                            commitment, conditions = condition_sets[manifest["condition_commitment"]]
                            condition = conditions[entry["condition"]]
                            check(condition["image"] == captured["image"]
                                  and condition["wheels"] == entry["wheels"] == result["wheels"]
                                  and condition["program"].encode() == (root/entry["program"]).read_bytes()
                                  and datetime.fromisoformat(commitment["committed_at"]) <
                                      datetime.fromisoformat(after["State"]["StartedAt"].replace("Z", "+00:00"))
                                  and any(episodes[parent]["claim_kind"] == "execution_precondition"
                                      and episodes[parent]["content"] == condition for parent in ep["evidence"]),
                                  "EXEC_READ_CONDITION_BINDING")
                            check(ep["execution_conditions"] == dict(condition=entry["condition"],
                                commitment=manifest["condition_commitment"], inputs=commitment["inputs"],
                                committed_at=commitment["committed_at"], started=after["State"]["StartedAt"],
                                scope="conditions_committed_before_new_occurrence", prediction_present=False),
                                "EXEC_READ_CONDITION_EPISODE")
                    episodes[key] = ep
                    states[key] = dict(active=True, grounded_scope=None, review="unassessed", receipt=None)
                    for parent in ep["evidence"]:
                        children.setdefault(parent, set()).add(key)
                else:
                    key = event["episode"]
                    state = states[key]
                    if event["op"] == "check_source":
                        receipt = event["result"]
                        check(receipt["id"] == digest({k:v for k,v in receipt.items() if k != "id"})
                              and receipt["episode"] == key, "EXEC_READ_RECEIPT")
                        state.update(receipt=receipt, review="source_checked")
                    elif event["op"] == "promote":
                        check(event["receipt"] == state["receipt"]["id"], "EXEC_READ_PROMOTION")
                        state["grounded_scope"] = state["receipt"]["scope"]
                    elif event["op"] == "invalidate":
                        affected, frontier = set(), [key]
                        while frontier:
                            next_key = frontier.pop()
                            if next_key not in affected:
                                affected.add(next_key)
                                frontier.extend(children.get(next_key, ()))
                        check(sorted(affected) == event["affected"], "EXEC_READ_INVALIDATION")
                        for item in affected:
                            states[item].update(grounded_scope=None, receipt=None, review="needs_review")
                    else:
                        raise ValueError("EXEC_READ_UNEXPECTED_MEMORY_ACTION")
                memory_head, memory_count = record["sha256"], memory_count+1
            echo = None
            if "assessment" in payload:
                key = pending.pop(body["request"]["context"])
                if "execution" in payload:
                    pin = providers[payload["execution"]["observation"]][2]
                    echo = dict(payload["execution"], forecast=key, observation_set=pin)
            if body["kind"] == "execution_observation" or echo is not None:
                attachment = body if echo is None else echo
                original_body = body
                body = attachment
                root, manifest, pin = providers[body["observation"]]
                entry = manifest["observations"][body["observation"]]
                ep = episodes[body["episode"]]
                check(ep["source"]["reexecution_of"] == entry["target"]
                      and body["observation_set"] == pin, "EXEC_READ_LINK")
                report = inputs.population.source.texts[entry["target"]["task"], entry["target"]["turn"]]
                report = report.split("\n\n(Open file:")[0]
                if body["comparison"]["comparison"]["kind"] == "exact_stdout":
                    agrees = ep["what"]["returncode"] == 0 and ep["content"] == dict(stdout=report+"\n", stderr="")
                else:
                    agrees = (ep["what"]["returncode"] == 1 and ep["content"]["stdout"] == ""
                              and ep["content"]["stderr"].rstrip().splitlines()[-1:] == [report.rstrip().splitlines()[-1]])
                check(body["comparison"]["concordant"] == agrees
                      and body["comparison"]["historical_occurrence_established"] is False,
                      "EXEC_READ_COMPARISON")
                attached[body["forecast"]] = dict(observation=body["observation"], episode=body["episode"],
                                                 comparison=body["comparison"])
                if echo is not None:
                    attached[body["forecast"]]["native"] = body["native"]
                body = original_body
    check(states == snapshot["memory"]["states"] and memory_head == snapshot["memory"]["head"]
          and attached == snapshot["process_observations"]["forecasts"], "EXEC_READ_FINAL")
    grounded = sorted(k for k,v in states.items() if v["grounded_scope"] == "observed_process_in_pinned_runtime")
    check(grounded == snapshot["memory"]["grounded_world_claims"], "EXEC_READ_SCOPE")
    return dict(forecasts=forecasts, independently_read_observations=len(attached),
                grounded_observations=len(grounded), memory_events=memory_count,
                historical_occurrence_established=False, native_feedback_available=execution_inputs is not None)
