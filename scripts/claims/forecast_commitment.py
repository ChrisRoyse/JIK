"""Durable pre-execution predictions, backed by a closed SQLite snapshot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest, ZERO
from claims.process_conditions import write_committed


def publish(session, root, keys):
    session._fresh()
    check("process_observations" not in session.journal.contract and keys and len(keys) == len(set(keys)),
          "FORECAST_SEAL_SCOPE")
    check(all(k not in session.executions and
              all(k not in e["links"] for e in session.execution_enrollments) for k in keys),
          "FORECAST_SEAL_PRIOR_OUTCOME")
    records = {}
    for key in keys:
        item = session.forecasts[key]
        req = item["record"]["request"]
        check(item["status"] == "pending" and not item["invalidated"]
              and session.pending.get(req["context"]) == key
              and "execution_conditions" in req, "FORECAST_SEAL_PENDING")
        session.native_action(req)
        records[key] = item["record"]
    root = Path(root)
    check(not root.exists(), "FORECAST_SEAL_EXISTS")
    root.mkdir(parents=True)
    with sqlite3.connect(root/"journal.sqlite3") as db:
        session.journal.db.backup(db)
    with (root/"journal.sqlite3").open("rb") as f:
        os.fsync(f.fileno())
    body = dict(schema="pre-execution-forecast-v1", journal=sha((root/"journal.sqlite3").read_bytes()),
        head=session.head, events=session.journal.count, contract=session.journal.pin,
        source=session.source.pin, conditions=session.condition_inputs.identity,
        forecasts=records, sealed_at=datetime.now(timezone.utc).isoformat(),
        clock="host_utc_after_durable_prediction_before_execute", exposed_program=True,
        unseen_task_skill_established=False, scientific_admission=False)
    pin = write_committed(root/"forecast.json", body)
    with (root/"COMMITTED").open("x") as f:
        f.write(pin)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    read(root, pin)
    return pin


def read(root, pin):
    root = Path(root)
    raw = (root/"forecast.json").read_bytes()
    check(sha(raw) == pin == (root/"COMMITTED").read_text(), "FORECAST_SEAL_PIN")
    body = json.loads(raw)
    check(canonical(body) == raw and body["schema"] == "pre-execution-forecast-v1"
          and body["scientific_admission"] is False
          and body["unseen_task_skill_established"] is False
          and sha((root/"journal.sqlite3").read_bytes()) == body["journal"], "FORECAST_SEAL_BYTES")
    pending, forecasts, head, count = {}, {}, ZERO, 0
    enrolled = set()
    with sqlite3.connect(f"file:{root/'journal.sqlite3'}?mode=ro", uri=True) as db:
        check(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "FORECAST_SEAL_SQLITE")
        contract = json.loads(db.execute("SELECT contract FROM metadata").fetchone()[0])
        check(digest(contract) == body["contract"] and contract["source"] == body["source"]
              and contract["condition_inputs"] == body["conditions"]
              and "process_observations" not in contract, "FORECAST_SEAL_CONTRACT")
        for seq, previous, raw, event_pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            event = json.loads(raw)
            check(seq == count and previous == head and canonical(event) == raw
                  and digest(dict(seq=seq, previous=previous, body=event)) == event_pin,
                  "FORECAST_SEAL_CHAIN")
            if event["kind"] == "execution_enrollment":
                check(not enrolled.intersection(event["links"])
                      and all(k in pending.values() for k in event["links"]), "FORECAST_SEAL_ENROLLMENT")
                enrolled.update(event["links"])
            if event["kind"] == "execution_observation":
                enrolled.add(event["forecast"])
            if event["kind"] == "repair_bank":
                for context, key in event["canceled"].items():
                    check(pending.pop(context) == key, "FORECAST_SEAL_CANCEL")
            if event.get("payload"):
                value = db.execute("SELECT data FROM payloads WHERE hash=?", (event["payload"],)).fetchone()
                check(value is not None and sha(value[0]) == event["payload"], "FORECAST_SEAL_PAYLOAD")
                payload = json.loads(value[0])
                if "forecast" in payload:
                    record = payload["forecast"]
                    key = digest(record)
                    forecasts[key] = record
                    pending[record["request"]["context"]] = key
                if "assessment" in payload:
                    pending.pop(event["request"]["context"])
            head, count = event_pin, count+1
    check(head == body["head"] and count == body["events"] and body["forecasts"], "FORECAST_SEAL_HEAD")
    for key, record in body["forecasts"].items():
        check(key not in enrolled, "FORECAST_SEAL_PRIOR_OUTCOME")
        check(forecasts.get(key) == record and key == digest(record)
              and pending.get(record["request"]["context"]) == key
              and "execution_conditions" in record["request"], "FORECAST_SEAL_RECORD")
    return body


def match(seal, record, entry, observed):
    check(seal["forecasts"].get(digest(record)) == record, "FORECAST_SEAL_RECORD")
    req = record["request"]
    condition = req["execution_conditions"]["native"]
    check(entry["target"]["task"] == req["task"] and entry["target"]["turn"] == req["turn"]+1
          and entry["condition"] == condition["condition"]
          and observed["conditions"]["commitment"] == condition["commitment"], "FORECAST_SEAL_OUTCOME")
    sealed = datetime.fromisoformat(seal["sealed_at"])
    started = datetime.fromisoformat(observed["started"].replace("Z", "+00:00"))
    check(sealed.tzinfo is not None and started.tzinfo is not None and sealed < started,
          "FORECAST_NOT_PREEXECUTION")
