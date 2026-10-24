"""Seal a controlled field forecast before execution; read without loading a model."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from claims.codata_source import check, sha
from claims.episodic_memory import ZERO, canonical, digest
from claims.process_conditions import write_committed
from production.publication import sync_dir


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {name: sha((root/name).read_bytes()) for name in (
        'claims/field_forecast.py', 'model/field_experiment.py', 'oracle/field_experiment.py')}


def publish(session, root, forecast):
    session._fresh()
    root = Path(root)
    check(not root.exists() and forecast['payload'] in session.experiments, 'FIELD_SEAL_STATE')
    check(session.experiments[forecast['payload']]['status'] == 'pending', 'FIELD_SEAL_STATE')
    root.mkdir()
    path = root/'journal.sqlite3'
    with sqlite3.connect(path) as db:
        session.journal.db.backup(db)
    with path.open('rb') as f:
        os.fsync(f.fileno())
    seal = dict(schema='controlled-field-forecast-v1', head=session.head,
        contract=session.journal.pin, events=session.journal.count, forecast=forecast,
        journal=sha(path.read_bytes()), committed_at=datetime.now(timezone.utc).isoformat(),
        clock='host_utc_after_durable_forecast_before_execute', code=code_pins(),
        known_program=True, scientific_admission=False, serving=False)
    pin = write_committed(root/'forecast.json', seal)
    with (root/'COMMITTED').open('x') as f:
        f.write(pin)
        f.flush()
        os.fsync(f.fileno())
    sync_dir(root)
    read(root, pin)
    return pin


def read(root, pin):
    root = Path(root)
    raw = (root/'forecast.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(), 'FIELD_FORECAST_PIN')
    seal = json.loads(raw)
    check(raw == canonical(seal) and seal['schema'] == 'controlled-field-forecast-v1'
          and seal['code'] == code_pins() and seal['known_program'] is True
          and seal['scientific_admission'] is False and seal['serving'] is False,
          'FIELD_FORECAST_SCOPE')
    path = root/'journal.sqlite3'
    check(sha(path.read_bytes()) == seal['journal'], 'FIELD_FORECAST_JOURNAL_BYTES')
    head, count, last = ZERO, 0, None
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        contract, = db.execute('SELECT contract FROM metadata').fetchone()
        check(sha(contract) == seal['contract'], 'FIELD_FORECAST_CONTRACT')
        for seq, previous, body, actual in db.execute('SELECT * FROM events ORDER BY seq'):
            value = json.loads(body)
            check(seq == count and previous == head and canonical(value) == body
                  and digest(dict(seq=seq, previous=previous, body=value)) == actual,
                  'FIELD_FORECAST_CHAIN')
            count, head, last = count+1, actual, value
        check(head == seal['head'] and count == seal['events'] and last == seal['forecast']
              and last['kind'] == 'field_experiment_forecast', 'FIELD_FORECAST_LAST_EVENT')
        payload, = db.execute('SELECT data FROM payloads WHERE hash=?', (last['payload'],)).fetchone()
        check(sha(payload) == last['payload'], 'FIELD_FORECAST_PAYLOAD')
    return seal, json.loads(payload)


def match(seal, payload, condition, observed):
    check(condition in payload['predictions']
          and observed['conditions']['condition'] == condition
          and observed['conditions']['commitment'] == payload['conditions']['pin'],
          'FIELD_FORECAST_CONDITION')
    start = datetime.fromisoformat(observed['started'].replace('Z', '+00:00'))
    committed = datetime.fromisoformat(seal['committed_at'])
    check(start > committed, 'FIELD_FORECAST_LATE')
    for row in payload['predictions'][condition]['candidate']['bank']['rows']:
        if row['observation'] is not None:
            check(datetime.fromisoformat(row['finished'].replace('Z', '+00:00')) < committed,
                  'FIELD_FORECAST_FUTURE_BANK')
    return (start-committed).total_seconds()
