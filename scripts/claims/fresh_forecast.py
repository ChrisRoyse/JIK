"""Durable forecasts of a recorded proposal's future, scoped to fixed context."""
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
    return {n:sha((root/n).read_bytes()) for n in (
        'claims/fresh_forecast.py', 'claims/fresh_observation.py',
        'model/fresh_experiment.py', 'model/fresh_input.py',
        'oracle/fresh_experiment.py', 'audit/corrective_open.py',
        'audit/fresh_experiment.py', 'claims/proposal_observation.py',
        'claims/proposal_forecast.py', 'model/proposal_input.py')}


def publish(session, root, forecast):
    session._fresh()
    root = Path(root)
    check(not root.exists() and session.fresh_proposals[forecast['payload']]['status'] == 'pending',
          'PROPOSAL_SEAL_STATE')
    root.mkdir()
    path = root/'journal.sqlite3'
    with sqlite3.connect(path) as db:
        session.journal.db.backup(db)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    seal = dict(schema='fresh-forecast-v1', head=session.head, contract=session.journal.pin,
        events=session.journal.count, forecast=forecast, journal=sha(path.read_bytes()),
        committed_at=datetime.now(timezone.utc).isoformat(), code=code_pins(),
        clock='host_utc_after_durable_forecast_before_execute', scientific_admission=False,
        serving=False, exposure_closure_complete=False, fixed_historical_context=True)
    pin = write_committed(root/'forecast.json', seal)
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(root)
    sync_dir(root.parent)
    read(root, pin)
    return pin


def read(root, pin):
    root = Path(root)
    raw = (root/'forecast.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(), 'PROPOSAL_FORECAST_PIN')
    seal = json.loads(raw)
    check(raw == canonical(seal) and seal['schema'] == 'fresh-forecast-v1'
          and seal['code'] == code_pins() and seal['fixed_historical_context'] is True
          and all(seal[k] is False for k in ('scientific_admission','serving','exposure_closure_complete')),
          'PROPOSAL_FORECAST_SCOPE')
    path = root/'journal.sqlite3'
    check(sha(path.read_bytes()) == seal['journal'], 'PROPOSAL_FORECAST_JOURNAL')
    head, count, last = ZERO, 0, None
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        contract, = db.execute('SELECT contract FROM metadata').fetchone()
        check(sha(contract) == seal['contract'], 'PROPOSAL_FORECAST_CONTRACT')
        for seq, previous, body, actual in db.execute('SELECT * FROM events ORDER BY seq'):
            value = json.loads(body)
            check(seq == count and previous == head and canonical(value) == body
                  and digest(dict(seq=seq, previous=previous, body=value)) == actual,
                  'PROPOSAL_FORECAST_CHAIN')
            if value.get('payload') is not None:
                row = db.execute('SELECT data FROM payloads WHERE hash=?', (value['payload'],)).fetchone()
                check(row is not None, 'PROPOSAL_FORECAST_PAYLOAD_MISSING')
                payload, = row
                check(sha(payload) == value['payload'] and canonical(json.loads(payload)) == payload,
                      'PROPOSAL_FORECAST_PAYLOAD')
            count, head, last = count+1, actual, value
        check(head == seal['head'] and count == seal['events'] and last == seal['forecast']
              and last['kind'] == 'fresh_forecast', 'PROPOSAL_FORECAST_LAST')
        raw_payload, = db.execute('SELECT data FROM payloads WHERE hash=?', (last['payload'],)).fetchone()
    return seal, json.loads(raw_payload)


def match(seal, payload, condition, observed):
    check(condition in payload['predictions'] and observed['conditions']['condition'] == condition
          and observed['conditions']['commitment'] == payload['conditions']['pin'],
          'PROPOSAL_FORECAST_CONDITION')
    start = datetime.fromisoformat(observed['started'].replace('Z', '+00:00'))
    committed = datetime.fromisoformat(seal['committed_at'])
    check(start > committed, 'PROPOSAL_FORECAST_LATE')
    for prediction in payload['predictions'][condition]['models'].values():
        for row in prediction['termination']['bank']['rows']:
            if row['observation'] is not None:
                check(datetime.fromisoformat(row['finished'].replace('Z', '+00:00')) < committed,
                      'PROPOSAL_FORECAST_FUTURE_BANK')
    return (start-committed).total_seconds()
