"""Physical journal corruption and re-signed stale-publication controls on retained copies."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_live_readback import LiveReader
from audit.repository_memory_readback import chain
from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed


def read(path): return json.loads(Path(path).read_bytes())


def mutate(path, kind):
    with sqlite3.connect(path) as db:
        seq, previous, raw = db.execute(
            'SELECT seq,previous,body FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        body = json.loads(raw)
        need(body['kind'] == 'task_kernel_publish', 'TASK_CONTROL_LAST_EVENT')
        old_pin = body['payload']
        payload_raw = db.execute('SELECT data FROM payloads WHERE hash=?', (old_pin,)).fetchone()[0]
        if kind == 'payload_corruption':
            changed = bytearray(payload_raw); changed[-1] ^= 1
            db.execute('UPDATE payloads SET data=? WHERE hash=?', (bytes(changed), old_pin))
        else:
            payload = json.loads(payload_raw)
            # A valid chain with a stale source must still fail semantic readback.
            need(kind == 'resigned_stale_publication', 'TASK_CONTROL_KIND')
            payload['source']['ids'] = payload['source']['ids'][1:]
            body['payload'] = digest(payload)
            db.execute('INSERT INTO payloads VALUES (?,?)', (body['payload'], canonical(payload)))
            pin = digest(dict(seq=seq, previous=previous, body=body))
            db.execute('UPDATE events SET body=?,hash=? WHERE seq=?', (canonical(body), pin, seq))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); config, proofs = read(a.config), read(a.proofs)
    need(not a.output.exists(), 'TASK_CONTROL_EXISTS'); a.output.mkdir()
    journal = Path(config['root'])/'experience.sqlite3'
    before = sha(journal.read_bytes())
    cases = []
    for name, expected in [('payload_corruption', 'REPOSITORY_READER_PAYLOAD'),
                           ('resigned_stale_publication', 'TASK_READER_PUBLICATION_SOURCE')]:
        path = a.output/(name+'.sqlite3'); shutil.copyfile(journal, path)
        mutate(path, name)
        if name.startswith('resigned'): chain(path)
        try: LiveReader(config, proofs).run(path)
        except ValueError as exc:
            need(expected in str(exc), 'TASK_CONTROL_WRONG_REFUSAL:'+str(exc))
            cases.append(dict(name=name, error=str(exc), bytes=path.stat().st_size,
                              sha256=sha(path.read_bytes()), source=before))
        else: raise AssertionError('TASK_CONTROL_ACCEPTED:'+name)
    need(sha(journal.read_bytes()) == before, 'TASK_CONTROL_SOURCE_CHANGED')
    result = dict(controls=cases, journal_unchanged=True, journal=before,
                  number_status='measured', acceptance='author-checked',
                  scope='independent_reader_not_model_consumer')
    pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(pin=pin, controls=len(cases))), flush=True)


if __name__ == '__main__': main()
