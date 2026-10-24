"""Read three intake boundaries and deliberately corrupt disposable journal copies."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_journal_reader import JournalReader, read
from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed


def altered_copy(source, output, mutate, *, delete_payload=False):
    """Rehash the successor tail so semantic controls cannot pass on hashes alone."""
    shutil.copyfile(source, output)
    with sqlite3.connect(output) as db:
        rows = db.execute('SELECT seq,previous,body,hash FROM events ORDER BY seq').fetchall()
        changed, previous = False, None
        for seq, parent, raw, old_pin in rows:
            body = json.loads(raw)
            if body['kind'] == 'capability_intake' and not changed:
                payload_raw = db.execute('SELECT data FROM payloads WHERE hash=?', (body['payload'],)).fetchone()[0]
                if delete_payload:
                    db.execute('DELETE FROM payloads WHERE hash=?', (body['payload'],))
                    return
                payload = json.loads(payload_raw)
                mutate(payload)
                new_raw = canonical(payload)
                body['payload'] = sha(new_raw)
                db.execute('INSERT INTO payloads VALUES (?,?)', (body['payload'], new_raw))
                changed = True
            if changed:
                if 'prior_head' in body: body['prior_head'] = previous
                pin = digest(dict(seq=seq, previous=previous, body=body))
                db.execute('UPDATE events SET previous=?,body=?,hash=? WHERE seq=?',
                           (previous, canonical(body), pin, seq))
                previous = pin
            else:
                previous = old_pin
        need(changed, 'READBACK_NO_SUCCESSOR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    config, proofs = read(a.root / 'config.json'), read(a.proofs)
    journal = a.root / 'experience.sqlite3'
    before = sha(journal.read_bytes())
    reader = JournalReader(config, proofs)
    result = reader.read(journal)
    snapshot = read(a.root / 'snapshot-final.json')
    memory = snapshot['capability_memory']
    need((result['head'], result['events'], result['memory_head'], result['states']) ==
         (snapshot['head'], snapshot['events'], snapshot['memory']['head'], snapshot['memory']['states']),
         'READBACK_SNAPSHOT_STATE')
    for name in ('tasks', 'fields', 'relationships'):
        need(result[name] == memory[name], 'READBACK_SNAPSHOT_' + name)
    need(sorted(result['tasks']) == memory['eligible_tasks'], 'READBACK_FINAL_ELIGIBILITY')
    counts = {family: dict(tasks=sum(t['family'] == family for t in result['tasks'].values()),
        intakes=sum(t['family'] == family for t in result['intakes']))
        for family in ('coding', 'conversation', 'reasoning')}
    need(all(v['intakes'] == 2 and v['tasks'] > 0 for v in counts.values()), 'READBACK_FAMILY_COVERAGE')
    stages = {s: read(a.root / (s + '-stage.json')) for s in ('intake', 'cold')}
    need(all(s['journal'] == before and s['snapshot'] == digest(snapshot) and s['legacy_retained']
             for s in stages.values()), 'READBACK_COLD_PROOF')
    write_committed(a.output / 'state-readback.json', dict(counts=counts, head=result['head'],
        memory_head=result['memory_head'], intakes=result['intakes'], invalidations=result['invalidations'],
        reviews=result['reviews'], snapshot=digest(snapshot)))
    controls = []

    def target(payload): return next(iter(payload['tasks'].values()))['selections']
    def missing(payload): target(payload)['targets']['fields'] = []
    def swapped(payload):
        s = target(payload)
        s['context']['fields'], s['targets']['fields'] = s['targets']['fields'], s['context']['fields']
    def future(payload): target(payload)['context']['fields'].extend(target(payload)['targets']['fields'])
    def lost(payload): payload['relationships'].pop()
    def vectors(payload):
        field = next(iter(payload['fields'].values()))
        next(iter(field['slots'].values()))['sha256'] = '0' * 64
    def source(payload): next(iter(payload['fields'].values()))['episode'] = '0' * 64

    for name, mutate, expected in (
        ('missing_target', missing, 'READER_INPUT_TARGET_MASK'),
        ('swapped_roles', swapped, 'READER_INPUT_TARGET_MASK'),
        ('future_answer', future, 'READER_INPUT_TARGET_MASK'),
        ('lost_relationship', lost, 'READER_LOST_RELATIONSHIP'),
        ('changed_native_reading', vectors, 'READER_NATIVE_BYTES'),
        ('wrong_source_episode', source, 'READER_FIELD_EPISODE'),
        ('missing_payload_bytes', lambda _: None, 'REPOSITORY_READER_PAYLOAD')):
        path = a.output / (name + '.sqlite3')
        altered_copy(journal, path, mutate, delete_payload=name == 'missing_payload_bytes')
        try: reader.read(path)
        except ValueError as exc:
            need(expected in str(exc), 'READBACK_CONTROL_ERROR:' + str(exc))
            code = str(exc)
        else: raise AssertionError('READBACK_ACCEPTED:' + name)
        controls.append(dict(name=name, path=str(path), sha256=sha(path.read_bytes()), error=code,
            hashes_recomputed=name != 'missing_payload_bytes', source_before=before,
            source_after=sha(journal.read_bytes())))
    need(all(c['source_after'] == before for c in controls), 'READBACK_SOURCE_CHANGED')
    report = dict(schema='capability-journal-readback-v1', counts=counts,
        journal=before, snapshot=digest(snapshot), parent=config['source_pin'],
        events=result['events'], episodes_added=len(result['states']) - len(reader.old['memory']['states']),
        task_records=len(result['tasks']), native_field_records=len(result['fields']),
        relationships=len(result['relationships']), controls=controls,
        source_native_episode_joins=True, execution_episodes_connected=True, unchanged_parent_prefix=True,
        cold_replay_equal=True, number_status='measured', acceptance='author-checked',
        kernel_ready=False, learned_capability=False, scientific_admission=False, serving=False,
        wall_seconds=time.perf_counter() - start)
    pin = write_committed(a.output / 'report.json', report)
    print(json.dumps(dict(pin=pin, counts=counts, controls=len(controls), events=result['events'])), flush=True)


if __name__ == '__main__': main()
