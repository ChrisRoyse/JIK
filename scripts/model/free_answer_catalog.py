"""Source-bound population enumeration and disk-backed complete-answer atoms.

Only cold-replayed surface outputs are eligible. Completion is a mechanical
property here; neither a finished answer nor a matching vector establishes truth.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3

from model.free_surface_atoms import read
from text.constellation_corpus import canonical, digest, file_sha, need

SCHEMA = 'free-answer-atom-catalog-v1'
BASE = Path(__file__).resolve().parents[1]
NEW = ('model/free_answer_catalog.py', 'audit/free_answer_catalog.py',
       'audit/free_answer_catalog_readback.py')


def code():
    return {name: file_sha(BASE / name) for name in NEW + ('model/free_surface_atoms.py',)}


def child(root, name):
    path = (Path(root) / name).resolve()
    need(path.is_relative_to(Path(root).resolve()) and path != Path(root).resolve(),
         'FREE_CATALOG_CHILD_PATH')
    return path


def selection(case, trace):
    return dict(case=case, **{k: trace[k] for k in ('architecture', 'proposal', 'mode')})


class Sources:
    """Small metadata manifest; generated text is never collected in this object."""
    def __init__(self, source, source_pin, readback, readback_pin):
        self.root = Path(source); self.source = dict(path=str(source), pin=source_pin)
        self.readback = dict(path=str(readback), pin=readback_pin); self.pins = {}
        r = self.get(self.root / 'report.json', source_pin)
        cold = self.get(Path(readback) / 'report.json', readback_pin)
        need(r['complete'] is True and cold['complete'] is True
             and cold['all_free_logits_exact'] is True and cold['all_terminals_recounted'] is True
             and r['reference_visible'] is False and cold['reference_visible'] is False
             and cold['capability_demonstrated'] is False, 'FREE_CATALOG_COLD_SCOPE')
        for name, h in cold['code'].items():
            need(file_sha(BASE / name) == h, 'FREE_CATALOG_COLD_CODE')
        self.cases = []; self.arms = []; self.full_query = False
        self.full_training = cold['full_training']; self.declared_cases = 170
        if cold.get('scope') == 'one_complete_sealed_trajectory_not_full_population':
            need(cold['all_bytes_exact'] is True and cold['source'] == self.source
                 and cold['trace'] == r['trace'], 'FREE_CATALOG_SINGLE_SOURCE')
            self.scope = cold['scope']; self.declared_cases = None
            self.cases.append(dict(case=cold['case'], unavailable=None,
                                   unavailable_scope='not_in_single_trajectory_proof'))
            item = selection(cold['case'], cold['selection'])
            self.add(self.root, source_pin, item, {
                n: v['summary'] for n, v in cold['results'].items()}, cold['results'])
        else:
            need(cold['input_pin'] == source_pin and cold['all_emitted_bytes_exact'] is True
                 and cold['source_traces_drained'] is True and r['free_generation_tested'] is True
                 and r['teacher_forced'] is False and r['capability_demonstrated'] is False
                 and r['reference_open_refused'] is True and r['declared_cases'] == 170
                 and cold['qualification'] == r['qualification']
                 and cold['full_training'] == r['full_training'], 'FREE_CATALOG_POPULATION_SCOPE')
            self.full_query = r['full_query']
            self.scope = 'full_original_task_population' if self.full_query else 'three_case_qualification'
            need(not self.full_query or (r['gpu_used'] and r['full_training'] and not r['qualification']),
                 'FREE_CATALOG_FULL_SCOPE')
            cold_rows = {digest(canonical(selection(v['case'], v))): v for v in cold['results']}
            need(len(cold_rows) == len(cold['results']) == cold['trajectories'], 'FREE_CATALOG_COLD_DUPLICATE')
            ordinals = []; consumed = set()
            for ref in r['results']:
                root = child(self.root, ref['path']); case = self.get(root / 'report.json', ref['pin'])
                need(case['complete'] is True, 'FREE_CATALOG_CASE_INCOMPLETE')
                ordinals.append(case['case']['ordinal'])
                self.cases.append(dict(case=case['case'], unavailable=case['unavailable'],
                                       unavailable_scope='complete_case_record'))
                for trace in case['traces']:
                    item = selection(case['case'], trace); key = digest(canonical(item))
                    need(key in cold_rows and key not in consumed, 'FREE_CATALOG_MISSING_COLD_TRAJECTORY')
                    consumed.add(key)
                    self.add(child(root, trace['path']), trace['pin'], item, cold_rows[key]['arms'])
            need(consumed == set(cold_rows), 'FREE_CATALOG_TRAJECTORY_COVERAGE')
            expected = list(range(170)) if self.full_query else [0, 1, 3]
            need(ordinals == expected and len(ordinals) == r['evaluated_cases'], 'FREE_CATALOG_CASE_COVERAGE')
        ids = [a['id'] for a in self.arms]
        need(ids and len(ids) == len(set(ids)), 'FREE_CATALOG_DUPLICATE_ARM')

    def get(self, path, pin):
        path = Path(path)
        need(str(path) not in self.pins or self.pins[str(path)] == pin, 'FREE_CATALOG_CONFLICTING_PIN')
        result = read(path, pin); self.pins[str(path)] = pin; return result

    def add(self, root, pin, item, summaries, proof=None):
        r = self.get(root / 'report.json', pin)
        need(r['complete'] is True and r['reference_visible'] is False
             and set(r['arms']) == set(summaries) == {'parent', 'corrected'}, 'FREE_CATALOG_ARM_ROSTER')
        for name in ('parent', 'corrected'):
            ref = r['arms'][name]; path = child(root, ref['path'])
            own = self.get(path / 'report.json', ref['pin'])
            request = self.get(path / 'request.json', own['request'])
            need(all(own.get(k) == v for k, v in ref.items() if k not in ('path', 'pin'))
                 and all(own.get(k) == v for k, v in summaries[name].items())
                 and request['arm'] == name and request['trace'] == r['trace'], 'FREE_CATALOG_ARM_BINDING')
            if proof is not None:
                need(proof[name]['report'] == ref['pin'] and proof[name]['request'] == own['request']
                     and proof[name]['model'] == request['decoder'], 'FREE_CATALOG_SINGLE_ARM')
            identity = dict(**item, arm=name, source_report=ref['pin'])
            self.arms.append(dict(id=digest(canonical(identity)), **identity, path=str(path),
                                  summary=summaries[name], trace=r['trace']))

    def fresh(self):
        need(all(file_sha(path) == pin for path, pin in self.pins.items()), 'FREE_CATALOG_SOURCE_CHANGED')


@contextmanager
def database(path, *, create=False):
    path = Path(path)
    need(not create or not path.exists(), 'FREE_CATALOG_DATABASE_EXISTS')
    db = sqlite3.connect(str(path) if create else path.resolve().as_uri() + '?mode=ro', uri=not create)
    try:
        db.execute('PRAGMA cache_size=-8192'); db.execute('PRAGMA mmap_size=0'); db.execute('PRAGMA temp_store=FILE')
        if create:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE inputs (id TEXT PRIMARY KEY, raw BLOB NOT NULL) WITHOUT ROWID')
        yield db
    finally:
        db.close()


def put(db, key, raw):
    need(raw and digest(raw) == key, 'FREE_CATALOG_INPUT_DIGEST')
    raw.decode('utf-8', errors='strict')
    db.execute('INSERT OR IGNORE INTO inputs VALUES (?, ?)', (key, raw))
    need(db.execute('SELECT raw FROM inputs WHERE id=?', (key,)).fetchone()[0] == raw,
         'FREE_CATALOG_HASH_COLLISION')


def records(db):
    for key, raw in db.execute('SELECT id,raw FROM inputs ORDER BY id'):
        need(digest(raw) == key, 'FREE_CATALOG_DATABASE_BYTES')
        yield dict(id=key, text=raw.decode('utf-8'))


def cohorts(db, byte_limit, max_atom_bytes):
    need(type(byte_limit) is int and byte_limit > 0, 'FREE_CATALOG_COHORT_SIZE')
    batch = []; size = 0
    for row in records(db):
        n = len(row['text'].encode('utf-8'))
        need(n <= max_atom_bytes, 'FREE_CATALOG_ATOM_MEMORY')
        if batch and (len(batch) == 32 or size + n > byte_limit):
            yield batch; batch = []; size = 0
        batch.append(row); size += n
    if batch:
        yield batch


class Lines:
    def __init__(self, path):
        self.path = Path(path); self.stream = self.path.open('xb'); self.count = 0

    def add(self, value):
        self.stream.write(canonical(value) + b'\n'); self.count += 1

    def close(self):
        if not self.stream.closed:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()


def lines(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            yield json.loads(line)


def totals():
    return {name: 0 for name in ('completed_atom_occurrences', 'refused_atom_occurrences',
        'complete_bytes', 'refused_bytes', 'unattempted_atoms', 'native_atoms', 'generated_bytes',
        'completed_answers', 'byte_decisions', 'arms')}


def count_arm(counts, summary):
    counts['arms'] += 1
    for name in ('unattempted_atoms', 'native_atoms', 'generated_bytes', 'byte_decisions'):
        counts[name] += summary[name]
    counts['completed_answers'] += int(summary['answer_ended'])


def count_atom(counts, row):
    complete = row['kind'] == 'atom'
    counts['completed_atom_occurrences' if complete else 'refused_atom_occurrences'] += 1
    counts['complete_bytes' if complete else 'refused_bytes'] += row['bytes']


def finish_counts(db, counts):
    counts['unique_measurement_inputs'], counts['unique_input_bytes'] = db.execute(
        'SELECT COUNT(*), COALESCE(SUM(LENGTH(raw)),0) FROM inputs').fetchone()
    counts['exact_duplicate_occurrences'] = counts['completed_atom_occurrences'] - counts['unique_measurement_inputs']
    need(counts['completed_atom_occurrences'] + counts['refused_atom_occurrences'] + counts['unattempted_atoms']
         == counts['native_atoms'] and counts['complete_bytes'] + counts['refused_bytes'] == counts['generated_bytes']
         and counts['exact_duplicate_occurrences'] >= 0, 'FREE_CATALOG_DENOMINATOR')


def row_binding(row, arm):
    return dict(row, catalog_arm=arm['id'], source_surface=str(Path(arm['path']) / 'surface.bin'),
                measurement_input=row['sha256'] if row['kind'] == 'atom' else False)
