"""Bounded native work descriptions; exact cache matches never need new inference."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import numpy as np
import torch
from claims.process_conditions import write_committed as write
from model import free_answer_catalog as catalog
from text.answer_native_plan import authority
from text.constellation_corpus import digest, file_sha, need
from text.legal_reuse_input import ReuseInputs

SCHEMA = 'free-catalog-native-work-v1'
NEW = ('text/free_catalog_native_work.py', 'audit/free_catalog_native_work_readback.py')


def code():
    return {n: file_sha(catalog.BASE / n) for n in (*NEW, *catalog.NEW,
        'text/answer_native_plan.py', 'text/legal_reuse_input.py', 'text/canonical_windows.py')}


def source_catalog(root, pin, cold_root, cold_pin):
    root = Path(root); cold_root = Path(cold_root)
    r = catalog.read(root / 'report.json', pin); cold = catalog.read(cold_root / 'report.json', cold_pin)
    need(r['schema'] == catalog.SCHEMA and r['complete'] is True and cold['complete'] is True
         and cold['input'] == dict(path=str(root), pin=pin) and cold['counts'] == r['counts']
         and cold['source_scope'] == r['scope'] and cold['full_query'] == r['full_query']
         and all(cold[n] is True for n in ('all_raw_spans_exact', 'all_native_joins_exact',
             'all_singleton_token_counts_exact', 'no_orphan_or_missing_inputs', 'population_coverage_exact'))
         and cold['serving'] is False and r['serving'] is False
         and r['capability_demonstrated'] is False and r['measurements_pending'] is True,
         'FREE_NATIVE_WORK_SOURCE')
    for n, h in cold['code'].items():
        need(file_sha(catalog.BASE / n) == h, 'FREE_NATIVE_WORK_SOURCE_CODE')
    for n, ref in r['files'].items():
        path = catalog.child(root, n)
        need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'FREE_NATIVE_WORK_SOURCE_FILE')
    return r, cold


@contextmanager
def cache(root, pin):
    source = ReuseInputs(root, pin)
    try:
        yield source
        source.fresh()
    finally:
        source.table.close(); source.cached = None
        for value in source.arrays.values():
            value._mmap.close()


def lookup(source, rows):
    """Hash index lookup followed by exact bytes; a digest alone cannot authorize reuse."""
    if not rows:
        return []
    keys = np.frombuffer(b''.join(bytes.fromhex(v['id']) for v in rows), dtype='V32')
    positions = np.searchsorted(source.arrays['hashes'], keys); found = []
    for row, key, pos in zip(rows, keys, positions, strict=True):
        need(digest(row['text'].encode('utf-8')) == row['id'], 'FREE_NATIVE_WORK_TEXT')
        uid = None
        if pos < len(source.arrays['hashes']) and source.arrays['hashes'][pos] == key:
            need(source.input(int(pos))['text'].encode('utf-8') == row['text'].encode('utf-8'),
                 'FREE_NATIVE_WORK_CACHE_BYTES')
            uid = int(pos)
        found.append(uid)
    return found


@contextmanager
def work_db(path, *, create=False):
    path = Path(path); need(not create or not path.exists(), 'FREE_NATIVE_WORK_EXISTS')
    db = sqlite3.connect(str(path) if create else path.resolve().as_uri() + '?mode=ro', uri=not create)
    try:
        db.execute('PRAGMA cache_size=-8192'); db.execute('PRAGMA mmap_size=0'); db.execute('PRAGMA temp_store=FILE')
        if create:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE mapping (ordinal INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, bytes INTEGER NOT NULL, uid INTEGER)')
            db.execute('CREATE TABLE tokens (instrument TEXT, ordinal INTEGER, windows INTEGER NOT NULL, tokens INTEGER NOT NULL, PRIMARY KEY(instrument,ordinal)) WITHOUT ROWID')
        yield db
    finally:
        db.close()


def entries(db, name):
    for i, key, size, uid, windows, tokens in db.execute(
        'SELECT m.ordinal,m.id,m.bytes,m.uid,t.windows,t.tokens FROM mapping m JOIN tokens t USING(ordinal) '
        'WHERE t.instrument=? ORDER BY m.ordinal', (name,)):
        yield dict(ordinal=i, id=key, bytes=size, uid=uid, windows=windows, active_tokens=tokens)


def batches(rows, *, max_inputs, text_bytes, vector_bytes, dims):
    group = []; size = windows = 0
    for row in rows:
        need(row['windows'] > 0 and row['active_tokens'] > 0
             and row['windows'] * dims * 4 <= vector_bytes, 'FREE_NATIVE_WORK_SINGLE_INPUT_WINDOW_TILING_REQUIRED')
        if group and (len(group) == max_inputs or size + row['bytes'] > text_bytes
                      or (windows + row['windows']) * dims * 4 > vector_bytes):
            yield group; group = []; size = windows = 0
        group.append(row); size += row['bytes']; windows += row['windows']
    if group:
        yield group


def build(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); pins = code()
    source, cold = source_catalog(a.catalog, a.catalog_pin, a.readback, a.readback_pin)
    need(type(a.max_inputs) is int and 0 < a.max_inputs <= 32 and 0 < a.text_bytes <= 4 << 20
         and 0 < a.vector_bytes <= 64 << 20, 'FREE_NATIVE_WORK_PHYSICAL_POLICY')
    bank, specs, contracts = authority(source['native_bank']); dims = {s.name: s.dims for s in specs}
    identity = json.loads((a.catalog / 'contracts.json').read_bytes())
    expected = {n: v['lens_id'] for n, v in bank.config['instruments'].items()}
    need(identity['contracts'] == contracts and identity['lens_ids'] == expected, 'FREE_NATIVE_WORK_IDENTITY')
    phases = dict(catalog_and_instrument_authentication=time.perf_counter() - tick); tick = time.perf_counter()
    totals = {}; reused = missing = count = 0
    with work_db(a.output / 'mapping.sqlite', create=True) as db, catalog.database(a.catalog / 'inputs.sqlite') as inputs:
        with cache(bank.config['source']['path'], bank.config['source']['pin']) as existing:
            source_bytes = sum(v['bytes'] for v in existing.report['files'].values())
            source_counts = existing.report['counts']
            phases['single_canonical_cache_authentication'] = time.perf_counter() - tick; tick = time.perf_counter()
            for batch in catalog.cohorts(inputs, a.text_bytes, source['memory']['atom_bytes']):
                for row, uid in zip(batch, lookup(existing, batch), strict=True):
                    db.execute('INSERT INTO mapping VALUES (?,?,?,?)', (count, row['id'], len(row['text'].encode('utf-8')), uid))
                    count += 1; reused += int(uid is not None); missing += int(uid is None)
                db.commit()
            # An actual canonical input is a positive probe; it is never added to the generated population.
            probe = existing.input(0); probe_uid = lookup(existing, [dict(id=probe['text_sha256'], text=probe['text'])])
            need(probe_uid == [0], 'FREE_NATIVE_WORK_CACHE_POSITIVE')
            positive = dict(uid=0, input_sha256=probe['text_sha256'], bytes=len(probe['text'].encode()),
                            included_in_generated_population=False)
        need(count == source['counts']['unique_measurement_inputs'] and count == reused + missing, 'FREE_NATIVE_WORK_INPUTS')
        phases['logarithmic_lookup_and_exact_cached_bytes'] = time.perf_counter() - tick; tick = time.perf_counter()
        (a.output / 'work').mkdir()
        for name in bank.config['roster']:
            n = 0; windows = tokens = 0
            for row in catalog.lines(a.catalog / 'tokens' / (name + '.jsonl')):
                found = db.execute('SELECT id FROM mapping WHERE ordinal=?', (n,)).fetchone()
                need(found == (row['id'],) and row['windows'] > 0 and row['active_tokens'] > 0,
                     'FREE_NATIVE_WORK_TOKEN_JOIN')
                db.execute('INSERT INTO tokens VALUES (?,?,?,?)', (name, n, row['windows'], row['active_tokens']))
                n += 1; windows += row['windows']; tokens += row['active_tokens']
            need(n == count and windows == source['instrument_totals'][name]['total_windows']
                 and tokens == source['instrument_totals'][name]['total_tokens'], 'FREE_NATIVE_WORK_TOKEN_TOTAL')
            db.commit(); stream = catalog.Lines(a.output / 'work' / (name + '.jsonl')); number = 0
            try:
                for group in batches(entries(db, name), max_inputs=a.max_inputs, text_bytes=a.text_bytes,
                                     vector_bytes=a.vector_bytes, dims=dims[name]):
                    stream.add(dict(batch=number, instrument=name, lens_id=expected[name], inputs=group,
                        source_bytes=sum(v['bytes'] for v in group), windows=sum(v['windows'] for v in group),
                        active_tokens=sum(v['active_tokens'] for v in group),
                        vector_payload_bytes=sum(v['windows'] for v in group) * dims[name] * 4))
                    number += 1
            finally:
                stream.close()
            totals[name] = dict(inputs=n, reused=reused, missing=missing, windows=windows,
                                active_tokens=tokens, batches=number, dims=dims[name], lens_id=expected[name])
        phases['persist_bounded_work_and_all_window_counts'] = time.perf_counter() - tick; tick = time.perf_counter()
    need(code() == pins and file_sha(a.catalog / 'report.json') == a.catalog_pin
         and file_sha(a.readback / 'report.json') == a.readback_pin
         and all(file_sha(a.catalog / n) == ref['sha256'] for n, ref in source['files'].items()), 'FREE_NATIVE_WORK_CHANGED')
    need(not torch.cuda.is_initialized(), 'FREE_NATIVE_WORK_CPU_ONLY')
    files = [p for p in a.output.rglob('*') if p.is_file()]
    for path in files:
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    refs = {str(p.relative_to(a.output)): dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in files}
    phases['seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_NATIVE_WORK_PHASES')
    result = dict(schema=SCHEMA, complete=True, catalog=dict(path=str(a.catalog), pin=a.catalog_pin),
        catalog_readback=dict(path=str(a.readback), pin=a.readback_pin), source_counts=source['counts'],
        source_scope=source['scope'], full_query=source['full_query'], code=pins, files=refs, native_bank=source['native_bank'],
        cache_source=bank.config['source'], cache_source_counts=source_counts, cache_authenticated_bytes=source_bytes,
        cache_authentications=1, inputs=count, reused=reused, missing=missing, positive_probe=positive, instruments=totals,
        policy=dict(max_inputs=a.max_inputs, text_bytes=a.text_bytes, vector_bytes=a.vector_bytes,
                    sqlite_cache_bytes=8 << 20, individual_atom_bytes=source['memory']['atom_bytes']),
        population_text_copied=False, all_native_windows_planned=True, worker_peak_memory_not_established=True,
        native_measurement_pending=True, outcome_checks_pending=True, serving=False, scientific_admission=False,
        capability_demonstrated=False, encoder_calls=0, gpu_used=False, phases=phases, seconds=seconds,
        acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, inputs=count, reused=reused, missing=missing, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('catalog', 'readback', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    for n in ('catalog-pin', 'readback-pin'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--max-inputs', type=int, default=32); p.add_argument('--text-bytes', type=int, default=4 << 20)
    p.add_argument('--vector-bytes', type=int, default=64 << 20)
    a = p.parse_args(); torch.set_num_threads(1)
    try:
        build(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
