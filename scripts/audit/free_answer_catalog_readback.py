"""Independently recount streamed full-answer catalogs from raw persisted spans."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from claims.process_conditions import write_committed as write
from model import free_answer_catalog as catalog
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.answer_native_plan import authority, tokenizer
from text.canonical_windows import tokenize
from text.constellation_corpus import digest, file_sha, need


def recount(arm, atom_limit):
    """Separate index/raw-byte reader; does not call free_surface_atoms.atoms."""
    root = Path(arm['path']); own = catalog.read(root / 'report.json', arm['source_report'])
    for name, pin in own['files'].items():
        need(file_sha(root / name) == pin, 'FREE_CATALOG_COLD_SOURCE_FILE')
    request = catalog.read(root / 'request.json', own['request'])
    path = Path(request['trace_path'])
    native = catalog.read(path / 'report.json', request['trace'])
    source = catalog.read(path / 'request.json', request['trace_request'])
    predictions = iter(replay(path, request['trace']))
    attempted = completed = decisions = consumed = size = 0; body = None; refusal = None
    with (root / 'surface.bin').open('rb') as raw:
        for row in catalog.lines(root / 'events.jsonl'):
            decisions += int('logits' in row)
            if row['kind'] == 'atom_start':
                need(body is None and refusal is None and row['atom'] == attempted, 'FREE_CATALOG_COLD_ORDER')
                prediction = next(predictions); consumed += 1
                need(prediction['kind'] == 'constellation' and prediction['index'] == attempted,
                     'FREE_CATALOG_COLD_NATIVE_ORDER')
                region = request['region_pin'] + ':' + '.'.join(map(str, prediction['codes'][0].tolist()))
                need(row['regions'] == [region], 'FREE_CATALOG_COLD_REGION')
                binding = dict(arm=request['arm'], source_report=arm['source_report'], source_request=own['request'],
                    trace=request['trace'], trace_path=str(path), atom=attempted, prediction=fingerprint(prediction),
                    decoder=request['decoder'], input_native=row['native'], region=region,
                    lens_ids=source['authority']['lens_ids'])
                attempted += 1; body = bytearray(); offset = raw.tell()
            elif row['kind'] == 'byte':
                need(body is not None and len(body) < atom_limit and row['atom'] == attempted - 1
                     and row['position'] == len(body), 'FREE_CATALOG_COLD_BYTE_ORDER')
                value = raw.read(1); need(value == bytes([row['value']]), 'FREE_CATALOG_COLD_RAW_BYTE')
                body.extend(value)
            else:
                need(row['kind'] in ('atom_end', 'refuse') and body is not None and row['atom'] == attempted - 1
                     and row['bytes'] == len(body) and row['sha256'] == digest(body), 'FREE_CATALOG_COLD_SPAN')
                done = row['kind'] == 'atom_end'
                if done:
                    need(body and row['token'] == 257, 'FREE_CATALOG_COLD_EOS')
                    bytes(body).decode('utf-8'); completed += 1
                else:
                    refusal = row['code']
                yield dict(kind='atom' if done else 'refused_atom', offset=offset, bytes=len(body),
                    sha256=digest(body), refusal=None if done else refusal, binding=binding,
                    source_surface=str(root / 'surface.bin'), measurement_input=digest(body) if done else False,
                    catalog_arm=arm['id']), bytes(body)
                size += len(body); body = None
        need(body is None and raw.read(1) == b'', 'FREE_CATALOG_COLD_RAW_TRAILING')
    for _ in predictions:
        consumed += 1
    need(consumed == native['events'], 'FREE_CATALOG_COLD_NATIVE_COVERAGE')
    summary = dict(kind='summary', attempted_atoms=attempted, completed_atoms=completed, generated_bytes=size,
        byte_decisions=decisions, surface_refusal=refusal, native_events=native['events'],
        native_atoms=native['generated_atoms'], native_ended=native['model_ended'], native_status=native['status'],
        native_refusal=native['refusal'], answer_ended=native['model_ended'] and refusal is None
            and completed == native['generated_atoms'], unattempted_atoms=native['generated_atoms'] - attempted,
        accepted=False, serving=False, semantic_validity_established=False)
    need(all(summary[k] == v for k, v in arm['summary'].items()), 'FREE_CATALOG_COLD_TERMINAL')
    yield summary, None


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    plan = catalog.read(a.input / 'report.json', a.pin)
    need(plan['schema'] == catalog.SCHEMA and plan['complete'] is True
         and plan['measurements_pending'] is True and plan['outcome_checks_pending'] is True
         and plan['all_native_windows_planned'] is True and plan['input_truncation'] is False
         and plan['encoder_calls'] == plan['optimizer_updates'] == 0 and plan['serving'] is False
         and plan['reference_visible'] is False and plan['capability_demonstrated'] is False
         and plan['semantic_validity_established'] is False, 'FREE_CATALOG_COLD_SCOPE')
    for name, ref in plan['files'].items():
        path = catalog.child(a.input, name)
        need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'FREE_CATALOG_COLD_FILE')
    for name, pin in plan['code'].items():
        need(file_sha(catalog.BASE / name) == pin, 'FREE_CATALOG_COLD_CODE')
    identity = json.loads((a.input / 'contracts.json').read_bytes())
    source = catalog.Sources(plan['source']['path'], plan['source']['pin'], plan['readback']['path'], plan['readback']['pin'])
    need(plan['scope'] == source.scope and plan['full_query'] == source.full_query
         and plan['full_training'] == source.full_training and plan['declared_cases'] == source.declared_cases
         and plan['evaluated_cases'] == len(source.cases) and plan['trajectories'] == len(source.arms) // 2,
         'FREE_CATALOG_COLD_POPULATION')
    need(list(catalog.lines(a.input / 'cases.jsonl')) == source.cases, 'FREE_CATALOG_COLD_CASES')
    phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
    counts = catalog.totals(); seen = sqlite3.connect(a.output / 'seen.sqlite')
    try:
        seen.execute('PRAGMA cache_size=-8192'); seen.execute('PRAGMA temp_store=FILE')
        seen.execute('CREATE TABLE seen (id TEXT PRIMARY KEY) WITHOUT ROWID')
        with catalog.database(a.input / 'inputs.sqlite') as db, ExitStack() as stack:
            streams = {name: catalog.lines(a.input / (name + '.jsonl')) for name in ('occurrences', 'refused', 'arms')}
            for stream in streams.values():
                stack.callback(stream.close)
            for arm in source.arms:
                saved_arm = next(streams['arms'], None)
                need(saved_arm is not None and {k: saved_arm[k] for k in arm} == arm, 'FREE_CATALOG_COLD_ARM')
                terminal = False
                for row, raw in recount(arm, plan['memory']['atom_bytes']):
                    if row['kind'] == 'summary':
                        need(row == saved_arm['terminal'], 'FREE_CATALOG_COLD_SAVED_TERMINAL')
                        catalog.count_arm(counts, row); terminal = True; continue
                    complete = row['kind'] == 'atom'
                    need(row['binding']['lens_ids'] == identity['lens_ids'], 'FREE_CATALOG_COLD_ROW_LENSES')
                    need(next(streams['occurrences' if complete else 'refused'], None) == row,
                         'FREE_CATALOG_COLD_OCCURRENCE')
                    if complete:
                        found = db.execute('SELECT raw FROM inputs WHERE id=?', (row['sha256'],)).fetchone()
                        need(found == (raw,), 'FREE_CATALOG_COLD_INPUT_BYTES')
                        seen.execute('INSERT OR IGNORE INTO seen VALUES (?)', (row['sha256'],))
                    catalog.count_atom(counts, row)
                need(terminal, 'FREE_CATALOG_COLD_NO_TERMINAL'); seen.commit()
            need(all(next(stream, None) is None for stream in streams.values()), 'FREE_CATALOG_COLD_EXTRA_ROWS')
            catalog.finish_counts(db, counts)
            need(seen.execute('SELECT COUNT(*) FROM seen').fetchone()[0] == counts['unique_measurement_inputs'],
                 'FREE_CATALOG_COLD_ORPHAN_INPUT')
            need(counts == plan['counts'], 'FREE_CATALOG_COLD_COUNTS')
            phases['independent_raw_spans_and_native_joins'] = time.perf_counter() - tick; tick = time.perf_counter()
            bank, specs, contracts = authority(plan['native_bank'])
            need(identity == dict(contracts=contracts, instruments={s.name: s.identity for s in specs},
                lens_ids={n: v['lens_id'] for n, v in bank.config['instruments'].items()}), 'FREE_CATALOG_COLD_IDENTITY')
            recomputed = {}
            for name, contract in contracts.items():
                lens = SimpleNamespace(kind='dense', tok=tokenizer(contract),
                    spec=dict(prefix=contract['prefix'], max_len=contract['max_len']))
                totals = dict(total_windows=0, total_tokens=0, max_windows=0,
                              inputs_exceeding_old_eight_window_limit=0, inputs=0)
                saved = catalog.lines(a.input / 'tokens' / (name + '.jsonl'))
                try:
                    for record in catalog.records(db):
                        _, _, traces = tokenize(lens, [record['text']])
                        windows = traces[0]['windows']; n = sum(sum(w['attention_mask']) for w in windows)
                        need(next(saved, None) == dict(id=record['id'], windows=len(windows), active_tokens=n),
                             'FREE_CATALOG_COLD_TOKEN')
                        totals['total_windows'] += len(windows); totals['total_tokens'] += n
                        totals['max_windows'] = max(totals['max_windows'], len(windows))
                        totals['inputs_exceeding_old_eight_window_limit'] += int(len(windows) > 8); totals['inputs'] += 1
                    need(next(saved, None) is None, 'FREE_CATALOG_COLD_EXTRA_TOKEN')
                finally:
                    saved.close()
                need(totals == {k: plan['instrument_totals'][name][k] for k in totals}, 'FREE_CATALOG_COLD_TOKEN_TOTAL')
                recomputed[name] = totals
    finally:
        seen.close()
    phases['canonical_singleton_token_readback'] = time.perf_counter() - tick; tick = time.perf_counter()
    source.fresh()
    need(all(file_sha(a.input / name) == ref['sha256'] for name, ref in plan['files'].items())
         and catalog.code() == plan['code'], 'FREE_CATALOG_COLD_INPUT_CHANGED')
    need(file_sha(a.input / 'report.json') == a.pin and not torch.cuda.is_initialized(),
                         'FREE_CATALOG_COLD_FINAL')
    phases['seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_CATALOG_COLD_PHASES')
    result = dict(complete=True, input=dict(path=str(a.input), pin=a.pin), code=catalog.code(), counts=counts,
        all_raw_spans_exact=True, all_native_joins_exact=True, all_singleton_token_counts_exact=True,
        no_orphan_or_missing_inputs=True, population_coverage_exact=True, instrument_totals=recomputed,
        source_scope=source.scope, full_query=source.full_query, measurements_pending=True, outcome_checks_pending=True,
        gpu_used=False, encoder_calls=0, serving=False, capability_demonstrated=False,
        semantic_validity_established=False, phases=phases, seconds=seconds, acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, counts=counts, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args(); torch.set_num_threads(1)
    try:
        run(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
