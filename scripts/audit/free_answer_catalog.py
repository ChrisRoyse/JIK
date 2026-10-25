"""Stream cold-replayed whole answers into a bounded native-measurement catalog."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from claims.process_conditions import write_committed as write
from model import free_answer_catalog as catalog
from model.free_surface_atoms import atoms
from text.answer_native_plan import BANK, authority, tokenizer
from text.constellation_corpus import file_sha, need


def token_census(db, contract, output, byte_limit, atom_limit):
    tok = tokenizer(contract); stream = catalog.Lines(output)
    total = dict(total_windows=0, total_tokens=0, max_windows=0,
                 inputs_exceeding_old_eight_window_limit=0, inputs=0, cohorts=0)
    try:
        for batch in catalog.cohorts(db, byte_limit, atom_limit):
            texts = [contract['prefix'] + row['text'] for row in batch]
            encoded = tok(texts, truncation=True, max_length=contract['max_len'], padding=False,
                          return_overflowing_tokens=True)
            windows = [0] * len(batch); tokens = [0] * len(batch)
            for index, ids in zip(encoded['overflow_to_sample_mapping'], encoded['input_ids'], strict=True):
                windows[index] += 1; tokens[index] += len(ids)
            need(all(w > 0 for w in windows), 'FREE_CATALOG_TOKEN_COVERAGE')
            for row, w, n in zip(batch, windows, tokens, strict=True):
                stream.add(dict(id=row['id'], windows=w, active_tokens=n))
                total['total_windows'] += w; total['total_tokens'] += n
                total['max_windows'] = max(total['max_windows'], w)
                total['inputs_exceeding_old_eight_window_limit'] += int(w > 8); total['inputs'] += 1
            total['cohorts'] += 1
    finally:
        stream.close()
    return total


def build(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    code = catalog.code()
    source = catalog.Sources(a.source, a.source_pin, a.readback, a.readback_pin)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    need(0 < a.atom_bytes <= 16 << 20 and 0 < a.cohort_bytes <= 4 << 20
         and max(a.atom_bytes, a.cohort_bytes) * 64 + (8 << 20) <= available // 8,
         'FREE_CATALOG_PHYSICAL_MEMORY')
    phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
    counts = catalog.totals(); lenses = None
    streams = {name: catalog.Lines(a.output / (name + '.jsonl')) for name in ('occurrences', 'refused', 'arms', 'cases')}
    try:
        with catalog.database(a.output / 'inputs.sqlite', create=True) as db:
            for case in source.cases:
                streams['cases'].add(case)
            for arm in source.arms:
                terminal = False
                for row in atoms(arm['path'], arm['source_report'], buffer_bytes=a.atom_bytes):
                    need(not terminal, 'FREE_CATALOG_AFTER_SUMMARY')
                    if row['kind'] == 'summary':
                        need(all(row[k] == v for k, v in arm['summary'].items()), 'FREE_CATALOG_COLD_COUNTS')
                        streams['arms'].add(dict(arm, terminal=row)); catalog.count_arm(counts, row)
                        terminal = True; continue
                    raw = row.pop('raw'); binding = row['binding']
                    need(binding['arm'] == arm['arm'] and binding['trace'] == arm['trace']
                         and (lenses is None or binding['lens_ids'] == lenses), 'FREE_CATALOG_NATIVE_BINDING')
                    lenses = binding['lens_ids']; item = catalog.row_binding(row, arm)
                    if row['kind'] == 'atom':
                        catalog.put(db, row['sha256'], raw); streams['occurrences'].add(item)
                    else:
                        streams['refused'].add(item)
                    catalog.count_atom(counts, row)
                need(terminal, 'FREE_CATALOG_MISSING_SUMMARY'); db.commit()
            catalog.finish_counts(db, counts)
            phases['stream_atoms_and_disk_deduplication'] = time.perf_counter() - tick; tick = time.perf_counter()
            bank, specs, contracts = authority(BANK)
            expected = {n: v['lens_id'] for n, v in bank.config['instruments'].items()}
            # Empty outputs remain real denominator records. Their instrument
            # identity still comes from each authenticated native request.
            for arm in source.arms:
                own = source.get(Path(arm['path']) / 'report.json', arm['source_report'])
                request = source.get(Path(arm['path']) / 'request.json', own['request'])
                native = source.get(Path(request['trace_path']) / 'request.json', request['trace_request'])
                need(native['authority']['lens_ids'] == expected, 'FREE_CATALOG_INSTRUMENT_IDENTITY')
            need(lenses is None or lenses == expected, 'FREE_CATALOG_LENS_IDENTITY')
            write(a.output / 'contracts.json', dict(contracts=contracts, instruments={s.name: s.identity for s in specs},
                                                   lens_ids=expected))
            phases['native_authority'] = time.perf_counter() - tick; tick = time.perf_counter()
            (a.output / 'tokens').mkdir()
            tokens = {name: token_census(db, contracts[name], a.output / 'tokens' / (name + '.jsonl'),
                a.cohort_bytes, a.atom_bytes) for name in bank.config['roster']}
            phases['bounded_complete_token_census'] = time.perf_counter() - tick; tick = time.perf_counter()
    finally:
        for stream in streams.values():
            stream.close()
    source.fresh(); need(not torch.cuda.is_initialized(), 'FREE_CATALOG_CPU_ONLY')
    need(catalog.code() == code, 'FREE_CATALOG_CODE_CHANGED')
    files = [p for p in a.output.rglob('*') if p.is_file()]
    for path in files:
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    refs = {str(p.relative_to(a.output)): dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in files}
    phases['seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_CATALOG_PHASES')
    result = dict(schema=catalog.SCHEMA, complete=True, source=source.source, readback=source.readback,
        scope=source.scope, full_query=source.full_query, full_training=source.full_training,
        declared_cases=source.declared_cases, evaluated_cases=len(source.cases), trajectories=len(source.arms) // 2,
        files=refs, counts=counts, instrument_totals=tokens, native_bank=BANK, code=code,
        memory=dict(atom_bytes=a.atom_bytes, cohort_bytes=a.cohort_bytes, sqlite_cache_bytes=8 << 20,
                    available_before=available, population_text_resident=False),
        measurements_pending=True, outcome_checks_pending=True, all_native_windows_planned=True,
        input_truncation=False, reference_visible=False, encoder_calls=0, optimizer_updates=0, gpu_used=False,
        scientific_admission=False, serving=False, capability_demonstrated=False, semantic_validity_established=False,
        phases=phases, seconds=seconds, rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, counts=counts, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'readback', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('source-pin', 'readback-pin'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--atom-bytes', type=int, default=16 << 20)
    p.add_argument('--cohort-bytes', type=int, default=4 << 20)
    a = p.parse_args(); torch.set_num_threads(1)
    try:
        build(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
