"""Independent scalar cache search and complete batch census for native work."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from claims.process_conditions import write_committed as write
from model import free_answer_catalog as catalog
from text import free_catalog_native_work as work
from text.answer_native_plan import authority
from text.constellation_corpus import file_sha, need


def scalar_lookup(source, key):
    expected = bytes.fromhex(key); lo = 0; hi = len(source.arrays['hashes']); comparisons = 0
    while lo < hi:
        mid = (lo + hi) // 2; value = source.arrays['hashes'][mid].tobytes(); comparisons += 1
        if value < expected:
            lo = mid + 1
        else:
            hi = mid
    found = lo if lo < len(source.arrays['hashes']) and source.arrays['hashes'][lo].tobytes() == expected else None
    return found, comparisons


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    r = catalog.read(a.input / 'report.json', a.pin)
    need(r['schema'] == work.SCHEMA and r['complete'] is True and r['native_measurement_pending'] is True
         and r['outcome_checks_pending'] is True and r['all_native_windows_planned'] is True
         and r['serving'] is False and r['capability_demonstrated'] is False and r['encoder_calls'] == 0
         and r['population_text_copied'] is False and r['cache_authentications'] == 1, 'FREE_WORK_COLD_SCOPE')
    need(work.code() == r['code'], 'FREE_WORK_COLD_CODE')
    for name, item in r['files'].items():
        path = catalog.child(a.input, name)
        need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'FREE_WORK_COLD_FILE')
    root = Path(r['catalog']['path']); cr = r['catalog_readback']
    parent, cold = work.source_catalog(root, r['catalog']['pin'], cr['path'], cr['pin'])
    need(r['source_counts'] == parent['counts'] and r['source_scope'] == parent['scope']
         and r['full_query'] == parent['full_query'] and r['native_bank'] == parent['native_bank'], 'FREE_WORK_COLD_POPULATION')
    bank, specs, contracts = authority(parent['native_bank']); dims = {s.name: s.dims for s in specs}
    need(r['cache_source'] == bank.config['source'] and set(r['instruments']) == set(bank.config['roster']), 'FREE_WORK_COLD_BANK')
    phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
    comparisons = reused = missing = count = 0
    with work.work_db(a.input / 'mapping.sqlite') as db, catalog.database(root / 'inputs.sqlite') as inputs:
        with work.cache(r['cache_source']['path'], r['cache_source']['pin']) as existing:
            need(existing.report['counts'] == r['cache_source_counts']
                 and sum(v['bytes'] for v in existing.report['files'].values()) == r['cache_authenticated_bytes'],
                 'FREE_WORK_COLD_CACHE_SOURCE')
            phases['one_cache_authentication'] = time.perf_counter() - tick; tick = time.perf_counter()
            mapping = iter(db.execute('SELECT ordinal,id,bytes,uid FROM mapping ORDER BY ordinal'))
            for record in catalog.records(inputs):
                raw = record['text'].encode('utf-8'); uid, steps = scalar_lookup(existing, record['id']); comparisons += steps
                need(next(mapping, None) == (count, record['id'], len(raw), uid), 'FREE_WORK_COLD_MAPPING')
                if uid is not None:
                    need(existing.input(uid)['text'].encode('utf-8') == raw, 'FREE_WORK_COLD_CACHE_BYTES')
                count += 1; reused += int(uid is not None); missing += int(uid is None)
            need(next(mapping, None) is None and count == r['inputs'] == parent['counts']['unique_measurement_inputs']
                 and reused == r['reused'] and missing == r['missing'], 'FREE_WORK_COLD_PARTITION')
            positive = r['positive_probe']; probe = existing.input(positive['uid'])
            uid, steps = scalar_lookup(existing, probe['text_sha256']); comparisons += steps
            need(uid == 0 == positive['uid'] and probe['text_sha256'] == positive['input_sha256']
                 and len(probe['text'].encode('utf-8')) == positive['bytes']
                 and positive['included_in_generated_population'] is False, 'FREE_WORK_COLD_POSITIVE')
        phases['independent_scalar_search_and_byte_equality'] = time.perf_counter() - tick; tick = time.perf_counter()
        need(db.execute('SELECT COUNT(*) FROM tokens').fetchone()[0] == count * len(bank.config['roster']),
             'FREE_WORK_COLD_TOKEN_POPULATION')
        totals = {}
        for name in bank.config['roster']:
            source_tokens = catalog.lines(root / 'tokens' / (name + '.jsonl'))
            index = windows = tokens = batches = 0
            try:
                for batch in catalog.lines(a.input / 'work' / (name + '.jsonl')):
                    group = batch['inputs']; payload = sum(v['windows'] for v in group) * dims[name] * 4
                    size = sum(v['bytes'] for v in group)
                    need(batch['batch'] == batches and batch['instrument'] == name
                         and batch['lens_id'] == bank.config['instruments'][name]['lens_id'], 'FREE_WORK_COLD_BATCH_IDENTITY')
                    need(0 < len(group) <= r['policy']['max_inputs']
                         and (len(group) == 1 or size <= r['policy']['text_bytes'])
                         and payload <= r['policy']['vector_bytes'], 'FREE_WORK_COLD_BATCH_MEMORY')
                    need(batch['source_bytes'] == size and batch['vector_payload_bytes'] == payload
                         and batch['windows'] == sum(v['windows'] for v in group)
                         and batch['active_tokens'] == sum(v['active_tokens'] for v in group), 'FREE_WORK_COLD_BATCH_TOTALS')
                    for item in group:
                        row = db.execute('SELECT id,bytes,uid FROM mapping WHERE ordinal=?', (index,)).fetchone()
                        t = next(source_tokens, None)
                        need(row is not None and t is not None and item == dict(ordinal=index, id=row[0], bytes=row[1],
                             uid=row[2], windows=t['windows'], active_tokens=t['active_tokens'])
                             and t['id'] == row[0], 'FREE_WORK_COLD_BATCH_COVERAGE')
                        need(db.execute('SELECT windows,tokens FROM tokens WHERE instrument=? AND ordinal=?',
                             (name, index)).fetchone() == (t['windows'], t['active_tokens']), 'FREE_WORK_COLD_TOKEN_JOIN')
                        index += 1; windows += item['windows']; tokens += item['active_tokens']
                    batches += 1
                need(next(source_tokens, None) is None and index == count, 'FREE_WORK_COLD_MISSING_INPUTS')
            finally:
                source_tokens.close()
            totals[name] = dict(inputs=index, reused=reused, missing=missing, windows=windows, active_tokens=tokens,
                batches=batches, dims=dims[name], lens_id=bank.config['instruments'][name]['lens_id'])
            need(totals[name] == r['instruments'][name] and windows == cold['instrument_totals'][name]['total_windows']
                 and tokens == cold['instrument_totals'][name]['total_tokens'], 'FREE_WORK_COLD_TOTALS')
    phases['independent_full_batch_and_window_census'] = time.perf_counter() - tick; tick = time.perf_counter()
    need(work.code() == r['code'] and file_sha(a.input / 'report.json') == a.pin
         and all(file_sha(a.input / n) == v['sha256'] for n, v in r['files'].items()), 'FREE_WORK_COLD_CHANGED')
    need(not torch.cuda.is_initialized(), 'FREE_WORK_COLD_CPU_ONLY')
    phases['seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_WORK_COLD_PHASES')
    result = dict(complete=True, input=dict(path=str(a.input), pin=a.pin), code=work.code(), inputs=count, reused=reused,
        missing=missing, scalar_comparisons=comparisons, all_cache_decisions_exact=True, exact_cached_bytes=True,
        all_batch_members_and_windows_exact=True, all_source_denominators_retained=True, positive_probe_checked=True,
        instruments=totals, source_counts=parent['counts'], source_scope=parent['scope'], full_query=parent['full_query'],
        native_measurement_pending=True, outcome_checks_pending=True, capability_demonstrated=False, serving=False,
        gpu_used=False, encoder_calls=0, phases=phases, seconds=seconds, acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, inputs=count, reused=reused, missing=missing,
                         comparisons=comparisons, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('input', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args(); torch.set_num_threads(1)
    try:
        run(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
