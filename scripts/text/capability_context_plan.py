"""Measure only uncached repository context; reuse immutable historical native banks."""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need
from claims.capability_context_population import ContextPopulation
from claims.process_conditions import write_committed
from senses.trajectory_population import read_pinned
from text.capability_delta_plan import retained_cache
from text.evaluation_multiscale import file_sha


def historical_cache(config):
    old = read_pinned(Path(config['old_config']['path']), config['old_config']['pin'])
    cache, banks = retained_cache(old['populations'])
    # Previous delta rows already have frozen readings; do not measure them again.
    for source in config['deltas']:
        root = Path(source['plan'])
        report = read_pinned(root / 'report.json', source['plan_pin'])
        need(report['complete'] and not report['errors'], 'CONTEXT_CACHE_PLAN')
        summary = read_pinned(root / 'delta/summary.json', report['families']['delta']['summary_sha256'])
        path = root / 'delta/chunks/atoms.parquet'
        need(file_sha(path) == summary['chunks_sha256'], 'CONTEXT_CACHE_ROWS')
        rows = pq.read_table(path).to_pylist()
        need(len(rows) == summary['chunks'], 'CONTEXT_CACHE_COUNT')
        native = read_pinned(Path(source['native']) / 'report.json', source['native_pin'])
        need(native['complete'] and native['calibration_report_sha256'] == source['plan_pin'],
             'CONTEXT_CACHE_NATIVE')
        banks[source['native']] = dict(native=source['native'], native_pin=source['native_pin'],
                                       source_pin=source['plan_pin'])
        for row in rows:
            need(row['entity'] not in cache or cache[row['entity']]['text'] == row['text'],
                 'CONTEXT_CACHE_COLLISION')
            cache.setdefault(row['entity'], dict(text=row['text'], bank=source['native'], row=row['row'],
                source=dict(plan=source['plan_pin'], row=row['row'])))
    return cache, banks


def build(a):
    start = tick = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    pop = ContextPopulation(config['population'], config['population_pin'])
    cache, banks = historical_cache(config)
    phases = dict(source_and_cache_binding=time.perf_counter()-tick); tick = time.perf_counter()
    fresh = sorted(u['entity'] for u in pop.units if u['entity'] not in cache)
    texts = {u['entity']:u['text'] for u in pop.units}
    need(fresh, 'CONTEXT_NO_NEW_CONTENT')
    positions = {pin:i for i,pin in enumerate(fresh)}
    rows = [dict(row=i, chunk=0, entity=pin, text=texts[pin], start=0, end=len(texts[pin]))
            for i,pin in enumerate(fresh)]
    candidates = [(pin, v) for pin,v in cache.items() if 1800 <= len(v['text']) <= 3000]
    need(candidates, 'CONTEXT_OVERFLOW_PROBE')
    probe_pin, probe = max(candidates, key=lambda item:(len(item[1]['text']), item[0]))
    rows.append(dict(row=len(rows), chunk=0, entity=probe_pin, text=probe['text'], start=0, end=len(probe['text'])))
    mapping, used = [], set()
    for u in pop.units:
        key = u['entity']
        if key in positions: mapping.append(dict(bank='fresh', row=positions[key], sha256=key))
        else:
            old = cache[key]
            need(old['text'] == u['text'], 'CONTEXT_CACHE_TEXT')
            mapping.append(dict(bank=old['bank'], row=old['row'], sha256=key, source=old['source']))
            used.add(old['bank'])
    phases['unique_delta_mapping'] = time.perf_counter()-tick; tick = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    folder = a.output / 'delta'; (folder / 'chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / 'chunks/atoms.parquet')
    write_committed(folder / 'summary.json', dict(chunks=len(rows), chunks_sha256=file_sha(folder / 'chunks/atoms.parquet')))
    write_committed(a.output / 'mapping.json', dict(schema='capability-context-row-map-v1',
        config=dict(path=str(a.config), pin=file_sha(a.config)),
        populations={pop.pin:dict(population=str(pop.root), family=pop.family, units=mapping)},
        banks={key:banks[key] for key in sorted(used)}))
    write_committed(a.output / 'probe.json', dict(row=len(rows)-1, sha256=probe_pin,
        retained_source={k:v for k,v in probe.items() if k != 'text'}, model_input=False,
        purpose='mandatory real-source overflow/determinism diagnostic'))
    phases['publication'] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema='capability-context-measurement-plan-v1', complete=True, errors=[], calibration_only=True,
        families=dict(delta=dict(ready=True, summary_sha256=file_sha(folder / 'summary.json'))),
        inputs=len(rows), task_unique_fragments=len(pop.units), new_fragments=len(fresh),
        reused_fragments=len(pop.units)-len(fresh), new_utf8_bytes=sum(len(texts[k].encode()) for k in fresh),
        mapping_sha256=file_sha(a.output / 'mapping.json'), probe_sha256=file_sha(a.output / 'probe.json'),
        producer_sha256=file_sha(Path(__file__)), phases_seconds=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, training_ready=False, scientific_admission=False, serving=False)
    need(report['phase_coverage'] >= .9, 'CONTEXT_PLAN_TIMING')
    pin = write_committed(a.output / 'report.json', report)
    print(json.dumps(dict(report_sha256=pin, **report)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    build(p.parse_args())
