"""Content-addressed delta measurements shared by every capability source and family."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import CapabilityPopulation, need
from claims.capability_revision_population import RevisionPopulation
from claims.process_conditions import write_committed
from text.evaluation_multiscale import file_sha


def retained_cache(specs):
    """Resolve source bytes once; no native forward or arrays are needed for cache lookup."""
    cache, banks, plans = {}, {}, {}

    def add(text, bank, row, provenance):
        pin = hashlib.sha256(text.encode()).hexdigest()
        if pin in cache:
            need(cache[pin]['text'] == text, 'DELTA_HASH_COLLISION')
        else:
            cache[pin] = dict(text=text, bank=bank, row=row, source=provenance)

    for spec in specs:
        p = CapabilityPopulation(spec['population'], spec['population_pin'])
        if p.family == 'coding':
            for f in p.fields.values():
                source = f['source']; bank = source['native_path']
                record = dict(native=bank, native_pin=source['native_pin'], source_pin=source['population_pin'])
                need(bank not in banks or banks[bank] == record, 'DELTA_BANK_ALIAS')
                banks[bank] = record
                raw = f['text'].encode()
                for part in source['native_parts']:
                    piece = raw[part['start']:part['end']]
                    need(hashlib.sha256(piece).hexdigest() == part['sha256'], 'DELTA_RETAINED_SPAN')
                    add(piece.decode(), bank, part['unit'], dict(population=p.pin, field=f['id'], part=part))
        else:
            plan_key = (spec['plan'], spec['plan_pin'], spec['native'], spec['native_pin'])
            if plan_key in plans:
                continue
            root = Path(spec['plan'])
            need(file_sha(root / 'report.json') == spec['plan_pin'], 'DELTA_RETAINED_PLAN')
            report = json.loads((root / 'report.json').read_bytes())
            need(report['complete'] and not report['errors'], 'DELTA_RETAINED_PLAN_SCOPE')
            bank = spec['native']
            record = dict(native=bank, native_pin=spec['native_pin'], source_pin=spec['plan_pin'])
            need(bank not in banks or banks[bank] == record, 'DELTA_BANK_ALIAS')
            banks[bank] = record
            offset = 0
            for name, item in sorted(report['families'].items()):
                summary_path = root / name / 'summary.json'
                need(file_sha(summary_path) == item['summary_sha256'], 'DELTA_RETAINED_SUMMARY')
                summary = json.loads(summary_path.read_bytes())
                path = root / name / 'chunks/atoms.parquet'
                need(file_sha(path) == summary['chunks_sha256'], 'DELTA_RETAINED_CHUNKS')
                rows = pq.read_table(path).to_pylist()
                need(len(rows) == summary['chunks'], 'DELTA_RETAINED_COUNT')
                for i, row in enumerate(rows):
                    add(row['text'], bank, offset + i, dict(plan=spec['plan_pin'], family=name, row=i))
                offset += len(rows)
            plans[plan_key] = True
    return cache, banks


def build(old_config, delta_config, output):
    start = tick = time.perf_counter()
    old = json.loads(old_config.read_bytes())['populations']
    config = json.loads(delta_config.read_bytes())
    pops = {name: RevisionPopulation(s['population'], s['population_pin']) for name, s in sorted(config.items())}
    cache, banks = retained_cache(old)
    phases = dict(source_validation_and_cache=time.perf_counter() - tick)
    tick = time.perf_counter()
    unique = {}
    for pop in pops.values():
        for unit in pop.units:
            pin, text = unit['entity'], unit['text']
            need(pin not in unique or unique[pin] == text, 'DELTA_HASH_COLLISION')
            unique[pin] = text
            need(pin not in cache or cache[pin]['text'] == text, 'DELTA_HASH_COLLISION')
    fresh = sorted(unique.keys() - cache.keys())
    need(fresh, 'DELTA_NO_NEW_CONTENT')
    units = [dict(row=i, chunk=0, entity=pin, text=unique[pin], start=0, end=len(unique[pin]))
             for i, pin in enumerate(fresh)]
    positions = {pin: i for i, pin in enumerate(fresh)}
    candidates = [(pin, value) for pin, value in cache.items() if 1800 <= len(value['text']) <= 3000]
    need(candidates, 'DELTA_OVERFLOW_PROBE')
    probe_pin, probe = max(candidates, key=lambda item: (len(item[1]['text']), item[0]))
    units.append(dict(row=len(units), chunk=0, entity=probe_pin, text=probe['text'], start=0, end=len(probe['text'])))
    mappings, used_banks = {}, set()
    for name, pop in pops.items():
        rows = []
        for u in pop.units:
            pin = u['entity']
            if pin in positions:
                record = dict(bank='fresh', row=positions[pin], sha256=pin)
            else:
                hit = cache[pin]
                record = dict(bank=hit['bank'], row=hit['row'], sha256=pin, source=hit['source'])
                used_banks.add(hit['bank'])
            rows.append(record)
        mappings[pop.pin] = dict(name=name, population=str(pop.root), family=pop.family, units=rows)
    phases['unique_delta_and_source_mapping'] = time.perf_counter() - tick
    tick = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    folder = output / 'delta'
    (folder / 'chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(units), folder / 'chunks/atoms.parquet')
    write_committed(folder / 'summary.json', dict(chunks=len(units), chunks_sha256=file_sha(folder / 'chunks/atoms.parquet')))
    mapping = dict(schema='capability-delta-row-map-v1', populations=mappings,
        banks={k: banks[k] for k in sorted(used_banks)},
        retained_config=dict(path=str(old_config), sha256=file_sha(old_config)),
        delta_config=dict(path=str(delta_config), sha256=file_sha(delta_config)))
    write_committed(output / 'mapping.json', mapping)
    write_committed(output / 'probe.json', dict(sha256=probe_pin, row=len(units)-1,
        retained_source={k: v for k, v in probe.items() if k != 'text'},
        purpose='mandatory overflow/determinism diagnostic; not a task field', model_input=False))
    phases['publication'] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    report = dict(schema='capability-delta-measurement-plan-v1', complete=True, errors=[], calibration_only=True,
        families=dict(delta=dict(ready=True, summary_sha256=file_sha(folder / 'summary.json'))),
        inputs=len(units), task_unique_fragments=len(unique), new_fragments=len(fresh),
        reused_fragments=len(unique)-len(fresh), fragment_occurrences=sum(len(p.units) for p in pops.values()),
        new_utf8_bytes=sum(len(unique[k].encode()) for k in fresh), probe_utf8_bytes=len(probe['text'].encode()),
        mapping_sha256=file_sha(output / 'mapping.json'), probe_sha256=file_sha(output / 'probe.json'),
        training_ready=False, scientific_admission=False, serving=False, producer_sha256=file_sha(Path(__file__)),
        phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values()) / wall)
    need(report['phase_coverage'] >= .9, 'DELTA_TIMING')
    pin = write_committed(output / 'report.json', report)
    print(json.dumps(dict(report_sha256=pin, inputs=len(units), new_fragments=len(fresh),
                         reused_fragments=report['reused_fragments'], wall_seconds=wall)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-config', 'delta-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    build(a.old_config, a.delta_config, a.output)
