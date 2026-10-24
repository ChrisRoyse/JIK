"""Reuse retained code readings; project only fresh dialogue/arithmetic source fields."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import CapabilityPopulation, need
from claims.process_conditions import write_committed
from text.evaluation_multiscale import file_sha


def build(roots, output):
    populations = {}
    for root in roots:
        pop = CapabilityPopulation(root, (root / 'COMMITTED').read_text())
        need(pop.family not in populations, 'PLAN_DUPLICATE_FAMILY')
        populations[pop.family] = pop
    need(set(populations) == {'coding', 'conversation', 'reasoning'}, 'PLAN_FAMILIES')
    code = populations['coding']
    candidates = [u for u in code.units if 1800 <= len(u['text']) <= 3000]
    need(candidates, 'PLAN_REAL_PROBE')
    probe = max(candidates, key=lambda u: (len(u['text']), u['entity']))
    owners = [f for f in code.fields.values() if any(p['unit'] == probe['row'] for p in f['parts'])]
    need(owners, 'PLAN_PROBE_SOURCE')
    output.mkdir(parents=True, exist_ok=False)
    families, mapping = {}, {}
    for family in ('conversation', 'reasoning'):
        pop = populations[family]
        (output / family / 'chunks').mkdir(parents=True)
        for suffix in ('chunks/atoms.parquet', 'summary.json'):
            shutil.copyfile(pop.root / family / suffix, output / family / suffix)
        families[family] = dict(ready=True, summary_sha256=file_sha(output / family / 'summary.json'))
        mapping[family] = dict(population=str(pop.root), pin=pop.pin, native_origin='fresh',
                               row_mapping='identity; source parquet copied byte-for-byte')
    (output / 'probe/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(probe, row=0)]), output / 'probe/chunks/atoms.parquet')
    write_committed(output / 'probe/summary.json', dict(chunks=1,
                    chunks_sha256=file_sha(output / 'probe/chunks/atoms.parquet')))
    families['probe'] = dict(ready=True, summary_sha256=file_sha(output / 'probe/summary.json'))
    mapping['coding'] = dict(population=str(code.root), pin=code.pin, native_origin='retained',
        field_reading_sources={key: {k: f['source'][k] for k in
            ('native_path', 'native_pin', 'native_parts', 'population', 'population_pin')}
                              for key, f in code.fields.items()})
    write_committed(output / 'mapping.json', mapping)
    write_committed(output / 'probe.json', dict(population=code.pin, original_unit=probe['row'],
        text_sha256=probe['entity'], field_ids=[f['id'] for f in owners],
        purpose='retained source-span overflow/determinism probe', model_input=False, target=False))
    inputs = sum(len(populations[k].units) for k in ('conversation', 'reasoning')) + 1
    report = dict(schema='capability-fresh-measurement-view-v1', complete=True, errors=[],
        calibration_only=True, families=families, inputs=inputs,
        mapping_sha256=file_sha(output / 'mapping.json'), probe_sha256=file_sha(output / 'probe.json'),
        fresh_text_bytes=sum(len(u['text'].encode()) for k in ('conversation', 'reasoning')
                             for u in populations[k].units) + len(probe['text'].encode()),
        retained_code_fragments=len(code.units), retained_code_fields=len(code.fields),
        all_task_fields_measured_but_inputs_masked=True, training_ready=False,
        scientific_admission=False, serving=False, code_sha256=file_sha(Path(__file__)))
    pin = write_committed(output / 'report.json', report)
    print(json.dumps(dict(report_sha256=pin, inputs=inputs, fresh_text_bytes=report['fresh_text_bytes'],
                          retained_code_fragments=len(code.units))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.population, args.output)
