"""New proposal surfaces plus content-addressed reuse of measured wheel names."""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from oracle.prepare_proposal import PreparedProposal
from production.publication import sync_dir

FILES = ('proposal/chunks/atoms.parquet', 'proposal/summary.json', 'fields.json', 'probe.json')


def cached(root, pin):
    root = Path(root)
    raw = (root/'report.json').read_bytes()
    report = json.loads(raw)
    check(sha(raw) == pin == (root/'COMMITTED').read_text()
          and report['schema'] == 'execution-condition-population-v1'
          and report['complete'] and not report['errors'], 'PROPOSAL_CACHE_PIN')
    for name, item in report['files'].items():
        check(name in ('condition/chunks/atoms.parquet', 'condition/summary.json', 'fields.json', 'probe.json'),
              'PROPOSAL_CACHE_PATH')
        value = (root/name).read_bytes()
        check(len(value) == item['bytes'] and sha(value) == item['sha256'], 'PROPOSAL_CACHE_BYTES')
    rows = pq.read_table(root/'condition/chunks/atoms.parquet').to_pylist()
    fields = json.loads((root/'fields.json').read_bytes())
    probe = json.loads((root/'probe.json').read_bytes())
    summary = json.loads((root/'condition/summary.json').read_bytes())
    check(len(rows) == summary['chunks'] == report['inputs']
          and sha((root/'condition/chunks/atoms.parquet').read_bytes()) == summary['chunks_sha256'],
          'PROPOSAL_CACHE_ROWS')
    wheels = {}
    for item in fields:
        if item['field'] == 'wheel_filename':
            unit = item['unit']
            check(type(unit) is int and 0 <= unit < len(rows) and unit != probe['unit'], 'PROPOSAL_CACHE_UNIT')
            row = rows[unit]
            check(sha(row['text'].encode()) == item['text_sha256'] == row['entity'], 'PROPOSAL_CACHE_SURFACE')
            check(row['text'] not in wheels or wheels[row['text']] == unit, 'PROPOSAL_CACHE_DUPLICATE')
            wheels[row['text']] = unit
    check(rows[probe['unit']]['text'] == probe['atom']['text']
          and probe['purpose'] == 'overflow_determinism_only_excluded_from_condition_input', 'PROPOSAL_CACHE_PROBE')
    return rows, wheels, probe


def expected(prepared, cache_root, cache_pin):
    old, wheels, old_probe = cached(cache_root, cache_pin)
    rows = []
    for text in (prepared.source.action.decode(), prepared.source.program.decode()):
        rows.append(dict(row=len(rows), chunk=0, entity=sha(text.encode()), text=text, start=0, end=len(text)))
    fields = []
    for key in sorted(prepared.conditions.inputs):
        public = prepared.conditions.read(key)
        for ordinal, role in enumerate(('action', 'program')):
            fields.append(dict(condition=key, role=role, ordinal=ordinal, origin='new', unit=ordinal,
                               text_sha256=rows[ordinal]['entity']))
        for ordinal, name in enumerate(sorted(public['wheels']), 2):
            check(name in wheels, 'PROPOSAL_UNMEASURED_WHEEL')
            fields.append(dict(condition=key, role='wheel_filename', ordinal=ordinal, origin='cache',
                               unit=wheels[name], text_sha256=sha(name.encode())))
    original = old[old_probe['unit']]
    probe = dict(unit=len(rows), source_population=cache_pin, source_unit=old_probe['unit'],
                 original=old_probe, purpose='overflow_determinism_only_excluded_from_model')
    rows.append(dict(original, row=len(rows)))
    return rows, fields, probe


def build(prepared, cache_root, cache_pin, root):
    start = time.perf_counter()
    rows, fields, probe = expected(prepared, cache_root, cache_pin)
    ready = time.perf_counter()
    check(not root.exists(), 'PROPOSAL_POPULATION_EXISTS')
    (root/'proposal/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root/FILES[0])
    write_committed(root/FILES[1], dict(chunks=len(rows), chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2], fields)
    write_committed(root/FILES[3], probe)
    published = time.perf_counter()
    report = dict(schema='proposed-field-population-v1', prepared=dict(root=str(prepared.root), pin=prepared.conditions.pin),
        cache=dict(root=str(cache_root), pin=cache_pin), inputs=len(rows), field_instances=len(fields),
        new_model_surfaces=2, cached_model_surfaces=len({r['unit'] for r in fields if r['origin']=='cache'}),
        utf8_bytes=sum(len(r['text'].encode()) for r in rows),
        complete=True, errors=[], calibration_only=True, scientific_admission=False,
        families=dict(proposal=dict(ready=True, summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size, sha256=sha((root/n).read_bytes())) for n in FILES},
        code_sha256=sha(Path(__file__).read_bytes()), phases=dict(source=ready-start, publication=published-ready),
        wall_seconds=published-start)
    pin = write_committed(root/'report.json', report)
    with (root/'COMMITTED').open('x') as f:
        f.write(pin)
    sync_dir(root)
    return pin


class ProposalPopulation:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        raw = (self.root/'report.json').read_bytes()
        report = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and raw == canonical(report)
              and report['schema'] == 'proposed-field-population-v1' and report['complete']
              and not report['errors'] and report['scientific_admission'] is False
              and report['code_sha256'] == sha(Path(__file__).read_bytes()), 'PROPOSAL_POPULATION_PIN')
        check(set(report['files']) == set(FILES), 'PROPOSAL_POPULATION_FILES')
        for name, item in report['files'].items():
            value = (self.root/name).read_bytes()
            check(len(value) == item['bytes'] and sha(value) == item['sha256'], 'PROPOSAL_POPULATION_BYTES')
        self.prepared = PreparedProposal(report['prepared']['root'], report['prepared']['pin'])
        self.cache_root, self.cache_pin = Path(report['cache']['root']), report['cache']['pin']
        rows, fields, probe = expected(self.prepared, self.cache_root, self.cache_pin)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows
              and (self.root/FILES[2]).read_bytes() == canonical(fields)
              and (self.root/FILES[3]).read_bytes() == canonical(probe), 'PROPOSAL_POPULATION_CONTENT')
        check(len(rows) == report['inputs'] and len(fields) == report['field_instances']
              and sum(len(r['text'].encode()) for r in rows) == report['utf8_bytes'], 'PROPOSAL_POPULATION_COUNTS')
        self.rows, self.fields, self.probe, self.report = rows, fields, probe, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('prepared-pin', 'cache-pin'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    prepared = PreparedProposal(a.prepared, a.prepared_pin)
    pin = build(prepared, a.cache, a.cache_pin, a.output)
    population = ProposalPopulation(a.output, pin)
    print(json.dumps(dict(pin=pin, **{k:population.report[k] for k in (
        'inputs','field_instances','new_model_surfaces','cached_model_surfaces','utf8_bytes','phases','wall_seconds')})))
