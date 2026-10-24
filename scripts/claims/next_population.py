"""Native-measurement population for the next prospective program occurrence."""
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from claims.proposal_population import FILES, expected
from oracle.prepare_next import PreparedNext
from production.publication import sync_dir


def code_pins():
    scripts = Path(__file__).resolve().parents[1]
    return {n:sha((scripts/n).read_bytes()) for n in ('claims/next_population.py',
        'claims/proposal_population.py','oracle/prepare_next.py')}


def build(prepared, cache_root, cache_pin, root):
    check(not root.exists(), 'NEXT_POPULATION_EXISTS')
    rows,fields,probe = expected(prepared,cache_root,cache_pin)
    (root/'proposal/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),root/FILES[0])
    write_committed(root/FILES[1],dict(chunks=len(rows),chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2],fields)
    write_committed(root/FILES[3],probe)
    report = dict(schema='next-field-population-v1',prepared=dict(root=str(prepared.root),pin=prepared.conditions.pin),
        cache=dict(root=str(cache_root),pin=cache_pin),inputs=len(rows),field_instances=len(fields),
        new_model_surfaces=2,cached_model_surfaces=len({f['unit'] for f in fields if f['origin']=='cache'}),
        utf8_bytes=sum(len(r['text'].encode()) for r in rows),complete=True,errors=[],calibration_only=True,
        scientific_admission=False,families=dict(proposal=dict(ready=True,summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},code=code_pins())
    pin = write_committed(root/'report.json',report)
    for name in FILES:
        with (root/name).open('rb') as stream:
            os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(root)
    return NextPopulation(root,pin)


class NextPopulation:
    def __init__(self, root, pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        report = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and raw == canonical(report)
              and report['schema'] == 'next-field-population-v1' and report['complete']
              and not report['errors'] and report['scientific_admission'] is False
              and report['code'] == code_pins() and set(report['files']) == set(FILES), 'NEXT_POPULATION_PIN')
        for name,item in report['files'].items():
            value = (self.root/name).read_bytes()
            check(dict(bytes=len(value),sha256=sha(value)) == item, 'NEXT_POPULATION_BYTES')
        self.prepared = PreparedNext(**dict(root=report['prepared']['root'],pin=report['prepared']['pin']))
        self.cache_root,self.cache_pin = Path(report['cache']['root']),report['cache']['pin']
        rows,fields,probe = expected(self.prepared,self.cache_root,self.cache_pin)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows
              and (self.root/FILES[2]).read_bytes() == canonical(fields)
              and (self.root/FILES[3]).read_bytes() == canonical(probe), 'NEXT_POPULATION_CONTENT')
        check(len(rows) == report['inputs'] and len(fields) == report['field_instances']
              and sum(len(r['text'].encode()) for r in rows) == report['utf8_bytes'], 'NEXT_POPULATION_COUNTS')
        summary = json.loads((self.root/FILES[1]).read_bytes())
        check(summary == dict(chunks=len(rows),chunks_sha256=sha((self.root/FILES[0]).read_bytes()))
              and report['families'] == dict(proposal=dict(ready=True,
                  summary_sha256=sha((self.root/FILES[1]).read_bytes()))), 'NEXT_POPULATION_SUMMARY')
        self.rows,self.fields,self.probe,self.report = rows,fields,probe,report
