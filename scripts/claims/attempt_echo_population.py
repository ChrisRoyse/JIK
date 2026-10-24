"""Exact proposal execution streams, with explicit emptiness and excluded probe."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from claims.attempt_observation import AttemptObservations
from claims.attempt_population import AttemptPopulation
from production.publication import sync_dir

from claims.proposal_echo_population import FILES, expected


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'claims/attempt_echo_population.py','claims/proposal_echo_population.py',
        'claims/attempt_population.py','claims/attempt_observation.py','claims/attempt_forecast.py')}


def build(observations, source, root, forecast_root, forecast_pin):
    started = time.perf_counter()
    check(not root.exists(),'PROPOSAL_ECHO_POPULATION_EXISTS')
    rows, refs, probe = expected(observations,source)
    (root/'process/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),root/FILES[0])
    write_committed(root/FILES[1],dict(chunks=len(rows),chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2],refs)
    write_committed(root/FILES[3],probe)
    report = dict(schema='attempt-output-population-v1',
        observations=dict(root=str(observations.root),pin=observations.pin),
        forecast=dict(root=str(forecast_root),pin=forecast_pin),
        source_population=dict(root=str(source.root),pin=source.pin),
        inputs=len(rows),stream_references=len(refs),nonempty_streams=sum(r['unit'] is not None for r in refs),
        utf8_bytes=sum(len(r['text'].encode()) for r in rows),
        complete=True,errors=[],calibration_only=True,scientific_admission=False,
        families=dict(process=dict(ready=True,summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},
        code=code_pins(),wall_seconds=time.perf_counter()-started)
    pin = write_committed(root/'report.json',report)
    for p in root.rglob('*'):
        if p.is_file():
            with p.open('rb') as stream:
                os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(root)
    sync_dir(root.parent)
    return pin


class AttemptEchoPopulation:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        report = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and raw == canonical(report)
              and report['schema'] == 'attempt-output-population-v1' and report['complete']
              and not report['errors'] and report['scientific_admission'] is False
              and report['code'] == code_pins(),'PROPOSAL_ECHO_POPULATION_PIN')
        check(set(report['files']) == set(FILES),'PROPOSAL_ECHO_POPULATION_FILES')
        for name,item in report['files'].items():
            value = (self.root/name).read_bytes()
            check(len(value) == item['bytes'] and sha(value) == item['sha256'],'PROPOSAL_ECHO_POPULATION_BYTES')
        self.observations = AttemptObservations(report['observations']['root'],report['observations']['pin'],
                                                 report['forecast']['root'],report['forecast']['pin'])
        source = AttemptPopulation(report['source_population']['root'],report['source_population']['pin'])
        rows, refs, probe = expected(self.observations,source)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows
              and (self.root/FILES[2]).read_bytes() == canonical(refs)
              and (self.root/FILES[3]).read_bytes() == canonical(probe),'PROPOSAL_ECHO_POPULATION_CONTENT')
        check(len(rows) == report['inputs'] and len(refs) == report['stream_references']
              and sum(len(r['text'].encode()) for r in rows) == report['utf8_bytes'],
              'PROPOSAL_ECHO_POPULATION_COUNTS')
        self.rows, self.refs, self.probe, self.report = rows,refs,probe,report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations','forecast','source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('observation-pin','forecast-pin','source-pin'):
        parser.add_argument('--'+name,required=True)
    a = parser.parse_args()
    observations = AttemptObservations(a.observations,a.observation_pin,a.forecast,a.forecast_pin)
    source = AttemptPopulation(a.source,a.source_pin)
    pin = build(observations,source,a.output,a.forecast,a.forecast_pin)
    p = AttemptEchoPopulation(a.output,pin)
    print(json.dumps(dict(pin=pin,**{k:p.report[k] for k in
        ('inputs','stream_references','nonempty_streams','utf8_bytes','wall_seconds')})))
