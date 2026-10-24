"""Lossless native-measurement population for the frozen repository depth increment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from claims.repository_depth_observation import DepthObservations
from claims.repository_cohort_population import FILES,LINES,SPAN_BYTES,expected
from production.publication import sync_dir


def code_pins():
    root=Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'claims/repository_depth_population.py','claims/repository_depth_observation.py',
        'claims/repository_cohort_population.py','claims/repository_cohort_observation.py')}


def build(source,root):
    start=time.perf_counter();root=Path(root);check(not root.exists(),'DEPTH_POPULATION_EXISTS')
    ob=DepthObservations(**source);bound=time.perf_counter();rows,fields=expected(ob);projected=time.perf_counter()
    (root/'repository/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),root/FILES[0])
    write_committed(root/FILES[1],dict(chunks=len(rows),chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2],fields);write_committed(root/FILES[3],dict(entries=ob.entries,gaps=ob.gaps))
    report=dict(schema='repository-depth-native-population-v1',observations=source,inputs=len(rows),
        unique_utf8_bytes=sum(len(r['text'].encode()) for r in rows),
        field_occurrences=sum(len(v[s]) for v in fields.values() for s in ('input','echo')),
        occurrence_bytes=sum(f['bytes'] for v in fields.values() for s in ('input','echo') for f in v[s]),
        action_counts={p:sum(v['partition']==p for v in fields.values()) for p in ('learn_1','learn_2')},
        source=ob.receipts,source_files=ob.files,source_lines_per_span=LINES,max_utf8_bytes_per_span=SPAN_BYTES,
        fragments_are_independent_atoms=False,complete=True,errors=[],calibration_only=True,
        scientific_admission=False,serving=False,partitions=['learn_1','learn_2'],oracle_reference_inputs=False,
        raw_oracle_fields_in_inputs=False,families=dict(repository=dict(ready=True,summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},code=code_pins())
    pin=write_committed(root/'report.json',report)
    for n in (*FILES,'report.json'):
        with (root/n).open('rb') as stream:os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:stream.write(pin);stream.flush();os.fsync(stream.fileno())
    sync_dir(root);end=time.perf_counter()
    write_committed(root/'timings.json',dict(wall_seconds=end-start,
        phases=dict(binding=bound-start,projection=projected-bound,publication=end-projected)))
    print(json.dumps(dict(path=str(root),pin=pin,inputs=len(rows),bytes=report['unique_utf8_bytes'],
                         actions=report['action_counts'],gaps=len(ob.gaps),wall_seconds=end-start)),flush=True)
    return pin


class DepthPopulation:
    def __init__(self,root,pin):
        self.root,self.pin=Path(root),pin;raw=(self.root/'report.json').read_bytes();r=json.loads(raw)
        check(sha(raw)==pin==(self.root/'COMMITTED').read_text() and raw==canonical(r) and
              r['schema']=='repository-depth-native-population-v1' and r['code']==code_pins() and r['complete'] and
              not r['scientific_admission'] and not r['oracle_reference_inputs'] and
              not r['raw_oracle_fields_in_inputs'] and set(r['files'])==set(FILES),'DEPTH_POPULATION_PIN')
        for n,v in r['files'].items():
            value=(self.root/n).read_bytes()
            check(dict(bytes=len(value),sha256=sha(value))==v,'DEPTH_POPULATION_BYTES')
        self.observations=DepthObservations(**r['observations']);rows,fields=expected(self.observations)
        check(pq.read_table(self.root/FILES[0]).to_pylist()==rows and json.loads((self.root/FILES[2]).read_bytes())==fields and
              json.loads((self.root/FILES[3]).read_bytes())==dict(entries=self.observations.entries,gaps=self.observations.gaps) and
              r['source']==self.observations.receipts and r['source_files']==self.observations.files,'DEPTH_POPULATION_SOURCE')
        check(len(rows)==r['inputs'] and sum(len(x['text'].encode()) for x in rows)==r['unique_utf8_bytes'],
              'DEPTH_POPULATION_COUNTS')
        self.rows,self.fields,self.report=rows,fields,r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_bytes());build(config['source'],config['root'])
