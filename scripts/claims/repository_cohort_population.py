"""Verbatim cohort surfaces, separately available before action and after consequence."""
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
from claims.repository_cohort_observation import CohortObservations
from production.publication import sync_dir

FILES = ('repository/chunks/atoms.parquet','repository/summary.json','fields.json','anchors.json')
LINES = 32
SPAN_BYTES = 2048


def fragments(raw):
    """Transport fragments of a retained field, never independent semantic atoms."""
    pending,lines = b'',0
    for line in raw.splitlines(keepends=True):
        cursor = 0
        while cursor < len(line):
            room = SPAN_BYTES-len(pending)
            end = min(cursor+room,len(line))
            while end < len(line) and end > cursor and line[end] & 0xc0 == 0x80:
                end -= 1
            if end == cursor:
                yield pending
                pending,lines = b'',0
                continue
            pending += line[cursor:end]
            cursor = end
            if cursor < len(line):
                yield pending
                pending,lines = b'',0
        lines += 1
        if len(pending) == SPAN_BYTES or lines == LINES:
            yield pending
            pending,lines = b'',0
    if pending:
        yield pending


def code_pins():
    return {n:sha(Path(__file__).with_name(n).read_bytes()) for n in
            ('repository_cohort_population.py','repository_cohort_observation.py')}


def expected(ob):
    rows,fields,index = [],{},{}

    def field(item):
        raw = item['raw']
        raw.decode('utf-8')
        parts,offset = [],0
        for piece in fragments(raw):
            pin = sha(piece)
            if pin not in index:
                unit = len(rows);index[pin] = unit
                text = piece.decode('utf-8')
                rows.append(dict(row=unit,chunk=0,entity=pin,text=text,start=0,end=len(text)))
            unit = index[pin]
            check(rows[unit]['text'].encode() == piece,'COHORT_POPULATION_COLLISION')
            parts.append(dict(unit=unit,start=offset,end=offset+len(piece),sha256=pin,
                starts_mid_line=bool(offset and raw[offset-1:offset] not in (b'\n',b'\r')),
                ends_mid_line=bool(offset+len(piece)<len(raw) and piece[-1:] not in (b'\n',b'\r'))))
            offset += len(piece)
        check(offset == len(raw),'COHORT_POPULATION_CONSERVATION')
        return dict(role=item['role'],bytes=len(raw),sha256=sha(raw),source=item['source'],
                    source_span=item.get('source_span',[0,len(raw)]),parts=parts,empty=not raw,
                    fragments_are_independent_atoms=False)

    for key,surfaces in ob.surfaces.items():
        entry = ob.entries[key]
        fields[key] = dict(input=[field(v) for v in surfaces['input']],echo=[field(v) for v in surfaces['echo']],
            action='recorded_patch',task=entry['task'],partition=entry['partition'],ordinal=entry['ordinal'],
            container=entry['container'],input_available_by=entry['started'],echo_available_after=entry['finished'],
            missing=surfaces['missing'],excluded_oracle_paths=surfaces['excluded_oracle_paths'],
            historical_occurrence_established=False)
    return rows,fields


def build(source,root):
    start = time.perf_counter()
    root = Path(root)
    check(not root.exists(),'COHORT_POPULATION_EXISTS')
    ob = CohortObservations(**source)
    bound = time.perf_counter()
    rows,fields = expected(ob)
    projected = time.perf_counter()
    (root/'repository/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),root/FILES[0])
    write_committed(root/FILES[1],dict(chunks=len(rows),chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2],fields)
    write_committed(root/FILES[3],dict(entries=ob.entries,gaps=ob.gaps))
    report = dict(schema='repository-cohort-native-population-v2',observations=source,
        inputs=len(rows),unique_utf8_bytes=sum(len(r['text'].encode()) for r in rows),
        field_occurrences=sum(len(v[s]) for v in fields.values() for s in ('input','echo')),
        occurrence_bytes=sum(f['bytes'] for v in fields.values() for s in ('input','echo') for f in v[s]),
        action_counts={p:sum(v['partition']==p for v in fields.values()) for p in ('learn_1','learn_2')},
        source=ob.receipts,source_files=ob.files,source_lines_per_span=LINES,
        max_utf8_bytes_per_span=SPAN_BYTES,fragments_are_independent_atoms=False,complete=True,errors=[],
        calibration_only=True,scientific_admission=False,serving=False,
        partitions=['learn_1','learn_2'],oracle_reference_inputs=False,raw_oracle_fields_in_inputs=False,
        families=dict(repository=dict(ready=True,summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},
        code=code_pins())
    pin = write_committed(root/'report.json',report)
    for name in (*FILES,'report.json'):
        with (root/name).open('rb') as stream:
            os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin);stream.flush();os.fsync(stream.fileno())
    sync_dir(root)
    end = time.perf_counter()
    write_committed(root/'timings.json',dict(wall_seconds=end-start,
        phases=dict(binding=bound-start,projection=projected-bound,publication=end-projected)))
    print(json.dumps(dict(path=str(root),pin=pin,inputs=len(rows),bytes=report['unique_utf8_bytes'],
        actions=report['action_counts'],wall_seconds=end-start)),flush=True)
    return pin


class CohortPopulation:
    def __init__(self,root,pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        r = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and raw == canonical(r) and
              r['schema'] == 'repository-cohort-native-population-v2' and r['code'] == code_pins() and
              r['complete'] and not r['scientific_admission'] and not r['oracle_reference_inputs'] and
              not r['raw_oracle_fields_in_inputs'] and set(r['files']) == set(FILES),'COHORT_POPULATION_PIN')
        for name,entry in r['files'].items():
            value = (self.root/name).read_bytes()
            check(dict(bytes=len(value),sha256=sha(value)) == entry,'COHORT_POPULATION_BYTES')
        self.observations = CohortObservations(**r['observations'])
        rows,fields = expected(self.observations)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows and
              json.loads((self.root/FILES[2]).read_bytes()) == fields and
              json.loads((self.root/FILES[3]).read_bytes()) == dict(entries=self.observations.entries,gaps=self.observations.gaps)
              and r['source'] == self.observations.receipts and r['source_files'] == self.observations.files,
              'COHORT_POPULATION_SOURCE')
        check(len(rows) == r['inputs'] and sum(len(x['text'].encode()) for x in rows) == r['unique_utf8_bytes'],
              'COHORT_POPULATION_COUNTS')
        self.rows,self.fields,self.report = rows,fields,r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_bytes())
    build(config['source'],Path(config['root']))
