"""Verbatim task/source and process-text spans, with separate pre/post occurrence roles."""
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from claims.codata_source import check,sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from claims.repository_observation import RepositoryObservations
from oracle.bound_rows import BoundRows
from production.publication import sync_dir

MODULE = 'src/apig_wsgi/__init__.py'
FILES = ('repository/chunks/atoms.parquet','repository/summary.json','fields.json')
LINES = 32


def code_pins():
    scripts = Path(__file__).resolve().parents[1]
    return {n:sha((scripts/n).read_bytes()) for n in
            ('claims/repository_population.py','claims/repository_observation.py')}


def expected(ob):
    case = json.loads(ob.file('case.json'))
    source_root = Path('/zfs/hot/mnlm')
    bound = BoundRows(source_root,[(case['benchmark'],['problem_statement'])])
    problem = bound.read(case['benchmark'],['problem_statement'])['problem_statement'].encode()
    check('apig_wsgi/__init__.py' in problem.decode(),'REPOSITORY_POPULATION_MODULE_SCOPE')
    rows,fields,index = [],{},{}

    def field(raw,role,source,source_span):
        raw.decode('utf-8')
        parts,offset = [],0
        lines = raw.splitlines(keepends=True)
        for start in range(0,len(lines),LINES):
            part = b''.join(lines[start:start+LINES])
            pin = sha(part)
            if pin not in index:
                unit = len(rows)
                index[pin] = unit
                text = part.decode('utf-8')
                rows.append(dict(row=unit,chunk=0,entity=pin,text=text,start=0,end=len(text)))
            unit = index[pin]
            check(rows[unit]['text'].encode() == part,'REPOSITORY_POPULATION_COLLISION')
            parts.append(dict(unit=unit,start=offset,end=offset+len(part),sha256=pin))
            offset += len(part)
        check(offset == len(raw),'REPOSITORY_POPULATION_BYTE_CONSERVATION')
        return dict(role=role,bytes=len(raw),sha256=sha(raw),source=source,
                    source_span=source_span,parts=parts,empty=not raw)

    shared = field(problem,'task_report',dict(benchmark=case['benchmark'],field='problem_statement'),
                   [0,len(problem)])
    for name in ('base','reference','agent'):
        prefix = ob.prepared.relative_to(ob.root).as_posix()+'/'
        filename = prefix+name+'/'+MODULE
        raw = ob.file(filename)
        inputs = [shared,field(raw,'candidate_module',dict(path=filename,sha256=sha(raw)),[0,len(raw)])]
        if name != 'base':
            filename = prefix+name+'.patch'
            raw = ob.file(filename)
            inputs.append(field(raw,'recorded_patch',dict(path=filename,sha256=sha(raw)),[0,len(raw)]))
        outputs = []
        for stream in ('stdout','stderr'):
            filename = ob.prefix+'/'+name+'/'+stream+'.bin'
            raw = ob.file(filename)
            if stream == 'stdout':
                before,marker,envelope = raw.partition(b'MNLM_EXECUTION_RESULT=')
                check(marker and envelope.count(b'MNLM_EXECUTION_RESULT=') == 0,
                      'REPOSITORY_POPULATION_ENVELOPE')
                # Machine results remain explicit anchors, outside the text instrument inputs.
                text = before
            else:
                text = raw
            outputs.append(field(text,stream,dict(path=filename,sha256=sha(raw)),[0,len(text)]))
        observed = ob.entries[name]
        fields[name] = dict(input=inputs,echo=outputs,action='unchanged_source' if name=='base' else 'recorded_patch',
            input_available_by=observed['started'],echo_available_after=observed['finished'],
            container=observed['container'],task=ob.spec['task'],historical_occurrence_established=False)
    receipts = [{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in bound.receipts]
    return rows,fields,receipts


def build(observation_path,observation_pin,root):
    root = Path(root)
    check(not root.exists(),'REPOSITORY_POPULATION_EXISTS')
    ob = RepositoryObservations(observation_path,observation_pin)
    rows,fields,receipts = expected(ob)
    (root/'repository/chunks').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),root/FILES[0])
    write_committed(root/FILES[1],dict(chunks=len(rows),chunks_sha256=sha((root/FILES[0]).read_bytes())))
    write_committed(root/FILES[2],fields)
    report = dict(schema='repository-native-population-v1',observations=dict(path=str(observation_path),pin=observation_pin),
        inputs=len(rows),unique_utf8_bytes=sum(len(r['text'].encode()) for r in rows),
        field_occurrences=sum(len(v[s]) for v in fields.values() for s in ('input','echo')),
        occurrence_bytes=sum(f['bytes'] for v in fields.values() for s in ('input','echo') for f in v[s]),
        source=receipts,source_lines_per_span=LINES,complete=True,errors=[],calibration_only=True,
        scientific_admission=False,serving=False,partition='development_inspection',
        families=dict(repository=dict(ready=True,summary_sha256=sha((root/FILES[1]).read_bytes()))),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},code=code_pins())
    pin = write_committed(root/'report.json',report)
    for name in (*FILES,'report.json'):
        with (root/name).open('rb') as f:
            os.fsync(f.fileno())
    with (root/'COMMITTED').open('x') as f:
        f.write(pin)
        f.flush()
        os.fsync(f.fileno())
    sync_dir(root)
    return pin


class RepositoryPopulation:
    def __init__(self,root,pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        r = json.loads(raw)
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text() and raw == canonical(r) and
              r['schema'] == 'repository-native-population-v1' and r['code'] == code_pins() and
              r['complete'] and r['scientific_admission'] is False and set(r['files']) == set(FILES),
              'REPOSITORY_POPULATION_PIN')
        for name,value in r['files'].items():
            raw = (self.root/name).read_bytes()
            check(dict(bytes=len(raw),sha256=sha(raw)) == value,'REPOSITORY_POPULATION_BYTES')
        self.observations = RepositoryObservations(r['observations']['path'],r['observations']['pin'])
        rows,fields,receipts = expected(self.observations)
        check(pq.read_table(self.root/FILES[0]).to_pylist() == rows and
              json.loads((self.root/FILES[2]).read_bytes()) == fields and r['source'] == receipts,
              'REPOSITORY_POPULATION_CONTENT')
        check(len(rows) == r['inputs'] and sum(len(x['text'].encode()) for x in rows) == r['unique_utf8_bytes'],
              'REPOSITORY_POPULATION_COUNTS')
        self.rows,self.fields,self.report = rows,fields,r
