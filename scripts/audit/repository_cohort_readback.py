"""Rebuild the selection from raw columns, then exercise persisted partition boundaries."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain
from claims.codata_source import check,sha
from claims.process_conditions import write_committed
from claims.repository_cohort_input import RepositoryCohort,partition_check
from senses.trajectory_population import open_bound,read_pinned


def audit(cohort):
    c,r = cohort.config,cohort.report
    proof = read_pinned(Path(c['proof']),c['proof_pin'])
    journal = Path(c['journal'])
    check(sha(journal.read_bytes()) == proof['journal'],'COHORT_READER_JOURNAL')
    _,events = chain(journal)
    check(len(events) == proof['events'] and events[-1][2] == proof['head'],'COHORT_READER_CHAIN')
    with sqlite3.connect(journal.as_uri()+'?mode=ro',uri=True) as db:
        records = [v[0] for v in db.execute('SELECT body FROM events')]
        records += [v[0] for v in db.execute('SELECT data FROM payloads')]
    tasks = sorted({m.decode() for raw in records for m in
                    re.findall(rb'[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+',raw)})
    contracts = read_pinned(Path(c['contracts']),c['contracts_pin'])
    excluded = {t.rsplit('-',1)[0].replace('__','/',1).casefold() for t in tasks}
    excluded |= {row['repo'].casefold() for row in contracts['contracts']}
    check(cohort.exposure == dict(tasks=tasks,projects=sorted(excluded)),'COHORT_READER_EXPOSURE')
    census = read_pinned(Path(c['census']),c['census_pin'])
    check(r['scope'] == census['source_scope'],'COHORT_READER_SOURCE_SCOPE')
    benchmark = contracts['benchmark']
    root = Path(benchmark['root'])
    with open_bound(root,benchmark) as stream:
        rows = pq.read_table(stream,columns=['instance_id','repo','base_commit','license','created_at']).to_pylist()
    meta = {v['instance_id']:(i,v) for i,v in enumerate(rows)}
    check(len(meta) == len(rows) == benchmark['rows'],'COHORT_READER_METADATA')
    retained = defaultdict(dict)
    scanned = 0
    for shard in r['scope']['shards']:
        with open_bound(root/'raw/swe-agent-trajectories',shard) as stream:
            table = pq.read_table(stream,columns=['instance_id','generated_patch'])
            check(table.num_rows == shard['rows'],'COHORT_READER_ROWS')
            for address in range(table.num_rows):
                scanned += 1
                task = table['instance_id'][address].as_py()
                raw = table['generated_patch'][address].as_py()
                if task not in meta or meta[task][1]['repo'].casefold() in excluded:
                    continue
                if not isinstance(raw,str) or not raw.strip() or len(raw.encode()) > (1 << 20):
                    continue
                data = raw.encode()
                pin = hashlib.sha256(data).hexdigest()
                if pin not in retained[task] and len(retained[task]) < 4:
                    retained[task][pin] = dict(binding=dict(shard,
                        path='raw/swe-agent-trajectories/'+shard['path'],row=address),raw=data)
    eligible = [t for t,v in retained.items() if len(v) >= 2 and
                isinstance(meta[t][1]['license'],str) and meta[t][1]['license']]
    ordered = sorted(eligible,key=lambda t:hashlib.sha256(('repository-learning-cohort-v1:20261021\0'+t).encode()).hexdigest())
    chosen,projects = [],set()
    for task in ordered:
        project = meta[task][1]['repo'].casefold()
        if project not in projects:
            chosen.append(task)
            projects.add(project)
        if len(chosen) == 12:
            break
    check(chosen == [v['task'] for v in cohort.cases] and scanned == r['counts']['attempt_rows'] and
          len(eligible) == r['counts']['eligible_tasks'],'COHORT_READER_SELECTION')
    patch_bytes = 0
    for case in cohort.cases:
        address,metadata = meta[case['task']]
        check(case['benchmark'] == dict(benchmark,row=address) and all(case[k] == metadata[v] for k,v in
            (('repo','repo'),('base_commit','base_commit'),('source_license','license'),('created_at','created_at'))),
            'COHORT_READER_CASE')
        expected = retained[case['task']]
        check([a['sha256'] for a in case['attempts']] == list(expected),'COHORT_READER_ATTEMPTS')
        for attempt in case['attempts']:
            source = expected[attempt['sha256']]
            raw = (cohort.root/attempt['path']).read_bytes()
            check(attempt['binding'] == source['binding'] and raw == source['raw'] and
                  len(raw) == attempt['bytes'],'COHORT_READER_PATCH_SOURCE')
            patch_bytes += len(raw)
    return dict(raw_attempt_rows=scanned,eligible_tasks=len(eligible),selected_tasks=len(chosen),
                selected_patch_bytes=patch_bytes,exposed_projects=len(excluded))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    start = time.perf_counter()
    cohort = RepositoryCohort(args.root,args.pin)
    before = {n:sha((args.root/n).read_bytes()) for n in cohort.report['files']}
    result = audit(cohort)
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as exc:
            check(code in str(exc),'COHORT_READER_CONTROL_CODE:'+str(exc))
        else:
            raise AssertionError('COHORT_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    refuse('reserved_used_for_training','COHORT_TRAINING_PARTITION',
           lambda:cohort.training_cases('reserved_evaluation'))
    refuse('development_used_for_training','COHORT_TRAINING_PARTITION',
           lambda:cohort.training_cases('development'))
    refuse('unaudited_evaluation','COHORT_EVALUATION_EXPOSURE_AUDIT_REQUIRED',cohort.evaluation_cases)
    bad = copy.deepcopy(cohort.cases)
    bad[0]['repo'] = bad[-1]['repo']
    refuse('cross_partition_project','COHORT_PROJECT_SEPARATION',lambda:partition_check(bad,cohort.exposure))
    bad = copy.deepcopy(cohort.cases)
    bad[-1]['partition'] = 'learn_1'
    refuse('changed_partition','COHORT_PARTITION_ORDER',lambda:partition_check(bad,cohort.exposure))
    with tempfile.TemporaryDirectory(prefix='repository-cohort-control-',dir=args.root.parent) as tmp:
        clone = Path(tmp)/'cohort'
        shutil.copytree(args.root,clone)
        patch = clone/cohort.cases[0]['attempts'][0]['path']
        value = bytearray(patch.read_bytes());value[0] ^= 1;patch.write_bytes(value)
        refuse('same_size_changed_patch','COHORT_FILE',lambda:RepositoryCohort(clone,args.pin))
        refuse('wrong_commitment','COHORT_PIN',lambda:RepositoryCohort(args.root,'0'*64))
    check(before == {n:sha((args.root/n).read_bytes()) for n in before},'COHORT_READER_MUTATED')
    check(len(cohort.training_cases('learn_1')) == len(cohort.training_cases('learn_2')) == 3,
          'COHORT_READER_INCREMENT_SIZE')
    report = dict(schema='repository-cohort-readback-v1',cohort=args.pin,results=result,controls=controls,
        files=before,wall_seconds=time.perf_counter()-start,acceptance='author-checked',
        evaluation_status='reserved_pending_training_exposure_audit',scientific_admission=False,serving=False)
    pin = write_committed(args.output,report)
    print(json.dumps(dict(pin=pin,results=result,controls=len(controls),wall_seconds=report['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()
