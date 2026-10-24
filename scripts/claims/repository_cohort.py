"""Freeze a project-separated learning cohort without consulting outcome columns."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain
from claims.codata_source import check,sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from production.publication import sync_dir
from senses.trajectory_population import open_bound,read_pinned

SEED = 'repository-learning-cohort-v1:20261021'
BENCHMARK_FIELDS = ('instance_id','repo','base_commit','license','created_at')
ATTEMPT_FIELDS = ('instance_id','generated_patch')
TASK = re.compile(rb'[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+')
PATCH_LIMIT = 1 << 20
POLICY = dict(seed=SEED,projects=12,attempts_per_task=4,minimum_distinct_patches=2,
    patch_limit=PATCH_LIMIT,selection='sha256_seed_task_then_first_task_per_project',
    roles=['learn_1']*3+['learn_2']*3+['development']*3+['reserved_evaluation']*3,
    outcomes_read=False,evaluation_status='reserved_pending_training_exposure_audit',
    scientific_admission=False,serving=False)


def rank(task):
    return sha((SEED+'\0'+task).encode())


def inputs(config):
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    journal = Path(config['journal'])
    check(sha(journal.read_bytes()) == proof['journal'],'COHORT_JOURNAL_PIN')
    _,records = chain(journal)
    check(len(records) == proof['events'] and records[-1][2] == proof['head'],'COHORT_JOURNAL_HEAD')
    exposed = sorted({m.decode() for body,payload,_ in records
        for m in TASK.findall(canonical(dict(body=body,payload=payload)))})
    contracts = read_pinned(Path(config['contracts']),config['contracts_pin'])
    census = read_pinned(Path(config['census']),config['census_pin'])
    excluded = {t.rsplit('-',1)[0].replace('__','/',1).casefold() for t in exposed}
    excluded.update(c['repo'].casefold() for c in contracts['contracts'])
    benchmark = contracts['benchmark']
    root = Path(benchmark['root'])
    return root,benchmark,census['source_scope'],exposed,sorted(excluded)


def collect(config):
    start = time.perf_counter()
    root,benchmark,scope,exposed,excluded = inputs(config)
    tick = time.perf_counter()
    with open_bound(root,benchmark) as stream:
        parquet = pq.ParquetFile(stream)
        check(parquet.metadata.num_rows == benchmark['rows'],'COHORT_BENCHMARK_ROWS')
        rows = parquet.read(columns=list(BENCHMARK_FIELDS)).to_pylist()
    metadata = {}
    for i,row in enumerate(rows):
        task = row['instance_id']
        check(task not in metadata,'COHORT_DUPLICATE_BENCHMARK_TASK')
        metadata[task] = dict(row=row,address=i)
    phases = dict(binding=tick-start,benchmark=time.perf_counter()-tick)
    tick = time.perf_counter()
    candidates,rejected = defaultdict(dict),Counter()
    counts = Counter()
    for shard in scope['shards']:
        with open_bound(root/'raw/swe-agent-trajectories',shard) as stream:
            parquet = pq.ParquetFile(stream)
            check(parquet.metadata.num_rows == shard['rows'],'COHORT_ATTEMPT_ROWS')
            offset = 0
            for batch in parquet.iter_batches(batch_size=512,columns=list(ATTEMPT_FIELDS)):
                for local,row in enumerate(batch.to_pylist()):
                    counts['attempt_rows'] += 1
                    task,patch = row['instance_id'],row['generated_patch']
                    if task not in metadata:
                        rejected['task_not_in_benchmark'] += 1
                        continue
                    meta = metadata[task]['row']
                    if meta['repo'].casefold() in excluded:
                        rejected['exposed_project'] += 1
                        continue
                    if not isinstance(patch,str) or not patch.strip():
                        rejected['missing_recorded_patch'] += 1
                        continue
                    raw = patch.encode()
                    counts['projected_patch_bytes'] += len(raw)
                    if len(raw) > PATCH_LIMIT:
                        rejected['patch_above_byte_cap'] += 1
                        continue
                    pin = sha(raw)
                    if pin in candidates[task]:
                        rejected['duplicate_patch'] += 1
                        continue
                    if len(candidates[task]) == POLICY['attempts_per_task']:
                        rejected['additional_patch_occurrence'] += 1
                        continue
                    binding = dict(shard,path='raw/swe-agent-trajectories/'+shard['path'],row=offset+local)
                    candidates[task][pin] = dict(binding=binding,raw=raw,sha256=pin,bytes=len(raw))
                offset += batch.num_rows
            check(offset == shard['rows'],'COHORT_ATTEMPT_COVERAGE')
    phases['attempt_projection'] = time.perf_counter()-tick
    eligible = [t for t,v in candidates.items() if len(v) >= POLICY['minimum_distinct_patches']
                and isinstance(metadata[t]['row']['license'],str) and metadata[t]['row']['license']]
    selected,projects = [],set()
    for task in sorted(eligible,key=rank):
        project = metadata[task]['row']['repo'].casefold()
        if project in projects:
            continue
        projects.add(project)
        selected.append(task)
        if len(selected) == POLICY['projects']:
            break
    check(len(selected) == POLICY['projects'],'COHORT_INSUFFICIENT_PROJECTS')
    cases,patches = [],{}
    for i,task in enumerate(selected):
        row = metadata[task]['row']
        check(re.fullmatch('[0-9a-f]{40}',row['base_commit']) is not None,'COHORT_BASE_COMMIT')
        attempts = []
        for pin,item in candidates[task].items():
            patches[pin] = item['raw']
            attempts.append(dict(binding=item['binding'],path='patches/'+pin+'.diff',
                                 sha256=pin,bytes=item['bytes']))
        cases.append(dict(index=i,task=task,repo=row['repo'],base_commit=row['base_commit'],
            source_license=row['license'],created_at=row['created_at'],partition=POLICY['roles'][i],
            rank=rank(task),benchmark=dict(benchmark,row=metadata[task]['address']),attempts=attempts))
    return dict(cases=cases,patches=patches,exposed=exposed,excluded=excluded,scope=scope,
        counts=dict(counts,benchmark_rows=len(rows),candidate_tasks=len(candidates),
                    eligible_tasks=len(eligible),selected_tasks=len(cases),
                    selected_attempts=sum(len(c['attempts']) for c in cases),
                    selected_patch_bytes=sum(map(len,patches.values()))),
        rejected=dict(rejected),phases=phases)


def build(config,root):
    start = time.perf_counter()
    check(not root.exists(),'COHORT_OUTPUT_EXISTS')
    result = collect(config)
    root.mkdir()
    (root/'patches').mkdir()
    files = {}
    for pin,raw in result.pop('patches').items():
        path = 'patches/'+pin+'.diff'
        (root/path).write_bytes(raw)
        files[path] = dict(bytes=len(raw),sha256=pin)
    for name,value in (('cases.json',result.pop('cases')),
                       ('exposure.json',dict(tasks=result.pop('exposed'),projects=result.pop('excluded'))),
                       ('config.json',config)):
        pin = write_committed(root/name,value)
        files[name] = dict(bytes=(root/name).stat().st_size,sha256=pin)
    wall = time.perf_counter()-start
    result['phases']['selection_and_publication'] = wall-sum(result['phases'].values())
    report = dict(schema='repository-learning-cohort-v1',policy=POLICY,files=files,**result,
        wall_seconds=wall,code=sha(Path(__file__).read_bytes()),acceptance='author-checked',
        columns_read=dict(benchmark=list(BENCHMARK_FIELDS),attempt=list(ATTEMPT_FIELDS)))
    pin = write_committed(root/'report.json',report)
    for name in (*files,'report.json'):
        with (root/name).open('rb') as stream:
            os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(root/'patches')
    sync_dir(root)
    print(json.dumps(dict(pin=pin,counts=result['counts'],wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    build(json.loads(args.config.read_bytes()),args.output)
