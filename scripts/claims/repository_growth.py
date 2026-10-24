"""Extend a frozen repository cohort by excluding all earlier project partitions.

The catalogue is a selection exclusion input, never an execution contract. Existing
cohort selection, native population and isolated oracle implementations stay pinned.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.process_conditions import write_committed
from claims.repository_cohort import inputs
from claims.repository_cohort_input import RepositoryCohort
from senses.trajectory_population import open_bound,read_pinned


def sources(recipe):
    original = RepositoryCohort(**recipe['original'])
    prior = [RepositoryCohort(**entry) for entry in recipe['prior']]
    check(prior and len({c.pin for c in prior}) == len(prior) and
          original.pin in {c.pin for c in prior},'GROWTH_PRIOR_ROSTER')
    projects = [c['repo'].casefold() for cohort in prior for c in cohort.cases]
    check(len(projects) == len(set(projects)),'GROWTH_PRIOR_OVERLAP')
    for cohort in prior:
        check(all(cohort.config[k] == original.config[k] for k in ('census','census_pin')),
              'GROWTH_SOURCE_SCOPE')
    contracts = read_pinned(Path(original.config['contracts']),original.config['contracts_pin'])
    entries = list(contracts['contracts'])
    for cohort in prior:
        for case in cohort.cases:
            entries.append(dict(kind='previous_cohort_project_exclusion',repo=case['repo'],
                task=case['task'],partition=case['partition'],
                source=dict(root=str(cohort.root),pin=cohort.pin,index=case['index'])))
    catalogue = dict(schema='repository-selection-exclusions-v1',
        benchmark=contracts['benchmark'],contracts=entries,execution_contract=False,
        original=dict(path=original.config['contracts'],sha256=original.config['contracts_pin']),
        prior=recipe['prior'])
    config = {k:original.config[k] for k in ('census','census_pin')}
    config.update({k:recipe[k] for k in ('journal','proof','proof_pin')})
    return original,prior,catalogue,config


def census(config):
    root,benchmark,scope,exposed,excluded = inputs(config)
    with open_bound(root,benchmark) as stream:
        table = pq.read_table(stream,columns=['instance_id','repo'])
    check(table.num_rows == benchmark['rows'],'GROWTH_METADATA_ROWS')
    meta = dict(zip(table['instance_id'].to_pylist(),table['repo'].to_pylist(),strict=True))
    check(len(meta) == table.num_rows,'GROWTH_DUPLICATE_TASK')
    tasks,rows = set(),0
    for shard in scope['shards']:
        with open_bound(root/'raw/swe-agent-trajectories',shard) as stream:
            table = pq.read_table(stream,columns=['instance_id'])
        check(table.num_rows == shard['rows'],'GROWTH_ATTEMPT_ROWS')
        rows += table.num_rows
        tasks.update(table['instance_id'].to_pylist())
    remaining = {t:meta[t] for t in tasks if t in meta and meta[t].casefold() not in excluded}
    return dict(benchmark_rows=benchmark['rows'],attempt_rows=rows,retained_tasks=len(tasks),
        remaining_tasks=len(remaining),remaining_projects=len({v.casefold() for v in remaining.values()}),
        excluded_projects=len(excluded),exposed_task_ids=len(exposed),
        eligibility='metadata_intersection_only; distinct_patch_eligibility_pending_selection',
        columns_read=dict(benchmark=['instance_id','repo'],attempt=['instance_id']))


def build(recipe,output):
    start = time.perf_counter()
    check(not output.exists(),'GROWTH_OUTPUT_EXISTS')
    original,prior,catalogue,config = sources(recipe)
    binding = time.perf_counter()-start
    output.mkdir(parents=True)
    catalogue_pin = write_committed(output/'exclusions.json',catalogue)
    config.update(contracts=str(output/'exclusions.json'),contracts_pin=catalogue_pin)
    tick = time.perf_counter()
    counted = census(config)
    counting = time.perf_counter()-tick
    files = {}
    for name,value in (('recipe.json',recipe),('config.json',config),('exclusions.json',catalogue)):
        if name != 'exclusions.json':
            write_committed(output/name,value)
        data = (output/name).read_bytes()
        files[name] = dict(bytes=len(data),sha256=sha(data))
    wall = time.perf_counter()-start
    report = dict(schema='repository-growth-v1',files=files,counts=counted,
        prior_projects=sum(len(c.cases) for c in prior),
        protected_prior_projects=sum(c['partition'] in ('development','reserved_evaluation')
                                     for cohort in prior for c in cohort.cases),
        wall_seconds=wall,phases=dict(binding=binding,census=counting,publication=wall-binding-counting),
        code=sha(Path(__file__).read_bytes()),outcomes_read=False,scientific_admission=False,
        execution_contract=False,serving=False,acceptance='author-checked')
    pin = write_committed(output/'report.json',report)
    write_committed(output/'COMMITTED.json',dict(sha256=pin))
    print(json.dumps(dict(pin=pin,counts=counted,wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    build(json.loads(a.recipe.read_bytes()),a.output.resolve())
