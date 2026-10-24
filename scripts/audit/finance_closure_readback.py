"""Bounded CPU execution of all predeclared finance cold-readback groups (#12)."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_closure_readback import closure, load_report, require, sha, write


def group_job(task):
    source, destination, names = task
    start = time.perf_counter(); cpu = time.process_time()
    destination.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1): result = closure(source, destination, names)
    result['wall_seconds'] = time.perf_counter()-start
    result['cpu_seconds'] = time.process_time()-cpu
    write(destination/'report.json', result)
    write(destination/'manifest.json', dict(report_sha256=sha(destination/'report.json')))
    return result


def run(a):
    start = time.perf_counter(); report = load_report(a.closure)
    pin = sha(a.closure/'report.json'); groups = defaultdict(list)
    for row in report['rows']: groups[row['name'].rsplit('-',1)[0]].append(row['name'])
    require(len(groups) == 54 and all(len(names) == 5 for names in groups.values()), 'FINANCE_COMPLETE_GROUPS')
    a.output.mkdir(parents=True, exist_ok=False)
    tasks = [(a.closure, a.output/name, names) for name,names in sorted(groups.items())]
    # First real group is the CPU cost measurement, not a discarded benchmark.
    first = group_job(tasks[0]); ratio = first['cpu_seconds']/first['wall_seconds']
    workers = min(32, os.cpu_count() or 1, len(tasks)-1) if ratio >= .8 else 1
    write(a.output/'intent.json', dict(source_report_sha256=pin, groups=54, cases=270,
        first_group=tasks[0][1].name, measured_cpu_seconds=first['cpu_seconds'],
        measured_wall_seconds=first['wall_seconds'], cpu_wall_ratio=ratio, workers=workers,
        invariants='frozen design fits, all native profile coordinates, same five resample seeds and complete count scans',
        scientific_admission=False, served=False))
    print({'first_group_seconds':first['wall_seconds'],'cpu_wall_ratio':ratio,'workers':workers},flush=True)
    results = [first]
    if workers == 1:
        for task in tasks[1:]: results.append(group_job(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(group_job, tasks[1:]):
                results.append(result)
                print({'groups_read_back':len(results),'total':len(tasks)},flush=True)
    require(sha(a.closure/'report.json') == pin, 'FINANCE_AFTER_PUBLICATION')
    rows = []; summary = {}; controls = {}; timings = defaultdict(float)
    for task,result in zip(tasks,results):
        require(result['report_sha256'] == pin and len(result['rows']) == 5 and result['geometry_builds'] == 1,
                'FINANCE_GROUP_BINDING')
        for row in result['rows']:
            row['artifact_file'] = str(Path(task[1].name)/(row['name']+'.npz'))
            rows.append(row)
        summary.update(result['summary']); controls[task[1].name] = result['negative_controls']
        for key,value in result['timing'].items(): timings[key] += value
    require(len(rows) == 270 and {r['name'] for r in rows} == {r['name'] for r in report['rows']}, 'FINANCE_ALL_ROWS')
    merged = dict(report_sha256=pin, rows=rows, summary=summary, negative_controls=controls,
                  geometry_builds=sum(r['geometry_builds'] for r in results),
                  original_source_files_rebound=len({r['source']['fixed_profile'] for r in report['rows']}),
                  timing=dict(timings))
    result = dict(schema='finance-closure-cold-v1', closure=merged, complete=True, author_checked=True,
        independently_accepted=False, scientific_admission=False, conditional_information_certified=False,
        served=False, wall_seconds=time.perf_counter()-start, workers=workers,
        code_sha256={str(p.relative_to(Path(__file__).parents[1])):sha(p) for p in
            (Path(__file__),Path(__file__).with_name('calyx_closure_readback.py'))})
    write(a.output/'report.json',result)
    write(a.output/'manifest.json',dict(report_sha256=sha(a.output/'report.json')))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('closure','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
