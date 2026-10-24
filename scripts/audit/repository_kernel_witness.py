"""Real journal prefixes -> sparse native associations -> recall-gated retrieval."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import sha
from claims.process_conditions import write_committed
from kernel.native_route import check
from kernel.repository_population import RepositoryPopulation
from kernel.repository_kernel import AssociationCompanion,RepositoryKernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    check(not args.output.exists(),'REPOSITORY_WITNESS_EXISTS')
    args.output.mkdir(parents=True)
    lease = args.output/'lease.json'
    write_committed(lease,dict(pid=os.getpid(),started=time.time(),scope='repository_association_cpu'))
    start = time.perf_counter()
    config = dict(journal=str(args.journal),pin=args.pin,initial_groups=12)
    write_committed(args.output/'config.json',config)
    from kernel.repository_kernel import code_pins
    write_committed(args.output/'producer-code.json',code_pins())
    population = RepositoryPopulation(args.journal,args.pin)
    loaded = time.perf_counter()
    excluded,queries = population.holdout(config['initial_groups'])
    check(len(population.groups) == 14,'REPOSITORY_WITNESS_GROUPS')
    split = dict(excluded_tasks=excluded,queries=queries,rule='first_two_SHA256_task_from_initial_12_intakes')
    write_committed(args.output/'split-before-scoring.json',split)
    companion = AssociationCompanion(population,excluded,queries)
    stages = []
    for groups in (12,13,14):
        root = args.output/('prefix-'+str(groups))
        value,pin,timing = companion.sync(groups,root)
        row = dict(root=str(root),pin=pin,groups=groups,source=value['source'],timings=timing,
            recall={n:r['recall'] for n,r in value['calibration'].items()},passes=value['recall_passes'],
            work=value['work'],nodes=len(value['ids']),absent=len(value['absent']))
        stages.append(row)
        print(json.dumps(row),flush=True)
    last = stages[-1]
    before = {n:sha((Path(last['root'])/n).read_bytes()) for n in ('manifest.json','cache.json','COMMITTED')}
    _,same,timing = companion.sync(14,args.output/'no-change')
    check(same == last['pin'] and timing['reused_artifact'] and not (args.output/'no-change').exists(),
          'REPOSITORY_NOOP')
    check(before == {n:sha((Path(last['root'])/n).read_bytes()) for n in before},'REPOSITORY_NOOP_BYTES')
    withdrawn = [sorted(companion.last['ids'])[0]]
    removed,removed_pin,removed_timing = companion.sync(14,args.output/'withdrawn',withdrawn=withdrawn)
    stages.append(dict(root=str(args.output/'withdrawn'),pin=removed_pin,groups=14,
        source=removed['source'],timings=removed_timing,withdrawn=withdrawn,
        recall={n:r['recall'] for n,r in removed['calibration'].items()},passes=removed['recall_passes'],
        work=removed['work'],nodes=len(removed['ids']),absent=len(removed['absent'])))
    controls,routes = [],[]
    def refuses(name,code,fn):
        try:fn()
        except ValueError as error:
            check(code in str(error),'REPOSITORY_CONTROL_CODE')
        else:raise AssertionError('MNLM_REPOSITORY_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    cutoff = max(r['available'] for r in population.records.values())
    for stage in stages:
        if not stage['passes']:
            refuses('recall_gate_'+Path(stage['root']).name,'REPOSITORY_RECALL_GATE',lambda:
                RepositoryKernel(population,stage['root'],stage['pin'],expected_source=stage['source']))
            continue
        kernel = RepositoryKernel(population,stage['root'],stage['pin'],expected_source=stage['source'])
        trace = kernel.route(queries[0],source=stage['source'],cutoff=cutoff)
        check(not set(withdrawn if 'withdrawn' in stage else ()) & set(trace['sources']),'REPOSITORY_WITHDRAWN_ROUTE')
        routes.append(dict(root=stage['root'],trace=trace))
        bad = copy.deepcopy(stage['source']);bad['head'] = '0'*64
        refuses('stale_head_'+Path(stage['root']).name,'REPOSITORY_STALE_HEAD',lambda:
            kernel.route(queries[0],source=bad,cutoff=cutoff))
        refuses('future_source_'+Path(stage['root']).name,'REPOSITORY_ROUTE_CUTOFF',lambda:
            kernel.route(queries[0],source=stage['source'],cutoff='1970-01-01T00:00:00Z'))
    population.fresh()
    end = time.perf_counter()
    result = dict(schema='repository-kernel-witness-v1',source_journal=args.pin,
        source_head=population.rows[-1][2],groups=len(population.groups),constellations=len(population.records),
        split=split,stages=stages,noop=timing,controls=controls,routes=routes,
        wall_seconds=end-start,load_seconds=loaded-start,stage_seconds=sum(s['timings']['wall_seconds'] for s in stages),
        source_unchanged=sha(args.journal.read_bytes()) == args.pin,number_status='measured',
        acceptance='author-checked',scientific_admission=False,learned_capability=False,serving=False)
    pin = write_committed(args.output/'witness.json',result)
    write_committed(args.output/'finished.json',dict(pin=pin,pid=os.getpid(),finished=time.time()))
    print(json.dumps(dict(pin=pin,passes=[s['passes'] for s in stages],wall_seconds=end-start)),flush=True)


if __name__ == '__main__':
    try:main()
    except BaseException:
        if '--output' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--output')+1])
            if root.is_dir():write_committed(root/'failure.json',dict(traceback=traceback.format_exc()))
        raise
