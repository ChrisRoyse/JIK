"""Execute only recovered dependencies, retaining unchanged cohort witnesses and actions."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle import repository_cohort_actions as actions
from oracle import repository_growth_witness as witness
from senses.contract import require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_RECOVERY_EXECUTION'


def run(recipe,output):
    start = time.perf_counter()
    source = read_pinned(Path(recipe['sources']),recipe['sources_pin'])
    proof = read_pinned(Path(recipe['proof']),recipe['proof_pin'])
    old = read_pinned(Path(recipe['witness']),recipe['witness_pin'])
    old_actions = read_pinned(Path(recipe['actions']),recipe['actions_pin'])
    old_check = read_pinned(Path(recipe['actions_proof']),recipe['actions_proof_pin'])
    require(proof['source_report'] == recipe['sources_pin'] and
            proof['counts'] == source['counts'] and old['sources']['sha256'] ==
            source['recovery']['recipe']['pin'] and old['cohort'] == source['cohort'] and
            old_actions['witness'] == dict(path=recipe['witness'],sha256=recipe['witness_pin']) and
            old_check['source'] == dict(path=recipe['actions'],sha256=recipe['actions_pin']),
            CODE,'recovery and prior execution proof chain')
    root = Path(old['root'])
    require(output.resolve().is_relative_to(root) and not output.exists(),CODE,'fresh MNLM output')
    cohort = RepositoryCohort(old['cohort']['path'],old['cohort']['sha256'])
    cases = cohort.training_cases('learn_1')+cohort.training_cases('learn_2')
    indexes = {c['index'] for c in source['recovery']['recipe']['recover']}
    require(indexes <= {c['index'] for c in cases},CODE,'learning only')
    selected,requests = [],[]
    for case in cases:
        if case['index'] not in indexes:
            continue
        entry = source['cases'][case['index']]
        actual = read_pinned(Path(entry['path']),entry['sha256'])
        require(actual['status'] == 'git_exact' and actual['task'] == case['task'],CODE,'recovered source')
        selected.append(dict(index=case['index'],task=case['task'],partition=case['partition'],
            benchmark=case['benchmark'],attempt=case['attempts'][0]['binding'],src=actual['src']))
        requests += [(case['benchmark'],witness.base.BENCHMARK_COLUMNS),
                     (case['benchmark'],witness.base.BENCHMARK_COLUMNS+['created_at']),
                     (case['benchmark'],actions.META)]
        requests.extend((a['binding'],actions.ATTEMPT) for a in case['attempts'])
    output.mkdir(parents=True)
    rows = witness.base.BoundRows(root,requests)
    publish(output/'bound-rows.json',rows.receipts)
    fetcher = witness.base.Fetcher(root,root/'raw/swe-runtime-v2',64 << 20,256 << 20)
    require(fetcher.budget == old['budgets'],CODE,'budget identity')
    # This is a continuation of the same cohort allowance, not a fresh allowance.
    fetcher.used = dict(old['fetched_bytes'])
    builds = witness.base.Builds(root,output,fetcher)
    prior_sources = read_pinned(Path(old['sources']['path']),old['sources']['sha256'])
    results,contracts = copy.deepcopy(old['results']),copy.deepcopy(old['contracts'])
    positions = {c['index']:i for i,c in enumerate(cases)}
    for case,first,later in zip(cases,old['results'],old_actions['results'],strict=True):
        require(case['task'] == first['task'] == later['task'] and
                case['partition'] == first['partition'] == later['partition'],CODE,'old selection')
        if case['index'] not in indexes:
            before = prior_sources['cases'][case['index']]
            after = source['cases'][case['index']]
            require(read_pinned(Path(before['path']),before['sha256']) ==
                    read_pinned(Path(after['path']),after['sha256']),CODE,'unchanged dependency')
    binding = time.perf_counter()-start
    tick = time.perf_counter()
    for case in selected:
        pos = positions[case['index']]
        require(results[pos]['status'] == 'gap',CODE,'do not rerun observed witness')
        target = output/'tasks'/str(case['index']).zfill(2)
        target.mkdir(parents=True)
        began = time.perf_counter()
        try:
            contracts[pos] = witness.contract(case,rows)
            result = witness.base.run_case(root,output,case,rows,fetcher,builds,
                                          root/'raw/swe-runtime-v2/uv-cache',target)
        except Exception as exc:
            result = dict(index=case['index'],task=case['task'],status='gap',runs=[],error=witness.error(exc))
        result.update(partition=case['partition'],wall_seconds=time.perf_counter()-began)
        pin = publish(target/'result.json',result)
        results[pos] = dict(index=case['index'],task=case['task'],partition=case['partition'],
            status=result['status'],path=str(target/'result.json'),sha256=pin,
            wall_seconds=result['wall_seconds'])
        print(json.dumps(results[pos]),flush=True)
    witness_seconds = time.perf_counter()-tick
    cumulative = dict(fetcher.used)
    require(all(cumulative[k] <= fetcher.budget[k] for k in cumulative),CODE,'cumulative fetch budget')
    summary = dict(old,producer='repository_recovery_execution.py',
        sources=dict(path=recipe['sources'],sha256=recipe['sources_pin']),
        proof=dict(path=recipe['proof'],sha256=recipe['proof_pin']),results=results,contracts=contracts,
        recovery=dict(recipe=recipe,writer_sha256=actions.digest(Path(__file__)),
                      executed_indexes=sorted(indexes),reused_indexes=sorted(set(positions)-indexes)),
        counts=dict(Counter(r['status'] for r in results)),fetched_bytes=cumulative,
        additional_fetched_bytes={k:cumulative[k]-old['fetched_bytes'][k] for k in cumulative},
        wall_seconds=binding+witness_seconds,phases=dict(binding=binding,witness=witness_seconds))
    witness_pin = publish(output/'witness.json',summary)
    code = {p.name:actions.digest(p) for p in Path(__file__).parent.glob('*.py')}
    results = copy.deepcopy(old_actions['results'])
    tick = time.perf_counter()
    for case in cases:
        if case['index'] not in indexes:
            continue
        pos = positions[case['index']]
        first = summary['results'][pos]
        observed = read_pinned(Path(first['path']),first['sha256'])
        if observed['status'] != 'observed':
            results[pos] = dict(task=case['task'],partition=case['partition'],status='gap',
                error='ORIGINAL_TASK_WITNESS_REQUIRED',source_error=observed['error'])
            continue
        plan,prepared,prep_pin,spec,before,modes,files,shared = actions.open_witness(root,first,observed,code)
        later = []
        for ordinal,attempt in enumerate(case['attempts'][1:],1):
            target = output/'actions'/str(case['index']).zfill(2)/str(ordinal)
            target.parent.mkdir(parents=True,exist_ok=True)
            try:
                result = actions.action(root,cohort,case,attempt,rows,plan,prepared,spec,files,modes,shared,target)
            except Exception as exc:
                target.mkdir(exist_ok=True)
                error = witness.error(exc)
                gap_pin = publish(target/'gap.json',dict(task=case['task'],attempt=attempt,error=error))
                result = dict(status='gap',path=str(target/'gap.json'),sha256=gap_pin,error=error)
            later.append(dict(ordinal=ordinal,**result))
            print(json.dumps(later[-1]),flush=True)
        results[pos] = dict(task=case['task'],partition=case['partition'],status='executed',actions=later,witness=shared)
    action_seconds = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(old_actions,witness=dict(path=str(output/'witness.json'),sha256=witness_pin),
        results=results,recovery=summary['recovery'],
        action_counts=dict(Counter(a['status'] for r in results for a in r.get('actions',[]))),
        wall_seconds=wall,phases=dict(binding=binding,witness=witness_seconds,actions=action_seconds,
                                    other=wall-binding-witness_seconds-action_seconds))
    action_pin = publish(output/'actions.json',report)
    print(json.dumps(dict(witness=witness_pin,actions=action_pin,counts=summary['counts'],
                          additional_fetched_bytes=summary['additional_fetched_bytes'],wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(json.loads(a.recipe.read_bytes()),a.output.resolve())
