"""Commission original task runtimes for both fixed learning increments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.repository_cohort_input import RepositoryCohort
from oracle.bound_rows import BoundRows
from oracle.cohort_pipeline import Builds,run_case
from oracle.complete_ids import truncated
from oracle.prepare_swe import ATTEMPT_COLUMNS,BENCHMARK_COLUMNS
from oracle.resolve_runtime import Fetcher
from oracle.test_contract import selectors
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_COHORT_WITNESS'


def run(root,cohort_path,cohort_pin,sources_path,sources_pin,proof_path,proof_pin,output):
    start = time.perf_counter()
    root,output = root.resolve(),output.resolve()
    require(output.is_relative_to(root) and not output.exists(),CODE,'fresh MNLM output required')
    cohort = RepositoryCohort(cohort_path,cohort_pin)
    source_report = read_pinned(sources_path,sources_pin)
    proof = read_pinned(proof_path,proof_pin)
    require(source_report['schema'] == 'repository-cohort-sources-v1' and
            source_report['cohort']['sha256'] == proof['cohort'] == cohort_pin and
            proof['source_report'] == sources_pin and proof['counts'] == source_report['counts'],
            CODE,'source proof binding')
    selected = cohort.training_cases('learn_1')+cohort.training_cases('learn_2')
    require(len(selected) == 6,CODE,'fixed learning increments')
    cases,requests = [],[]
    for item in selected:
        row = source_report['cases'][item['index']]
        source = read_pinned(Path(row['path']),row['sha256'])
        require(row['task'] == source['task'] == item['task'] and source['status'] == 'git_exact' and
                source['partition'] == item['partition'] and source['licensing_files'],
                CODE,'training source unavailable')
        case = {k:item[k] for k in ('index','task','partition','benchmark')}
        case.update(src=source['src'],attempt=item['attempts'][0]['binding'])
        cases.append(case)
        requests += [(case['benchmark'],BENCHMARK_COLUMNS),
                     (case['benchmark'],BENCHMARK_COLUMNS+['created_at']),
                     (case['attempt'],ATTEMPT_COLUMNS)]
    output.mkdir(parents=True)
    rows = BoundRows(root,requests)
    publish(output/'bound-rows.json',rows.receipts)
    contracts,contract_errors = [],{}
    for case in cases:
        try:
            metadata = rows.read(case['benchmark'],BENCHMARK_COLUMNS)
            required = {}
            for role in ('FAIL_TO_PASS','PASS_TO_PASS'):
                values = metadata[role]
                require(isinstance(values,list) and len(values) == len(set(values)) and
                        not set(required)&set(values),CODE,'original test role contract')
                required.update((n,role) for n in values)
            # Validate module/function syntax now; only real reference collection may complete a parameter ID.
            require('FAIL_TO_PASS' in required.values(),CODE,'empty fail-to-pass contract')
            for node,role in required.items():
                # selectors checks a complete contract's roles too; this singleton call checks syntax only.
                selectors({node.split('[',1)[0] if truncated(node) else node:'FAIL_TO_PASS'})
            case['test_modules'] = sorted({n.split('::',1)[0] for n in required})
            contracts.append(dict(task=case['task'],partition=case['partition'],tests=len(required),
                roles=dict(Counter(required.values())),modules=case['test_modules'],
                truncated_test_ids=sum(truncated(n) for n in required)))
        except (SenseError,ValueError,KeyError,TypeError) as exc:
            error = exc.envelope() if isinstance(exc,SenseError) else dict(
                code=CODE+'_CONTRACT',message='Invalid retained test contract.',exception=type(exc).__name__)
            contract_errors[case['index']] = error
            contracts.append(dict(task=case['task'],partition=case['partition'],error=error))
    publish(output/'cases.json',cases)
    publish(output/'contracts.json',contracts)
    binding_seconds = time.perf_counter()-start
    print(json.dumps(dict(stage='bound',cases=contracts,binding_seconds=binding_seconds)),flush=True)
    fetcher = Fetcher(root,root/'raw/swe-runtime-v2',64 << 20,256 << 20)
    builds = Builds(root,output,fetcher)
    results = []
    tick = time.perf_counter()
    for case in cases:
        case_start = time.perf_counter()
        destination = output/'tasks'/str(case['index']).zfill(2)
        destination.mkdir(parents=True)
        try:
            if case['index'] in contract_errors:
                result = dict(index=case['index'],task=case['task'],status='gap',
                              error=contract_errors[case['index']],runs=[])
            else:
                result = run_case(root,output,case,rows,fetcher,builds,
                                  root/'raw/swe-runtime-v2/uv-cache',destination)
        except Exception as exc:
            error = exc.envelope() if isinstance(exc,SenseError) else dict(
                code=CODE+'_CASE',message='Case failed; inspect retained artifacts.',exception=type(exc).__name__)
            result = dict(index=case['index'],task=case['task'],status='gap',error=error,runs=[])
        result.update(partition=case['partition'],wall_seconds=time.perf_counter()-case_start)
        pin = publish(destination/'result.json',result)
        entry = dict(index=case['index'],task=case['task'],partition=case['partition'],
                     status=result['status'],path=str(destination/'result.json'),sha256=pin,
                     wall_seconds=result['wall_seconds'])
        results.append(entry)
        print(json.dumps(entry),flush=True)
    execution_seconds = time.perf_counter()-tick
    wall = time.perf_counter()-start
    summary = dict(schema='repository-cohort-witness-v1',root=str(root),
        cohort=dict(path=str(cohort_path),sha256=cohort_pin),
        sources=dict(path=str(sources_path),sha256=sources_pin),
        proof=dict(path=str(proof_path),sha256=proof_pin),results=results,contracts=contracts,
        counts=dict(Counter(r['status'] for r in results)),fetched_bytes=fetcher.used,budgets=fetcher.budget,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=wall,phases=dict(binding=binding_seconds,execution=execution_seconds,
                                     other=wall-binding_seconds-execution_seconds),
        evaluation_executed=False,action_origin='retained_recorded_patches',model_generated_action=False,
        learned_skill=False,admitted=False,acceptance='author-checked')
    pin = publish(output/'summary.json',summary)
    print(json.dumps(dict(path=str(output/'summary.json'),sha256=pin,counts=summary['counts'],
                          wall_seconds=wall)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort','sources','proof','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('cohort-pin','sources-pin','proof-pin'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    run(a.root,a.cohort,a.cohort_pin,a.sources,a.sources_pin,a.proof,a.proof_pin,a.output)
