"""Use the existing isolated oracle while preserving unavailable-source gaps per task."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle import repository_cohort_witness as base
from claims.repository_cohort_input import RepositoryCohort
from senses.contract import SenseError,require
from senses.trajectory_population import publish,read_pinned

CODE = 'MNLM_REPOSITORY_GROWTH_WITNESS'


def error(exc):
    return exc.envelope() if isinstance(exc,SenseError) else dict(
        code=CODE+'_CASE',message='Task unavailable; inspect retained evidence.',exception=type(exc).__name__)


def contract(case,rows):
    metadata = rows.read(case['benchmark'],base.BENCHMARK_COLUMNS)
    required = {}
    for role in ('FAIL_TO_PASS','PASS_TO_PASS'):
        values = metadata[role]
        require(isinstance(values,list) and len(values) == len(set(values)) and
                not set(required)&set(values),CODE,'original test role contract')
        required.update((n,role) for n in values)
    require('FAIL_TO_PASS' in required.values(),CODE,'empty fail-to-pass contract')
    for node in required:
        base.selectors({node.split('[',1)[0] if base.truncated(node) else node:'FAIL_TO_PASS'})
    case['test_modules'] = sorted({n.split('::',1)[0] for n in required})
    return dict(task=case['task'],partition=case['partition'],tests=len(required),
        roles=dict(Counter(required.values())),modules=case['test_modules'],
        truncated_test_ids=sum(base.truncated(n) for n in required))


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
    cases,requests,errors = [],[],{}
    for item in selected:
        row = source_report['cases'][item['index']]
        source = read_pinned(Path(row['path']),row['sha256'])
        require(row['task'] == source['task'] == item['task'] and
                row['status'] == source['status'] and source['partition'] == item['partition'],
                CODE,'source identity differs')
        case = {k:item[k] for k in ('index','task','partition','benchmark')}
        if source['status'] == 'unavailable':
            errors[case['index']] = dict(code=CODE+'_SOURCE',message='Original source unavailable.',
                source_error=source['error'],source=dict(path=row['path'],sha256=row['sha256']))
        else:
            require(source['status'] == 'git_exact' and source['licensing_files'],CODE,'source scope')
            case.update(src=source['src'],attempt=item['attempts'][0]['binding'])
            requests += [(case['benchmark'],base.BENCHMARK_COLUMNS),
                         (case['benchmark'],base.BENCHMARK_COLUMNS+['created_at']),
                         (case['attempt'],base.ATTEMPT_COLUMNS)]
        cases.append(case)
    output.mkdir(parents=True)
    rows = base.BoundRows(root,requests)
    publish(output/'bound-rows.json',rows.receipts)
    contracts = []
    for case in cases:
        if case['index'] not in errors:
            try:
                contracts.append(contract(case,rows))
                continue
            except (SenseError,ValueError,KeyError,TypeError) as exc:
                errors[case['index']] = error(exc)
        contracts.append(dict(task=case['task'],partition=case['partition'],error=errors[case['index']]))
    publish(output/'cases.json',cases)
    publish(output/'contracts.json',contracts)
    binding = time.perf_counter()-start
    print(json.dumps(dict(stage='bound',cases=contracts,seconds=binding)),flush=True)
    fetcher = base.Fetcher(root,root/'raw/swe-runtime-v2',64 << 20,256 << 20)
    builds = base.Builds(root,output,fetcher)
    results = []
    tick = time.perf_counter()
    for case in cases:
        case_start = time.perf_counter()
        destination = output/'tasks'/str(case['index']).zfill(2)
        destination.mkdir(parents=True)
        try:
            if case['index'] in errors:
                result = dict(index=case['index'],task=case['task'],status='gap',
                              error=errors[case['index']],runs=[])
            else:
                result = base.run_case(root,output,case,rows,fetcher,builds,
                    root/'raw/swe-runtime-v2/uv-cache',destination)
        except Exception as exc:
            result = dict(index=case['index'],task=case['task'],status='gap',error=error(exc),runs=[])
        result.update(partition=case['partition'],wall_seconds=time.perf_counter()-case_start)
        pin = publish(destination/'result.json',result)
        entry = dict(index=case['index'],task=case['task'],partition=case['partition'],
            status=result['status'],path=str(destination/'result.json'),sha256=pin,
            wall_seconds=result['wall_seconds'])
        results.append(entry)
        print(json.dumps(entry),flush=True)
    execution = time.perf_counter()-tick
    wall = time.perf_counter()-start
    summary = dict(schema='repository-cohort-witness-v1',producer='repository_growth_witness.py',
        root=str(root),cohort=dict(path=str(cohort_path),sha256=cohort_pin),
        sources=dict(path=str(sources_path),sha256=sources_pin),
        proof=dict(path=str(proof_path),sha256=proof_pin),results=results,contracts=contracts,
        counts=dict(Counter(r['status'] for r in results)),fetched_bytes=fetcher.used,budgets=fetcher.budget,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_writer_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        wall_seconds=wall,phases=dict(binding=binding,execution=execution,other=wall-binding-execution),
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
