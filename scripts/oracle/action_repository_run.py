"""Commission a recorded action's original task oracle; never a learned repair claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.bound_rows import BoundRows
from oracle.cohort_pipeline import Builds, run_case
from oracle.prepare_swe import ATTEMPT_COLUMNS, BENCHMARK_COLUMNS
from oracle.resolve_runtime import Fetcher
from oracle.test_contract import selectors
from senses.contract import require
from senses.trajectory_population import publish, read_pinned

CODE = 'MNLM_ACTION_REPOSITORY_RUN'


def run(root, source, pin, output):
    start = time.perf_counter()
    root, output = root.resolve(), output.resolve()
    require(output.is_relative_to(root) and not output.exists(), CODE, 'fresh MNLM output required')
    scope = read_pinned(source, pin)
    require(scope['schema'] == 'action-repository-source-v1' and scope['status'] == 'git_exact'
            and scope['independent_archive_readback_equal'], CODE, 'source not commissioned')
    case = scope['case']
    require(case['partition'] == 'development_inspection', CODE, 'development only')
    output.mkdir(parents=True)
    tick = time.perf_counter()
    rows = BoundRows(root, [(case['benchmark'], BENCHMARK_COLUMNS),
        (case['benchmark'], BENCHMARK_COLUMNS+['created_at']), (case['attempt'], ATTEMPT_COLUMNS)])
    metadata = rows.read(case['benchmark'], BENCHMARK_COLUMNS)
    required = {}
    for role in ('FAIL_TO_PASS','PASS_TO_PASS'):
        value = metadata[role]
        values = json.loads(value) if isinstance(value,str) else value
        require(len(values) == len(set(values)) and not set(required)&set(values), CODE, 'test identities')
        required.update((n,role) for n in values)
    case['test_modules'] = sorted({n.split('::',1)[0] for n in selectors(required)})
    publish(output/'bound-rows.json', rows.receipts)
    publish(output/'case.json', case)
    binding_seconds = time.perf_counter()-tick
    fetcher = Fetcher(root, root/'raw/swe-runtime-v2', 64 << 20, 256 << 20)
    builds = Builds(root, output, fetcher)
    tick = time.perf_counter()
    result = run_case(root, output, case, rows, fetcher, builds,
                      root/'raw/swe-runtime-v2/uv-cache', output)
    execution_seconds = time.perf_counter()-tick
    wall = time.perf_counter()-start
    phases = dict(binding=binding_seconds, runtime_execution_readback=execution_seconds,
                  other=wall-binding_seconds-execution_seconds)
    summary = dict(schema='action-repository-run-v1', source=dict(path=str(source),sha256=pin),
        result=result, fetched_bytes=fetcher.used, budgets=fetcher.budget,
        phases=phases, wall_seconds=wall, required_test_count=len(required),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        partition='development_inspection', observed_action_origin='recorded_source_attempt',
        model_generated_action=False, learned_skill=False, admitted=False, serving=False,
        acceptance='author-checked only')
    summary_pin = publish(output/'summary.json', summary)
    return dict(path=str(output/'summary.json'), pin=summary_pin, status=result['status'],
                agent_outcome=result.get('agent_outcome'), phases=phases)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','source','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-pin', required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root,a.source,a.source_pin,a.output),sort_keys=True))
