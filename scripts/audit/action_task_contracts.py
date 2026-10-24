"""Bind inspected proposals to original requirements and evaluator-only test contracts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from senses.trajectory_population import publish


BENCHMARK = dict(path='raw/trajectory-benchmark-metadata/nebius-SWE-bench-extra/'
    '11dcbfb30e19552df2a2f8030bd764adc95c92a5/data/train-00000-of-00001.parquet',
    bytes=24675794, rows=6376,
    sha256='c8633563d41c1a2fde0361650f96b5103b11f74d7feac5c9abf33f58e486d946')
INPUT_FIELDS = ['instance_id', 'repo', 'base_commit', 'problem_statement', 'license']
ORACLE_FIELDS = ['test_patch', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit']


def validate(request, row):
    check(request['task'] == row['instance_id'], 'TASK_CONTRACT_IDENTITY')
    check(request['issue'].strip() == row['problem_statement'].strip(), 'TASK_CONTRACT_REQUIREMENT')
    tests = {k:json.loads(row[k]) if isinstance(row[k], str) else row[k]
             for k in ('FAIL_TO_PASS', 'PASS_TO_PASS')}
    check(all(isinstance(v, list) and all(isinstance(x, str) and x for x in v)
              and len(v) == len(set(v)) for v in tests.values())
          and tests['FAIL_TO_PASS'] and isinstance(row['test_patch'], str) and row['test_patch'],
          'TASK_CONTRACT_TESTS')
    check(not set(tests['FAIL_TO_PASS']) & set(tests['PASS_TO_PASS']), 'TASK_CONTRACT_TEST_OVERLAP')
    commit = row['base_commit']
    check(isinstance(commit, str) and len(commit) == 40 and
          all(c in '0123456789abcdef' for c in commit), 'TASK_CONTRACT_COMMIT')
    return tests


def requests(census, pin):
    raw = (census/'report.json').read_bytes()
    check(sha(raw) == pin == (census/'COMMITTED').read_text(), 'TASK_CONTRACT_CENSUS_PIN')
    report = json.loads(raw)
    raw = (census/'chains.json').read_bytes()
    check(dict(bytes=len(raw), sha256=sha(raw)) == report['files']['chains.json'],
          'TASK_CONTRACT_CHAINS')
    result = []
    for c in json.loads(raw):
        m = c['messages'][1]
        check(m['turn'] == 1 and m['role'] == 'user' and m['text_present'], 'TASK_CONTRACT_REQUEST')
        raw = (census/m['file']).read_bytes()
        check(dict(bytes=len(raw), sha256=sha(raw)) == report['files'][m['file']],
              'TASK_CONTRACT_REQUEST_PIN')
        text = raw.decode()
        check(text.count('\nISSUE:\n') == 1 and text.count('\nINSTRUCTIONS:') == 1,
              'TASK_CONTRACT_REQUEST_ENVELOPE')
        issue = text.split('\nISSUE:\n', 1)[1].split('\nINSTRUCTIONS:', 1)[0]
        result.append(dict(task=c['task'], chain=c['id'], request_file=m['file'],
            request_sha256=sha(raw), issue=issue, proposal_occurrences=len(c['programs'])))
    return result


def run(census, census_pin, data_root, output):
    start = time.perf_counter()
    reqs = requests(census, census_pin)
    targets = {r['task'] for r in reqs}
    path = data_root/BENCHMARK['path']
    check(path.stat().st_size == BENCHMARK['bytes'], 'TASK_CONTRACT_SOURCE_BYTES')
    with path.open('rb') as stream:
        check(hashlib.file_digest(stream, 'sha256').hexdigest() == BENCHMARK['sha256'],
              'TASK_CONTRACT_SOURCE_PIN')
        stream.seek(0)
        parquet = pq.ParquetFile(stream)
        check(parquet.metadata.num_rows == BENCHMARK['rows'], 'TASK_CONTRACT_ROWS')
        offset, located = 0, {}
        for batch in parquet.iter_batches(batch_size=1024, columns=['instance_id']):
            for i,task in enumerate(batch.column(0).to_pylist()):
                if task in targets:
                    check(task not in located, 'TASK_CONTRACT_DUPLICATE_TASK')
                    located[task] = offset+i
            offset += batch.num_rows
        check(set(located) == targets, 'TASK_CONTRACT_MISSING_TASK')
        offset, projected = 0, {}
        for group in range(parquet.num_row_groups):
            length = parquet.metadata.row_group(group).num_rows
            wanted = {t:i for t,i in located.items() if offset <= i < offset+length}
            if wanted:
                table = parquet.read_row_group(group, columns=INPUT_FIELDS+ORACLE_FIELDS)
                for task,address in wanted.items():
                    projected[task] = table.slice(address-offset,1).to_pylist()[0]
            offset += length
    contracts = []
    for request in reqs:
        row = projected[request['task']]
        tests = validate(request, row)
        contracts.append(dict(task=request['task'], chain=request['chain'],
            request_file=request['request_file'], request_sha256=request['request_sha256'],
            requirement_matches_after_outer_whitespace=True,
            requirement_sha256=sha(row['problem_statement'].encode()),
            benchmark_row=located[request['task']], repo=row['repo'], base_commit=row['base_commit'],
            source_license=row['license'],
            evaluator_only=dict(field_digests={k:digest(row[k]) for k in ORACLE_FIELDS},
                tests={k:len(v) for k,v in tests.items()}, test_patch_bytes=len(row['test_patch'].encode())),
            original_repository_runtime_prepared=False, original_repository_tests_executed=False,
            task_success_established=False))
    controls = {}
    for name, change in (
        ('changed_task', lambda r:r.update(instance_id='wrong-task')),
        ('changed_requirement', lambda r:r.update(problem_statement='different requested behavior')),
        ('missing_selectors', lambda r:r.update(FAIL_TO_PASS='[]'))):
        row = copy.deepcopy(projected[reqs[0]['task']])
        change(row)
        try:
            validate(reqs[0], row)
        except ValueError as exc:
            controls[name] = str(exc)
        else:
            raise ValueError('TASK_CONTRACT_CONTROL_ACCEPTED:'+name)
    result = dict(schema='action-task-contracts-v1', census=dict(root=str(census), pin=census_pin),
        benchmark=dict(root=str(data_root), **BENCHMARK), columns_read=INPUT_FIELDS+ORACLE_FIELDS,
        model_inputs_written=False, reference_patch_read=False, source_outcome_labels_read=False,
        partition='development_inspection', scientific_admission=False, serving=False,
        contracts=contracts, controls=controls, code_sha256=sha(Path(__file__).read_bytes()),
        wall_seconds=time.perf_counter()-start, acceptance='author-checked only')
    pin = publish(output, result)
    persisted = json.loads(output.read_bytes())
    check(persisted == result, 'TASK_CONTRACT_PERSISTENCE')
    # Independent raw-column batch path reads back the persisted bindings.
    with path.open('rb') as stream:
        check(hashlib.file_digest(stream,'sha256').hexdigest() == BENCHMARK['sha256'],
              'TASK_CONTRACT_REPLAY_SOURCE')
        stream.seek(0)
        offset, matched = 0, 0
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=256, columns=INPUT_FIELDS+ORACLE_FIELDS):
            for c in persisted['contracts']:
                if offset <= c['benchmark_row'] < offset+batch.num_rows:
                    raw = batch.slice(c['benchmark_row']-offset,1).to_pylist()[0]
                    check(all(raw[k] == c[out] for k,out in
                              [('instance_id','task'),('repo','repo'),('base_commit','base_commit')])
                          and sha(raw['problem_statement'].encode()) == c['requirement_sha256']
                          and {k:digest(raw[k]) for k in ORACLE_FIELDS} == c['evaluator_only']['field_digests'],
                          'TASK_CONTRACT_RAW_READBACK')
                    matched += 1
            offset += batch.num_rows
        check(matched == len(reqs), 'TASK_CONTRACT_READBACK_COVERAGE')
    readback = dict(schema='action-task-contract-readback-v1', report_pin=pin,
        contracts=matched, distinct_tasks=len(targets), controls=controls,
        raw_batch_readback_equal=True, acceptance='author-checked only')
    readback_pin = publish(output.with_name(output.stem+'-readback.json'), readback)
    return dict(path=str(output), pin=pin, readback_pin=readback_pin, **readback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--census-pin', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.census,a.census_pin,a.data_root,a.output),sort_keys=True))
