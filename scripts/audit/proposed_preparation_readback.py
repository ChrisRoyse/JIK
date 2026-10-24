"""Physical preparation readback and rejection controls; never execute the program."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from claims.proposed_program import validate as source_check
from oracle.prepare_proposal import PreparedProposal, validate, inherited


def physical(root):
    return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob('*')) if p.is_file()}


def run(root, pin, output):
    start = time.perf_counter()
    check(not output.exists(), 'PROPOSAL_AUDIT_EXISTS')
    before = physical(root)
    value = PreparedProposal(root, pin)
    source, spec = value.source, value.spec
    source_before = physical(source.root)
    runtime_root = Path(spec['runtime_source']['root'])
    runtime_before = physical(runtime_root)
    authenticated = time.perf_counter()
    # Rebuild each boundary from bytes rather than from the producer's return value.
    left, right = source.row['utf8_span']
    actual_program = (source.root/'action.txt').read_bytes()[left:right]
    check(actual_program == (source.root/'program.py').read_bytes(), 'PROPOSAL_AUDIT_SPAN')
    c = json.loads((root/'condition-commitment.json').read_bytes())
    public = json.loads((root/'condition-inputs.json').read_bytes())['inputs']
    counts, boundaries = {}, {}
    for name, row in spec['conditions'].items():
        work, runtime = root/name/'work', root/name/'runtime'
        check((work/'program.py').read_bytes() == actual_program, 'PROPOSAL_AUDIT_PROGRAM')
        wheels = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.iterdir()}
        original = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (runtime_root/name/'runtime').iterdir()}
        check(wheels == original == row['wheels'], 'PROPOSAL_AUDIT_RUNTIME')
        observation = json.loads((work/'observation.json').read_bytes())
        check(observation == dict(program_sha256=hashlib.sha256(actual_program).hexdigest(), wheels=wheels),
              'PROPOSAL_AUDIT_OBSERVATION')
        projection = public[c['routing'][name]]
        check(projection['program'].encode() == actual_program and projection['wheels'] == wheels
              and projection['image'] == spec['image'] and digest(projection) == c['routing'][name]
              and set(projection) == {'schema', 'program', 'program_sha256', 'image', 'wheels',
                                      'harness_sha256', 'mode', 'scope'}, 'PROPOSAL_AUDIT_PUBLIC')
        counts[name] = dict(program_bytes=len(actual_program), wheel_copies=len(wheels),
                           wheel_bytes=sum(p.stat().st_size for p in runtime.iterdir()))
        boundaries[name] = dict(program=hashlib.sha256(actual_program).hexdigest(),
            runtime=wheels, observation=hashlib.sha256((work/'observation.json').read_bytes()).hexdigest(),
            public=c['routing'][name], no_historical_target=True)
    rebuilt = time.perf_counter()
    output.mkdir()
    scratch = output/'control-copy'
    shutil.copytree(root, scratch)
    controls = {}

    def reject(name, code, fn):
        try:
            fn()
        except ValueError as exc:
            check(code in str(exc), 'PROPOSAL_AUDIT_CONTROL_CODE')
        else:
            raise AssertionError('PROPOSAL_AUDIT_CONTROL_ACCEPTED_'+name)
        controls[name] = code

    def change_file(relative, code):
        path = scratch/relative
        old = path.read_bytes()
        try:
            path.write_bytes(old+b' ')
            reject(relative, code, lambda:PreparedProposal(scratch, pin))
        finally:
            path.write_bytes(old)

    change_file('with_dependency/work/program.py', 'PROPOSAL_PREPARED_BYTES')
    change_file('without_dependency/runtime/packaging-24.1-py3-none-any.whl', 'PROPOSAL_PREPARED_BYTES')
    change_file('process_inside.py', 'PROPOSAL_PREPARED_BYTES')
    change_file('prepared.json', 'PROPOSAL_PREPARED_PIN')
    changed = copy.deepcopy(source.row)
    changed['utf8_span'][0] += 1
    reject('source_span', 'PROPOSAL_SPAN', lambda:source_check(changed, source.action, source.program))
    reject('source_program', 'PROPOSAL_SPAN', lambda:source_check(source.row, source.action, source.program+b' '))
    runtime_spec = inherited(runtime_root, spec['runtime_source']['pin'])
    changed_spec = copy.deepcopy(spec)
    changed_spec['target'] = dict(kind='recorded_response')
    reject('historical_target', 'PROPOSAL_PREPARED_FIELDS', lambda:validate(changed_spec, source, runtime_spec))
    changed_condition = copy.deepcopy(spec)
    changed_condition['conditions']['with_dependency']['target'] = dict(kind='recorded_response')
    reject('condition_target', 'PROPOSAL_PREPARED_CONDITION',
           lambda:validate(changed_condition, source, runtime_spec))
    # Even coherently rehashed public metadata cannot become model input.
    files = ['condition-inputs.json', 'condition-commitment.json', 'conditions-COMMITTED']
    saved = {n:(scratch/n).read_bytes() for n in files}
    try:
        altered_public = json.loads(saved[files[0]])
        key = c['routing']['with_dependency']
        entry = altered_public['inputs'].pop(key)
        entry['target'] = 'historical-response-injection'
        replacement = digest(entry)
        altered_public['inputs'][replacement] = entry
        encoded = canonical(altered_public)
        altered_c = copy.deepcopy(c)
        altered_c['inputs'] = sha(encoded)
        altered_c['routing']['with_dependency'] = replacement
        committed = canonical(altered_c)
        (scratch/files[0]).write_bytes(encoded)
        (scratch/files[1]).write_bytes(committed)
        (scratch/files[2]).write_text(sha(committed))
        reject('rehashed_public_target', 'CONDITION_INPUT_FIELDS',
               lambda:PreparedProposal(scratch, sha(committed)))
    finally:
        for name, raw in saved.items():
            (scratch/name).write_bytes(raw)
    check(physical(scratch) == before, 'PROPOSAL_AUDIT_CONTROL_RESTORE')
    PreparedProposal(scratch, pin)
    controlled = time.perf_counter()
    check(physical(root) == before and physical(source.root) == source_before
          and physical(runtime_root) == runtime_before, 'PROPOSAL_AUDIT_SOURCE_MUTATION')
    now = time.perf_counter()
    phases = dict(authentication=authenticated-start, independent_boundaries=rebuilt-authenticated,
                  controls=controlled-rebuilt, final_readback=now-controlled)
    report = dict(schema='proposed-preparation-readback-v1', acceptance='author-checked',
        root=str(root), pin=pin, source=source.reference(), counts=counts, boundaries=boundaries,
        inventory=before, controls=controls, original_and_scratch_restored=True,
        phases=phases, wall_seconds=now-start, phase_coverage=sum(phases.values())/(now-start),
        forecast_present=False, execution_present=False, scientific_admission=False,
        code_sha256=sha(Path(__file__).read_bytes()))
    report_pin = write_committed(output/'readback.json', report)
    return report, report_pin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, pin = run(args.root, args.pin, args.output)
    print(json.dumps(dict(report_pin=pin, **{k:v for k,v in report.items() if k not in ('inventory','boundaries')})))
