"""Prepare a new proposed-program occurrence without inventing a prior response."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.process_conditions import ConditionInputs, condition_input, publish, write_committed
from claims.proposed_program import ProposedProgram
from oracle.prepare_swe import inventory
from production.publication import sync_dir


NAMES = {'with_dependency', 'without_dependency'}
FLAGS = {'historical_occurrence_established':False, 'task_correctness_established':False,
         'forecast_present':False, 'execution_present':False, 'scientific_admission':False}


def code_pins():
    scripts = Path(__file__).resolve().parents[1]
    return {n:sha((scripts/n).read_bytes()) for n in (
        'claims/proposed_program.py', 'oracle/prepare_proposal.py', 'claims/process_conditions.py',
        'oracle/prepare_swe.py', 'oracle/process_inside.py', 'oracle/container_run.py')}


def inherited(root, pin):
    root = Path(root)
    raw = (root/'prepared.json').read_bytes()
    check(sha(raw) == pin, 'PROPOSAL_RUNTIME_PIN')
    value = json.loads(raw)
    check(raw == canonical(value) and value['schema'] == 'tool-process-prepared-v1'
          and set(value['conditions']) == NAMES, 'PROPOSAL_RUNTIME_CONTRACT')
    for name, row in value['conditions'].items():
        check(inventory(root/name) == row['files'], 'PROPOSAL_RUNTIME_BYTES')
        expected = {'runtime/'+n:p for n,p in row['wheels'].items()}
        check({n:p for n,p in row['files'].items() if n.startswith('runtime/')} == expected,
              'PROPOSAL_RUNTIME_ROSTER')
    return value


def validate(spec, source, runtime):
    check(set(spec) == {'schema', 'source', 'source_root', 'program_source', 'program_span',
        'program_sha256', 'runtime_source', 'image', 'harness_sha256', 'conditions', 'code', *FLAGS},
        'PROPOSAL_PREPARED_FIELDS')
    check(spec['schema'] == 'proposed-process-prepared-v1' and spec['source'] == source.pin
          and spec['source_root'] == str(source.root) and spec['program_source'] == source.reference()
          and spec['program_span'] == source.row['utf8_span']
          and spec['program_sha256'] == sha(source.program)
          and all(spec[k] is v for k,v in FLAGS.items()), 'PROPOSAL_PREPARED_SOURCE')
    check(spec['image'] == runtime['image'] and spec['code'] == code_pins()
          and spec['harness_sha256'] == spec['code']['oracle/process_inside.py'],
          'PROPOSAL_PREPARED_RUNTIME')
    check(set(spec['conditions']) == NAMES, 'PROPOSAL_PREPARED_ROSTER')
    for name, row in spec['conditions'].items():
        check(set(row) == {'wheels', 'files'} and row['wheels'] == runtime['conditions'][name]['wheels'],
              'PROPOSAL_PREPARED_CONDITION')
        observation = dict(program_sha256=spec['program_sha256'], wheels=row['wheels'])
        expected = {'runtime/'+n:p for n,p in row['wheels'].items()}
        expected.update({'work/program.py':sha(source.program), 'work/observation.json':sha(canonical(observation))})
        check(row['files'] == expected, 'PROPOSAL_PREPARED_INVENTORY')


def build(source_root, source_pin, runtime_root, runtime_pin, root):
    started = time.perf_counter()
    check(not root.exists(), 'PROPOSAL_OUTPUT_EXISTS')
    source = ProposedProgram(source_root, source_pin)
    source_at = time.perf_counter()
    runtime = inherited(runtime_root, runtime_pin)
    runtime_at = time.perf_counter()
    root.mkdir()
    spec = dict(schema='proposed-process-prepared-v1', source=source.pin, source_root=str(source.root),
        program_source=source.reference(), program_span=source.row['utf8_span'],
        program_sha256=sha(source.program), runtime_source=dict(root=str(runtime_root), pin=runtime_pin),
        image=runtime['image'], harness_sha256=code_pins()['oracle/process_inside.py'],
        conditions={}, code=code_pins(), **FLAGS)
    for name in sorted(NAMES):
        for part in ('work', 'runtime'):
            (root/name/part).mkdir(parents=True)
        wheels = runtime['conditions'][name]['wheels']
        for filename, pin in wheels.items():
            raw = (runtime_root/name/'runtime'/filename).read_bytes()
            check(sha(raw) == pin, 'PROPOSAL_RUNTIME_COPY')
            (root/name/'runtime'/filename).write_bytes(raw)
        (root/name/'work/program.py').write_bytes(source.program)
        write_committed(root/name/'work/observation.json', dict(program_sha256=sha(source.program), wheels=wheels))
        spec['conditions'][name] = dict(wheels=wheels, files=inventory(root/name))
    (root/'action.txt').write_bytes(source.action)
    harness = Path(__file__).with_name('process_inside.py').read_bytes()
    check(sha(harness) == spec['harness_sha256'], 'PROPOSAL_HARNESS_COPY')
    (root/'process_inside.py').write_bytes(harness)
    validate(spec, source, runtime)
    prepared_pin = write_committed(root/'prepared.json', spec)
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as f:
                os.fsync(f.fileno())
    copied_at = time.perf_counter()
    condition_pin = publish(root, spec, prepared_pin, spec['harness_sha256'])
    sync_dir(root)
    read = PreparedProposal(root, condition_pin)
    check(read.spec == spec, 'PROPOSAL_PREPARED_READBACK')
    now = time.perf_counter()
    phases = dict(source=source_at-started, runtime=runtime_at-source_at,
        copy_and_commit=copied_at-runtime_at, public_commit_and_readback=now-copied_at)
    return dict(schema='proposed-process-preparation-v1', root=str(root), pin=condition_pin,
        prepared_pin=prepared_pin, source=source.reference(), phases=phases, wall_seconds=now-started,
        phase_coverage=sum(phases.values())/(now-started), code=code_pins(), **FLAGS)


class PreparedProposal:
    def __init__(self, root, pin):
        self.root = Path(root)
        self.conditions = ConditionInputs(root, pin)
        raw = (self.root/'prepared.json').read_bytes()
        check(sha(raw) == self.conditions.commitment['prepared'], 'PROPOSAL_PREPARED_PIN')
        self.spec = json.loads(raw)
        check(raw == canonical(self.spec), 'PROPOSAL_PREPARED_CANONICAL')
        spec = self.spec
        self.source = ProposedProgram(spec['source_root'], spec['source'])
        runtime = inherited(spec['runtime_source']['root'], spec['runtime_source']['pin'])
        validate(spec, self.source, runtime)
        check((self.root/'action.txt').read_bytes() == self.source.action
              and sha((self.root/'process_inside.py').read_bytes()) == spec['harness_sha256'],
              'PROPOSAL_PREPARED_BYTES')
        c = self.conditions.commitment
        check(c['source'] == spec['source'] and c['program_source'] == spec['program_source']
              and c['program_span'] == spec['program_span'] and set(c['routing']) == NAMES,
              'PROPOSAL_PUBLIC_SOURCE')
        for name, item in spec['conditions'].items():
            check(inventory(self.root/name) == item['files'], 'PROPOSAL_PREPARED_BYTES')
            public = condition_input(spec, name, self.source.program, spec['harness_sha256'])
            check(public == self.conditions.read(c['routing'][name]), 'PROPOSAL_PUBLIC_INPUT')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'runtime', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('source-pin', 'runtime-pin'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    report = build(args.source, args.source_pin, args.runtime, args.runtime_pin, args.output)
    pin = write_committed(args.output.parent/(args.output.name+'-preparation.json'), report)
    print(json.dumps(dict(report_pin=pin, **{k:v for k,v in report.items() if k != 'code'})))
