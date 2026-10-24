"""Prepare a distinct proposal using the existing pinned execution runtime."""
import json
import os
from pathlib import Path

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.fresh_proposal import FreshProposal
from claims.process_conditions import ConditionInputs, condition_input, publish, write_committed
from oracle.prepare_proposal import FLAGS, NAMES, inherited
from oracle.prepare_swe import inventory
from production.publication import sync_dir


def code_pins():
    scripts = Path(__file__).resolve().parents[1]
    return {n:sha((scripts/n).read_bytes()) for n in ('oracle/prepare_fresh.py',
        'claims/fresh_proposal.py','claims/process_conditions.py','oracle/prepare_proposal.py',
        'oracle/prepare_swe.py','oracle/process_inside.py','oracle/container_run.py')}


def validate(spec, source, runtime):
    check(set(spec) == {'schema','source','source_root','program_source','program_span',
        'program_sha256','runtime_source','image','harness_sha256','conditions','code',*FLAGS},
        'FRESH_PREPARED_FIELDS')
    check(spec['schema'] == 'fresh-process-prepared-v1' and spec['source'] == source.pin
          and spec['source_root'] == str(source.root) and spec['program_source'] == source.reference()
          and spec['program_span'] == source.row['utf8_span']
          and spec['program_sha256'] == sha(source.program) and all(spec[k] is v for k,v in FLAGS.items()),
          'FRESH_PREPARED_SOURCE')
    check(spec['image'] == runtime['image'] and spec['code'] == code_pins()
          and spec['harness_sha256'] == spec['code']['oracle/process_inside.py'], 'FRESH_PREPARED_RUNTIME')
    check(set(spec['conditions']) == NAMES, 'FRESH_PREPARED_ROSTER')
    for name,row in spec['conditions'].items():
        wheels = runtime['conditions'][name]['wheels']
        files = {'runtime/'+n:p for n,p in wheels.items()}
        files.update({'work/program.py':sha(source.program),'work/observation.json':sha(canonical(
            dict(program_sha256=sha(source.program),wheels=wheels)))})
        check(row == dict(wheels=wheels,files=files), 'FRESH_PREPARED_INVENTORY')


def build(source, runtime_root, runtime_pin, root):
    check(not root.exists(), 'FRESH_PREPARED_EXISTS')
    runtime = inherited(runtime_root,runtime_pin)
    root.mkdir()
    code = code_pins()
    spec = dict(schema='fresh-process-prepared-v1',source=source.pin,source_root=str(source.root),
        program_source=source.reference(),program_span=source.row['utf8_span'],program_sha256=sha(source.program),
        runtime_source=dict(root=str(runtime_root),pin=runtime_pin),image=runtime['image'],
        harness_sha256=code['oracle/process_inside.py'],conditions={},code=code,**FLAGS)
    for name in sorted(NAMES):
        for part in ('work','runtime'):
            (root/name/part).mkdir(parents=True)
        wheels = runtime['conditions'][name]['wheels']
        for filename,pin in wheels.items():
            raw = (runtime_root/name/'runtime'/filename).read_bytes()
            check(sha(raw) == pin, 'FRESH_RUNTIME_COPY')
            (root/name/'runtime'/filename).write_bytes(raw)
        (root/name/'work/program.py').write_bytes(source.program)
        write_committed(root/name/'work/observation.json',dict(program_sha256=sha(source.program),wheels=wheels))
        spec['conditions'][name] = dict(wheels=wheels,files=inventory(root/name))
    (root/'action.txt').write_bytes(source.action)
    (root/'process_inside.py').write_bytes(Path(__file__).with_name('process_inside.py').read_bytes())
    validate(spec,source,runtime)
    pin = write_committed(root/'prepared.json',spec)
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    condition_pin = publish(root,spec,pin,spec['harness_sha256'])
    sync_dir(root)
    return PreparedFresh(root,condition_pin)


class PreparedFresh:
    def __init__(self, root, pin):
        self.root = Path(root)
        self.conditions = ConditionInputs(root,pin)
        raw = (self.root/'prepared.json').read_bytes()
        check(sha(raw) == self.conditions.commitment['prepared'], 'FRESH_PREPARED_PIN')
        self.spec = json.loads(raw)
        check(raw == canonical(self.spec), 'FRESH_PREPARED_CANONICAL')
        spec = self.spec
        self.source = FreshProposal(spec['source_root'],spec['source'])
        runtime = inherited(spec['runtime_source']['root'],spec['runtime_source']['pin'])
        validate(spec,self.source,runtime)
        check((self.root/'action.txt').read_bytes() == self.source.action
              and sha((self.root/'process_inside.py').read_bytes()) == spec['harness_sha256'],
              'FRESH_PREPARED_BYTES')
        c = self.conditions.commitment
        check(c['source'] == self.source.pin and c['program_source'] == self.source.reference()
              and c['program_span'] == self.source.row['utf8_span'] and set(c['routing']) == NAMES,
              'FRESH_PUBLIC_SOURCE')
        for name,row in spec['conditions'].items():
            check(inventory(self.root/name) == row['files'], 'FRESH_PREPARED_BYTES')
            check(condition_input(spec,name,self.source.program,spec['harness_sha256']) ==
                  self.conditions.read(c['routing'][name]), 'FRESH_PUBLIC_INPUT')
