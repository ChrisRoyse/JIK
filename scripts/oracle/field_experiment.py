"""Execute a sealed pair of field interventions with the existing confined runner."""
import json
import os
from pathlib import Path
import shutil
import time

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.field_forecast import read, match
from claims.process_conditions import ConditionInputs, publish, condition_input, write_committed
from oracle.container_run import execute
from oracle.prepare_swe import inventory


def prepare(prepared, pin, output):
    prepared, output = Path(prepared), Path(output)
    raw = (prepared/'prepared.json').read_bytes()
    check(sha(raw) == pin, 'FIELD_PREPARED_PIN')
    spec = json.loads(raw)
    check(spec['schema'] == 'tool-process-prepared-v1' and len(spec['conditions']) == 2,
          'FIELD_PREPARED_SCHEMA')
    for name, item in spec['conditions'].items():
        check(name in ('without_dependency', 'with_dependency') and inventory(prepared/name) == item['files'],
              'FIELD_PREPARED_BYTES')
    check(not output.exists(), 'FIELD_EXECUTION_OUTPUT_EXISTS')
    shutil.copytree(prepared, output)
    harness = Path(__file__).with_name('process_inside.py')
    shutil.copyfile(harness, output/'process_inside.py')
    return publish(output, spec, pin, sha(harness.read_bytes()))


def run(root, forecast_root, forecast_pin):
    started = time.perf_counter()
    root = Path(root)
    seal, payload = read(forecast_root, forecast_pin)
    check(payload['conditions']['root'] == str(root), 'FIELD_EXECUTION_ROOT')
    conditions = ConditionInputs(root, payload['conditions']['pin'])
    raw = (root/'prepared.json').read_bytes()
    check(sha(raw) == conditions.commitment['prepared'], 'FIELD_PREPARED_PIN')
    spec = json.loads(raw)
    for name, item in spec['conditions'].items():
        check(inventory(root/name) == item['files'], 'FIELD_PREPARED_BYTES')
        public = condition_input(spec, name, (root/name/'work/program.py').read_bytes(),
                                 sha((root/'process_inside.py').read_bytes()))
        check(public == conditions.read(conditions.commitment['routing'][name]), 'FIELD_EXECUTION_INPUT')
    validated = time.perf_counter()
    probe = execute(spec['image'], root/'without_dependency/work', root/'without_dependency/runtime', root/'probe', 'probe')
    check(probe['status'] == 'observed' and probe['exitcode'] == 0 and probe['result']['probe']['uid'] == 65534
          and all(v is True for k,v in probe['result']['probe'].items() if k != 'uid'), 'FIELD_EXECUTION_ISOLATION')
    probed = time.perf_counter()
    observations = {}
    for name in ('without_dependency', 'with_dependency'):
        item = spec['conditions'][name]
        execute(spec['image'], root/name/'work', root/name/'runtime', root/name/'execution', 'observe-script')
        stored = json.loads((root/name/'execution/execution.json').read_bytes())
        after = json.loads((root/name/'execution/after.json').read_bytes())
        check(stored['status'] == 'observed' and stored['exitcode'] == 0, 'FIELD_EXECUTION_CAPTURE')
        for part in ('work', 'runtime'):
            check(inventory(root/name/part) == {k.removeprefix(part+'/'):v for k,v in item['files'].items()
                  if k.startswith(part+'/')}, 'FIELD_EXECUTION_MUTATION')
        entry = dict(target=item['target'], program_source=spec['program_source'], program_span=spec['program_span'],
            program_sha256=spec['program_sha256'], program=name+'/work/program.py', runtime=name+'/runtime',
            wheels=item['wheels'], execution=name+'/execution', condition=conditions.commitment['routing'][name],
            experiment=forecast_pin, occurrence=stored['container_id'])
        observed = dict(result=stored['result'], image=stored['image'], started=after['State']['StartedAt'])
        observed['conditions'] = conditions.match(entry['condition'], entry, observed)
        match(seal, payload, entry['condition'], observed)
        observations[digest(entry)] = entry
    executed = time.perf_counter()
    files = {f.relative_to(root).as_posix():dict(bytes=f.stat().st_size, sha256=sha(f.read_bytes()))
             for f in sorted(root.rglob('*')) if f.is_file()}
    hashed = time.perf_counter()
    manifest = dict(schema='tool-process-observations-v1', source=spec['source'],
        prepared=conditions.commitment['prepared'], files=files, observations=observations,
        condition_commitment=conditions.pin, controlled_forecast=forecast_pin,
        historical_occurrence_established=False, task_correctness_established=False,
        phases=dict(validation=validated-started, confinement_probe=probed-validated,
                    executions=executed-probed, inventory=hashed-executed), wall_seconds=hashed-started)
    pin = write_committed(root/'manifest.json', manifest)
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    from production.publication import sync_dir
    sync_dir(root)
    return pin
