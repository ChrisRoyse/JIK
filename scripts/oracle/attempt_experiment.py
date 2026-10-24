"""Run the exact predeclared proposal twice after durable forecast publication."""
import json
import os
from pathlib import Path
import time

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims import attempt_forecast
from claims.process_conditions import write_committed
from claims.attempt_observation import AttemptObservations
from oracle.prepare_attempt import PreparedAttempt
from oracle.prepare_swe import inventory
from oracle.container_run import execute
from production.publication import sync_dir


def run(root, forecast_root, forecast_pin):
    started = time.perf_counter()
    root = Path(root)
    check(not root.exists(), 'PROPOSAL_EXECUTION_EXISTS')
    seal, payload = attempt_forecast.read(forecast_root,forecast_pin)
    check(str(root) == seal['forecast']['spec']['execution'], 'PROPOSAL_EXECUTION_ROOT')
    restart_raw = (root.parent/'forecast-restart.json').read_bytes()
    restart = json.loads(restart_raw)
    check(restart['forecast'] == forecast_pin and restart['cold_replay'] is True
          and restart['executed'] is False and restart['head'] == seal['head'],
          'ATTEMPT_EXECUTION_RESTART_REQUIRED')
    prepared = PreparedAttempt(payload['conditions']['root'],payload['conditions']['pin'])
    check(prepared.source.reference() == payload['source'], 'PROPOSAL_EXECUTION_SOURCE')
    root.mkdir()
    validated = time.perf_counter()
    probe = execute(prepared.spec['image'],prepared.root/'without_dependency/work',
                    prepared.root/'without_dependency/runtime',root/'probe','probe')
    check(probe['status'] == 'observed' and probe['exitcode'] == 0 and probe['result']['probe']['uid'] == 65534
          and all(v is True for k,v in probe['result']['probe'].items() if k != 'uid'),
          'PROPOSAL_EXECUTION_ISOLATION')
    probed = time.perf_counter()
    entries = {}
    for name in ('without_dependency','with_dependency'):
        execute(prepared.spec['image'],prepared.root/name/'work',prepared.root/name/'runtime',
                root/name/'execution','observe-script')
        execution = json.loads((root/name/'execution/execution.json').read_bytes())
        after = json.loads((root/name/'execution/after.json').read_bytes())
        check(execution['status'] == 'observed' and execution['exitcode'] == 0, 'PROPOSAL_EXECUTION_CAPTURE')
        check(inventory(prepared.root/name) == prepared.spec['conditions'][name]['files'],
              'PROPOSAL_EXECUTION_INPUT_MUTATED')
        condition = prepared.conditions.commitment['routing'][name]
        actual = dict(result=execution['result'],image=execution['image'],started=after['State']['StartedAt'])
        actual['conditions'] = prepared.conditions.match(condition,dict(program_sha256=prepared.spec['program_sha256'],
            wheels=prepared.spec['conditions'][name]['wheels']),actual)
        attempt_forecast.match(seal,payload,condition,actual)
        entry = dict(name=name,condition=condition,proposal=prepared.source.reference(),
                     execution=name+'/execution',occurrence=execution['container_id'])
        entries[digest(entry)] = entry
    executed = time.perf_counter()
    files = {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
             for p in sorted(root.rglob('*')) if p.is_file()}
    hashed = time.perf_counter()
    manifest = dict(schema='attempt-observations-v1',prepared=payload['conditions'],
        forecast_restart=sha(restart_raw),
        source=prepared.source.reference(),forecast=forecast_pin,observations=entries,files=files,
        historical_occurrence_established=False,task_correctness_established=False,scientific_admission=False,
        phases=dict(validation=validated-started,confinement_probe=probed-validated,
                    executions=executed-probed,inventory=hashed-executed),wall_seconds=hashed-started)
    pin = write_committed(root/'manifest.json',manifest)
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    with (root/'COMMITTED').open('x') as stream:
        stream.write(pin)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(root)
    AttemptObservations(root,pin,forecast_root,forecast_pin)
    return pin
