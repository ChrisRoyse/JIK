"""Commit, execute and cold-read a controlled pair in the existing experience journal."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.fsv_identity_loop import configure
import torch
from audit.fsv_field_world import authenticate
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import ConditionInputs
from claims.process_observation import ProcessObservations
from claims.tool_temporal_source import ToolSource
from claims import field_forecast
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.tool_temporal_input import ToolInputs
from model.tool_kernel import ToolKernel
from model.field_experiment import ExperimentLoop
from oracle.field_experiment import prepare, run
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('forecast', 'execute', 'feedback', 'recover', 'cold'), required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.output
    if a.stage == 'execute':
        forecast_pin = (root/'forecast/COMMITTED').read_text()
        pin = run(root/'execution', root/'forecast', forecast_pin)
        write(root/'executed.json', dict(pin=pin, wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(stage=a.stage, observation_pin=pin)), flush=True)
        return
    proof = authenticate(a.source, a.pin)
    prior = json.loads((a.source/'source.json').read_bytes())
    native, grounded = Path(prior['root']), Path(prior['retained']['root'])
    evidence = Path(prior['upstream']['root'])
    authenticate(native, prior['seal'])
    authenticate(grounded, prior['retained']['seal'])
    authenticate(evidence, prior['upstream']['seal'])
    upstream = json.loads((evidence/'summary.json').read_bytes())
    c = upstream['config']
    source_journal = sha((a.source/'experience.sqlite3').read_bytes())
    if a.stage == 'forecast':
        root.mkdir(exist_ok=False)
        shutil.copyfile(a.source/'experience.sqlite3', root/'experience.sqlite3')
        write(root/'source.json', dict(root=str(a.source), seal=a.pin, journal=source_journal))
        execution = c['executions'][0]
        condition_pin = prepare(execution['prepared'], execution['prepared_pin'], root/'execution')
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28', 'field-experiment-'+a.stage)
    with (root/('gpu-'+a.stage+'.txt')).open('x') as stream:
        stream.write(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source, Path(c['population']), Path(c['native']),
                        population_pin=c['population_pin'], native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']), v['condition_pin']), inputs,
        Path(v['condition_population']), Path(v['condition_native']),
        population_pin=v['condition_population_pin'], native_pin=v['condition_native_pin']) for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'], 'FIELD_PARENT_PIN')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs, Path(c['kernel']), (Path(c['kernel'])/'COMMITTED').read_text())
    snapshot_path = a.source/'snapshot.json' if a.stage == 'forecast' else root/(
        'snapshot-final.json' if a.stage == 'cold' else 'snapshot.json')
    saved = json.loads(snapshot_path.read_bytes())
    expected_head = saved['head']
    if a.stage == 'recover':
        from audit.prospective_execution_readback import journal
        contract, events, payloads, expected_head = journal(root/'experience.sqlite3')
        retained = json.loads((root/'feedback-result.json').read_bytes())
        check(digest(contract) == saved['contract'] and events[-1] == retained['body']
              and payloads[events[-1]['payload']] == retained['payload'], 'FIELD_RECOVERY_RETAINED_FEEDBACK')
    session = ExperimentLoop(root/'experience.sqlite3', source, inputs, parent_root/'initial', model_pin=parent['initial'],
        standard=parent['standard'], world=parent['world'], contexts={'train:1':c['providers'][0]['task']},
        device='cuda', condition_inputs=conditions, routing=routing, growth_root=Path(c['output'])/'growth',
        candidate_root=grounded/'candidates', native_root=native/'candidates', field_root=a.source/'candidates',
        expected_head=expected_head, expected_contract=saved['contract'])
    if a.stage != 'recover':
        check(session.snapshot() == saved, 'FIELD_EXPERIMENT_PREFIX_REPLAY')
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    controls = []
    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'FIELD_CONTROL_CODE_'+name+':'+str(error))
        else:
            raise AssertionError('FIELD_CONTROL_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before, 'FIELD_CONTROL_STATE')
        controls.append(dict(name=name, code=code))
    tick = time.perf_counter()
    if a.stage == 'forecast':
        train = [v['forecast'] for v in upstream['measured']['scored'][:2]]
        evaluate = [upstream['measured']['scored'][2]['forecast']]
        spec = dict(conditions=str(root/'execution'), condition_pin=condition_pin,
                    train=train, evaluate=evaluate, candidate='field-1', base_forecast=train[1])
        body = session.forecast_fields(spec)
        forecast_pin = field_forecast.publish(session, root/'forecast', body)
        write(root/'forecast-result.json', dict(body=body, pin=forecast_pin))
        refuse('duplicate_forecast', 'FIELD_EXPERIMENT_DUPLICATE', lambda:session.forecast_fields(spec))
        write(root/'snapshot.json', session.snapshot())
    elif a.stage in ('feedback', 'recover'):
        forecast_pin = (root/'forecast/COMMITTED').read_text()
        pin = (root/'execution/COMMITTED').read_text()
        seal, payload = field_forecast.read(root/'forecast', forecast_pin)
        observations = ProcessObservations(root/'execution', pin, source)
        ob = next(iter(observations.entries))
        condition = observations.entries[ob]['condition']
        other, = set(payload['predictions'])-{condition}
        actual = observations.read(ob)
        late = copy.deepcopy(seal)
        late['committed_at'] = actual['finished']
        refuse('late_forecast', 'FIELD_FORECAST_LATE', lambda:field_forecast.match(late, payload, condition, actual))
        refuse('swapped_condition', 'FIELD_FORECAST_CONDITION', lambda:field_forecast.match(seal, payload, other, actual))
        broken = root/'negative-forecast'
        if a.stage == 'feedback':
            shutil.copytree(root/'forecast', broken)
            raw = (broken/'forecast.json').read_bytes()
            (broken/'forecast.json').write_bytes(raw+b' ')
        refuse('changed_forecast_bytes', 'FIELD_FORECAST_PIN', lambda:field_forecast.read(broken, forecast_pin))
        if a.stage == 'feedback':
            body = session.observe_fields(root/'execution', pin, root/'forecast', forecast_pin)
            write(root/'feedback-result.json', dict(body=body, payload=session.journal.payload(body['payload'])))
        else:
            check(all(v['status'] == 'observed' for v in session.experiments.values()), 'FIELD_RECOVERY_STATUS')
        refuse('duplicate_feedback', 'FIELD_EXPERIMENT_NOT_PENDING',
               lambda:session.observe_fields(root/'execution', pin, root/'forecast', forecast_pin))
        refuse('failed_candidate_activation', 'TOOL_CANDIDATE_NOT_SELECTED', lambda:session.require_selection('field-1'))
        old = {k:v for k,v in saved.items() if k not in ('head', 'events', 'memory', 'controlled_experiments')}
        now = {k:v for k,v in session.snapshot().items() if k not in ('head', 'events', 'memory', 'controlled_experiments')}
        check(old == now, 'FIELD_EXPERIMENT_EXISTING_CONTEXT_DRIFT')
        write(root/'snapshot-final.json', session.snapshot())
    else:
        check(all(v['status'] == 'observed' for v in session.experiments.values()), 'FIELD_EXPERIMENT_COLD_STATUS')
        refuse('failed_candidate_activation', 'TOOL_CANDIDATE_NOT_SELECTED', lambda:session.require_selection('field-1'))
    phases[a.stage] = time.perf_counter()-tick
    tick = time.perf_counter()
    head, snapshot = session.head, session.snapshot()
    session.close()
    check(source_journal == sha((a.source/'experience.sqlite3').read_bytes()), 'FIELD_EXPERIMENT_SOURCE_MUTATION')
    phases['final_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True, acceptance='author-checked', stage=a.stage, source=proof, head=head,
        snapshot=digest(snapshot), journal_sha256=sha((root/'experience.sqlite3').read_bytes()), controls=controls,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        serving=False, scientific_admission=False, new_weight_update=False)
    write(root/(a.stage+'-report.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run_cli(main)
