"""Carry sealed controlled observations through distinct intrinsic feedback contexts."""
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
from claims.tool_temporal_source import ToolSource
from lens.lenscommon import run_cli, take_lease
from model.condition_input import ConditionNative
from model.condition_catalog import NativeConditions
from model.tool_temporal_input import ToolInputs
from model.tool_kernel import ToolKernel
from model.field_echo import EchoExperimentLoop, feedback_forward
from senses.contract import SenseError


def original(snapshot):
    return {k:v for k,v in snapshot.items() if k not in
            ('head', 'events', 'native_feedback_contexts', 'native_feedback_enrollments')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--stage', choices=('begin', 'finish', 'cold'), required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.output
    proof = authenticate(a.source, a.pin)
    f = json.loads((a.source/'source.json').read_bytes())
    field = Path(f['root'])
    authenticate(field, f['seal'])
    n = json.loads((field/'source.json').read_bytes())
    native, grounded, evidence = Path(n['root']), Path(n['retained']['root']), Path(n['upstream']['root'])
    for r,pin in ((native,n['seal']), (grounded,n['retained']['seal']), (evidence,n['upstream']['seal'])):
        authenticate(r, pin)
    c = json.loads((evidence/'summary.json').read_bytes())['config']
    before = json.loads((a.source/'snapshot-final.json').read_bytes())
    source_journal = sha((a.source/'experience.sqlite3').read_bytes())
    if a.stage == 'begin':
        check(not (root/'experience.sqlite3').exists(), 'FIELD_ECHO_OUTPUT_EXISTS')
        shutil.copyfile(a.source/'experience.sqlite3', root/'experience.sqlite3')
        write(root/'source.json', dict(root=str(a.source), seal=a.pin, journal=source_journal,
                                     field=f, config=c))
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28', 'field-echo-'+a.stage)
    with (root/('gpu-'+a.stage+'.txt')).open('x') as stream:
        stream.write(gpu)
    configure()
    source = ToolSource(Path(c['raw']))
    inputs = ToolInputs(source, Path(c['population']), Path(c['native']),
                        population_pin=c['population_pin'], native_pin=c['native_pin'])
    providers = [ConditionNative(ConditionInputs(Path(v['conditions']),v['condition_pin']), inputs,
        Path(v['condition_population']),Path(v['condition_native']),
        population_pin=v['condition_population_pin'],native_pin=v['condition_native_pin']) for v in c['providers']]
    conditions = NativeConditions(providers)
    parent_root = Path(c['parent'])
    check(sha((parent_root/'run.json').read_bytes()) == c['parent_pin'], 'FIELD_ECHO_PARENT_PIN')
    parent = json.loads((parent_root/'run.json').read_bytes())
    routing = ToolKernel(inputs, Path(c['kernel']), (Path(c['kernel'])/'COMMITTED').read_text())
    saved = before if a.stage == 'begin' else json.loads((root/(
        'snapshot-begin.json' if a.stage == 'finish' else 'snapshot-final.json')).read_bytes())
    session = EchoExperimentLoop(root/'experience.sqlite3', source, inputs, parent_root/'initial',
        model_pin=parent['initial'], standard=parent['standard'], world=parent['world'],
        contexts={'train:1':c['providers'][0]['task']}, device='cuda', condition_inputs=conditions,
        routing=routing, growth_root=Path(c['output'])/'growth', candidate_root=grounded/'candidates',
        native_root=native/'candidates', field_root=field/'candidates',
        expected_head=saved['head'], expected_contract=saved['contract'])
    check(session.snapshot() == saved, 'FIELD_ECHO_PREFIX_REPLAY')
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    controls, transitions, isolation = [], [], []

    def refuse(name, code, fn):
        prior = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'FIELD_ECHO_CONTROL_CODE_'+name+':'+str(error))
        else:
            raise AssertionError('FIELD_ECHO_CONTROL_ACCEPTED_'+name)
        check(digest(session.snapshot()) == prior, 'FIELD_ECHO_CONTROL_MUTATED')
        controls.append(dict(name=name, code=code))

    def advance(context, phase):
        old = copy.deepcopy(session.echo_contexts)
        ob = old[context]['observation']
        event = session.advance_echo(context, ob, phase)
        check(all(v == session.echo_contexts[k] for k,v in old.items() if k != context),
              'FIELD_ECHO_CONTEXT_INTERFERENCE')
        transitions.append(event)
        isolation.append(dict(changed=context, untouched={k:digest(v) for k,v in old.items() if k != context}))

    tick = time.perf_counter()
    if a.stage == 'begin':
        experiment, = session.experiments
        spec = dict(experiment=experiment, population=str(root/'population'),
            population_pin=(root/'population/COMMITTED').read_text(), native=str(a.native),
            native_pin=sha((a.native/'report.json').read_bytes()))
        event = session.enroll_echo(spec)
        write(root/'enrollment.json', dict(body=event, payload=session.journal.payload(event['payload'])))
        refuse('duplicate_enrollment', 'FIELD_ECHO_DUPLICATE_ENROLLMENT', lambda:session.enroll_echo(spec))
        first, second = sorted(session.echo_contexts)
        observation = session.echo_contexts[first]['observation']
        refuse('swapped_observation', 'FIELD_ECHO_OBSERVATION_JOIN',
               lambda:session.advance_echo(first, session.echo_contexts[second]['observation'], 3))
        refuse('correction_before_echo', 'FIELD_ECHO_PHASE_ORDER',
               lambda:session.advance_echo(first, observation, 4))
        refuse('generated_proposal_echo', 'FIELD_ECHO_CAPTURED_REQUIRED',
               lambda:session.advance_echo(first, observation, 3, origin='generated_proposal'))
        runtime = session.echo_runtime[first]
        from dataclasses import replace
        refuse('predicted_constellation_echo', 'FIELD_ECHO_OBSERVED_REQUIRED', lambda:feedback_forward(
            runtime['view'], replace(runtime['observed'], origin='predicted'), runtime['state'],
            allowed=runtime['allowed'], phase=3, event=digest('negative-prediction'),
            history=runtime['history'], origin='execution_receipt'))
        advance(first, 3)
        refuse('duplicate_echo', 'FIELD_ECHO_PHASE_ORDER', lambda:session.advance_echo(first, observation, 3))
        write(root/'snapshot-begin.json', session.snapshot())
    elif a.stage == 'finish':
        first, second = sorted(session.echo_contexts)
        check(session.echo_contexts[first]['status'] == 'awaiting_correction' and
              session.echo_contexts[second]['status'] == 'awaiting_echo', 'FIELD_ECHO_PENDING_RESTART')
        advance(second, 3)
        advance(first, 4)
        advance(second, 4)
        refuse('duplicate_correction', 'FIELD_ECHO_PHASE_ORDER', lambda:session.advance_echo(
            first, session.echo_contexts[first]['observation'], 4))
        write(root/'snapshot-final.json', session.snapshot())
    else:
        check(all(v['status'] == 'corrected_state' for v in session.echo_contexts.values()),
              'FIELD_ECHO_FINAL_REPLAY')
    refuse('failed_candidate_activation', 'TOOL_CANDIDATE_NOT_SELECTED', lambda:session.require_selection('field-1'))
    phases['feedback_and_controls'] = time.perf_counter()-tick
    tick = time.perf_counter()
    final = session.snapshot()
    check(original(before) == original(final), 'FIELD_ECHO_EXISTING_STATE_CHANGED')
    session.close()
    check(sha((a.source/'experience.sqlite3').read_bytes()) == source_journal, 'FIELD_ECHO_SOURCE_CHANGED')
    phases['readback_and_close'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True, stage=a.stage, acceptance='author-checked', source=proof,
        snapshot=digest(final), head=final['head'], journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        contexts=final['native_feedback_contexts'], controls=controls, transitions=transitions, isolation=isolation,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        learned_understanding=False, new_weight_update=False, serving=False, scientific_admission=False)
    write(root/(a.stage+'-report.json'), report)
    print(json.dumps({k:report[k] for k in ('complete','stage','head','snapshot','phases','phase_coverage')}), flush=True)


if __name__ == '__main__':
    run_cli(main)
