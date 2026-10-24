"""Actual controlled-feedback weight updates, exact recovery and inherited refusal gates."""
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
from audit.controlled_open import open_session
from audit.fsv_process_feedback import write
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from lens.lenscommon import run_cli, take_lease
from model.controlled_learning import ControlledData
from model.controlled_update import increment
from model import field_checkpoint
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', action='store_true')
    a = p.parse_args()
    started = time.perf_counter()
    root = a.output
    original = json.loads((a.source/'snapshot-final.json').read_bytes())
    journal_pin = sha((a.source/'experience.sqlite3').read_bytes())
    if not a.cold:
        root.mkdir(exist_ok=False)
        shutil.copyfile(a.source/'experience.sqlite3', root/'experience.sqlite3')
        write(root/'source.json', dict(root=str(a.source), seal=a.pin, journal=journal_pin))
    stage = 'cold' if a.cold else 'run'
    gpu = subprocess.check_output(['nvidia-smi']).decode()
    take_lease('28', 'controlled-learning-'+stage)
    with (root/('gpu-'+stage+'.txt')).open('x') as stream:
        stream.write(gpu)
    configure()
    saved = json.loads((root/'snapshot.json').read_bytes()) if a.cold else original
    session, proof = open_session(a.source, a.pin, root, saved)
    phases = dict(authentication_and_replay=time.perf_counter()-started)
    tick = time.perf_counter()
    controls = []

    def refuse(name, code, fn):
        before = digest(session.snapshot())
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'CONTROLLED_NEGATIVE_CODE_'+name+':'+str(error))
        else:
            raise AssertionError('CONTROLLED_NEGATIVE_ACCEPTED_'+name)
        check(digest(session.snapshot()) == before, 'CONTROLLED_NEGATIVE_STATE')
        controls.append(dict(name=name, code=code))

    if not a.cold:
        contexts = sorted(session.echo_contexts)
        experiment = session.experiments[session.echo_contexts[contexts[0]]['experiment']]
        prior = experiment['forecast']['spec']
        spec = dict(train=prior['train'], evaluate=prior['evaluate'], foundation=prior['candidate'], contexts=contexts)
        first, second = contexts

        def changed_entry(changes):
            old = session.echo_contexts[first]
            altered = copy.deepcopy(old)
            altered.update(changes)
            session.echo_contexts[first] = altered
            try:
                ControlledData(session, spec)
            finally:
                session.echo_contexts[first] = old

        refuse('feedback_not_corrected', 'CONTROLLED_FEEDBACK_INCOMPLETE',
               lambda:changed_entry(dict(status='awaiting_correction')))
        other = session.echo_contexts[second]
        refuse('condition_output_swap', 'CONTROLLED_CONDITION_JOIN',
               lambda:changed_entry(dict(observation=other['observation'], native=other['native'])))
        fork = copy.deepcopy(session.echo_contexts[first]['fork'])
        fork['prestate'] = digest(session.echo_contexts[first]['state']['hidden'])
        refuse('feedback_state_as_forecast_input', 'CONTROLLED_PREACT_STATE',
               lambda:changed_entry(dict(fork=fork)))
        phases['negative_binding_controls'] = time.perf_counter()-tick
        tick = time.perf_counter()
        result = increment(session, root/'candidates', name='controlled-1', spec=spec)
        write(root/'learning.json', result)
        write(root/'snapshot.json', session.snapshot())
        phases['incremental_learning'] = time.perf_counter()-tick
    else:
        result = json.loads((root/'learning.json').read_bytes())
        spec = result['decision']['spec']
        check(session.controlled_candidates['controlled-1'] == result['decision'], 'CONTROLLED_COLD_DECISION')
        phases['cold_result_join'] = time.perf_counter()-tick
    tick = time.perf_counter()
    refuse('failed_candidate_activation', 'TOOL_CANDIDATE_NOT_SELECTED', lambda:session.require_selection('controlled-1'))
    if not a.cold:
        # A separate, deliberately broken checkpoint. The valid candidate is immutable.
        broken = root/'negative-checkpoint'
        shutil.copytree(root/'candidates/controlled-1', broken)
        with (broken/'weights.pt').open('ab') as stream:
            stream.write(b'corrupt-control')
        # Current data binds the next update; historical load authenticates this exact saved version.
        data = ControlledData(session, spec)
        manifest = json.loads((broken/'manifest.json').read_bytes())
        refuse('changed_checkpoint_bytes', 'NATIVE_WORLD_CHECKPOINT_BYTES', lambda:field_checkpoint.load(
            broken, data, expected=result['candidate_pin'], historical=manifest))
    current = session.snapshot()
    excluded = {'head','events','controlled_feedback_candidates'}
    check({k:v for k,v in current.items() if k not in excluded} ==
          {k:v for k,v in original.items() if k not in excluded}, 'CONTROLLED_EXISTING_STATE_DRIFT')
    session.close()
    check(sha((a.source/'experience.sqlite3').read_bytes()) == journal_pin, 'CONTROLLED_SOURCE_CHANGED')
    phases['final_controls_and_readback'] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(complete=True, acceptance='author-checked', stage=stage, source=proof,
        head=current['head'], snapshot=digest(current), journal_sha256=sha((root/'experience.sqlite3').read_bytes()),
        controls=controls, phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        metric_gate_passed=result['decision']['result']['metric_gate_passed'],
        serving=False, scientific_admission=False)
    write(root/(stage+'-report.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run_cli(main)
