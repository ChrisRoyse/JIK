"""Actual complete-answer updates, persisted feedback, and autonomous comparison."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import answer_history_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_history_training as history
from model.native_population_training import cpu, fingerprint
from model.native_requested_answer import update as original_update
from model.native_request_input import NativeRequest, legal
from text.constellation_corpus import file_sha, need


def controls(state):
    model, opt = source.make(state, 'shared'); value, request = source.item(state, source.TRAIN[0])
    initial = fingerprint((model.state_dict(), opt.state_dict())); out = {}
    def run(v=value, q=request, p=False):
        return history.update(model, opt, v, 5, state['policy'], request=q, target=state['target'], predicted=p)
    held = next(d['document'] for d in state['items'].values() if d['side'] != 'train')
    out['held_gradient'] = rejected(lambda: run(v=source.item(state, held)[0]), 'CONDITIONED_NONTRAIN')
    out['wrong_request'] = rejected(lambda: run(q=NativeRequest(dict(legal(2), operation='respond_to_source_question'))),
                                    'ANSWER_HISTORY_REQUEST')
    out['policy_type'] = rejected(lambda: run(p=1), 'ANSWER_HISTORY_POLICY')
    bad = deepcopy(value); bad['codes'][0, 0] = (bad['codes'][0, 0]+1) % 4
    out['changed_labels'] = rejected(lambda: run(v=bad), 'CONDITIONED_LABEL_PIN')
    bad = deepcopy(value); next(iter(bad['answers'][0].values())).add_(1)
    out['changed_native'] = rejected(lambda: run(v=bad), 'CONDITIONED_NATIVE_PIN')
    bad = deepcopy(value); bad['provenance']['observed'] += 1
    out['changed_binding'] = rejected(lambda: run(v=bad), 'CONDITIONED_SOURCE_BINDING')
    bad = deepcopy(state['initial']['shared']); bad['optimizer']['state'] = {}
    out['missing_moments'] = rejected(lambda: source.make(state, 'shared', bad),
        'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')
    out['wrong_step'] = rejected(lambda: source.make(state, 'shared', step=6), 'OPTIMIZER')
    try:
        history.model_feedback(model, None, None, model.proposals[0], target=state['target'],
            generator=None, target_length=1)
    except TypeError as exc: out['future_length'] = str(exc)
    else: raise ValueError('HISTORY_FUTURE_LENGTH_ACCEPTED')
    stream = model.stream(request); cache = None
    with torch.no_grad():
        for pos, frame in enumerate(value['context']):
            hidden, cache = stream.step((frame,), context='history-control', offset=pos,
                                        state=cache, detach_history=True)
    head, hidden = model.forecaster.contexts(hidden)['assembled']; wrong = deepcopy(state['target'])
    out['stale_target'] = rejected(lambda: history.model_feedback(model, head, hidden, 'assembled',
        target=wrong, generator=torch.Generator().manual_seed(99)), 'PLACEMENT_TARGET_CHANGED')
    need(fingerprint((model.state_dict(), opt.state_dict())) == initial, 'HISTORY_CONTROLS_MUTATED')
    separate, other = source.make(state, 'shared')
    next(iter(other.state.values()))['exp_avg'].add_(1)
    need(fingerprint((model.state_dict(), opt.state_dict())) == initial, 'HISTORY_MOMENT_ALIAS')
    return out


def differences(left, right):
    need(left.keys() == right.keys(), 'HISTORY_COMPARISON_KEYS')
    return dict(maximum=max(float((left[n]-right[n]).abs().max()) for n in left),
                differing_elements=sum(int((left[n] != right[n]).sum()) for n in left))


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); cpu0 = time.process_time(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    state = source.load(a.root); write_committed(a.output/'census.json', state['census'])
    write_committed(a.output/'source.json', dict(parent=source.PARENT, cold=source.COLD, code=state['code'],
        target=fingerprint(state['target'].state_dict()), region=state['pin'], policy=state['policy']))
    phases['authentication'] = time.perf_counter()-tick; arms = {}; arithmetic = {}
    for architecture in source.ARCHITECTURES:
        initial = state['initial'][architecture]; tick = time.perf_counter()
        # Run the unchanged teacher learner once, solely to expose arithmetic differences.
        old, old_opt = source.make(state, architecture); value, request = source.item(state, source.TRAIN[0])
        old_row, old_grad = original_update(old, old_opt, value, 5, state['policy'], request=request, target=state['target'])
        persist(a.output/(architecture+'-original.pt'), dict(record=old_row, gradients=old_grad,
            model=old.state_dict(), optimizer=old_opt.state_dict()))
        phases[architecture+'_original'] = time.perf_counter()-tick
        for predicted in (False, True):
            tick = time.perf_counter(); key = architecture+('/mixed' if predicted else '/teacher')
            dest = a.output/key; dest.mkdir(parents=True)
            model, opt = source.make(state, architecture); before = fingerprint((model.state_dict(), opt.state_dict()))
            records = []
            for cursor, doc in enumerate(source.TRAIN):
                model.train(); value, request = source.item(state, doc)
                row, grad = history.update(model, opt, value, 5+cursor, state['policy'],
                    request=request, target=state['target'], predicted=predicted)
                artifact = dict(record=row, gradients=grad, model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()),
                    binding=dict(source=source.PARENT[1], region=state['pin'], code=state['code'],
                                 architecture=architecture, predicted=predicted, document=doc, step=6+cursor))
                path = dest/f'update-{6+cursor}.pt'; persist(path, artifact); records.append(row)
                if cursor == 0:
                    if not predicted:
                        arithmetic[architecture] = dict(parameters=differences(old.state_dict(), model.state_dict()),
                                                        gradients=differences(old_grad, grad))
                    saved = torch.load(path, map_location='cpu', weights_only=True)
                    model, opt = source.make(state, architecture, saved, step=6)
            phases[key+'_updates_restore'] = time.perf_counter()-tick; tick = time.perf_counter()
            evaluation = source.evaluate(state, model); persist(dest/'evaluation.pt', evaluation)
            arms[key] = dict(initial=before, updates=2, decisions=sum(r['decisions'] for r in records),
                predicted_inputs=sum(r['predicted_inputs'] for r in records), rollouts=len(evaluation),
                ended=sum(r['metrics']['ended'] for r in evaluation),
                exact_end=sum(r['metrics']['ended_at_reference_boundary'] for r in evaluation),
                refused=sum(r['refusal'] is not None for r in evaluation),
                final=fingerprint(model.state_dict()), optimizer=fingerprint(opt.state_dict()))
            phases[key+'_evaluation'] = time.perf_counter()-tick
            print(json.dumps(dict(arm=key, **arms[key])), flush=True)
        need(fingerprint(initial) == fingerprint(state['initial'][architecture]), 'HISTORY_INITIAL_CHANGED')
    tick = time.perf_counter(); negative = controls(state); state['fresh']()
    need(not torch.cuda.is_initialized(), 'HISTORY_CPU_PROOF_GPU')
    write_committed(a.output/'controls.json', negative); write_committed(a.output/'arithmetic.json', arithmetic)
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_readback_hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'HISTORY_PHASE_COVERAGE')
    result = dict(complete=True, acceptance='author-checked', number_status='measured', code=state['code'], files=files,
        source=source.PARENT, cold=source.COLD, training_documents=source.TRAIN, arms=arms,
        total_updates=12, teacher_arithmetic_updates=3, total_decisions=sum(r['decisions'] for r in arms.values()),
        controls=negative, arithmetic=arithmetic, seconds=seconds, phases=phases, cpu_seconds=time.process_time()-cpu0,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, scientific_admission=False,
        capability_demonstrated=False, promoted_to_training_or_serving=False, scope='CPU_engineering_qualification')
    pin = write_committed(a.output/'report.json', result); print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
