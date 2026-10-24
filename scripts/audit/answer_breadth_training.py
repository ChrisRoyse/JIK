"""Matched corpus warm-start and complete-answer qualification on real legal data."""
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
from audit import answer_breadth_source as source
from audit import answer_history_source as history
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.answer_breadth_transfer import install
from model.native_population_training import cpu, fingerprint
from model.native_requested_answer import update
from model.native_placement_checked import validate_optimizer
from text.constellation_corpus import file_sha, need


def binding(state, architecture, arm):
    return dict(architecture=architecture, arm=arm, source=source.SOURCE, cold=source.COLD,
        source_state=fingerprint(state['checkpoints'][arm]), region=state['pin'], code=state['code'],
        optimizer_policy='new_task_warm_start_reset_adamw_not_continuation')


def apply(state, model, donor, arm):
    return install(model, donor, source_reference=state['bundle']['target_ref'],
        destination_reference=state['bundle']['target_ref'],
        expected_source=fingerprint(state['checkpoints'][arm]['model']))


def restore(state, architecture, arm, saved, step):
    need(saved['binding'] == binding(state, architecture, arm) and saved['step'] == step,
         'BREADTH_RESTORE_BINDING')
    need(saved['model_pin'] == fingerprint(saved['model']) and
         saved['optimizer_pin'] == fingerprint(saved['optimizer']), 'BREADTH_RESTORE_STATE')
    model, opt = source.make(state, architecture)
    model.load_state_dict(deepcopy(saved['model']), strict=True)
    opt.load_state_dict(deepcopy(saved['optimizer']))
    validate_optimizer(model, opt, state['policy'], step)
    return model, opt


def checkpoint(state, model, opt, arm, step, **extra):
    return dict(model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()), step=step,
        model_pin=fingerprint(model.state_dict()), optimizer_pin=fingerprint(opt.state_dict()),
        binding=binding(state, model.forecaster.architecture, arm), **extra)


@torch.no_grad()
def shared_equivalence(state, model, donor):
    model.eval(); donor.eval(); values = []
    for doc in history.TRAIN:
        item, request = history.item(state, doc)
        streams = (donor.stream(), model.forecaster.predictor.stream(), model.stream(request))
        caches = [None]*3
        for pos, frame in enumerate(item['context']):
            hidden = []
            for i, stream in enumerate(streams):
                h, caches[i] = stream.step((frame,), context='breadth-function', offset=pos,
                                           state=caches[i], detach_history=True)
                hidden.append(h)
            need(all(torch.equal(hidden[0], h) for h in hidden[1:]), 'BREADTH_TRANSFER_SHARED_HIDDEN')
            outputs = (donor.region(hidden[0]), model.forecaster.predictor.region(hidden[1]),
                       model.forecaster.predictor.region(hidden[2]))
            need(len({fingerprint(v) for v in outputs}) == 1, 'BREADTH_TRANSFER_SHARED_REGION')
            values.append(dict(document=doc, position=pos, hidden=fingerprint(hidden[0]),
                               region=fingerprint(outputs[0])))
    return values


def controls(state):
    donor = source.donor(state, 'learned_corpus'); model, opt = source.make(state, 'shared')
    original = fingerprint((model.state_dict(), opt.state_dict())); ref = state['bundle']['target_ref']
    expected = fingerprint(donor.state_dict()); out = {}
    def run(d=donor, src=ref, dst=ref, pin=expected):
        return install(model, d, source_reference=src, destination_reference=dst, expected_source=pin)
    out['different_codebook'] = rejected(lambda: run(dst=dict(ref, rows=ref['rows']+1)), 'BREADTH_TRANSFER_TARGET_IDENTITY')
    out['changed_donor_pin'] = rejected(lambda: run(pin='0'*64), 'BREADTH_TRANSFER_SOURCE_STATE')
    bad = deepcopy(donor); del bad.encoder.query
    out['missing_donor_field'] = rejected(lambda: run(d=bad), 'BREADTH_TRANSFER_SOURCE_STATE')
    bad = deepcopy(donor); bad.register_buffer('unexpected', torch.ones(1))
    out['extra_donor_field'] = rejected(lambda: run(d=bad), 'BREADTH_TRANSFER_SOURCE_STATE')
    bad = deepcopy(donor)
    with torch.no_grad(): next(bad.parameters()).flatten()[0] = float('nan')
    out['nonfinite_donor'] = rejected(lambda: run(d=bad, pin=fingerprint(bad.state_dict())), 'BREADTH_TRANSFER_LAYOUT')
    bad = deepcopy(donor); bad.encoder.query = torch.nn.Parameter(bad.encoder.query[:1].clone(), requires_grad=False)
    out['wrong_shape'] = rejected(lambda: run(d=bad, pin=fingerprint(bad.state_dict())), 'BREADTH_TRANSFER_LAYOUT')
    held = next(v['document'] for v in state['items'].values() if v['side'] != 'train')
    value, request = history.item(state, held)
    out['heldout_gradient'] = rejected(lambda: update(model, opt, value, 0, state['policy'],
        request=request, target=state['target']), 'CONDITIONED_NONTRAIN')
    out['stale_target_runtime'] = rejected(lambda: model.forecaster.placement['assembled'].check_target(deepcopy(state['target'])),
                                          'PLACEMENT_TARGET_CHANGED')
    need(fingerprint((model.state_dict(), opt.state_dict())) == original, 'BREADTH_CONTROLS_MUTATED')
    apply(state, model, donor, 'learned_corpus')
    with torch.no_grad(): model.forecaster.predictor.encoder.query.add_(1)
    need(fingerprint(donor.state_dict()) == expected, 'BREADTH_TRANSFER_MUTATION_ALIAS')
    out['receiver_mutation_does_not_mutate_donor'] = True
    return out


def summary(rows):
    return dict(rollouts=len(rows), ended=sum(r['metrics']['ended'] for r in rows),
        exact_end=sum(r['metrics']['ended_at_reference_boundary'] for r in rows),
        refused=sum(r['refusal'] is not None for r in rows))


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); cpu0 = time.process_time(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    state = source.load(a.root); donors = {arm:source.donor(state, arm) for arm in source.ARMS}
    write_committed(a.output/'census.json', state['census'])
    write_committed(a.output/'source.json', dict(source=source.SOURCE, cold=source.COLD, code=state['code'],
        target=fingerprint(state['target'].state_dict()), region=state['pin'], policy=state['policy'], refs=state['source_refs']))
    phases['authentication'] = time.perf_counter()-tick; results = {}; preserved = {}; transfers = {}
    for architecture in history.ARCHITECTURES:
        for arm in source.ARMS:
            tick = time.perf_counter(); key = architecture+'/'+arm; dest = a.output/key; dest.mkdir(parents=True)
            model, opt = source.make(state, architecture); result = apply(state, model, donors[arm], arm)
            validate_optimizer(model, opt, state['policy'], 0); transfers[key] = result
            need(not opt.state, 'BREADTH_WARM_START_HAS_MOMENTS')
            if architecture == 'shared': persist(dest/'shared-function.pt', shared_equivalence(state, model, donors[arm]))
            if architecture in preserved: need(preserved[architecture] == result['preserved'], 'BREADTH_UNMATCHED_NEW_HEADS')
            else: preserved[architecture] = result['preserved']
            write_committed(dest/'transfer.json', result)
            persist(dest/'initial.pt', checkpoint(state, model, opt, arm, 0))
            phases[key+'_transfer'] = time.perf_counter()-tick; tick = time.perf_counter()
            initial = history.evaluate(state, model); persist(dest/'before.pt', initial)
            phases[key+'_before'] = time.perf_counter()-tick; tick = time.perf_counter(); records = []
            for step, doc in enumerate(history.TRAIN):
                model.train(); value, request = history.item(state, doc)
                row, grads = update(model, opt, value, step, state['policy'], request=request, target=state['target'])
                saved = checkpoint(state, model, opt, arm, step+1, record=row, gradients=grads)
                path = dest/f'update-{step+1}.pt'; persist(path, saved); records.append(row)
                saved = torch.load(path, map_location='cpu', weights_only=True)
                model, opt = restore(state, architecture, arm, saved, step+1)
            phases[key+'_updates_restart'] = time.perf_counter()-tick; tick = time.perf_counter()
            after = history.evaluate(state, model); persist(dest/'after.pt', after)
            results[key] = dict(updates=2, decisions=sum(r['decisions'] for r in records),
                before=summary(initial), after=summary(after), final=fingerprint(model.state_dict()),
                optimizer=fingerprint(opt.state_dict()))
            phases[key+'_after'] = time.perf_counter()-tick
            print(json.dumps(dict(arm=key, **results[key])), flush=True)
    tick = time.perf_counter(); negative = controls(state); state['fresh']()
    for arm, d in donors.items(): need(fingerprint(d.state_dict()) == fingerprint(state['checkpoints'][arm]['model']),
                                     'BREADTH_DONOR_MUTATED')
    need(not torch.cuda.is_initialized(), 'BREADTH_CPU_GPU')
    write_committed(a.output/'controls.json', negative)
    files = {str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_readback_hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'BREADTH_PHASE_COVERAGE')
    result = dict(complete=True, acceptance='author-checked', number_status='measured', code=state['code'], files=files,
        source=source.SOURCE, cold=source.COLD, arms=results, total_updates=12,
        total_decisions=sum(v['decisions'] for v in results.values()),
        total_rollouts=sum(v['before']['rollouts']+v['after']['rollouts'] for v in results.values()),
        controls=negative, seconds=seconds, phases=phases, cpu_seconds=time.process_time()-cpu0,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scientific_admission=False, capability_demonstrated=False, promoted_to_training_or_serving=False,
        scope='CPU_engineering_qualification', architecture_fairness='shared_pretraining_layout_favors_shared_no_winner_claim')
    pin = write_committed(a.output/'report.json', result); print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
