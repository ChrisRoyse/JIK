"""Matched real complete-answer updates with and without full history gradients."""
import argparse
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
from audit import full_history_source as source
from audit.full_history_controls import controls, finite_difference
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model.answer_full_history import update
from model.native_population_training import fingerprint
from model.native_requested_answer import update as original_update
from text.constellation_corpus import file_sha, need


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}; arms = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    state = source.load(a.root)
    write_committed(a.output/'source.json', dict(parent=source.prior.PARENT, cold=source.prior.COLD,
        census=state['census'], train=state['train'], policy=state['policy'], code=state['code']))
    phases['authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
    negative = controls(state); write_committed(a.output/'controls.json', negative)
    phases['negative_controls'] = time.perf_counter()-tick; initial_forwards = {}; arithmetic = {}
    for architecture in source.prior.ARCHITECTURES:
        tick = time.perf_counter(); probe = finite_difference(state, architecture)
        persist(a.output/(architecture+'-direction.pt'), probe)
        model, opt = source.prior.make(state, architecture)
        value, request = source.prior.item(state, state['train'][0])
        row, gradients = original_update(model, opt, value, 5, state['policy'], request=request, target=state['target'])
        original = dict(records=row['records'], gradients=gradients, model=model.state_dict(), optimizer=opt.state_dict())
        original_pin = fingerprint(original); persist(a.output/(architecture+'-original.pt'), original)
        del model, opt, original, gradients
        phases[architecture+'_derivative_original'] = time.perf_counter()-tick
        for full in (False, True):
            key = architecture+('/full' if full else '/detached'); directory = a.output/key; directory.mkdir(parents=True)
            tick = time.perf_counter(); model, opt = source.prior.make(state, architecture); model.train()
            initial = fingerprint((model.state_dict(), opt.state_dict())); ledger = []
            for cursor, doc in enumerate(state['train']):
                value, request = source.prior.item(state, doc)
                row, gradients, credit = update(model, opt, value, 5+cursor, state['policy'], request=request,
                                                 target=state['target'], full=full)
                binding = dict(source=source.prior.PARENT[1], region=state['pin'], code=state['code'],
                               architecture=architecture, full=full, document=doc, step=6+cursor)
                persist(directory/f'update-{6+cursor}.pt', dict(record=row, gradients=gradients, early=credit,
                    model=model.state_dict(), optimizer=opt.state_dict(), binding=binding))
                if cursor == 0:
                    if not full:
                        need(fingerprint(dict(records=row['records'], gradients=gradients, model=model.state_dict(),
                            optimizer=opt.state_dict())) == original_pin, 'FULL_HISTORY_DETACHED_ARITHMETIC')
                        initial_forwards[architecture] = fingerprint(row['records']); arithmetic[architecture] = True
                    else: need(fingerprint(row['records']) == initial_forwards[architecture], 'FULL_HISTORY_FORWARD_MISMATCH')
                ledger.append(row)
            phases[key+'_training'] = time.perf_counter()-tick; tick = time.perf_counter()
            evaluation = source.evaluate(state, model); persist(directory/'evaluation.pt', evaluation)
            phases[key+'_autonomous'] = time.perf_counter()-tick
            arms[key] = dict(initial=initial, final=fingerprint(model.state_dict()), optimizer=fingerprint(opt.state_dict()),
                updates=len(ledger), decisions=sum(r['decisions'] for r in ledger), rollouts=len(evaluation),
                peak_saved_storage_bytes=max(r['storage']['peak_saved_storage_bytes'] for r in ledger),
                all_saved_storage_released=all(r['storage']['remaining_saved_storage_bytes'] == 0 for r in ledger))
            write_committed(directory/'ledger.json', [{n:v for n,v in row.items() if n != 'records'} for row in ledger])
            print(json.dumps(dict(arm=key, **arms[key])), flush=True)
    tick = time.perf_counter(); state['fresh'](); need(not torch.cuda.is_initialized(), 'FULL_HISTORY_CPU_SCOPE')
    phases['final_authentication'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(sum(r['decisions'] for r in arms.values()) == 396 and sum(r['rollouts'] for r in arms.values()) == 640,
         'FULL_HISTORY_DENOMINATORS')
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'FULL_HISTORY_PHASES')
    report = dict(complete=True, schema='full-history-answer-v1', acceptance='author-checked', code=state['code'],
        files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
        arms=arms, controls=negative, exact_detached_arithmetic=arithmetic, initial_forward_equal=True,
        train=state['train'], phases=phases, seconds=seconds, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        frozen_encoder_calls=0, gpu_used=False, promoted=False, scientific_admission=False,
        scope='development-exposed legal complete continuations; CPU runtime and backward-path qualification')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', report), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=source.ROOT)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
