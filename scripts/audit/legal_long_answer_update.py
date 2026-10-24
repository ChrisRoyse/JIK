"""Qualify a complete long-answer update, early credit and optimizer restart."""
import argparse
from copy import deepcopy
import gc
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
from audit import recomputed_answer_training as reference
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_full_history as learner
from model.guarded_transient_answer import guarded_transient_attention
from model.native_placement_checked import validate_optimizer
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need

ROOT = reference.source.ROOT
AUTHORITIES = {
    'short': ('guarded-transient-v1-20261024/shared',
              '0edbb817ba565866b2fe9eaa216326d643370dfc516ca07e8f9bef20cc3f8f7f'),
    'short_cold': ('guarded-transient-v1-20261024/shared-cold',
                   '076c5893de731a08f0ca13b15c4f396e606c43adde12ef8f6a0b387bee48316a'),
    'long': ('transient-answer-v1-20261024/long-shared-heads',
             'f916843285d6875f5c65529ccf1bb3557d55cba9e5e90a3610ee1d2a018cd882'),
    'long_cold': ('transient-answer-v1-20261024/long-shared-heads-cold',
                  '362b156f26760b857c64c7901ae13a4bafbd7baf4bc21b57721f0f240e0e4ee6')}


def read_report(path, pin):
    need(file_sha(path/'report.json') == pin, 'LONG_UPDATE_REPORT_PIN')
    return json.loads((path/'report.json').read_bytes())


def save(path, value):
    persist(path, value)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(saved) == fingerprint(value), 'LONG_UPDATE_PHYSICAL_VALUE')
    return file_sha(path), saved


def restore(path, pin, state, binding, step):
    need(file_sha(path) == pin, 'LONG_UPDATE_CHECKPOINT_PIN')
    value = torch.load(path, map_location='cpu', weights_only=True)
    need(value['binding'] == binding and value['step'] == step, 'LONG_UPDATE_CHECKPOINT_BINDING')
    return reference.source.prior.make(state, 'shared', value, step=step)


def controls(output, state, binding, value):
    initial = fingerprint(state['initial']); path = output/'control-checkpoint.pt'
    pin, saved = save(path, value); step = value['step']; rows = {}
    rows['pin'] = rejected(lambda:restore(path, '0'*64, state, binding, step), 'LONG_UPDATE_CHECKPOINT_PIN')
    rows['binding'] = rejected(lambda:restore(path, pin, state, {}, step), 'LONG_UPDATE_CHECKPOINT_BINDING')
    rows['cursor'] = rejected(lambda:restore(path, pin, state, binding, step+1), 'LONG_UPDATE_CHECKPOINT_BINDING')
    missing = deepcopy(saved); missing['optimizer']['state'].pop(next(iter(missing['optimizer']['state'])))
    bad = output/'missing-moment.pt'; bad_pin, _ = save(bad, missing)
    rows['moment'] = rejected(lambda:restore(bad, bad_pin, state, binding, step),
                              'PLACEMENT_OPTIMIZER_INCOMPLETE_STATE_RESTORE_FULL_CHECKPOINT')
    model, opt = restore(path, pin, state, binding, step)
    before = fingerprint((model.state_dict(), opt.state_dict()))
    held = next(k for k,d in state['items'].items() if d['side'] != 'train')
    item, request = reference.source.prior.item(state, held)
    with guarded_transient_attention(model, heads=True):
        rows['held_out'] = rejected(lambda:learner.update(model, opt, item, step, state['policy'],
            request=request, target=state['target'], full=True), 'CONDITIONED_NONTRAIN')
    need(before == fingerprint((model.state_dict(), opt.state_dict())) and
         initial == fingerprint(state['initial']), 'LONG_UPDATE_CONTROL_MUTATION')
    return rows


def update(model, opt, item, request, step, state, output, label):
    projected = [0]
    def progress(module, args, value):
        projected[0] += 1
        if projected[0] % 128 == 0:
            with (output/'progress.jsonl').open('a') as f:
                f.write(json.dumps(dict(label=label, projected_atoms=projected[0], time=time.time(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))+'\n')
                f.flush(); os.fsync(f.fileno())
    handle = model.forecaster.predictor.projectors[model.roster[0]].register_forward_hook(progress)
    try:
        with guarded_transient_attention(model, heads=True):
            row, gradients, early = learner.update(model, opt, item, step, state['policy'],
                                                   request=request, target=state['target'], full=True)
    finally:
        handle.remove()
    need(projected[0] == len(item['context'])+len(item['answers']), 'LONG_UPDATE_PROJECTION_COVERAGE')
    need(row['storage']['remaining_saved_storage_bytes'] == 0 and
         row['decisions'] == len(item['answers'])+1, 'LONG_UPDATE_COVERAGE')
    value = dict(record=row, gradients=gradients, early=early,
                 model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()))
    save(output/(label+'.pt'), value)
    print(json.dumps(dict(label=label, projected_atoms=projected[0], decisions=row['decisions'],
                          storage=row['storage'], phases=row['phases'])), flush=True)
    return value


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    reports = {}; physical = {}; base = Path(__file__).resolve().parents[1]
    for name,(directory,pin) in AUTHORITIES.items():
        path = a.root/directory; r = read_report(path,pin); reports[name] = r
        need(r['complete'] and r['qualified'] and r['architecture'] == 'shared' and r['heads'],
             'LONG_UPDATE_AUTHORITY_SCOPE')
        physical[path/'report.json'] = pin
        physical.update({path/n:h for n,h in r['files'].items()})
        need(all(file_sha(base/n) == h for n,h in r['code'].items()), 'LONG_UPDATE_AUTHORITY_CODE')
    for name in ('short', 'long'):
        need(reports[name+'_cold']['previous_pin'] == AUTHORITIES[name][1], 'LONG_UPDATE_COLD_AUTHORITY')
    for path,pin in physical.items(): need(file_sha(path) == pin, 'LONG_UPDATE_AUTHORITY_BYTES')
    state = reference.source.load(a.root)
    code = dict(state['code'], **reports['short']['code'], **{'audit/legal_long_answer_update.py':file_sha(__file__)})
    binding = dict(schema='legal-long-answer-update-v1', architecture='shared', code=code,
                   authorities=AUTHORITIES, region=state['pin'], policy=state['policy'])
    # Normalize tuple serialization before comparing a separately read JSON report.
    binding = json.loads(json.dumps(binding))
    previous = None
    if a.previous:
        previous = read_report(a.previous,a.previous_pin)
        need(previous['complete'] and previous['binding'] == binding and previous['mode'] == a.mode,
             'LONG_UPDATE_PREVIOUS_BINDING')
    if a.mode == 'long':
        need(a.probe is not None and a.probe_pin is not None, 'LONG_UPDATE_PROBE_REQUIRED')
        probe = read_report(a.probe,a.probe_pin)
        need(probe['complete'] and probe['mode'] == 'probe' and probe['binding'] == binding,
             'LONG_UPDATE_UNQUALIFIED')
        for n,h in probe['files'].items(): need(file_sha(a.probe/n) == h, 'LONG_UPDATE_PROBE_BYTES')
    phases['authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
    initial = state['initial']['shared']; start = 5
    if a.mode == 'probe':
        initial = torch.load(a.root/AUTHORITIES['short'][0]/'update-6.pt', map_location='cpu', weights_only=True)
        start = 6
    checkpoint = dict(model=initial['model'], optimizer=initial['optimizer'], step=start, binding=binding)
    pin, _ = save(a.output/'initial.pt', checkpoint)
    negatives = controls(a.output,state,binding,checkpoint)
    model,opt = restore(a.output/'initial.pt',pin,state,binding,start); model.train()
    phases['restore_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    payloads = {}; expected_steps = 0
    if a.mode == 'probe':
        item,request = reference.source.prior.item(state,27195)
        value = update(model,opt,item,request,start,state,a.output,'short-update')
        expected = torch.load(a.root/AUTHORITIES['short'][0]/'update-7.pt', map_location='cpu', weights_only=True)
        reference.exact(value,expected); payloads['short-update.pt'] = fingerprint(reference.numerical(value))
    else:
        limit = 24*1024**3
        available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:')).split()[1])*1024
        need(available >= 2*limit, 'LONG_UPDATE_PHYSICAL_HEADROOM')
        resource.setrlimit(resource.RLIMIT_AS, (limit,limit))
        item,request = reference.long_item(a.root,5796)
        need(len(item['context']) == 2 and len(item['answers']) == 3208, 'LONG_UPDATE_LONG_CENSUS')
        need(fingerprint((model.state_dict(),opt.state_dict())) == reports['long']['initial'],
             'LONG_UPDATE_INITIAL_REFERENCE')
        value = update(model,opt,item,request,start,state,a.output,'long-update')
        expected = torch.load(a.root/AUTHORITIES['long'][0]/'full-gradient.pt',map_location='cpu',weights_only=True)
        need(fingerprint(value['gradients']) == fingerprint(expected['gradients']) == reports['long']['long']['gradient_pin']
             and fingerprint(value['record']['records']) == fingerprint(expected['records'])
             == reports['long']['long']['records_pin'], 'LONG_UPDATE_REFERENCE_GRADIENT')
        expected_model,expected_opt = restore(a.output/'initial.pt',pin,state,binding,start)
        for name,p in expected_model.named_parameters(): p.grad = expected['gradients'][name].clone()
        expected_opt.step(); expected_steps = 1
        validate_optimizer(expected_model,expected_opt,state['policy'],start+1)
        need(fingerprint((model.state_dict(),opt.state_dict())) ==
             fingerprint((expected_model.state_dict(),expected_opt.state_dict())), 'LONG_UPDATE_EXPECTED_OPTIMIZER')
        del expected_model,expected_opt,expected
        payloads['long-update.pt'] = fingerprint(reference.numerical(value))
        checkpoint = dict(model=value['model'],optimizer=value['optimizer'],step=start+1,binding=binding)
        resume_pin,_ = save(a.output/'after-long.pt',checkpoint)
        restored,restored_opt = restore(a.output/'after-long.pt',resume_pin,state,binding,start+1)
        phases['long_update_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
        short,q = reference.source.prior.item(state,105748)
        uninterrupted = update(model,opt,short,q,start+1,state,a.output,'continuation')
        resumed = update(restored,restored_opt,short,q,start+1,state,a.output,'restored-continuation')
        reference.exact(uninterrupted,resumed)
        payloads.update({n+'.pt':fingerprint(reference.numerical(v)) for n,v in
                         (('continuation',uninterrupted),('restored-continuation',resumed))})
    phases['continuation_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    if previous:
        for name,pin in payloads.items():
            need(file_sha(a.previous/name) == previous['files'][name], 'LONG_UPDATE_PREVIOUS_BYTES')
            saved = torch.load(a.previous/name,map_location='cpu',weights_only=True)
            need(fingerprint(reference.numerical(saved)) == pin, 'LONG_UPDATE_COLD_NUMERICAL')
        need(previous['controls'] == negatives, 'LONG_UPDATE_COLD_CONTROLS')
    state['fresh'](); learner.check(item)
    for path,pin in physical.items(): need(file_sha(path) == pin, 'LONG_UPDATE_AUTHORITY_CHANGED')
    need(all(file_sha(base/n) == h for n,h in code.items()) and not torch.cuda.is_initialized(),
         'LONG_UPDATE_RUNTIME_CHANGED')
    gc.collect()
    files = {p.name:file_sha(p) for p in a.output.iterdir() if p.is_file()}
    phases['final_authentication'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.001, 'LONG_UPDATE_PHASES')
    report = dict(complete=True, mode=a.mode, binding=binding, code=code, files=files, controls=negatives,
        numerical_pins=payloads, previous_pin=a.previous_pin, probe_pin=a.probe_pin, phases=phases, seconds=seconds,
        optimizer_updates=len(payloads), reference_optimizer_steps=expected_steps,
        long_optimizer_update=a.mode == 'long', cold_reproduced=previous is not None,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, gpu_used=False,
        frozen_encoder_calls=0, promoted=False, capability_demonstrated=False, acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),mode=a.mode,seconds=seconds)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--mode',choices=('probe','long'),required=True)
    p.add_argument('--probe',type=Path); p.add_argument('--probe-pin')
    p.add_argument('--previous',type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json',dict(error=type(exc).__name__,detail=str(exc)))
        raise
