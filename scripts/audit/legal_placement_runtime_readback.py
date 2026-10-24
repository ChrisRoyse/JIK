"""Paired original/efficient placement updates and cold restart on a declared device."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import native_placement_checked as original, native_placement_efficient as efficient
from model.native_device_transport import transfer
from model.native_population_training import cpu, fingerprint
from model.native_rq_packed import PackedNativeRQ
from text.constellation_corpus import file_sha, need


def pinned(path, pin):
    need(file_sha(path) == pin, 'PLACEMENT_RUNTIME_PIN')
    return json.loads(path.read_bytes())


def persist(path, value):
    value = cpu(value)
    with path.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(loaded) == fingerprint(value), 'PLACEMENT_RUNTIME_PAYLOAD_READBACK')
    return dict(sha256=file_sha(path), state=fingerprint(loaded))


def reject(fn):
    try: fn()
    except (ValueError, RuntimeError) as error: return type(error).__name__+':'+str(error)
    raise AssertionError('PLACEMENT_RUNTIME_NEGATIVE_ACCEPTED')


def make(bundle, target, name, device):
    policy = bundle['policy']
    model = original.NativePlacementForecaster(target, target_pin=bundle['target_ref']['pin'], architecture=name,
                 seed=policy['seed'], placement_seed=policy['placement_seed'], cache_bytes=policy['cache_bytes'],
                 output_bytes=policy['output_bytes']).to(device)
    return model, original.optimizer_for(model, policy)


def controls(model, optimizer, item, held, target, step, policy):
    before = fingerprint((model.state_dict(), optimizer.state_dict()))
    invoke = lambda row=item, opt=optimizer, cursor=step, tt=target: efficient.update(model, opt, row, cursor, policy, target=tt)
    result = dict(nontrain=reject(lambda: invoke(row=held)),
                  native_pin=reject(lambda: invoke(row=dict(item, native_pin='0'*64))),
                  label_pin=reject(lambda: invoke(row=dict(item, label_pin='0'*64))),
                  cursor=reject(lambda: invoke(cursor=step+1)),
                  weights_only=reject(lambda: invoke(opt=original.optimizer_for(model, policy))))
    parameter = next(model.parameters()); state = optimizer.state.pop(parameter)
    try: result['missing_optimizer_state'] = reject(invoke)
    finally: optimizer.state[parameter] = state
    values = transfer(item['frames'][:2], next(model.parameters()).device)
    broken = tuple({n: x.clone() for n, x in f.items()} for f in values)
    broken[0][model.roster[0]][0, 0] = float('nan')
    stream = efficient.stream_for(model)
    result['nonfinite_native'] = reject(lambda: stream.step(broken, context='negative', offset=0, detach_history=True))
    old = state['exp_avg'].clone(); state['exp_avg'].view(-1)[0] = float('nan')
    try: result['nonfinite_moment'] = reject(invoke)
    finally: state['exp_avg'].copy_(old)
    other = deepcopy(target)
    result['wrong_target'] = reject(lambda: invoke(tt=other))
    optimizer.zero_grad(set_to_none=True)
    need(fingerprint((model.state_dict(), optimizer.state_dict())) == before, 'PLACEMENT_RUNTIME_CONTROL_MUTATION')
    return result


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False); device = 'cpu' if a.cpu_proof else 'cuda'
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    if device == 'cuda':
        take_lease('29', 'placement-architecture-runtime-qualification')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        need(not active.strip() and torch.cuda.is_available(), 'PLACEMENT_RUNTIME_GPU_BUSY')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = pinned(a.source/'report.json', a.source_pin)
    need(source['complete'] and source['train_documents'] == 5 and source['train_targets'] == 66,
         'PLACEMENT_RUNTIME_SOURCE_SCOPE')
    root = Path(__file__).resolve().parents[1]
    for name, pin in source['code'].items():
        need(file_sha(root/name) == pin and file_sha(a.source/'source'/name) == pin, 'PLACEMENT_RUNTIME_CODE_CHANGED')
    need(file_sha(a.source/'bundle.pt') == source['bundle_sha256'], 'PLACEMENT_RUNTIME_BUNDLE_PIN')
    bundle = torch.load(a.source/'bundle.pt', map_location='cpu', weights_only=True)
    need(fingerprint(bundle) == source['bundle_state'], 'PLACEMENT_RUNTIME_BUNDLE_STATE')
    target = PackedNativeRQ(bundle['dimensions'], bundle['config'], bundle['sizes'],
                            'no_encoder_quantizer', bundle['statistics'])
    target.load_state_dict(bundle['target'], strict=True); target.eval().requires_grad_(False)
    need(fingerprint(target.state_dict()) == bundle['target_fingerprint'], 'PLACEMENT_RUNTIME_TARGET_RECONSTRUCTION')
    target = target.to(device); frozen = fingerprint(target.state_dict())
    fitting = [i for i in bundle['items'] if i['side'] == 'train']
    held = next(i for i in bundle['items'] if i['side'] == 'query')
    policy = dict(bundle['policy'], device=device)
    binding = dict(source=a.source_pin, code=source['code'], policy=policy, device=device,
                   torch_version=str(torch.__version__), target=bundle['target_ref'],
                   arithmetic='same_GEMM_shapes_addition_order_objective_and_AdamW')
    previous = pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding'] == binding, 'PLACEMENT_RUNTIME_COLD_BINDING')
        for n, pin in previous['files'].items(): need(file_sha(a.previous/n) == pin, 'PLACEMENT_RUNTIME_COLD_BYTES')
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    results = {}; timings = []
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    for name in original.ARCHITECTURES:
        path = a.output/name; path.mkdir()
        models = {n: make(bundle, target, name, device) for n in ('original', 'efficient')}
        initial = {n: fingerprint((m.state_dict(), o.state_dict())) for n, (m, o) in models.items()}
        need(initial['original'] == initial['efficient'], 'PLACEMENT_RUNTIME_INITIAL')
        if device == 'cpu':
            m, o = models['original']
            initial_payload = dict(model=m.state_dict(), optimizer=o.state_dict(), step=0,
                                   noise_cursor=policy['noise_seed'])
            need(fingerprint(initial_payload) == bundle['cpu_reference'][name]['initial_state'],
                 'PLACEMENT_RUNTIME_PRIOR_INITIAL')
        steps = []; states = []; negatives = None
        for step, item in enumerate(fitting):
            outputs = {}; snapshots = {}
            for arm in (('original', 'efficient') if step % 2 == 0 else ('efficient', 'original')):
                m, o = models[arm]; sync(); started = time.perf_counter()
                fn = original.update if arm == 'original' else efficient.update
                record, gradients, noise = fn(m, o, item, step, policy, target=target)
                sync(); timings.append(dict(architecture=name, arm=arm, step=step+1, document=item['document'],
                                            targets=len(item['codes']), seconds=time.perf_counter()-started))
                outputs[arm] = record
                snapshots[arm] = cpu(dict(model=m.state_dict(), optimizer=o.state_dict(), gradients=gradients,
                                          noise=noise, record=record, step=step+1, noise_cursor=policy['noise_seed']+step+1))
            need(outputs['original'] == outputs['efficient'], 'PLACEMENT_RUNTIME_UPDATE_NOT_EXACT')
            need(fingerprint(snapshots['original']) == fingerprint(snapshots['efficient']), 'PLACEMENT_RUNTIME_STATE_NOT_EXACT')
            if device == 'cpu':
                need(outputs['original'] == bundle['cpu_reference'][name]['updates'][step]['record'], 'PLACEMENT_RUNTIME_PRIOR_CPU_NOT_EXACT')
            cp = persist(path/f'checkpoint-{step+1}.pt', snapshots['efficient']); states.append(cp['state']); steps.append(outputs['efficient'])
            # Replace the efficient model with a real full-state restoration after every update.
            models['efficient'] = make(bundle, target, name, device); m, o = models['efficient']
            saved = torch.load(path/f'checkpoint-{step+1}.pt', map_location='cpu', weights_only=True)
            need(file_sha(path/f'checkpoint-{step+1}.pt') == cp['sha256'] and fingerprint(saved) == cp['state'],
                 'PLACEMENT_RUNTIME_RESTORE_BYTES')
            m.load_state_dict(saved['model'], strict=True); o.load_state_dict(saved['optimizer'])
            efficient.validate_optimizer(m, o, policy, step+1)
            need(fingerprint((m.state_dict(), o.state_dict())) == fingerprint((saved['model'], saved['optimizer'])),
                 'PLACEMENT_RUNTIME_RESTORE_STATE')
            print(json.dumps(dict(architecture=name, step=step+1, exact=True, timings=timings[-2:])), flush=True)
        m, o = models['efficient']; negatives = controls(m, o, fitting[-1], held, target, len(fitting), policy)
        need(fingerprint(target.state_dict()) == frozen, 'PLACEMENT_RUNTIME_TARGET_CHANGED')
        results[name] = dict(initial=initial['original'], steps=steps, checkpoint_states=states, controls=negatives,
                             parameters=sum(p.numel() for p in m.parameters()))
        del models, m, o, outputs, snapshots
    phases['paired_updates_resume_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    result_pin = write_committed(a.output/'results.json', results)
    if previous: need(result_pin == previous['results_pin'], 'PLACEMENT_RUNTIME_COLD_NOT_EXACT')
    for name, pin in source['code'].items(): need(file_sha(root/name) == pin, 'PLACEMENT_RUNTIME_CODE_CHANGED')
    need(device != 'cpu' or not torch.cuda.is_initialized(), 'PLACEMENT_RUNTIME_UNEXPECTED_GPU')
    files = {str(f.relative_to(a.output)): file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.000001, 'PLACEMENT_RUNTIME_TIMING_COVERAGE')
    report = dict(complete=True, binding=binding, results_pin=result_pin, previous_pin=a.previous_pin, files=files,
                  timings=timings, phases=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu0,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                  all_updates_exact=True, cpu_reference_exact=device == 'cpu', gpu_qualified=device == 'cuda',
                  capacity_matched=False, scientific_admission=False, capability_demonstrated=False,
                  number_status='measured', acceptance='author-checked')
    print(json.dumps(dict(complete=True, pin=write_committed(a.output/'report.json', report))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--source-pin', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--cpu-proof', action='store_true')
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    a = p.parse_args(); need(bool(a.previous) == bool(a.previous_pin), 'PLACEMENT_RUNTIME_PREVIOUS_ARGUMENTS')
    try: run(a)
    except BaseException as error:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(error)))
        raise
    finally: release_lease()
