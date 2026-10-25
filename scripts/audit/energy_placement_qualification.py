"""Qualify explicit energy batching against the active batched-check update path."""
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_grouped_placement_training as source
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import native_placement_batched_checks as active
from model import native_placement_training as base
from model.native_energy_forecaster import EnergyBatchedForecaster
from model.native_population_training import cpu, fingerprint
from model.native_rq_packed import PackedNativeRQ
from text.constellation_corpus import file_sha, need

REPO = Path(__file__).resolve().parents[1]
NEW = ('audit/energy_placement_qualification.py', 'audit/energy_placement_readback.py',
       'model/native_energy_forecaster.py', 'model/native_energy_batched.py',
       'model/native_placement_batched_checks.py', 'model/native_batched_validation.py')


def sources(root, device):
    report, bundle = source.source_bundle(root)
    target = PackedNativeRQ(bundle['dimensions'], bundle['config'], bundle['sizes'],
                            'no_encoder_quantizer', bundle['statistics'])
    target.load_state_dict(bundle['target'], strict=True); target.eval().requires_grad_(False)
    need(fingerprint(target.state_dict()) == bundle['target_fingerprint'], 'ENERGY_QUAL_TARGET')
    items = [i for i in bundle['items'] if i['side'] == 'train']
    need(len(items) == 5 and sum(len(i['codes']) for i in items) == 66, 'ENERGY_QUAL_COHORT')
    pins = dict(source.code(report))
    pins.update({n: file_sha(REPO/n) for n in NEW})
    return report, bundle, target.to(device), items, pins


def make(bundle, target, name, device, arm):
    need(arm in ('reference', 'candidate'), 'ENERGY_QUAL_ARM')
    policy = bundle['policy']; cls = base.NativePlacementForecaster if arm == 'reference' else EnergyBatchedForecaster
    model = cls(target, target_pin=bundle['target_ref']['pin'], architecture=name, seed=policy['seed'],
                placement_seed=policy['placement_seed'], cache_bytes=policy['cache_bytes'],
                output_bytes=policy['output_bytes']).to(device)
    return model, base.optimizer_for(model, policy)


def configure(device):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    need(device in ('cpu', 'cuda'), 'ENERGY_QUAL_DEVICE')


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def payload(model, optimizer, record, gradients, noise):
    return cpu(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    record=record, gradients=gradients, noise=noise))


def checked(root, name, ref):
    path = root/name; need(file_sha(path) == ref['sha256'], 'ENERGY_QUAL_ARTIFACT_PIN')
    value = torch.load(path, map_location='cpu', weights_only=True)
    need(fingerprint(value) == ref['state'], 'ENERGY_QUAL_ARTIFACT_STATE')
    return value


def restore(bundle, target, architecture, device, arm, saved, step):
    model, optimizer = make(bundle, target, architecture, device, arm)
    model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    active.validate_optimizer(model, optimizer, dict(bundle['policy'], device=device), step)
    need(fingerprint((model.state_dict(), optimizer.state_dict()))
         == fingerprint((saved['model'], saved['optimizer'])), 'ENERGY_QUAL_RESTORE')
    return model, optimizer


def controls(model, optimizer, item, held, target, step, policy):
    reject = source.runtime.reject; before = fingerprint((model.state_dict(), optimizer.state_dict()))
    invoke = lambda row=item, opt=optimizer, cursor=step, tt=target, config=policy: active.update(model, opt, row, cursor, config, target=tt)
    result = dict(nontrain=reject(lambda: invoke(row=held)),
        native_pin=reject(lambda: invoke(row=dict(item, native_pin='0'*64))),
        label_pin=reject(lambda: invoke(row=dict(item, label_pin='0'*64))),
        cursor=reject(lambda: invoke(cursor=step+1)),
        memory=reject(lambda: invoke(config=dict(policy, working_bytes=1))),
        weights_only=reject(lambda: invoke(opt=base.optimizer_for(model, policy))))
    parameter = next(model.parameters()); state = optimizer.state.pop(parameter)
    try: result['missing_moment'] = reject(invoke)
    finally: optimizer.state[parameter] = state
    old = state['exp_avg'].clone(); state['exp_avg'].reshape(-1)[0] = float('nan')
    try: result['nonfinite_moment'] = reject(invoke)
    finally: state['exp_avg'].copy_(old)
    other = deepcopy(target); result['wrong_target'] = reject(lambda: invoke(tt=other))
    broken = deepcopy(item); broken['frames'][0][model.roster[0]][0, 0] = float('nan')
    broken['native_pin'] = fingerprint(broken['frames'])
    result['nonfinite_native'] = reject(lambda: invoke(row=broken))
    broken = deepcopy(item); broken['frames'][0][model.roster[0]] = broken['frames'][0][model.roster[0]].reshape(-1)
    broken['native_pin'] = fingerprint(broken['frames'])
    result['native_layout'] = reject(lambda: invoke(row=broken))
    optimizer.zero_grad(set_to_none=True)
    need(fingerprint((model.state_dict(), optimizer.state_dict())) == before, 'ENERGY_QUAL_CONTROL_MUTATION')
    return result


def cpu_gate(a, pins):
    need(a.cpu_source is not None and a.cpu_cold is not None and a.cpu_pin and a.cpu_cold_pin, 'ENERGY_QUAL_CPU_REQUIRED')
    first = source.pinned(a.cpu_source, a.cpu_pin); cold = source.pinned(a.cpu_cold, a.cpu_cold_pin)
    need(first['complete'] and first['all_updates_exact'] and first['device'] == 'cpu'
         and cold['complete'] and cold['all_updates_exact'] and cold['device'] == 'cpu'
         and cold['source_pin'] == a.cpu_pin and first['code'] == cold['code'] == pins
         and cold['restarts_exact'] and cold['reference_replay_exact'], 'ENERGY_QUAL_CPU_PROOF')
    for root, report in ((a.cpu_source, first), (a.cpu_cold, cold)):
        for n, pin in report['files'].items(): need(file_sha(root/n) == pin, 'ENERGY_QUAL_CPU_BYTES')


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time(); a.output.mkdir(exist_ok=False)
    device = 'cpu' if a.cpu_proof else 'cuda'; configure(device)
    report, bundle, target, items, pins = sources(a.root, 'cpu')
    lease = False
    try:
        (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        if device == 'cuda':
            cpu_gate(a, pins); take_lease('29', 'energy-placement-qualification'); lease = True
            need(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True).strip() and torch.cuda.is_available(), 'ENERGY_QUAL_GPU_BUSY')
            target = target.to(device)
        policy = dict(bundle['policy'], device=device); frozen = fingerprint(target.state_dict())
        phases = dict(authentication=time.perf_counter()-tick); files = {}; cases = {}; timings = []
        tick = time.perf_counter()
        for n, pin in pins.items():
            path = a.output/'source'/n; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO/n).read_bytes()); need(file_sha(path) == pin, 'ENERGY_QUAL_SOURCE_COPY')
        held = next(i for i in bundle['items'] if i['side'] == 'query')
        phases['source_capture'] = time.perf_counter()-tick
        tick = time.perf_counter()
        for architecture in base.ARCHITECTURES:
            models = {arm: make(bundle, target, architecture, device, arm) for arm in ('reference', 'candidate')}
            initial = {arm: fingerprint((m.state_dict(), o.state_dict())) for arm, (m, o) in models.items()}
            need(initial['reference'] == initial['candidate'], 'ENERGY_QUAL_INITIAL')
            states = []
            for step, item in enumerate(items):
                outputs = {}
                for arm in (('reference', 'candidate') if step % 2 == 0 else ('candidate', 'reference')):
                    model, optimizer = models[arm]; sync(device); start = time.perf_counter()
                    record, gradients, noise = active.update(model, optimizer, item, step, policy, target=target)
                    sync(device); elapsed = time.perf_counter()-start
                    timings.append(dict(architecture=architecture, arm=arm, step=step+1, seconds=elapsed))
                    value = payload(model, optimizer, record, gradients, noise)
                    name = f'{architecture}-{arm}-{step+1}.pt'; ref = source.runtime.persist(a.output/name, value)
                    files[name] = ref; outputs[arm] = value
                    if device == 'cpu':
                        need(record == bundle['cpu_reference'][architecture]['updates'][step]['record'], 'ENERGY_QUAL_PRIOR_CPU')
                need(fingerprint(outputs['reference']) == fingerprint(outputs['candidate']), 'ENERGY_QUAL_UPDATE_DIFFERENT')
                name = f'{architecture}-candidate-{step+1}.pt'
                models['candidate'] = restore(bundle, target, architecture, device, 'candidate',
                    checked(a.output, name, files[name]), step+1)
                states.append(files[name]['state'])
            m, opt = models['candidate']; negative = controls(m, opt, items[-1], held, target, len(items), policy)
            need(fingerprint(target.state_dict()) == frozen, 'ENERGY_QUAL_TARGET_CHANGED')
            cases[architecture] = dict(initial=initial['reference'], states=states, controls=negative)
            print(json.dumps(dict(architecture=architecture, exact=True, updates=10, restarts=5)), flush=True)
            del models, m, opt, outputs, value; gc.collect()
        phases['updates_persistence_and_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
        cases_pin = write_committed(a.output/'cases.json', cases)
        need(all(file_sha(REPO/n) == pin for n, pin in pins.items()), 'ENERGY_QUAL_CODE_CHANGED')
        need(device != 'cpu' or not torch.cuda.is_initialized(), 'ENERGY_QUAL_UNEXPECTED_GPU')
        physical = {str(p.relative_to(a.output)): file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['final_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'ENERGY_QUAL_PHASE_COVERAGE')
        result = dict(complete=True, pid=os.getpid(), device=device, code=pins, source_pin=source.SOURCE[1],
            source_state=report['bundle_state'], policy=policy, torch_version=str(torch.__version__),
            target=bundle['target_ref'], cases_pin=cases_pin, checkpoints=files, files=physical, timings=timings,
            documents=5, targets=66, architectures=3, paired_update_cases=15, executed_updates=30,
            restart_count=15, all_updates_exact=True, gpu_qualified=False, cuda_producer_exact=device == 'cuda',
            phases=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            cpu_qualification=None if device == 'cpu' else dict(source=str(a.cpu_source), pin=a.cpu_pin,
                cold=str(a.cpu_cold), cold_pin=a.cpu_cold_pin),
            promoted=False, accuracy_gain_claimed=False, acceptance='author-checked')
        write_committed(a.output/'report.json', result)
        print(json.dumps(dict(complete=True, pin=file_sha(a.output/'report.json'), seconds=seconds)), flush=True)
    finally:
        if lease: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('root', 'output'): p.add_argument('--'+n, type=Path, required=True)
    for n in ('cpu-source', 'cpu-cold'): p.add_argument('--'+n, type=Path)
    for n in ('cpu-pin', 'cpu-cold-pin'): p.add_argument('--'+n)
    p.add_argument('--cpu-proof', action='store_true'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
