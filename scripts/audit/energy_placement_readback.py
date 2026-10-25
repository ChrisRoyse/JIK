"""Cold reference replay, optimizer restart and paired complete-update timing."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import struct
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--cpu-proof' in sys.argv: os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import energy_placement_qualification as proof
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def exact(actual, expected, path='root'):
    """Independent recursive byte comparison without device/environment mutation."""
    need(type(actual) is type(expected), 'ENERGY_READ_TYPE:'+path)
    if isinstance(actual, torch.Tensor):
        need(actual.device.type == expected.device.type == 'cpu'
             and actual.layout == expected.layout == torch.strided
             and actual.dtype == expected.dtype and actual.shape == expected.shape,
             'ENERGY_READ_LAYOUT:'+path)
        if actual.is_floating_point():
            need(bool(actual.isfinite().all()) and bool(expected.isfinite().all()),
                 'ENERGY_READ_FINITE:'+path)
        a = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        b = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        need(torch.equal(a, b), 'ENERGY_READ_BYTES:'+path)
        return dict(tensors=1, bytes=a.numel())
    counts = dict(tensors=0, bytes=0)
    if isinstance(actual, dict):
        need(actual.keys() == expected.keys(), 'ENERGY_READ_KEYS:'+path)
        children = ((actual[k], expected[k], path+'/'+str(k)) for k in actual)
    elif isinstance(actual, (list, tuple)):
        need(len(actual) == len(expected), 'ENERGY_READ_LENGTH:'+path)
        children = ((a, b, path+'/'+str(i)) for i, (a, b) in enumerate(zip(actual, expected)))
    else:
        same = struct.pack('!d', actual) == struct.pack('!d', expected) if isinstance(actual, float) else actual == expected
        need(same, 'ENERGY_READ_VALUE:'+path)
        return counts
    for a, b, name in children:
        result = exact(a, b, name)
        for k in counts: counts[k] += result[k]
    return counts


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time(); a.output.mkdir(exist_ok=False)
    device = 'cpu' if a.cpu_proof else 'cuda'; proof.configure(device)
    report = proof.source.pinned(a.source, a.pin)
    need(report['complete'] and report['all_updates_exact'] and report['device'] == device
         and report['pid'] != os.getpid() and not report['promoted'], 'ENERGY_READ_SCOPE')
    for n, pin in report['files'].items(): need(file_sha(a.source/n) == pin, 'ENERGY_READ_BYTES')
    source, bundle, target, items, pins = proof.sources(a.root, 'cpu')
    need(pins == report['code'] and source['bundle_state'] == report['source_state'], 'ENERGY_READ_SOURCE')
    lease = False
    try:
        (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        if device == 'cuda':
            need(report['cuda_producer_exact'] and report['cpu_qualification'], 'ENERGY_READ_CUDA_PRODUCER')
            q = report['cpu_qualification']
            proof.cpu_gate(argparse.Namespace(cpu_source=Path(q['source']), cpu_pin=q['pin'],
                cpu_cold=Path(q['cold']), cpu_cold_pin=q['cold_pin']), pins)
            proof.take_lease('29', 'energy-placement-cold-replay'); lease = True
            need(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True).strip() and torch.cuda.is_available(), 'ENERGY_READ_GPU_BUSY')
            target = target.to(device)
        policy = dict(bundle['policy'], device=device); need(policy == report['policy'], 'ENERGY_READ_POLICY')
        frozen = fingerprint(target.state_dict()); phases = dict(authentication=time.perf_counter()-tick)
        timings = []; comparisons = dict(tensors=0, bytes=0); restarts = []; states = []; files = {}
        tick = time.perf_counter()
        for architecture in proof.base.ARCHITECTURES:
            for repetition in range(6):
                order = ('reference', 'candidate') if repetition % 2 == 0 else ('candidate', 'reference')
                for arm in order:
                    model, optimizer = proof.make(bundle, target, architecture, device, arm); elapsed = []
                    # First reference pass uses the original unbatched update implementation.
                    # Later timed reference passes use precisely the current batched-check path.
                    update = proof.base.update if repetition == 0 and arm == 'reference' else proof.active.update
                    for step, item in enumerate(items):
                        proof.sync(device); start = time.perf_counter()
                        record, gradients, noise = update(model, optimizer, item, step, policy, target=target)
                        proof.sync(device); elapsed.append(time.perf_counter()-start)
                        actual = proof.payload(model, optimizer, record, gradients, noise)
                        name = f'{architecture}-reference-{step+1}.pt'
                        expected = proof.checked(a.source, name, report['checkpoints'][name])
                        counts = exact(actual, expected)
                        for k in comparisons: comparisons[k] += counts[k]
                        name = f'{architecture}-candidate-{step+1}.pt'
                        peer = proof.checked(a.source, name, report['checkpoints'][name])
                        exact(actual, peer)
                        if repetition == 0:
                            states.append(dict(architecture=architecture, arm=arm, step=step+1, state=fingerprint(actual)))
                        if repetition == 5 and step == 2:
                            name = f'{architecture}-{arm}-restart.pt'
                            files[name] = proof.source.runtime.persist(a.output/name, actual)
                            reopened = proof.checked(a.output, name, files[name])
                            model, optimizer = proof.restore(bundle, target, architecture, device, arm, reopened, step+1)
                            restarts.append(dict(architecture=architecture, arm=arm, step=step+1, state=files[name]['state']))
                    timings.append(dict(architecture=architecture, arm=arm, repetition=repetition,
                        order=order, seconds=elapsed, total=sum(elapsed), warmup=repetition == 0))
                    del model, optimizer, actual, expected, peer; gc.collect()
            print(json.dumps(dict(architecture=architecture, repeated_updates_exact=True, restarts=2)), flush=True)
        phases['replay_and_benchmark'] = time.perf_counter()-tick; tick = time.perf_counter()
        name = 'shared-candidate-1.pt'; saved = proof.checked(a.source, name, report['checkpoints'][name])
        bad_gradient = deepcopy(saved); next(iter(bad_gradient['gradients'].values())).reshape(-1)[0] += 1
        bad_moment = deepcopy(saved); next(iter(bad_moment['optimizer']['state'].values()))['exp_avg'].reshape(-1)[0] += 1
        negative = {}
        for label, call in (
            ('artifact_pin', lambda: proof.checked(a.source, name, dict(report['checkpoints'][name], sha256='0'*64))),
            ('gradient', lambda: exact(bad_gradient, saved)),
            ('moment', lambda: exact(bad_moment, saved))):
            try: call()
            except (ValueError, RuntimeError, AssertionError) as exc: negative[label] = str(exc)
            else: raise AssertionError('ENERGY_READ_CONTROL_ACCEPTED')
        speed = {}
        for name in proof.base.ARCHITECTURES:
            rows = [r for r in timings if r['architecture'] == name and not r['warmup']]
            ref = [r['total'] for r in rows if r['arm'] == 'reference']
            new = [r['total'] for r in rows if r['arm'] == 'candidate']
            speed[name] = dict(reference=ref, candidate=new, reference_median=statistics.median(ref),
                candidate_median=statistics.median(new), median_ratio=statistics.median(new)/statistics.median(ref))
        need(fingerprint(target.state_dict()) == frozen and all(file_sha(proof.REPO/n) == pin for n, pin in pins.items()),
             'ENERGY_READ_FINAL_PINS')
        need(device != 'cpu' or not torch.cuda.is_initialized(), 'ENERGY_READ_UNEXPECTED_GPU')
        timing_pin = write_committed(a.output/'timings.json', timings)
        state_pin = write_committed(a.output/'states.json', states)
        (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
        physical = {str(p.relative_to(a.output)): file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['controls_and_final_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'ENERGY_READ_PHASE_COVERAGE')
        result = dict(complete=True, pid=os.getpid(), source_pin=a.pin, code=pins, device=device,
            all_updates_exact=True, reference_replay_exact=True, restarts_exact=True,
            replayed_updates=180, distinct_update_cases=15, restart_count=len(restarts), restarts=restarts,
            comparisons=comparisons, controls=negative, speed=speed, timing_pin=timing_pin, state_pin=state_pin,
            checkpoint_files=files, files=physical, source_state=report['source_state'],
            gpu_qualified=device == 'cuda', promoted=False, phases=phases, seconds=seconds,
            cpu_seconds=time.process_time()-cpu0, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            accuracy_gain_claimed=False, acceptance='author-checked')
        write_committed(a.output/'report.json', result)
        print(json.dumps(dict(complete=True, pin=file_sha(a.output/'report.json'), speed=speed, seconds=seconds)), flush=True)
    finally:
        if lease: proof.release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('root', 'source', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--pin', required=True); p.add_argument('--cpu-proof', action='store_true'); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
