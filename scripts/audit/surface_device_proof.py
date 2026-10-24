"""Characterize original, mixed and gathered surface execution on one leased device."""
import argparse
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

from audit import surface_device_authority as authority
from claims.process_conditions import write_committed
from lenscommon import take_lease, release_lease
from model import surface_device_runtime as runtime
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import file_sha, need


def run(a):
    began = tick = time.perf_counter(); process = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    leased = False
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        need(os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'SURFACE_DEVICE_CUBLAS_CONTRACT')
        config, specs, examples, metadata, initial, shuffled, saved, code, fresh = authority.load(a.root)
        train = [e for e in examples if e.side == 'train']
        selected = [next(e for e in train if e.occurrence == i) for i in saved['selected']]
        device = torch.device('cpu' if a.cpu_proof else 'cuda:0')
        if not a.cpu_proof:
            observed = subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True)
            (a.output/'nvidia-before.txt').write_text(observed.stdout)
            need(os.environ.get('CUDA_VISIBLE_DEVICES') != '' and torch.cuda.is_available(), 'SURFACE_DEVICE_CUDA_REQUIRED')
            take_lease('29', 'surface-device-characterization'); leased = True
            write_committed(a.output/'lease.json', json.loads((Path.home()/'mnlm/.lease/gpu').read_bytes()))
        else: need(not torch.cuda.is_initialized(), 'SURFACE_DEVICE_CPU_INITIALIZED_CUDA')
        environment = dict(device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
            deterministic=torch.are_deterministic_algorithms_enabled(), cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            tf32=False, threads=torch.get_num_threads(), cpu_proof=a.cpu_proof,
            gpu=torch.cuda.get_device_name(device) if not a.cpu_proof else None)
        phases = dict(authentication_device=time.perf_counter()-tick); tick = time.perf_counter()
        packing = {}
        for label, mode, rows in runtime.ARMS:
            states = []
            for b in runtime.batches(selected, config, specs, mode, rows):
                actual = cpu(runtime.packed(b.to(device)))
                need(fingerprint(actual) == fingerprint(runtime.packed(b)), 'SURFACE_DEVICE_TRANSFER_CHANGED')
                states.append(actual)
            torch.save(states, a.output/f'packed-{label}.pt')
            manifest = [fingerprint(runtime.packed(b)) for b in runtime.batches(examples, config, specs, mode, rows)]
            write_committed(a.output/f'schedule-{label}.json', manifest)
            packing[label] = dict(calls=len(manifest), selected_calls=len(states), state=fingerprint(states))
            del states, b, actual
        phases['packing_transfer_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
        # Exercise CUDA kernels before timed comparisons; discard all warm-up state.
        warm, _ = runtime.execute(initial, config, specs, selected, selected, selected, 'reference', None, device)
        del warm; gc.collect(); runtime.sync(device)
        if not a.cpu_proof: torch.cuda.empty_cache()
        phases['warmup'] = time.perf_counter()-tick; tick = time.perf_counter()
        results = {}; order = list(reversed(runtime.ARMS)) if a.reverse else runtime.ARMS
        for label, mode, rows in order:
            if not a.cpu_proof:
                torch.cuda.reset_peak_memory_stats(device)
                before = torch.cuda.memory_allocated(device)
            else: before = 0
            started = time.perf_counter()
            artifact, profile = runtime.execute(initial, config, specs, train, examples, shuffled, mode, rows, device)
            runtime.sync(device); elapsed = time.perf_counter()-started
            gc.collect()
            memory = dict(allocated_before=before, allocated_after=torch.cuda.memory_allocated(device) if not a.cpu_proof else 0,
                peak_allocated=torch.cuda.max_memory_allocated(device) if not a.cpu_proof else 0,
                peak_reserved=torch.cuda.max_memory_reserved(device) if not a.cpu_proof else 0)
            need(memory['allocated_after'] == before, 'SURFACE_DEVICE_RETAINED_ALLOCATION')
            need(memory['peak_reserved'] <= 30000*1024**2, 'SURFACE_DEVICE_LEASE_MEMORY')
            torch.save(artifact, a.output/f'{label}.pt')
            need(fingerprint(torch.load(a.output/f'{label}.pt', map_location='cpu', weights_only=True)) == fingerprint(artifact),
                 'SURFACE_DEVICE_SAVED_STATE')
            if a.cpu_proof:
                prior = 'reference' if mode == 'reference' else f'rows-{rows}'
                need(fingerprint(artifact) == saved['results'][prior]['state'], 'SURFACE_DEVICE_CPU_PRIOR_DIFFERENCE')
            results[label] = dict(state=fingerprint(artifact), profile=profile, seconds=elapsed, memory=memory,
                conditioning_delta=[float((x['losses']-y['losses']).mean()) for x, y in zip(artifact['shuffled'], artifact['evaluation'])])
            del artifact
            print(json.dumps(dict(arm=label, seconds=elapsed, memory=memory)), flush=True)
        phases['actual_model_and_state_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
        baseline = torch.load(a.output/'reference.pt', map_location='cpu', weights_only=True)
        for label, _, _ in runtime.ARMS:
            actual = torch.load(a.output/f'{label}.pt', map_location='cpu', weights_only=True)
            results[label]['difference_from_reference'] = runtime.differences(actual, baseline)
            results[label]['optimizer_matches_reference'] = fingerprint(actual['optimizer']) == fingerprint(baseline['optimizer'])
        for rows in runtime.COHORTS:
            need(results[f'mixed-{rows}']['state'] == results[f'gathered-{rows}']['state'], 'SURFACE_DEVICE_SAME_COHORT_DIFFERENCE')
        fresh()
        write_committed(a.output/'results.json', dict(results=results, packing=packing, metadata=metadata,
            donors=saved['donors'], selected=saved['selected'], execution_order=[v[0] for v in order]))
        for n, h in code.items():
            p = a.output/'source'/n; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes()); need(file_sha(p) == h, 'SURFACE_DEVICE_CODE_COPY')
        files = {str(p.relative_to(a.output)): file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['comparisons_seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'SURFACE_DEVICE_TOTAL_PHASE_COVERAGE')
        need(not a.cpu_proof or not torch.cuda.is_initialized(), 'SURFACE_DEVICE_CPU_GPU_LEAK')
        report = dict(complete=True, code=code, files=files, sources=authority.REFS, environment=environment,
            same_cohort_exact=True, original_per_atom_exact=all(v['state'] == results['reference']['state'] for v in results.values()),
            seconds=seconds, cpu_seconds=time.process_time()-process, phases=phases,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            promoted_to_training=False, scientific_admission=False, capability_demonstrated=False,
            acceptance='author-checked', number_status='measured')
        pin = write_committed(a.output/'report.json', report); print(json.dumps(dict(complete=True, pin=pin)), flush=True)
    finally:
        if leased: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=authority.ROOT); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-proof', action='store_true'); p.add_argument('--reverse', action='store_true'); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(e)))
        raise
