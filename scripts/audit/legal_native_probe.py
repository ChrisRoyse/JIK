"""Score-blind legal input selection and canonical native runtime evidence."""
import argparse
from collections import Counter
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import take_lease, release_lease
from text.canonical_windows import contract, exact_reference, forward
from text.constellation_corpus import canonical, digest, file_sha, need
from text.corpus_readings import ROSTER
from text.legal_reuse_input import ReuseInputs


def select(bank):
    selected = set(range(min(128, bank.report['counts']['unique_inputs'])))
    histogram = Counter(); per_range = Counter(); longest = []; limits = (16, 32, 64, 128, 256, 512, 1024)
    for batch in bank.table.iter_batches(batch_size=8192, columns=['index', 'text']):
        data = batch.to_pydict()
        for i, text in zip(data['index'], data['text']):
            length = len(text); bucket = next((limit for limit in limits if length <= limit), 'over1024')
            histogram[bucket] += 1
            if per_range[bucket] < 4: selected.add(i); per_range[bucket] += 1
            if len(longest) < 8: heapq.heappush(longest, (length, -i))
            elif (length, -i) > longest[0]: heapq.heapreplace(longest, (length, -i))
    selected.update(-i for _, i in longest)
    counts = np.asarray(bank.arrays['counts'])
    threshold = np.partition(counts, len(counts)-8)[-8]
    frequent = np.flatnonzero(counts >= threshold)
    frequent = sorted(map(int, frequent), key=lambda i: (-int(counts[i]), i))[:8]
    selected.update(frequent)
    rows = [bank.input(int(i)) for i in sorted(selected)]
    return rows, dict(character_histogram={str(k): v for k, v in histogram.items()},
        longest=[dict(characters=n, unique=-i) for n, i in sorted(longest, reverse=True)], most_frequent=frequent,
        selection='first128 SHA order + first4 per length range + longest8 + most frequent8; before any score')


def setup_gpu():
    need(torch.cuda.is_available(), 'CANONICAL_GPU_REQUIRED')
    resident = subprocess.check_output(['nvidia-smi'], text=True)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    need(not active.strip(), 'CANONICAL_GPU_BUSY')
    take_lease('29', 'legal-canonical-native-runtime-proof')
    torch.set_num_threads(32); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return resident


def measure(name, texts, destination):
    start = tick = time.perf_counter(); cpu = time.process_time(); destination.mkdir()
    lens = LearnedLens(by_name(name), 'cuda')
    need(not lens.model.training and all(not p.requires_grad for p in lens.model.parameters()), 'CANONICAL_FROZEN')
    identity = contract(lens); lens_id = digest(canonical(identity)); torch.cuda.reset_peak_memory_stats()
    phases = dict(load=time.perf_counter()-tick); tick = time.perf_counter()
    values, bounds, traces, timing = forward(lens, texts)
    np.savez(destination/'reference.npz', vectors=values, bounds=bounds)
    write_committed(destination/'traces.json', traces)
    phases['reference_forward_and_persistence'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = {}; control_times = {}
    orders = dict(replay=list(range(len(texts))), reversed=list(reversed(range(len(texts)))),
                  duplicate_mixed=list(range(len(texts)))+list(reversed(range(len(texts)))))
    for label, order in orders.items():
        v, b, t, perf = forward(lens, [texts[i] for i in order])
        controls[label] = exact_reference(values, bounds, traces, v, b, t, order); control_times[label] = perf
    singleton_seconds = 0.
    for i, text in enumerate(texts):
        v, b, t, perf = forward(lens, [text]); singleton_seconds += perf['seconds']
        exact_reference(values, bounds, traces, v, b, t, [i])
    controls['singletons'] = True; control_times['singletons'] = dict(seconds=singleton_seconds, inputs=len(texts))
    phases['batch_identity_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
    need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
         'CANONICAL_INSTRUMENT_DRIFT')
    files = {n: dict(bytes=(destination/n).stat().st_size, sha256=file_sha(destination/n))
             for n in ('reference.npz', 'traces.json')}
    phases['frozen_readback_and_pins'] = time.perf_counter()-tick
    result = dict(name=name, lens_id=lens_id, contract=identity, inputs=len(texts), windows=len(values),
        maximum_windows=int(np.diff(bounds).max()), all_windows_retained=True, controls=controls,
        timing=timing, control_timings=control_times, files=files,
        phases_seconds=phases, seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        frozen_unchanged=True, scientific_admission=False, runtime_ready_pending_cold_replay=True)
    need(sum(phases.values())/result['seconds'] >= .9, 'CANONICAL_TIMING')
    pin = write_committed(destination/'report.json', result)
    print(json.dumps(dict(name=name, pin=pin, inputs=len(texts), windows=len(values), seconds=result['seconds'])), flush=True)
    del lens; torch.cuda.empty_cache()
    return dict(pin=pin, lens_id=lens_id)


def run(root, pin, output):
    start = tick = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    bank = ReuseInputs(root, pin); rows, selection = select(bank)
    write_committed(output/'inputs.json', rows); write_committed(output/'selection.json', selection)
    source_seconds = time.perf_counter()-tick
    resident = setup_gpu()
    results, errors = {}, []
    try:
        (output/'gpu-before.txt').write_text(resident)
        for name in ROSTER:
            try: results[name] = measure(name, [r['text'] for r in rows], output/name)
            except Exception as exc:
                error = dict(name=name, error=type(exc).__name__+':'+str(exc)); errors.append(error)
                write_committed(output/name/'failure.json', error); print(json.dumps(error), flush=True)
                torch.cuda.empty_cache()
        bank.fresh()
        report = dict(schema='legal-canonical-native-probe-v1', complete=not errors, errors=errors, instruments=results,
            source=dict(path=str(root), pin=pin), inputs=len(rows), source_seconds=source_seconds,
            files={n: dict(bytes=(output/n).stat().st_size, sha256=file_sha(output/n)) for n in ('inputs.json', 'selection.json')},
            roster=list(ROSTER), number_status='measured', acceptance='author-checked',
            full_corpus_measured=False, model_updates=0, scientific_admission=False,
            code_sha256=file_sha(Path(__file__)), seconds=time.perf_counter()-start)
        result_pin = write_committed(output/'report.json', report)
        print(json.dumps(dict(pin=result_pin, complete=not errors, inputs=len(rows))), flush=True)
        need(not errors, 'CANONICAL_RUNTIME_REFUSED')
    finally: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.root, a.pin, a.output)
