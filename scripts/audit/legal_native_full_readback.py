"""Qualify and execute a full-bank source/window audit plus exact frozen GPU replay."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
import text_windows
from audit.legal_native_full_scan import POLICY, scan, controls
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import release_lease
from text.canonical_windows import contract, forward
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import stamp
from text.legal_native_measure import setup_gpu
from text.legal_native_store import NativeBank
from text.legal_reuse_input import ReuseInputs


def pinned(path, pin):
    need(file_sha(path) == pin, 'NATIVE_FULL_REPORT_PIN')
    return json.loads(path.read_bytes())


def code():
    base = Path(__file__).resolve().parents[1]
    names = ['audit/legal_native_full_readback.py', 'audit/legal_native_full_scan.py',
             'audit/legal_native_bank_readback.py', 'text/legal_native_store.py',
             'text/legal_native_publish.py', 'text/legal_reuse_input.py',
             'text/canonical_windows.py', 'text/legal_native_measure.py',
             'lens/learned.py', 'lens/catalogue.py']
    return dict({n:file_sha(base/n) for n in names}, window_provenance=file_sha(Path(text_windows.__file__)))


def cohort(root, report):
    for n, pin in report['files'].items():
        need(file_sha(root/n) == pin, 'NATIVE_FULL_COHORT_PIN')
    rows = json.loads((root/'samples.json').read_bytes())
    with np.load(root/'expected.npz', allow_pickle=False) as saved:
        values, bounds = saved['vectors'].copy(), saved['bounds'].copy()
    need(values.dtype == np.float32 and values.shape == (report['replay_windows'], 768)
         and len(rows) == report['replay_inputs'] and len(bounds) == len(rows)+1
         and bounds[0] == 0 and bounds[-1] == len(values) and bool((np.diff(bounds) > 0).all()),
         'NATIVE_FULL_COHORT_SHAPE')
    return rows, values, bounds


def independent_counts(report):
    # Python integer counts/fsum are separate from the scan's NumPy reductions.
    windows = sum(int(r['windows']) for r in report['shards'])
    inputs = sum(int(r['path'].split('-')[1])-int(r['path'].split('-')[0]) for r in report['shards'])
    need(windows == report['windows'] and inputs == report['inputs'], 'NATIVE_FULL_COUNT_REPLAY')
    sums = [math.fsum(r['sum'][i] for r in report['shards']) for i in range(768)]
    squares = [math.fsum(r['square_sum'][i] for r in report['shards']) for i in range(768)]
    variance = [max(0., q/windows-(s/windows)**2) for q, s in zip(squares, sums)]
    std = sorted(math.sqrt(v) for v in variance)
    return dict(inputs=inputs, windows=windows,
        coordinate_sd_min=std[0], coordinate_sd_median=(std[383]+std[384])/2, coordinate_sd_max=std[-1],
        mean_distinct_pair_cosine=(math.fsum(x*x for x in sums)-math.fsum(squares))/(windows*(windows-1)))


def replay(name, bank, source_root, scanned, output):
    began = tick = time.perf_counter(); output.mkdir()
    rows, expected, bounds = cohort(source_root, scanned)
    phases = dict(cohort_readback=time.perf_counter()-tick); tick = time.perf_counter()
    lens = LearnedLens(by_name(name), 'cuda')
    try:
        need(digest(canonical(contract(lens))) == bank.config['instruments'][name]['lens_id'], 'NATIVE_FULL_RUNTIME')
        need(not lens.model.training and all(not p.requires_grad for p in lens.model.parameters()), 'NATIVE_FULL_FROZEN')
        phases['load_runtime'] = time.perf_counter()-tick; tick = time.perf_counter()
        values = np.empty_like(expected); all_traces = []
        timings = dict(tokenize=0., forward=0.)
        for lo in range(0, len(rows), POLICY['replay_tile']):
            batch = rows[lo:lo+POLICY['replay_tile']]
            native, offsets, traces, timing = forward(lens, [r['source']['text'] for r in batch])
            need(np.array_equal(offsets, bounds[lo:lo+len(batch)+1]-bounds[lo]), 'NATIVE_FULL_REPLAY_WINDOWS')
            need(np.array_equal(native, expected[bounds[lo]:bounds[lo+len(batch)]]), 'NATIVE_FULL_REPLAY_VALUES')
            need(traces == [r['trace'] for r in batch], 'NATIVE_FULL_REPLAY_TRACES')
            values[bounds[lo]:bounds[lo+len(batch)]] = native; all_traces.extend(traces)
            for key in timings: timings[key] += timing['phases_seconds'][key]
        # A different I/O shape and reversed order must leave canonical window bytes unchanged.
        picked = sorted(set(range(min(8, len(rows)))) | {len(rows)-1}, reverse=True)
        native, offsets, traces, _ = forward(lens, [rows[i]['source']['text'] for i in picked])
        for j, i in enumerate(picked):
            need(np.array_equal(native[offsets[j]:offsets[j+1]], expected[bounds[i]:bounds[i+1]])
                 and traces[j] == rows[i]['trace'], 'NATIVE_FULL_REVERSED_BATCH')
        phases['native_replay'] = time.perf_counter()-tick; tick = time.perf_counter()
        np.savez(output/'actual.npz', vectors=values, bounds=bounds)
        trace_pin = write_committed(output/'actual-traces.json', all_traces)
        with np.load(output/'actual.npz', allow_pickle=False) as reread:
            need(np.array_equal(reread['vectors'], expected) and np.array_equal(reread['bounds'], bounds),
                 'NATIVE_FULL_PERSISTED_REPLAY')
        need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
             'NATIVE_FULL_WEIGHTS_CHANGED')
        phases['persistence_and_frozen_check'] = time.perf_counter()-tick
        result = dict(instrument=name, inputs=len(rows), windows=len(expected), exact_arrays=True,
            exact_traces=True, reversed_batch_exact=True, diagnostics=independent_counts(scanned),
            replay_source=dict(path=str(source_root), pin=file_sha(source_root/'scan.json')),
            files={'actual.npz':file_sha(output/'actual.npz'), 'actual-traces.json':trace_pin},
            phases_seconds=phases, native_timing=timings, seconds=time.perf_counter()-began)
        need(sum(phases.values())/result['seconds'] >= .9, 'NATIVE_FULL_REPLAY_TIMING')
        return dict(pin=write_committed(output/'report.json', result), **result)
    finally:
        del lens; torch.cuda.empty_cache()


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    publication = pinned(a.publication/'report.json', a.pin)
    need(publication['stage'] == 'full' and publication['complete_scope'] and publication['full_corpus_complete']
         and not publication['errors'], 'NATIVE_FULL_REQUIRED')
    bank = NativeBank(Path(publication['bank']['path']), publication['bank']['configuration_pin'])
    source = ReuseInputs(Path(bank.config['source']['path']), bank.config['source']['pin'])
    binding = dict(publication_pin=a.pin, bank_pin=bank.pin, source_pin=source.pin, code=code(), policy=POLICY)
    previous = None
    if a.mode in ['full', 'cold']:
        need(a.previous is not None and a.previous_pin is not None, 'NATIVE_FULL_QUALIFICATION_REQUIRED')
        previous = pinned(a.previous/'report.json', a.previous_pin)
        need(previous['complete'] and previous['binding'] == binding and previous['mode'] == 'qualify',
             'NATIVE_FULL_QUALIFICATION_BINDING')
    if a.mode == 'full':
        need(a.cold is not None and a.cold_pin is not None, 'NATIVE_FULL_COLD_REQUIRED')
        proof = pinned(a.cold/'report.json', a.cold_pin)
        need(proof['complete'] and proof['binding'] == binding and proof['mode'] == 'cold'
             and proof['previous_pin'] == a.previous_pin, 'NATIVE_FULL_COLD_BINDING')
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    scans = {}; scan_roots = {}
    for name in bank.config['roster']:
        if a.mode == 'cold':
            root = a.previous/name/'stored'
            scans[name] = dict(pin=previous['scans'][name]['pin'],
                               **pinned(root/'scan.json', previous['scans'][name]['pin']))
        else:
            root = a.output/name/'stored'; root.parent.mkdir()
            scans[name] = scan(name, bank, source, publication['instruments'][name], a.mode, root)
        scan_roots[name] = root
        if a.mode == 'full':
            need(scans[name]['inputs'] == bank.config['inputs'] == publication['coverage'][name]['inputs']
                 and scans[name]['windows'] == publication['coverage'][name]['windows'], 'NATIVE_FULL_COVERAGE')
    phases['stored_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    negatives = controls(bank, source, a.output)
    first = bank.config['roster'][0]; bad = dict(scans[first], files={**scans[first]['files'], 'expected.npz':'0'*64})
    negatives['changed_replay_cohort'] = rejected(lambda: cohort(scan_roots[first], bad))
    wrong_count = dict(scans[first], windows=scans[first]['windows']+1)
    negatives['changed_window_count'] = rejected(lambda: independent_counts(wrong_count))
    phases['controls'] = time.perf_counter()-tick; tick = time.perf_counter(); results = {}
    resident = setup_gpu()
    try:
        (a.output/'gpu-before.txt').write_text(resident)
        for name in bank.config['roster']:
            root = a.output/name/'replay'; root.parent.mkdir(exist_ok=True)
            results[name] = replay(name, bank, scan_roots[name], scans[name], root)
            if a.mode == 'cold':
                prior = previous['replay'][name]
                need(results[name]['files'] == prior['files'] and results[name]['diagnostics'] == prior['diagnostics'],
                     'NATIVE_FULL_COLD_BYTES')
            print(json.dumps(dict(stage='native_replay', instrument=name, pin=results[name]['pin'],
                                  inputs=results[name]['inputs'], seconds=results[name]['seconds'])), flush=True)
    finally:
        release_lease()
    phases['gpu_replay'] = time.perf_counter()-tick; tick = time.perf_counter()
    source.fresh()
    need(file_sha(a.publication/'report.json') == a.pin and file_sha(bank.root/'configuration.json') == bank.pin,
         'NATIVE_FULL_FINAL_PUBLICATION')
    for name in bank.config['roster']:
        stamps = json.loads((scan_roots[name]/'stamps.json').read_bytes())
        need(all(list(stamp(Path(p))) == s for p, s in stamps.items()), 'NATIVE_FULL_FINAL_SHARD')
    phases['final_readback'] = time.perf_counter()-tick
    seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'NATIVE_FULL_PHASE_ATTRIBUTION')
    result = dict(schema='legal-native-full-cold-readback-v1', complete=True, mode=a.mode,
        binding=binding, previous_pin=a.previous_pin, scans={n:{k:v for k,v in r.items() if k != 'shards'} for n,r in scans.items()},
        replay=results, controls=negatives, phases_seconds=phases, seconds=seconds,
        cpu_seconds=time.process_time()-cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        full_corpus_readback=a.mode == 'full', model_updates=0, scientific_admission=False,
        acceptance='author-checked', number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json', result), seconds=seconds, complete=True)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['qualify', 'cold', 'full'], required=True)
    p.add_argument('--publication', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin')
    p.add_argument('--cold', type=Path); p.add_argument('--cold-pin')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'failure.json').exists():
            write_committed(args.output/'failure.json', dict(error=type(exc).__name__+':'+str(exc)))
        raise
