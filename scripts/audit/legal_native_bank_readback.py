"""Cold production-bank audit: every stored binding, indexed windows and native replay."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import release_lease
import text_windows
from text.canonical_windows import contract, forward
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_measure import setup_gpu
from text.legal_native_store import NativeBank, NativeShard
from text.legal_reuse_input import ReuseInputs


def check_record(record, source, vector_count):
    need(record['index'] == source['index'] and record['input_sha256'] == source['text_sha256'], 'NATIVE_AUDIT_ADDRESS')
    trace = json.loads(record['provenance'])
    text_windows.validate(trace, 'dense', vector_count, text=source['text'])
    return trace


def norms(values):
    need(np.isfinite(values).all() and np.max(np.abs(np.linalg.norm(values, axis=1)-1)) < 2e-6,
         'NATIVE_AUDIT_FINITE_UNIT')


def lane(name, bank, source, publication, output):
    start = tick = time.perf_counter(); records = []; samples = []; coverage = 0; windows = 0; stamps = {}
    running_sum = np.zeros(768, dtype=np.float64); squares = running_sum.copy(); squared_norm_sum = 0.
    largest = None; example = None; phases = dict(source_and_shard_authenticate=0., full_stored_readback=0.)
    for ref in publication['shards']:
        lo, hi = map(int, ref['path'].split('-')); saved = bank.shard(name, lo)
        need(saved.pin == ref['pin'] and lo == coverage, 'NATIVE_AUDIT_CONTIGUOUS')
        for path, value in saved.stamps.items(): stamps[str(path)] = list(value)
        g0, g1 = lo//source.report['group_rows'], (hi-1)//source.report['group_rows']
        source_rows = source.table.read_row_groups(list(range(g0, g1+1))).to_pylist()
        need(len(source_rows) == hi-lo, 'NATIVE_AUDIT_SOURCE_RANGE')
        phases['source_and_shard_authenticate'] += time.perf_counter()-tick; tick = time.perf_counter()
        cursor = 0; chosen = set(np.linspace(lo, hi-1, 16, dtype=np.int64).tolist())
        local_largest = (-1, -1)
        for batch in saved.table.iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                src = source_rows[cursor]; a, b = map(int, saved.offsets[cursor:cursor+2])
                trace = check_record(record, src, b-a)
                local_largest = max(local_largest, (len(src['text']), src['index']))
                if src['index'] in chosen: samples.append(dict(**src, instrument=name, windows=b-a))
                if example is None and cursor == 1: example = (deepcopy(record), deepcopy(src), b-a, saved)
                cursor += 1
        need(cursor == hi-lo, 'NATIVE_AUDIT_TRACE_COVERAGE')
        chosen_extra = local_largest[1]
        if chosen_extra not in chosen:
            src = source_rows[chosen_extra-lo]; a, b = map(int, saved.offsets[chosen_extra-lo:chosen_extra-lo+2])
            samples.append(dict(**src, instrument=name, windows=b-a))
        for offset in range(0, len(saved.vectors), 8192):
            values = np.asarray(saved.vectors[offset:offset+8192]); norms(values); high = values.astype(np.float64)
            running_sum += high.sum(0); squares += np.square(high).sum(0); squared_norm_sum += float(np.square(high).sum())
        records.append(dict(path=ref['path'], pin=saved.pin, inputs=hi-lo, windows=len(saved.vectors)))
        coverage = hi; windows += len(saved.vectors); saved.fresh()
        phases['full_stored_readback'] += time.perf_counter()-tick; tick = time.perf_counter()
    lens = LearnedLens(by_name(name), 'cuda')
    need(digest(canonical(contract(lens))) == bank.config['instruments'][name]['lens_id'], 'NATIVE_AUDIT_RUNTIME')
    phases['cold_load'] = time.perf_counter()-tick; tick = time.perf_counter()
    texts = [r['text'] for r in samples]; v, bounds, traces, timing = forward(lens, texts)
    for j, row in enumerate(samples):
        actual, trace = bank.input(name, row['index'], row['text_sha256'])
        need(np.array_equal(actual, v[bounds[j]:bounds[j+1]]) and trace == traces[j], 'NATIVE_AUDIT_COLD_REPLAY')
        # Follow the real reverse occurrence index; source contexts stay distinct.
        occurrences = source.occurrences(row['index'])
        need(len(occurrences) == row['occurrences'], 'NATIVE_AUDIT_CONTEXT_MULTIPLICITY')
    phases['cold_native_and_indexed_consumer'] = time.perf_counter()-tick; tick = time.perf_counter()
    proof_root = Path(bank.config['runtime_proof']['path']); proof_inputs = json.loads((proof_root/'inputs.json').read_bytes())
    proof_traces = json.loads((proof_root/name/'traces.json').read_bytes()); cached = 0
    with np.load(proof_root/name/'reference.npz', allow_pickle=False) as reference:
        for j, row in enumerate(proof_inputs):
            if row['index'] >= coverage: continue
            native, trace = bank.input(name, row['index'], row['text_sha256'])
            need(np.array_equal(native, reference['vectors'][reference['bounds'][j]:reference['bounds'][j+1]])
                 and trace == proof_traces[j], 'NATIVE_AUDIT_REUSED_BYTES')
            cached += 1
    need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
         'NATIVE_AUDIT_FROZEN')
    variance = np.maximum(0, squares/windows-np.square(running_sum/windows)); std = np.sqrt(variance)
    diagnostics = dict(native_windows=windows, coordinate_sd_min=float(std.min()), coordinate_sd_median=float(np.median(std)),
        coordinate_sd_max=float(std.max()), mean_distinct_pair_cosine=float((running_sum@running_sum-squared_norm_sum)/(windows*(windows-1))))
    phases['proof_cache_and_frozen_readback'] = time.perf_counter()-tick
    result = dict(instrument=name, inputs=coverage, windows=windows, shards=records, cold_inputs=len(samples), reused_proof_inputs=cached,
        all_vectors_finite_unit=True, all_source_bindings_checked=True, cold_arrays_exact=True, diagnostics=diagnostics,
        phases_seconds=phases, seconds=time.perf_counter()-start, replay_timing=timing)
    need(sum(phases.values())/result['seconds'] >= .9, 'NATIVE_AUDIT_TIMING')
    pin = write_committed(output/(name+'.json'), result)
    del lens; torch.cuda.empty_cache()
    print(json.dumps(dict(name=name, pin=pin, inputs=coverage, windows=windows, seconds=result['seconds'])), flush=True)
    return dict(pin=pin, inputs=coverage, windows=windows), stamps, example


def controls(bank, example, output):
    record, src, count, saved = example; altered = deepcopy(record); altered['input_sha256'] = '0'*64
    missing = deepcopy(record); trace = json.loads(missing['provenance']); trace['windows'].pop()
    missing['provenance'] = canonical(trace)
    partial = output/'uncommitted-real-shard'; partial.mkdir()
    shutil.copy2(saved.root/'intent.json', partial/'intent.json')
    corrupt = output/'corrupted-real-shard'; corrupt.mkdir()
    for name in ('COMMITTED', 'report.json', 'offsets.npy', 'traces.parquet'):
        shutil.copy2(saved.root/name, corrupt/name)
    shutil.copy2(saved.root/'vectors.npy', corrupt/'vectors.npy')
    with (corrupt/'vectors.npy').open('r+b') as f:
        f.seek(-1, 2); old = f.read(1); f.seek(-1, 2); f.write(bytes([old[0] ^ 1])); f.flush(); os.fsync(f.fileno())
    params = (bank.pin, saved.report['lens_id'], saved.report['start'], saved.report['end'])
    return dict(changed_source_binding=rejected(lambda: check_record(altered, src, count)),
        missing_window=rejected(lambda: check_record(missing, src, count)),
        corrupted_saved_vector=rejected(lambda: NativeShard(corrupt, *params)),
        uncommitted_shard=rejected(lambda: NativeShard(partial, *params)),
        wrong_runtime=rejected(lambda: NativeShard(saved.root, bank.pin, '0'*64, params[2], params[3])),
        missing_future_shard=rejected(lambda: bank.shard(bank.config['roster'][0], saved.report['end'])),
        wrong_consumer_input=rejected(lambda: saved.input(src['index'], '0'*64)),
        changed_config_pin=rejected(lambda: NativeBank(bank.root, '0'*64)),
        implicit_training_admission=rejected(bank.training_authority))


def run(a):
    start = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    need(file_sha(a.publication/'report.json') == a.pin, 'NATIVE_AUDIT_PUBLICATION_PIN')
    report = json.loads((a.publication/'report.json').read_bytes())
    need(report['complete_scope'] and not report['errors'] and report['stage'] == 'prefix', 'NATIVE_AUDIT_PROOF_SCOPE')
    bank = NativeBank(Path(report['bank']['path']), report['bank']['configuration_pin'])
    source = ReuseInputs(Path(bank.config['source']['path']), bank.config['source']['pin'])
    results = {}; stamps = {}; example = None; errors = []
    resident = setup_gpu()
    try:
        (a.output/'gpu-before.txt').write_text(resident)
        for name in bank.config['roster']:
            try:
                results[name], lane_stamps, lane_example = lane(name, bank, source, report['instruments'][name], a.output)
                stamps.update(lane_stamps)
                if example is None: example = lane_example
            except Exception as exc:
                error = dict(name=name, error=type(exc).__name__+':'+str(exc)); errors.append(error)
                write_committed(a.output/(name+'-failure.json'), error); print(json.dumps(error), flush=True)
                torch.cuda.empty_cache()
        negative = controls(bank, example, a.output) if not errors else {}
        source.fresh()
        write_committed(a.output/'bank-file-stamps.json', stamps)
        result = dict(schema='legal-native-prefix-cold-readback-v1', complete=not errors, errors=errors,
            publication=dict(path=str(a.publication), pin=a.pin), bank=report['bank'], instruments=results,
            controls=negative, seconds=time.perf_counter()-start, code_sha256=file_sha(Path(__file__)),
            model_updates=0, full_corpus_measured=False, scientific_admission=False,
            acceptance='author-checked', number_status='measured')
        pin = write_committed(a.output/'report.json', result)
        print(json.dumps(dict(pin=pin, complete=not errors, controls=negative)), flush=True)
        need(not errors, 'NATIVE_AUDIT_INCOMPLETE')
    finally: release_lease()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
