"""Sequential all-window readback and bounded replay cohorts for a full native bank."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from audit.legal_native_bank_readback import check_record, norms
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical, file_sha, need
from text.legal_native_publish import proof_cache
from text.legal_native_store import NativeBank, NativeShard, ranges

POLICY = dict(sample_per_shard=16, longest_per_shard=True, original_proof=True,
              vector_tile=8192, replay_tile=128, replay_bytes=128 << 20,
              qualification='first_last_and_all_original_proof_shards_per_lane',
              full_scope='every_committed_shard_every_input_every_window',
              sample_selection='uniform_source_indices_and_text_length_before_scores')


def close(saved):
    saved.table.close()
    saved.vectors._mmap.close()
    saved.offsets._mmap.close()


def refs_for(bank, publication, mode, proof_inputs):
    expected = [f'{lo:012d}-{hi:012d}' for lo, hi in ranges(bank.config['inputs'])]
    refs = publication['shards']
    need([r['path'] for r in refs] == expected, 'NATIVE_FULL_PUBLICATION_RANGES')
    selected = {0, len(refs)-1} | {r['index']//bank.config['shard_inputs'] for r in proof_inputs}
    return refs if mode == 'full' else [refs[i] for i in sorted(selected)]


def controls(bank, source, root):
    name = bank.config['roster'][0]
    saved = bank.shard(name, 0)
    src = source.input(1)
    record = saved.table.read_row_group(0).slice(1, 1).to_pylist()[0]
    count = int(saved.offsets[2]-saved.offsets[1])
    changed = deepcopy(record); changed['input_sha256'] = '0'*64
    missing = deepcopy(record); trace = json.loads(missing['provenance'])
    trace['windows'].pop(); missing['provenance'] = canonical(trace)
    partial = root/'uncommitted-real-shard'; partial.mkdir()
    shutil.copy2(saved.root/'intent.json', partial/'intent.json')
    corrupt = root/'corrupted-real-shard'; corrupt.mkdir()
    for n in ['COMMITTED', 'report.json', 'offsets.npy', 'traces.parquet', 'vectors.npy']:
        shutil.copy2(saved.root/n, corrupt/n)
    with (corrupt/'vectors.npy').open('r+b') as stream:
        stream.seek(-1, 2); byte = stream.read(1); stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1])); stream.flush(); os.fsync(stream.fileno())
    params = (bank.pin, saved.report['lens_id'], 0, saved.report['end'])
    result = dict(changed_source=rejected(lambda: check_record(changed, src, count)),
        missing_window=rejected(lambda: check_record(missing, src, count)),
        corrupt_vector=rejected(lambda: NativeShard(corrupt, *params)),
        uncommitted_shard=rejected(lambda: NativeShard(partial, *params)),
        wrong_runtime=rejected(lambda: NativeShard(saved.root, bank.pin, '0'*64, *params[2:])),
        full_bank_out_of_range=rejected(lambda: bank.shard(name, bank.config['inputs'])),
        wrong_input=rejected(lambda: saved.input(1, '0'*64)),
        wrong_config=rejected(lambda: NativeBank(bank.root, '0'*64)),
        implicit_admission=rejected(bank.training_authority))
    saved.fresh(); close(saved); bank.cached.clear()
    return result


def scan(name, bank, source, publication, mode, output):
    began = time.perf_counter(); output.mkdir()
    proof_root = Path(bank.config['runtime_proof']['path'])
    need(file_sha(proof_root/'report.json') == bank.config['runtime_proof']['pin'], 'NATIVE_FULL_PROOF_PIN')
    proof = json.loads((proof_root/'report.json').read_bytes())
    for n, item in proof['files'].items():
        need(file_sha(proof_root/n) == item['sha256'], 'NATIVE_FULL_PROOF_FILE')
    proof_inputs = json.loads((proof_root/'inputs.json').read_bytes())
    _, cache = proof_cache(proof_root, name, proof, proof_inputs)
    samples = []; blocks = []; bounds = [0]; shards = []; stamps = {}
    phases = dict(authentication_source=0., binding_numeric_readback=0., persistence=0.)
    copied = checked_proof = 0
    for ref in refs_for(bank, publication, mode, proof_inputs):
        tick = time.perf_counter(); lo, hi = map(int, ref['path'].split('-'))
        saved = NativeShard(bank.root/name/ref['path'], bank.pin,
                            bank.config['instruments'][name]['lens_id'], lo, hi)
        try:
            need(saved.pin == ref['pin'], 'NATIVE_FULL_SHARD_PIN')
            stamps.update({str(p): list(s) for p, s in saved.stamps.items()})
            g0, g1 = lo//source.report['group_rows'], (hi-1)//source.report['group_rows']
            rows = source.table.read_row_groups(list(range(g0, g1+1))).to_pylist()
            need(len(rows) == hi-lo and [r['index'] for r in rows] == list(range(lo, hi)),
                 'NATIVE_FULL_SOURCE_RANGE')
            chosen = set(np.linspace(lo, hi-1, POLICY['sample_per_shard'], dtype=np.int64).tolist())
            chosen.add(max(rows, key=lambda r: (len(r['text']), r['index']))['index'])
            chosen.update(uid for uid in cache if lo <= uid < hi)
            phases['authentication_source'] += time.perf_counter()-tick; tick = time.perf_counter()
            cursor = 0
            for batch in saved.table.iter_batches(batch_size=1024):
                for record in batch.to_pylist():
                    src = rows[cursor]; a, b = map(int, saved.offsets[cursor:cursor+2])
                    trace = check_record(record, src, b-a)
                    if src['index'] in chosen:
                        native = np.array(saved.vectors[a:b], copy=True)
                        copied += native.nbytes
                        need(copied <= POLICY['replay_bytes'], 'NATIVE_FULL_REPLAY_MEMORY')
                        need(len(source.occurrences(src['index'])) == src['occurrences'], 'NATIVE_FULL_OCCURRENCES')
                        samples.append(dict(source=src, trace=trace)); blocks.append(native); bounds.append(bounds[-1]+b-a)
                    if src['index'] in cache:
                        expected, old_trace, sha = cache[src['index']]
                        need(sha == src['text_sha256'] and trace == old_trace
                             and np.array_equal(saved.vectors[a:b], expected), 'NATIVE_FULL_PROOF_REUSE')
                        checked_proof += 1
                    cursor += 1
            need(cursor == hi-lo, 'NATIVE_FULL_TRACE_COUNT')
            sums = np.zeros(768, np.float64); squares = sums.copy()
            for offset in range(0, len(saved.vectors), POLICY['vector_tile']):
                values = np.asarray(saved.vectors[offset:offset+POLICY['vector_tile']]); norms(values)
                high = values.astype(np.float64); sums += high.sum(0); squares += np.square(high).sum(0)
            saved.fresh()
            shards.append(dict(path=ref['path'], pin=saved.pin, inputs=hi-lo, windows=len(saved.vectors),
                file_bytes=sum(f['bytes'] for f in saved.report['files'].values()),
                sum=sums.tolist(), square_sum=squares.tolist()))
            phases['binding_numeric_readback'] += time.perf_counter()-tick
            print(json.dumps(dict(stage='stored_readback', instrument=name, end=hi,
                                  shards=len(shards), seconds=time.perf_counter()-began)), flush=True)
        finally:
            close(saved)
    tick = time.perf_counter()
    need(checked_proof == len(cache), 'NATIVE_FULL_PROOF_COVERAGE')
    need(any(len(row['trace']['windows']) > 1 for row in samples), 'NATIVE_FULL_MULTIWIND_PROOF_REQUIRED')
    np.savez(output/'expected.npz', vectors=np.concatenate(blocks), bounds=np.asarray(bounds, dtype=np.int64))
    sample_pin = write_committed(output/'samples.json', samples)
    stamp_pin = write_committed(output/'stamps.json', stamps)
    phases['persistence'] += time.perf_counter()-tick
    result = dict(instrument=name, mode=mode, shards=shards, inputs=sum(r['inputs'] for r in shards),
        windows=sum(r['windows'] for r in shards), authenticated_file_bytes=sum(r['file_bytes'] for r in shards),
        replay_inputs=len(samples), replay_windows=bounds[-1], replay_vector_bytes=copied,
        reused_proof_inputs=checked_proof, all_source_bindings=True, all_windows_finite_unit=True,
        files={'samples.json':sample_pin, 'stamps.json':stamp_pin, 'expected.npz':file_sha(output/'expected.npz')},
        phases_seconds=phases, seconds=time.perf_counter()-began)
    need(sum(phases.values())/result['seconds'] >= .9, 'NATIVE_FULL_SCAN_TIMING')
    return dict(pin=write_committed(output/'scan.json', result), **result)
