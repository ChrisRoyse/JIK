"""Resumable complete legal native measurement, with a retained prefix proof mode."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from claims.process_conditions import write_committed
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import take_lease, release_lease
from text.canonical_windows import contract
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_publish import proof_cache, publish
from text.legal_native_store import SCHEMA, SHARD_INPUTS, IO_INPUTS, NativeShard, durable_directory, ranges, shard_name
from text.legal_reuse_input import ReuseInputs


def authenticated(path, pin):
    raw = path.read_bytes(); need(digest(raw) == pin, 'NATIVE_METADATA_PIN')
    return json.loads(raw)


def setup_gpu():
    need(torch.cuda.is_available(), 'NATIVE_GPU_REQUIRED')
    resident = subprocess.check_output(['nvidia-smi'], text=True)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    need(not active.strip(), 'NATIVE_GPU_BUSY')
    take_lease('29', 'legal-complete-native-measurement')
    torch.set_num_threads(32); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return resident


def configuration(source, proof_root, proof_pin, cold_root, cold_pin):
    proof = authenticated(proof_root/'report.json', proof_pin)
    cold = authenticated(cold_root/'report.json', cold_pin)
    need(proof['complete'] and not proof['errors'] and cold['complete'] and not cold['errors']
         and cold['source']['pin'] == proof_pin and proof['source']['pin'] == source.pin, 'NATIVE_RUNTIME_PROOF')
    for name in proof['roster']:
        need(cold['instruments'][name]['lens_id'] == proof['instruments'][name]['lens_id'], 'NATIVE_COLD_IDENTITY')
    for name, item in proof['files'].items():
        need((proof_root/name).stat().st_size == item['bytes'] and file_sha(proof_root/name) == item['sha256'],
             'NATIVE_PROOF_INPUT_FILE')
    inputs = json.loads((proof_root/'inputs.json').read_bytes())
    for row in inputs: need(row == source.input(row['index']), 'NATIVE_REUSED_SOURCE')
    result = dict(schema=SCHEMA, source=dict(path=str(source.root), pin=source.pin), inputs=source.report['counts']['unique_inputs'],
        runtime_proof=dict(path=str(proof_root), pin=proof_pin), cold_readback=dict(path=str(cold_root), pin=cold_pin),
        instruments=proof['instruments'], roster=proof['roster'], shard_inputs=SHARD_INPUTS, io_inputs=IO_INPUTS,
        native_window_reduction='none', scientific_admission=False, training_ready=False)
    return result, proof, inputs


def existing(root, config_pin, lens_id, requested):
    results = {}; pending = sorted(root.glob('*.pending'))
    need(not pending, 'NATIVE_SHARD_RECOVERY_REQUIRED:'+','.join(map(str, pending)))
    for start, end in requested:
        dest = root/shard_name(start, end)
        if dest.exists():
            saved = NativeShard(dest, config_pin, lens_id, start, end)
            results[start] = dict(path=dest.name, pin=saved.pin, inputs=end-start, windows=saved.report['windows'],
                measured_inputs=0, reused_inputs=end-start, existing_unchanged=True)
    return results


def measure_lane(name, source, output, config_pin, proof_root, proof, inputs, requested, already):
    began = tick = time.perf_counter(); lens = LearnedLens(by_name(name), 'cuda')
    need(not lens.model.training and all(not p.requires_grad for p in lens.model.parameters()), 'NATIVE_LENS_FROZEN')
    identity = contract(lens); lens_id = digest(canonical(identity))
    need(lens_id == proof['instruments'][name]['lens_id'], 'NATIVE_RUNTIME_CHANGED')
    old_report, cache = proof_cache(proof_root, name, proof, inputs)
    need(old_report['contract'] == identity, 'NATIVE_REUSE_CONTRACT')
    load_seconds = time.perf_counter()-tick; results = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for start, end in requested:
            if start in already: result = already[start]
            else:
                result = publish(lens, source, output/name, config_pin, lens_id, start, end, cache)
                print(json.dumps(dict(instrument=name, **result)), flush=True)
            results.append(result)
        tick = time.perf_counter()
        need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
             'NATIVE_FROZEN_DRIFT')
        return dict(shards=results, load_seconds=load_seconds, final_frozen_check_seconds=time.perf_counter()-tick,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.perf_counter()-began, model_loaded=True, frozen_unchanged=True)
    finally:
        del lens; torch.cuda.empty_cache()


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time()
    a.receipt.mkdir(parents=True, exist_ok=False)
    source = ReuseInputs(a.source, a.source_pin)
    config, proof, inputs = configuration(source, a.proof, a.proof_pin, a.cold, a.cold_pin)
    blob = canonical(config); config_pin = digest(blob)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'writer.lock').open('a+b')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: lock.close(); raise ValueError('MNLM_NATIVE_BANK_WRITER_BUSY')
    lease = False; results = {}; errors = []
    try:
        path = a.output/'configuration.json'
        if path.exists(): need(path.read_bytes() == blob, 'NATIVE_BANK_CONFIGURATION_CHANGED')
        else: write_committed(path, config); durable_directory(a.output)
        requested = ranges(config['inputs'])
        if a.stage == 'prefix': requested = requested[:1]
        phases = dict(authenticate_and_configure=time.perf_counter()-tick, existing_authentication=0., lane_measurement=0.)
        for name in config['roster']:
            destination = a.output/name; destination.mkdir(exist_ok=True); tick = time.perf_counter()
            try:
                already = existing(destination, config_pin, config['instruments'][name]['lens_id'], requested)
                phases['existing_authentication'] += time.perf_counter()-tick; tick = time.perf_counter()
                if len(already) == len(requested):
                    result = dict(shards=[already[start] for start, _ in requested], model_loaded=False, seconds=0.)
                else:
                    if not lease:
                        resident = setup_gpu(); lease = True; (a.receipt/'gpu-before.txt').write_text(resident)
                    result = measure_lane(name, source, a.output, config_pin, a.proof, proof, inputs, requested, already)
                results[name] = result
                phases['lane_measurement'] += time.perf_counter()-tick
            except Exception as exc:
                error = dict(instrument=name, error=type(exc).__name__+':'+str(exc), traceback=traceback.format_exc())
                errors.append(error); write_committed(a.receipt/(name+'-failure.json'), error)
                print(json.dumps(error), flush=True); torch.cuda.empty_cache()
        source.fresh()
        complete = not errors and all(sum(s['inputs'] for s in r['shards']) == config['inputs'] for r in results.values())
        coverage = {name: dict(inputs=sum(s['inputs'] for s in r['shards']), windows=sum(s['windows'] for s in r['shards']),
            newly_measured=sum(s['measured_inputs'] for s in r['shards']), reused=sum(s['reused_inputs'] for s in r['shards']))
                    for name, r in results.items()}
        report = dict(schema=SCHEMA, stage=a.stage, complete_scope=not errors, full_corpus_complete=complete, errors=errors,
            bank=dict(path=str(a.output), configuration_pin=config_pin), instruments=results, coverage=coverage,
            seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu, phases_seconds=phases,
            code_sha256=file_sha(Path(__file__)), publisher_sha256=file_sha(Path(__file__).with_name('legal_native_publish.py')),
            model_updates=0, scientific_admission=False, training_ready=False, acceptance='author-checked', number_status='measured')
        if not errors: need(sum(phases.values())/report['seconds'] >= .9, 'NATIVE_MEASUREMENT_TIMING')
        pin = write_committed(a.receipt/'report.json', report)
        if complete:
            complete_record = dict(schema=SCHEMA, configuration_pin=config_pin, full_corpus_complete=True,
                shards={name: [dict(path=s['path'], pin=s['pin']) for s in r['shards']] for name, r in results.items()})
            if (a.output/'bank.json').exists(): need(json.loads((a.output/'bank.json').read_bytes()) == complete_record, 'NATIVE_BANK_REPUBLICATION')
            else: write_committed(a.output/'bank.json', complete_record); durable_directory(a.output)
        print(json.dumps(dict(pin=pin, coverage=coverage, full_corpus_complete=complete, seconds=report['seconds'])), flush=True)
        need(not errors, 'NATIVE_MEASUREMENT_INCOMPLETE')
    finally:
        if lease: release_lease()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'proof', 'cold', 'output', 'receipt'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('source-pin', 'proof-pin', 'cold-pin'): p.add_argument('--'+name, required=True)
    p.add_argument('--stage', choices=('prefix', 'full'), required=True)
    run(p.parse_args())
