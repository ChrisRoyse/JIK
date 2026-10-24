"""Physical, matched qualification of an immutable native-reading cache."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_training_statistics as training
from audit.legal_native_io_readback import DOCUMENTS, PRIOR, controls, read, usage
from audit.legal_coalesced_io_readback import requests, trace
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_native_training_stream import CertifiedNativeFrames
from model.legal_native_resident_stream import ResidentNativeFrames
from model.legal_native_reading_cache import ReadingNativeFrames
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(training.code_pins(), **{n:file_sha(root/n) for n in [
        'model/legal_native_training_stream.py', 'model/legal_native_resident_stream.py',
        'model/legal_native_reading_cache.py', 'model/native_population_training.py',
        'audit/legal_native_io_readback.py', 'audit/legal_coalesced_io_readback.py',
        'audit/legal_reading_cache.py', 'audit/legal_reading_cache_readback.py']})


def state(reader):
    return dict(reader=reader.stats.copy(), handles=reader.io_stats.copy(),
                cache=reader.cache_stats.copy() if hasattr(reader, 'cache_stats') else None,
                accounted_bytes=reader.accounted_bytes() if hasattr(reader, 'accounted_bytes') else None)


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pins = code()
    for name, pin in pins.items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'READING_PROOF_SOURCE_COPY')
    _, _, data, _, _, _, _, _ = training.authenticate(a.root)
    prior = training.pinned(a.root/PRIOR[0]/'report.json', PRIOR[1])
    args = (data, a.root/training.DEFAULTS['audit'][0], training.DEFAULTS['audit'][1])
    reference = CertifiedNativeFrames(*args); plan = []
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    for number in DOCUMENTS:
        start, stop = map(int, data.documents[number]); uids = data.mapping[start:stop].tolist()
        frames = read(reference, uids)
        path = a.root/PRIOR[0]/f'labels-{number}.pt'
        need(file_sha(path) == prior['files'][path.name], 'READING_PROOF_PRIOR_PIN')
        labels = torch.load(path, map_location='cpu', weights_only=True)
        need(fingerprint(frames) == labels['native_pin'] and labels['uids'] == uids,
             'READING_PROOF_PRIOR_NATIVE')
        torch.save(dict(frames=frames, **labels), a.output/f'document-{number}.pt')
        plan.append(dict(document=number, uids=uids, native=labels['native_pin'], atoms=len(uids)))
    reference.close()
    phases['source_documents_and_shard_authentication'] = time.perf_counter()-tick
    tick = time.perf_counter()
    readers = dict(pread=ResidentNativeFrames(*args), cache=ReadingNativeFrames(*args, cache_bytes=256 << 20))
    measurements = []
    for repeat, order in enumerate([['pread','cache'], ['cache','pread'], ['pread','cache'], ['cache','pread']]):
        for mode in order:
            reader = readers[mode]
            for row in plan:
                before = usage(); s0 = state(reader); c0 = time.process_time(); t0 = time.perf_counter()
                frames = read(reader, row['uids']); seconds = time.perf_counter()-t0
                cpu_seconds = time.process_time()-c0; after = usage(); s1 = state(reader)
                need(fingerprint(frames) == row['native'], 'READING_PROOF_NATIVE_EXACT')
                measurements.append(dict(repeat=repeat, mode=mode, document=row['document'],
                    atoms=row['atoms'], seconds=seconds, cpu_seconds=cpu_seconds,
                    before=before, after=after, state_before=s0, state_after=s1, native=row['native']))
            print(json.dumps(dict(repeat=repeat, mode=mode, seconds=sum(r['seconds'] for r in
                measurements if r['repeat']==repeat and r['mode']==mode))), flush=True)
    phases['matched_timings'] = time.perf_counter()-tick; tick = time.perf_counter()
    measured_stats = {name:state(reader) for name,reader in readers.items()}
    # The producer's outputs are saved before a different implementation reads
    # every original native range independently in a separate process.
    uids = [u for row in plan for u in row['uids']]
    candidate = ReadingNativeFrames(*args, cache_bytes=256 << 20)
    cold = trace(candidate, uids, a.output/'fill-reads.jsonl')
    warm = trace(candidate, uids, a.output/'reuse-reads.jsonl')
    need(fingerprint(cold) == fingerprint(warm), 'READING_PROOF_REPEAT_EXACT')
    need((a.output/'reuse-reads.jsonl').stat().st_size == 0, 'READING_PROOF_REOPENED_NATIVE')
    torch.save(dict(uids=uids, frames=warm), a.output/'cached-frames.pt')
    write_committed(a.output/'requests.json', requests(data, uids))
    trace_state = state(candidate); candidate.close()
    need(candidate.owned_bytes == 0 and not candidate.readings and not candidate.maps,
         'READING_PROOF_TRACE_TEARDOWN')
    phases['persisted_source_and_cache_trace'] = time.perf_counter()-tick; tick = time.perf_counter()
    cached = readers['cache']; seed = plan[0]['uids'][:2]
    expected = fingerprint(cached.frames(seed))
    mutated = cached.frames(seed); mutated[0][data.roster[0]].zero_()
    need(fingerprint(cached.frames(seed)) == expected, 'READING_PROOF_OUTPUT_ALIAS')
    duplicated = cached.frames([seed[0],seed[0]])
    duplicate_pin = fingerprint(duplicated[1]); duplicated[0][data.roster[0]].zero_()
    need(fingerprint(duplicated[1]) == duplicate_pin, 'READING_PROOF_DUPLICATE_ALIAS')
    negatives = controls(cached, seed, a.output)
    need(not cached.readings and cached.owned_bytes == 0, 'READING_PROOF_REFUSED_CLOSE_LEAK')
    negatives['bad_capacity'] = rejected(lambda:ReadingNativeFrames(*args, cache_bytes=0))
    negatives['bool_capacity'] = rejected(lambda:ReadingNativeFrames(*args, cache_bytes=True))
    small = ReadingNativeFrames(*args, cache_bytes=1 << 20)
    for row in plan:
        need(fingerprint(read(small, row['uids'])) == row['native'], 'READING_PROOF_EVICTION_EXACT')
    need(small.cache_stats['evictions'] > 0 and small.accounted_bytes() <= small.cache_limit,
         'READING_PROOF_EVICTION_BOUND')
    eviction = state(small); small.close(); small.close(); readers['pread'].close()
    need(not small.readings and small.owned_bytes == 0 and not small.maps, 'READING_PROOF_CLOSE_LEAK')
    need(not torch.cuda.is_initialized(), 'READING_PROOF_UNEXPECTED_GPU')
    phases['controls_and_teardown'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'READING_PROOF_PHASE_ATTRIBUTION')
    result = dict(complete=True, binding=dict(code=pins, source=data.binding, prior=PRIOR[1],
        documents=DOCUMENTS, cache_bytes=256 << 20, native_checks_unchanged=True),
        roster=data.roster, plan=plan, measurements=measurements, measured_stats=measured_stats,
        trace_state=trace_state, eviction=eviction, controls=negatives, files=files,
        all_native_bytes_exact=True, returned_arrays_independently_owned=True,
        all_handles_and_cache_closed=True, phases_seconds=phases, seconds=seconds,
        cpu_seconds=time.process_time()-cpu0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_updates=0, scientific_admission=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
