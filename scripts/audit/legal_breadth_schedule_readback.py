"""Publish and cold-recount a complete TRAIN breadth/repetition comparison schedule."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_cached_target_readback as qualification
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model import legal_breadth_schedule as schedule
from model.legal_cached_targets import LegalCachedTargets
from text.constellation_corpus import file_sha, need


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(qualification.code(), **{n:file_sha(root/n) for n in
        ['model/legal_breadth_schedule.py', 'audit/legal_breadth_schedule_readback.py']})


def prepare(a):
    producer = qualification.producer
    data, continuation, reader, labels, _, authority = producer.prepare(a.root)
    proof = producer.qualification.training.pinned(a.qualification/'report.json', a.qualification_pin)
    cold = producer.qualification.training.pinned(a.cold/'report.json', a.cold_pin)
    need(proof['complete'] and cold['complete'] and proof['mode'] == 'qualify' and cold['mode'] == 'cold'
         and cold['previous_pin'] == a.qualification_pin and proof['binding'] == cold['binding']
         and proof['proof_state'] == cold['proof_state'] and proof['binding']['code'] == qualification.code(),
         'BREADTH_CONSUMER_QUALIFICATION')
    for directory, report in [(a.qualification, proof), (a.cold, cold)]:
        for name, pin in report['files'].items():
            need(file_sha(directory/name) == pin, 'BREADTH_CONSUMER_PROOF_BYTES')
    cache = LegalCachedTargets(a.cache, a.cache_pin, a.cache_cold, a.cache_cold_pin, data, authority)
    need(cache.binding == proof['binding']['cache'], 'BREADTH_CACHE_BINDING')
    return data, continuation, reader, labels, cache, schedule.Metadata(data)


def negatives(plan, data, meta, counts):
    result = {}; row = plan[0].copy()
    held = int(np.flatnonzero(data.doc_sides != 0)[0]); row['repeated'] = held
    result['held_exemplar'] = rejected(lambda: schedule.validate_row(row, data, meta, counts))
    document = int(plan[0]['full']); start, stop = map(int, data.documents[document])
    result['changed_boundary'] = rejected(lambda: schedule.document(data, meta, counts, document, (start, stop-1)))
    row = plan[0].copy()
    # Real wrong source/shape document, selected without consulting region labels.
    other = next(r for r in plan[1:] if r['layout'].tobytes() != row['layout'].tobytes())
    row['repeated'] = int(other['full'])
    result['wrong_shape_or_source'] = rejected(lambda: schedule.validate_row(row, data, meta, counts))
    row = plan[0].copy(); row['layout'] = bytes(32)
    result['changed_layout_pin'] = rejected(lambda: schedule.validate_row(row, data, meta, counts))
    return result


def run(a):
    began = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(), 'BREADTH_UNEXPECTED_GPU')
    for name, pin in code().items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'BREADTH_SOURCE_COPY')
    data, continuation, reader, labels, cache, meta = prepare(a)
    binding = dict(code=code(), policy=schedule.POLICY, cache=cache.binding, source=data.binding,
                   metadata=meta.binding, qualification=a.qualification_pin, cold=a.cold_pin,
                   numpy_version=str(np.__version__))
    previous = qualification.producer.qualification.training.pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    if previous: need(previous['complete'] and previous['binding'] == binding, 'BREADTH_COLD_BINDING')
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    plan = schedule.build(data, meta, cache.maps[1])
    np.save(a.output/'schedule.npy', plan, allow_pickle=False)
    saved = np.load(a.output/'schedule.npy', mmap_mode='r', allow_pickle=False)
    need(np.array_equal(saved, plan), 'BREADTH_SCHEDULE_READBACK')
    phases['construct_and_publish'] = time.perf_counter()-tick; tick = time.perf_counter()
    summary = schedule.recount(saved, data, meta, cache.maps[1], continuation.presence)
    if previous:
        need(file_sha(a.previous/'schedule.npy') == previous['files']['schedule.npy'], 'BREADTH_COLD_SCHEDULE_PIN')
        old = np.load(a.previous/'schedule.npy', mmap_mode='r', allow_pickle=False)
        need(np.array_equal(old, saved) and summary == previous['summary'], 'BREADTH_COLD_RECOUNT')
        old._mmap.close()
    phases['independent_complete_recount'] = time.perf_counter()-tick; tick = time.perf_counter()
    controls = negatives(plan, data, meta, cache.maps[1])
    for key, value in [('missing_full_document', plan[:-1]), ('repeated_full_document', np.r_[plan[:1], plan[:-1]])]:
        controls[key] = rejected(lambda: schedule.recount(value, data, meta, cache.maps[1], continuation.presence))
    if previous: need(controls == previous['controls'], 'BREADTH_COLD_CONTROLS')
    cache.close(); reader.close(); continuation.fresh(); meta.fresh(); saved._mmap.close()
    need(not torch.cuda.is_initialized(), 'BREADTH_UNEXPECTED_GPU')
    phases['negative_controls_and_final_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
    files = {str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
    need(sum(phases.values())/seconds >= .9, 'BREADTH_PHASE_ATTRIBUTION')
    result = dict(complete=True, mode='cold' if previous else 'publish', binding=binding,
                  previous_pin=a.previous_pin, summary=summary, controls=controls, files=files,
                  phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  optimizer_updates=0, scientific_admission=False, acceptance='author-checked', number_status='measured')
    print(json.dumps(dict(pin=qualification.producer.seal(a.output, result), summary=summary)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output', type=Path, required=True)
    for name in ['qualification', 'cold', 'cache', 'cache-cold']:
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    try: run(a)
    except BaseException as error:
        if a.output.exists(): write_committed(a.output/'failure.json', dict(error=repr(error), code=code()))
        raise
