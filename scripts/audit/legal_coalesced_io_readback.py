"""Qualify coalesced reads on persisted legal documents and real native ranges."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch

from audit import legal_training_statistics as training
from audit.legal_native_io_readback import DOCUMENTS, PRIOR, controls, read, usage
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_native_training_stream import CertifiedNativeFrames
from model.legal_native_resident_stream import ResidentNativeFrames, NativeFileView
from model.legal_native_coalesced_stream import CoalescedNativeFrames
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(training.code_pins(), **{n:file_sha(root/n) for n in [
        'model/legal_native_training_stream.py', 'model/legal_native_resident_stream.py',
        'model/legal_native_coalesced_stream.py', 'model/native_population_training.py',
        'audit/legal_native_io_readback.py', 'audit/legal_coalesced_io_readback.py',
        'audit/legal_coalesced_io_recount.py', 'text/legal_native_batch.py']})


@contextmanager
def traced_reads(path, cursor):
    """Observe real preads outside timed regions; never replace their bytes."""
    original = NativeFileView.__getitem__
    with path.open('x') as out:
        def observed(view, selected):
            result = original(view, selected)
            record = dict(batch=cursor[0], path=os.readlink(f'/proc/self/fd/{view.fd}'),
                          start=selected.start, stop=selected.stop, dimensions=view.shape[1],
                          byte_offset=view.offset+selected.start*view.shape[1]*4,
                          bytes=result.nbytes)
            out.write(json.dumps(record, sort_keys=True)+'\n')
            return result
        NativeFileView.__getitem__ = observed
        try: yield
        finally: NativeFileView.__getitem__ = original


def trace(reader, uids, path):
    cursor = [0]; frames = []
    with traced_reads(path, cursor):
        for batch, lo in enumerate(range(0, len(uids), 128)):
            cursor[0] = batch; frames.extend(reader.frames(uids[lo:lo+128]))
    return tuple(frames)


def requests(data, uids):
    result = []
    for batch, lo in enumerate(range(0, len(uids), 128)):
        for position, uid in enumerate(uids[lo:lo+128], lo):
            start = uid//16384*16384
            for name in data.roster:
                d = data.native.descriptors[name, start]
                a, b = map(int, d['offsets'][uid-start:uid-start+2])
                result.append(dict(batch=batch, position=position, uid=uid, name=name,
                                   path=str(d['root']/'vectors.npy'), start=a, stop=b,
                                   dimensions=d['report']['dimensions'], descriptor_pin=d['pin']))
    return result


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pins = code()
    for name, pin in pins.items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'COALESCED_SOURCE_COPY')
    _, _, data, _, _, _, _, _ = training.authenticate(a.root)
    prior = training.pinned(a.root/PRIOR[0]/'report.json', PRIOR[1])
    args = (data, a.root/training.DEFAULTS['audit'][0], training.DEFAULTS['audit'][1])
    reference = CertifiedNativeFrames(*args); plan = []
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    for number in DOCUMENTS:
        start, stop = map(int, data.documents[number]); uids = data.mapping[start:stop].tolist()
        frames = read(reference, uids)
        label = a.root/PRIOR[0]/f'labels-{number}.pt'
        need(file_sha(label) == prior['files'][label.name], 'COALESCED_PRIOR_PHYSICAL')
        labels = torch.load(label, map_location='cpu', weights_only=True)
        need(fingerprint(frames) == labels['native_pin'] and labels['uids'] == uids, 'COALESCED_PRIOR_NATIVE')
        torch.save(dict(frames=frames, **labels), a.output/f'document-{number}.pt')
        plan.append(dict(document=number, uids=uids, native=labels['native_pin'], atoms=len(uids)))
    reference.close()
    phases['source_documents'] = time.perf_counter()-tick; tick = time.perf_counter()
    readers = dict(pread=ResidentNativeFrames(*args), coalesced=CoalescedNativeFrames(*args))
    measurements = []
    for repeat, order in enumerate([['pread','coalesced'], ['coalesced','pread'], ['pread','coalesced']]):
        for mode in order:
            reader = readers[mode]
            for row in plan:
                before = usage(); c0 = time.process_time(); start = time.perf_counter()
                frames = read(reader, row['uids']); seconds = time.perf_counter()-start
                cpu_seconds = time.process_time()-c0; after = usage()
                need(fingerprint(frames) == row['native'], 'COALESCED_CANDIDATE_NATIVE')
                measurements.append(dict(repeat=repeat, mode=mode, document=row['document'],
                    atoms=row['atoms'], seconds=seconds, cpu_seconds=cpu_seconds,
                    before=before, after=after, native=row['native']))
    phases['matched_timings'] = time.perf_counter()-tick; tick = time.perf_counter()
    measured_stats = {name:dict(reader=reader.stats.copy(), handles=reader.io_stats.copy(),
                       ranges=reader.range_stats.copy() if hasattr(reader,'range_stats') else None)
                      for name,reader in readers.items()}
    # Real requests include cross-shard, arbitrary order, holes and repeated UIDs.
    seed = plan[0]['uids'][:8]
    cases = dict(shard_boundary=[16383,16384,16385,16382],
                 reversed=seed[::-1], duplicates=[seed[0],seed[0],seed[-1],seed[0]],
                 holes=[16380,16383,16381,16385])
    traces = []
    selected = [(f'document-{r["document"]}', r['uids'], r['native']) for r in plan]
    for label,uids in cases.items():
        frames = read(readers['pread'], uids)
        torch.save(dict(uids=uids,frames=frames), a.output/f'{label}.pt')
        selected.append((label,uids,fingerprint(frames)))
    for label, uids, expected in selected:
        for name, reader in readers.items():
            path = a.output/f'{label}-{name}-reads.jsonl'
            frames = trace(reader, uids, path)
            need(fingerprint(frames) == expected, 'COALESCED_TRACE_NATIVE')
            candidate = a.output/f'{label}-{name}-frames.pt'; torch.save(frames, candidate)
            need(fingerprint(torch.load(candidate,map_location='cpu',weights_only=True)) == expected,
                 'COALESCED_PERSISTED_NATIVE')
        req = requests(data, uids); reqpath = a.output/f'{label}-requests.json'
        write_committed(reqpath, req)
        traces.append(dict(label=label,uids=uids,native=expected,requests=reqpath.name))
    phases['range_and_persistence_boundaries'] = time.perf_counter()-tick; tick = time.perf_counter()
    duplicate = readers['coalesced'].frames([seed[0],seed[0]])
    for name in data.roster:
        original = duplicate[1][name].clone()
        need(duplicate[0][name].data_ptr() != duplicate[1][name].data_ptr(), 'COALESCED_OUTPUT_ALIAS')
        duplicate[0][name][0,0] += 1
        need(torch.equal(duplicate[1][name],original), 'COALESCED_DUPLICATE_MUTATION')
    readers['pread'].close()
    boolean_uid = rejected(lambda:readers['coalesced'].frames([True]))
    negatives = controls(readers['coalesced'], seed, a.output)
    negatives['boolean_uid'] = boolean_uid
    # Eviction exercises real six-instrument reads under a three-handle cache.
    small = CoalescedNativeFrames(*args,handle_limit=3); baseline = ResidentNativeFrames(*args)
    try:
        need(fingerprint(read(small,seed)) == fingerprint(read(baseline,seed)),
             'COALESCED_EVICTION_BYTES')
        need(small.io_stats['evictions'] > 0 and len(small.maps) <= 3, 'COALESCED_EVICTION_BOUND')
        held = list(small.maps.values()); evictions = small.io_stats.copy()
    finally:
        small.close(); baseline.close()
    need(all(v.fd is None for v in held), 'COALESCED_EVICTION_TEARDOWN')
    binding = dict(code=pins,source=data.binding,prior=PRIOR[1],documents=DOCUMENTS,
                   handle_limit=64,output_bytes=16<<20,slab_bytes=16<<20,
                   finite_scratch_bytes=4<<20,same_native_arithmetic=True)
    previous = training.pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding and previous['plan']==plan
             and previous['traces']==traces and previous['controls']==negatives,'COALESCED_COLD_BINDING')
        for row in traces:
            for suffix in ['requests.json','pread-reads.jsonl','coalesced-reads.jsonl',
                           'pread-frames.pt','coalesced-frames.pt']:
                name=f'{row["label"]}-{suffix}'
                need(file_sha(a.previous/name)==previous['files'][name]==file_sha(a.output/name),
                     'COALESCED_COLD_PHYSICAL')
    data.fresh(); data.source.fresh()
    need(not torch.cuda.is_initialized(),'COALESCED_UNEXPECTED_GPU')
    phases['controls_and_cold_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick; seconds=time.perf_counter()-began
    need(.9 <= sum(phases.values())/seconds <= 1.000001,'COALESCED_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,roster=list(data.roster),plan=plan,traces=traces,measurements=measurements,
        measured_stats=measured_stats,evictions=evictions,controls=negatives,files=files,
        previous_pin=a.previous_pin,phases_seconds=phases,seconds=seconds,
        cpu_seconds=time.process_time()-cpu0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        all_native_bytes_exact=True,all_handles_closed=True,optimizer_updates=0,
        scientific_admission=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    a=p.parse_args()
    try: run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
