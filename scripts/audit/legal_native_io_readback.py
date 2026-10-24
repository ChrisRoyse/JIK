"""Real five-document native I/O comparison, exact bytes and resource teardown."""
import argparse
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
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_native_training_stream import CertifiedNativeFrames
from model.legal_native_resident_stream import ResidentNativeFrames, NativeFileView
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

DOCUMENTS = [105748, 81127, 58348, 840, 5796]
PRIOR = ('legal-long-document-training-qualification-v1-20261023',
         '0916ae94ced1640727dbf1ad2193d212bd646f5007210102f7bd6e83614b62ed')


def usage():
    p = Path('/proc/self')
    return dict(io={k:int(v) for k,v in (r.split(':') for r in (p/'io').read_text().splitlines())},
                fds=len(list((p/'fd').iterdir())), status=(p/'status').read_text())


def read(reader, uids):
    result = []
    for lo in range(0, len(uids), 128): result.extend(reader.frames(uids[lo:lo+128]))
    return tuple(result)


def controls(reader, uids, root):
    n = reader.roster[0]
    original = reader.frames(uids[:2])
    need(fingerprint(reader.frames(uids[:2][::-1])[::-1]) == fingerprint(original), 'IO_REVERSE_EXACT')
    same = reader.frames([uids[0], uids[0]])
    need(fingerprint(same[0]) == fingerprint(same[1]) == fingerprint(original[0]), 'IO_DUPLICATE_EXACT')
    result = dict(negative_uid=rejected(lambda:reader.frames([-1])),
                  too_many_uids=rejected(lambda:reader.frames([uids[0]]*129)),
                  empty=rejected(lambda:reader.frames([])))
    key = (n, uids[0]//16384*16384)
    d = reader.data.native.descriptors[key]
    path = d['root']/'vectors.npy'; old = d['stamps'][path]
    d['stamps'][path] = (old[0], old[1], old[2]+1, old[3], old[4])
    try: result['changed_source'] = rejected(lambda:reader.frames(uids[:1]))
    finally: d['stamps'][path] = old
    # Persist actual source readings and exercise changed/open/truncated files.
    path = root/'real-native-window.npy'; np.save(path, original[0][n].numpy(), allow_pickle=False)
    saved = np.load(path, mmap_mode='r', allow_pickle=False)
    layout = (saved.offset, saved.shape); saved._mmap.close()
    descriptor = dict(stamps={path:stamp(path)})
    view = NativeFileView(path, descriptor, layout)
    need(np.array_equal(view[0:len(original[0][n])], original[0][n].numpy()), 'IO_DIRECT_REAL_BYTES')
    # Slice start is explicit in production; pass the same valid bounds here.
    view.close()
    result['closed_file'] = rejected(lambda:view[0:1])
    with path.open('r+b') as f: f.truncate(layout[0]+1)
    result['changed_open_identity'] = rejected(lambda:NativeFileView(path, descriptor, layout))
    descriptor = dict(stamps={path:stamp(path)})
    view = NativeFileView(path, descriptor, layout)
    try: result['short_read'] = rejected(lambda:view[0:1])
    finally: view.close()
    # Refusal must also close every retained handle.
    held = list(reader.maps.values())
    d['stamps'][d['root']/'vectors.npy'] = (old[0], old[1], old[2]+1, old[3], old[4])
    try: result['close_changed_source'] = rejected(reader.close)
    finally: d['stamps'][d['root']/'vectors.npy'] = old
    need(not reader.maps and reader.closed, 'IO_REFUSED_CLOSE_RESOURCE_LEAK')
    need(all(v.fd is None for v in held), 'IO_REFUSED_CLOSE_OPEN_FILE')
    result['closed_reader'] = rejected(lambda:reader.frames(uids[:1]))
    return result


def run(a):
    began = tick = time.perf_counter(); cpu0 = time.process_time()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    code = dict(training.code_pins(), **{name:file_sha(Path(__file__).resolve().parents[1]/name)
        for name in ['model/legal_native_training_stream.py', 'model/legal_native_resident_stream.py',
                     'audit/legal_native_io_readback.py', 'text/legal_native_batch.py']})
    for name, pin in code.items():
        dest = a.output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(dest) == pin, 'IO_SOURCE_COPY')
    _, _, data, _, _, _, _, _ = training.authenticate(a.root)
    prior = training.pinned(a.root/PRIOR[0]/'report.json', PRIOR[1])
    args = (data, a.root/training.DEFAULTS['audit'][0], training.DEFAULTS['audit'][1])
    reference = CertifiedNativeFrames(*args)
    plan = []; phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    for number in DOCUMENTS:
        start, stop = map(int, data.documents[number]); uids = data.mapping[start:stop].tolist()
        frames = read(reference, uids)
        label = a.root/PRIOR[0]/f'labels-{number}.pt'
        need(file_sha(label) == prior['files'][label.name], 'IO_PRIOR_PHYSICAL_PIN')
        labels = torch.load(label, map_location='cpu', weights_only=True)
        need(fingerprint(frames) == labels['native_pin'] and labels['uids'] == uids, 'IO_PRIOR_NATIVE_EXACT')
        payload = dict(frames=frames, **labels)
        path = a.output/f'document-{number}.pt'; torch.save(payload, path)
        need(fingerprint(torch.load(path, map_location='cpu', weights_only=True)) == fingerprint(payload), 'IO_SAVED_NATIVE_EXACT')
        plan.append(dict(document=number, uids=uids, native=labels['native_pin'], atoms=len(uids)))
    phases['cold_native_authentication_and_source_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    readers = dict(reference=reference, pread=ResidentNativeFrames(*args), mmap=ResidentNativeFrames(*args, mode='mmap'))
    measurements = []
    # Rotate order so no candidate always benefits from the previous reader's page cache.
    for repeat, order in enumerate([['reference','pread','mmap'], ['mmap','reference','pread'], ['pread','mmap','reference']]):
        for mode in order:
            reader = readers[mode]
            for row in plan:
                before = usage(); start = time.perf_counter(); frames = read(reader, row['uids']); seconds = time.perf_counter()-start
                after = usage(); need(fingerprint(frames) == row['native'], 'IO_CANDIDATE_NATIVE_EXACT')
                measurements.append(dict(repeat=repeat, mode=mode, document=row['document'], atoms=row['atoms'],
                    seconds=seconds, before=before, after=after, native=row['native']))
            print(json.dumps(dict(repeat=repeat,mode=mode,seconds=sum(r['seconds'] for r in measurements if r['repeat']==repeat and r['mode']==mode))),flush=True)
    phases['matched_readers'] = time.perf_counter()-tick; tick = time.perf_counter()
    io_stats = {n:dict(r.io_stats) for n,r in readers.items() if hasattr(r,'io_stats')}
    reference.close(); readers['mmap'].close()
    negatives = controls(readers['pread'], plan[0]['uids'], a.output)
    need(all(not r.maps for r in readers.values()), 'IO_MAPPING_LEAK')
    previous = training.pinned(a.previous/'report.json', a.previous_pin) if a.previous else None
    binding = dict(code=code, source=data.binding, prior=PRIOR[1], documents=DOCUMENTS,
                   handle_limit=64, mapped_bytes=1 << 30, same_native_arithmetic=True)
    if previous:
        need(previous['complete'] and previous['binding']==binding and previous['plan']==plan
             and previous['controls']==negatives, 'IO_COLD_BINDING')
        for row in plan:
            name=f"document-{row['document']}.pt"
            need(file_sha(a.previous/name)==previous['files'][name], 'IO_COLD_PHYSICAL')
            need(fingerprint(torch.load(a.previous/name,map_location='cpu',weights_only=True))
                 == fingerprint(torch.load(a.output/name,map_location='cpu',weights_only=True)), 'IO_COLD_NATIVE_EXACT')
    need(not torch.cuda.is_initialized(), 'IO_UNEXPECTED_GPU')
    phases['controls_and_cold_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['hashes']=time.perf_counter()-tick; seconds=time.perf_counter()-began
    need(sum(phases.values())/seconds>=.9,'IO_PHASE_ATTRIBUTION')
    result=dict(complete=True,binding=binding,plan=plan,measurements=measurements,io_stats=io_stats,
                controls=negatives,files=files,previous_pin=a.previous_pin,phases_seconds=phases,seconds=seconds,
                cpu_seconds=time.process_time()-cpu0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                all_native_bytes_exact=True,all_handles_closed=True,optimizer_updates=0,
                scientific_admission=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    run(p.parse_args())
