"""Separate direct NumPy readback; never imports the reading-cache implementation."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch


def require(ok, code):
    if not ok: raise ValueError('MNLM_READING_RECOUNT_'+code)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def check_reads(expected, observed, layouts):
    counts = Counter()
    for row in observed:
        offset, shape = layouts[row['path']]
        require(row['byte_offset'] == offset+row['start']*shape[1]*4
                and row['bytes'] == (row['stop']-row['start'])*shape[1]*4
                and row['dimensions'] == shape[1], 'READ_OFFSET')
        counts[row['path'],row['start'],row['stop']] += 1
    require(counts == Counter({key:1 for key in expected}), 'UNIQUE_READ_COVERAGE')


def refused(fn):
    try: fn()
    except ValueError as error: return str(error)
    raise ValueError('MNLM_READING_RECOUNT_CONTROL_ACCEPTED')


def run(a):
    began = time.perf_counter(); cpu0 = time.process_time()
    require(sha(a.input/'report.json') == a.pin, 'REPORT_PIN')
    report = json.loads((a.input/'report.json').read_bytes())
    require(report['complete'] and not report['scientific_admission'], 'AUTHORITY')
    for name,pin in report['files'].items(): require(sha(a.input/name) == pin, 'PHYSICAL_FILE')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cached = torch.load(a.input/'cached-frames.pt', map_location='cpu', weights_only=True)
    requests = json.loads((a.input/'requests.json').read_bytes())
    roster = report['roster']; frames = cached['frames']; uids = cached['uids']
    require(uids == [u for row in report['plan'] for u in row['uids']], 'UIDS')
    require(len(frames) == len(uids) and all(list(f) == roster for f in frames), 'ROSTER')
    require(len(requests) == len(uids)*len(roster) and
            {(r['position'],r['name']) for r in requests} ==
            {(i,n) for i in range(len(uids)) for n in roster}, 'REQUEST_COMPLETENESS')
    native = OrderedDict(); metadata = {}; stamps = {}; layouts = {}; expected = set(); checked = 0
    try:
        for row in requests:
            path = Path(row['path']); key = str(path); name = row['name']; i = row['position']
            if key not in metadata:
                root = path.parent; meta = root/'report.json'; seal = root/'COMMITTED'
                require(path.name == 'vectors.npy' and root.parent.name == name, 'NATIVE_ADDRESS')
                require(sha(meta) == row['descriptor_pin'], 'DESCRIPTOR_PIN')
                r = json.loads(meta.read_bytes())
                require(json.loads(seal.read_bytes()) == {'report_sha256':row['descriptor_pin']}
                        and r['configuration_pin'] == report['binding']['source']['bank']
                        and r['complete'], 'NATIVE_BINDING')
                for fname,proof in r['files'].items():
                    f = root/fname; before = identity(f)
                    require(f.stat().st_size == proof['bytes'] and sha(f) == proof['sha256']
                            and identity(f) == before, 'NATIVE_PHYSICAL')
                    stamps[str(f)] = before
                stamps[str(meta)] = identity(meta); stamps[str(seal)] = identity(seal)
                metadata[key] = (r,row['descriptor_pin'])
            r,pin = metadata[key]
            if key not in native:
                while len(native) >= 32:
                    _,(v,o) = native.popitem(last=False); v._mmap.close(); o._mmap.close()
                v = np.load(path,mmap_mode='r',allow_pickle=False)
                try: o = np.load(path.parent/'offsets.npy',mmap_mode='r',allow_pickle=False)
                except BaseException: v._mmap.close(); raise
                native[key] = (v,o); layouts[key] = (v.offset,v.shape)
                require(v.dtype == np.float32 and v.shape == (r['windows'],r['dimensions'])
                        and v.flags.c_contiguous, 'VECTOR_LAYOUT')
            native.move_to_end(key); v,o = native[key]
            require(row['uid'] == uids[i] and row['descriptor_pin'] == pin
                    and r['start'] <= row['uid'] < r['end'], 'REQUEST_IDENTITY')
            lo,hi = map(int,o[row['uid']-r['start']:row['uid']-r['start']+2])
            require((lo,hi) == (row['start'],row['stop']) and row['dimensions'] == v.shape[1], 'OFFSETS')
            x = frames[i][name]
            require(x.dtype == torch.float32 and tuple(x.shape) == (hi-lo,v.shape[1])
                    and x.numpy().tobytes() == np.ascontiguousarray(v[lo:hi]).tobytes(), 'ALL_NATIVE_BYTES')
            expected.add((key,lo,hi)); checked += 1
        reads = [json.loads(line) for line in (a.input/'fill-reads.jsonl').read_text().splitlines()]
        check_reads(expected, reads, layouts)
        require((a.input/'reuse-reads.jsonl').read_bytes() == b'', 'REUSE_REOPENED_NATIVE')
        cursor = 0
        for row in report['plan']:
            saved = torch.load(a.input/f'document-{row["document"]}.pt',map_location='cpu',weights_only=True)
            require(saved['uids'] == row['uids'] and len(saved['frames']) == row['atoms'], 'DOCUMENT_SCOPE')
            for j,frame in enumerate(saved['frames']):
                for name in roster:
                    require(frame[name].numpy().tobytes() == frames[cursor+j][name].numpy().tobytes(), 'DOCUMENT_BYTES')
            cursor += row['atoms']
        altered = [dict(r) for r in reads]; altered[0]['byte_offset'] += 4
        controls = dict(changed_offset=refused(lambda:check_reads(expected,altered,layouts)),
            missing_read=refused(lambda:check_reads(expected,reads[1:],layouts)),
            duplicate_read=refused(lambda:check_reads(expected,reads+reads[:1],layouts)))
        require(all(identity(Path(p)) == s for p,s in stamps.items()), 'SOURCE_CHANGED')
        require(not torch.cuda.is_initialized(), 'UNEXPECTED_GPU')
        timings = {mode:{str(repeat):sum(r['seconds'] for r in report['measurements']
            if r['mode'] == mode and r['repeat'] == repeat) for repeat in range(4)} for mode in ['pread','cache']}
        result = dict(complete=True, input=str(a.input), input_pin=a.pin, code=sha(Path(__file__)),
            reader_files=len(metadata), checked_native_slot_arrays=checked, checked_atoms=len(uids),
            all_native_bytes_exact=True, unique_source_reads=len(reads), source_bytes=sum(r['bytes'] for r in reads),
            reuse_source_reads=0, timings=timings, controls=controls, seconds=time.perf_counter()-began,
            cpu_seconds=time.process_time()-cpu0, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            acceptance='author-checked', number_status='measured', scientific_admission=False)
        with a.output.open('x') as out:
            out.write(json.dumps(result,sort_keys=True,indent=2)+'\n');out.flush();os.fsync(out.fileno())
        print(json.dumps(dict(complete=True,pin=sha(a.output),timings=timings,checked=checked)),flush=True)
    finally:
        for v,o in native.values(): v._mmap.close(); o._mmap.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
