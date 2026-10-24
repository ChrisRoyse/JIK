"""Independent NumPy/file readback of saved native frames and observed read ranges."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch


def require(ok, code):
    if not ok: raise ValueError('MNLM_COALESCED_RECOUNT_'+code)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def identity(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def check_ranges(requests, reads, mode, layouts):
    expected=Counter(); observed=Counter()
    for row in requests:
        for window in range(row['start'],row['stop']):
            expected[row['batch'],row['path'],window]+=1
    if mode=='coalesced': expected=Counter({k:1 for k in expected})
    for row in reads:
        offset,shape=layouts[row['path']]
        require(0<=row['start']<row['stop']<=shape[0] and row['dimensions']==shape[1], 'READ_SCOPE')
        require(row['byte_offset']==offset+row['start']*shape[1]*4
                and row['bytes']==(row['stop']-row['start'])*shape[1]*4
                and row['bytes']<=16<<20, 'READ_BYTES')
        for window in range(row['start'],row['stop']):
            observed[row['batch'],row['path'],window]+=1
    require(observed==expected,'RANGE_COVERAGE')
    return dict(calls=len(reads),bytes=sum(r['bytes'] for r in reads),windows=sum(observed.values()))


def refused(fn):
    try: fn()
    except ValueError as error: return str(error)
    raise ValueError('MNLM_COALESCED_RECOUNT_CONTROL_ACCEPTED')


def run(a):
    began=time.perf_counter();require(sha(a.input/'report.json')==a.pin,'REPORT_PIN')
    report=json.loads((a.input/'report.json').read_bytes())
    require(report['complete'] and not report['scientific_admission'],'AUTHORITY')
    for name,pin in report['files'].items():require(sha(a.input/name)==pin,'PHYSICAL_FILE')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    native=OrderedDict(); metadata={}; layouts={}; stamps={}; totals=[]; checked=0; controls={}
    try:
        for item in report['traces']:
            requests=json.loads((a.input/item['requests']).read_bytes())
            baseline=torch.load(a.input/f'{item["label"]}-pread-frames.pt',map_location='cpu',weights_only=True)
            candidate=torch.load(a.input/f'{item["label"]}-coalesced-frames.pt',map_location='cpu',weights_only=True)
            roster=report['roster']
            require(len(candidate)==len(baseline)==len(item['uids']),'FRAME_COUNT')
            require(all(list(f)==roster for f in baseline+candidate),'ROSTER')
            expected={(i,n) for i in range(len(item['uids'])) for n in roster}
            require(len(requests)==len(expected) and {(r['position'],r['name']) for r in requests}==expected,'REQUEST_COMPLETENESS')
            for row in requests:
                path=Path(row['path']);key=row['path'];position=row['position'];name=row['name']
                if key not in metadata:
                    root=path.parent; meta=root/'report.json'; seal=root/'COMMITTED'
                    require(path.name=='vectors.npy' and path.parent.parent.name==name,'NATIVE_ADDRESS')
                    require(sha(meta)==row['descriptor_pin'],'DESCRIPTOR_PIN')
                    r=json.loads(meta.read_bytes())
                    require(json.loads(seal.read_bytes())=={'report_sha256':row['descriptor_pin']}
                            and r['configuration_pin']==report['binding']['source']['bank'] and r['complete'],'NATIVE_BINDING')
                    for fname,proof in r['files'].items():
                        f=root/fname;stamps[str(f)]=identity(f)
                        require(f.stat().st_size==proof['bytes'] and sha(f)==proof['sha256'],'NATIVE_PHYSICAL')
                    stamps[str(meta)]=identity(meta);stamps[str(seal)]=identity(seal)
                    metadata[key]=(r,row['descriptor_pin'])
                r,pin=metadata[key]
                if key not in native:
                    while len(native)>=32:
                        _,(old_v,old_offsets)=native.popitem(last=False)
                        old_v._mmap.close();old_offsets._mmap.close()
                    v=np.load(path,mmap_mode='r',allow_pickle=False)
                    try: offsets=np.load(path.parent/'offsets.npy',mmap_mode='r',allow_pickle=False)
                    except BaseException:
                        v._mmap.close();raise
                    native[key]=(v,offsets);layouts[key]=(v.offset,v.shape)
                    require(v.dtype==np.float32 and v.shape==(r['windows'],r['dimensions']) and v.flags.c_contiguous,'VECTOR_LAYOUT')
                native.move_to_end(key);v,offsets=native[key]
                require(row['descriptor_pin']==pin and row['uid']==item['uids'][position]
                        and row['batch']==position//128 and r['start']<=row['uid']<r['end'],'REQUEST_IDENTITY')
                lo,hi=map(int,offsets[row['uid']-r['start']:row['uid']-r['start']+2])
                require((lo,hi)==(row['start'],row['stop']) and row['dimensions']==v.shape[1],'REQUEST_OFFSETS')
                original=np.ascontiguousarray(v[lo:hi]).tobytes()
                for frames in [baseline,candidate]:
                    x=frames[position][name]
                    require(x.dtype==torch.float32 and tuple(x.shape)==(hi-lo,v.shape[1])
                            and x.numpy().tobytes()==original,'ALL_NATIVE_BYTES')
                checked+=1
            counts={}
            for mode in ['pread','coalesced']:
                reads=[json.loads(line) for line in (a.input/f'{item["label"]}-{mode}-reads.jsonl').read_text().splitlines()]
                counts[mode]=check_ranges(requests,reads,mode,layouts)
                if not controls and mode=='coalesced':
                    bad=[dict(x) for x in reads];bad[0]['byte_offset']+=4
                    controls=dict(changed_offset=refused(lambda:check_ranges(requests,bad,mode,layouts)),
                                  missing_read=refused(lambda:check_ranges(requests,reads[1:],mode,layouts)),
                                  repeated_read=refused(lambda:check_ranges(requests,reads+reads[:1],mode,layouts)))
            totals.append(dict(label=item['label'],atoms=len(item['uids']),counts=counts))
        require(all(identity(Path(p))==s for p,s in stamps.items()),'SOURCE_CHANGED')
        require(not torch.cuda.is_initialized(),'UNEXPECTED_GPU')
        timings={}
        for mode in ['pread','coalesced']:
            rows=[r for r in report['measurements'] if r['mode']==mode]
            timings[mode]=dict(seconds=sum(r['seconds'] for r in rows),cpu_seconds=sum(r['cpu_seconds'] for r in rows),
                by_document={str(d):sum(r['seconds'] for r in rows if r['document']==d) for d in report['binding']['documents']})
        result=dict(complete=True,input=str(a.input),input_pin=a.pin,reader_files=len(metadata),
                    code=sha(Path(__file__)),map_pairs_limit=32,
                    checked_native_slot_arrays=checked,all_bytes_exact=True,no_unrequested_window_reads=True,
                    counts=totals,controls=controls,timings=timings,seconds=time.perf_counter()-began,
                    acceptance='author-checked',number_status='measured',scientific_admission=False)
        require(not a.output.exists(),'OUTPUT_EXISTS')
        with a.output.open('x') as out:
            out.write(json.dumps(result,sort_keys=True,indent=2)+'\n');out.flush();os.fsync(out.fileno())
        print(json.dumps(dict(complete=True,pin=sha(a.output),counts=totals,timings=timings)))
    finally:
        for v,offsets in native.values():v._mmap.close();offsets._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
