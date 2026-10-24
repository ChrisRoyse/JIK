"""Recount all indexed surface bytes against source Parquet and test real reads."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit import legal_surface_bytes_build as producer
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need
from text.legal_reuse_input import ReuseInputs
from text.legal_surface_bytes import SurfaceBytes,TILE


def rejected(fn):
    try:fn()
    except (ValueError,FileNotFoundError) as e:return str(e)
    raise ValueError('SURFACE_READBACK_NEGATIVE_ACCEPTED')


def corruption_controls(a, source):
    fixture=a.output/'private-corruption';shutil.copytree(a.input,fixture)
    reader=SurfaceBytes(fixture,a.input_pin,source);path=fixture/'surfaces.bin'
    with path.open('r+b') as f:
        raw=f.read(1);f.seek(0);f.write(bytes([raw[0]^1]));f.flush();os.fsync(f.fileno())
    controls=dict(stale_open=rejected(lambda:reader.read([0])),corrupt_open=rejected(lambda:SurfaceBytes(fixture,a.input_pin,source)))
    controls['stale_close']=rejected(reader.close)
    need(reader.closed and reader.fd is None and reader.offsets is None,'SURFACE_READBACK_HANDLE_LEAK')
    # A separate private path tests absence, without changing the publication.
    path.rename(fixture/'corrupt-surface-retained.bin')
    controls['missing_blob']=rejected(lambda:SurfaceBytes(fixture,a.input_pin,source))
    return controls


def benchmark(source, reader):
    # Identical real scattered requests with duplicates; no data-dependent selection.
    ids=[u for pair in zip(range(0,4096,64),range(4096,8192,64)) for u in pair]
    result=[]
    for rotation in range(3):
        order=ids[rotation:]+ids[:rotation];order+=order[:2]
        tick=time.perf_counter();old=tuple(source.input(u)['text'].encode('utf-8') for u in order)
        old_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        new=tuple(raw for lo in range(0,len(order),TILE) for raw in reader.read(order[lo:lo+TILE]))
        seconds=time.perf_counter()-tick
        need(old==new,'SURFACE_READBACK_BENCHMARK_BYTES')
        result.append(dict(rotation=rotation,requests=len(order),bytes=sum(map(len,old)),
            baseline_seconds=old_seconds,indexed_seconds=seconds,
            raw_sha256=hashlib.sha256(b''.join(old)).hexdigest()))
    return result


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    source=ReuseInputs(a.root/producer.SOURCE[0],producer.SOURCE[1])
    reader=SurfaceBytes(a.input,a.input_pin,source);r=reader.report;pins=producer.code()
    need(r['code']==pins,'SURFACE_READBACK_SOURCE_CODE')
    offsets=np.load(a.input/'offsets.npy',mmap_mode='r',allow_pickle=False)
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    cursor=total=0;counts=dict(empty=0,nonascii=0,over_640=0,maximum_bytes=0)
    for batch in source.table.iter_batches(batch_size=4096,columns=['text'],use_threads=False):
        texts=batch.column(0).to_pylist()[:r['inputs']-cursor]
        for lo in range(0,len(texts),TILE):
            expected=[s.encode('utf-8','strict') for s in texts[lo:lo+TILE]]
            actual=reader.read(list(range(cursor,cursor+len(expected))))
            for raw,saved in zip(expected,actual):
                need(raw==saved,'SURFACE_READBACK_SOURCE_BYTES')
                need(int(offsets[cursor])==total and int(offsets[cursor+1])==total+len(raw),'SURFACE_READBACK_OFFSET')
                counts['empty']+=not raw;counts['nonascii']+=not raw.decode('utf-8').isascii()
                counts['over_640']+=len(raw)>640;counts['maximum_bytes']=max(counts['maximum_bytes'],len(raw))
                total+=len(raw);cursor+=1
        if cursor==r['inputs']:break
        if cursor%262144==0:print(json.dumps(dict(recounted=cursor,bytes=total)),flush=True)
    need(cursor==r['inputs'] and total==r['bytes'] and counts==r['counts'],'SURFACE_READBACK_COVERAGE')
    offsets._mmap.close();recount_stats=deepcopy(reader.stats)
    phases['full_source_recount']=time.perf_counter()-tick;tick=time.perf_counter()
    controls={str(value):rejected(lambda v=value:reader.read(v)) for value in [[],[True],[-1],[reader.size],[0]*129]}
    controls['admission']=rejected(reader.training_authority)
    if r['scope']=='qualify':controls.update(corruption_controls(a,source))
    else:
        q=r['qualification'];p=Path(q['path'])/'report.json';need(file_sha(p)==q['pin'],'SURFACE_READBACK_QUALIFICATION')
        previous=json.loads(p.read_bytes());need(previous['all_bytes_exact'] and previous['all_offsets_exact'],'SURFACE_READBACK_PRIOR')
        controls['qualified_corruption_controls']=previous['negative_controls']
    timing=benchmark(source,reader) if r['scope']=='qualify' else []
    reader.close();controls['closed']=rejected(lambda:reader.read([0]));source.fresh()
    phases['controls_and_benchmark']=time.perf_counter()-tick;tick=time.perf_counter()
    need(producer.code()==pins,'SURFACE_READBACK_CODE_CHANGED')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.000001,'SURFACE_READBACK_PHASE_COVERAGE')
    result=dict(complete=True,input=str(a.input),input_pin=a.input_pin,source=source.pin,code=pins,scope=r['scope'],
        inputs=cursor,bytes=total,counts=counts,all_bytes_exact=True,all_offsets_exact=True,
        negative_controls=controls,benchmark=timing,recount_stats=recount_stats,files=files,
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-process,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        number_status='measured',acceptance='author-checked',scientific_admission=False)
    pin=write_committed(a.output/'report.json',result)
    print(json.dumps(dict(complete=True,pin=pin,inputs=cursor,bytes=total,benchmark=timing)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
