"""Publish a lossless indexed view of already-measured legal atom surfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need
from text.legal_reuse_input import ReuseInputs
from text.legal_surface_bytes import SCHEMA

SOURCE=('legal-input-reuse-v1-20261022','7dd93b1996f79768ead1bb419068b7eb62e3b15d3cf5f529e17c610d5279a02e')


def code():
    root=Path(__file__).resolve().parents[1]
    return {n:file_sha(root/n) for n in ['text/legal_surface_bytes.py','text/legal_reuse_input.py',
        'audit/legal_surface_bytes_build.py','audit/legal_surface_bytes_readback.py']}


def run(a):
    start=tick=time.perf_counter(); process=time.process_time(); a.output.mkdir(parents=True,exist_ok=False)
    source=ReuseInputs(a.root/SOURCE[0],SOURCE[1]); pins=code(); qualification=None
    if a.scope=='full':
        need(a.qualification and a.qualification_pin,'SURFACE_BUILD_QUALIFICATION_REQUIRED')
        p=a.qualification/'report.json';need(file_sha(p)==a.qualification_pin,'SURFACE_BUILD_QUALIFICATION_PIN')
        q=json.loads(p.read_bytes())
        need(q['complete'] and q['all_bytes_exact'] and q['all_offsets_exact'] and q['inputs']==8192
             and q['code']==pins and q['source']==source.pin and q['negative_controls'],'SURFACE_BUILD_QUALIFICATION')
        for name,pin in q['files'].items():need(file_sha(a.qualification/name)==pin,'SURFACE_BUILD_PROOF_BYTES')
        qp=Path(q['input']);need(file_sha(qp/'report.json')==q['input_pin'],'SURFACE_BUILD_PROOF_INPUT')
        published=json.loads((qp/'report.json').read_bytes())
        for name,pin in published['files'].items():need(file_sha(qp/name)==pin,'SURFACE_BUILD_PROOF_INPUT_BYTES')
        qualification=dict(path=str(a.qualification),pin=a.qualification_pin)
    n=8192 if a.scope=='qualify' else source.report['counts']['unique_inputs']
    phases=dict(authentication=time.perf_counter()-tick); tick=time.perf_counter()
    offsets=np.lib.format.open_memmap(a.output/'offsets.npy',mode='w+',dtype='<u8',shape=(n+1,));offsets[0]=0
    cursor=total=0; counts=dict(empty=0,nonascii=0,over_640=0,maximum_bytes=0)
    with (a.output/'surfaces.bin').open('xb') as blob:
        for batch in source.table.iter_batches(batch_size=128,columns=['index','text_sha256','text'],use_threads=False):
            rows=batch.to_pydict(); chunks=[]
            for uid,pin,text in zip(rows['index'],rows['text_sha256'],rows['text']):
                if cursor==n:break
                raw=text.encode('utf-8','strict')
                need(uid==cursor and hashlib.sha256(raw).hexdigest()==pin
                     and bytes.fromhex(pin)==source.arrays['hashes'][uid].tobytes(),'SURFACE_BUILD_SOURCE_ROW')
                total+=len(raw); cursor+=1; offsets[cursor]=total; chunks.append(raw)
                counts['empty']+=not raw; counts['nonascii']+=not text.isascii(); counts['over_640']+=len(raw)>640
                counts['maximum_bytes']=max(counts['maximum_bytes'],len(raw))
            blob.write(b''.join(chunks))
            if cursor==n:break
            if cursor%262144==0:print(json.dumps(dict(inputs=cursor,bytes=total)),flush=True)
        blob.flush();os.fsync(blob.fileno())
    need(cursor==n,'SURFACE_BUILD_COVERAGE'); offsets.flush(); offsets._mmap.close()
    with (a.output/'offsets.npy').open('rb') as f:os.fsync(f.fileno())
    if a.scope=='full':need(total==source.report['counts']['unique_text_bytes'],'SURFACE_BUILD_FULL_BYTES')
    phases['stream_and_publish']=time.perf_counter()-tick; tick=time.perf_counter()
    source.fresh(); need(code()==pins,'SURFACE_BUILD_CODE_CHANGED')
    for name,pin in pins.items():
        p=a.output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(p)==pin,'SURFACE_BUILD_CODE_COPY')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['seal']=time.perf_counter()-tick; seconds=time.perf_counter()-start
    need(.9<=sum(phases.values())/seconds<=1.000001,'SURFACE_BUILD_PHASE_COVERAGE')
    report=dict(schema=SCHEMA,complete=True,scope=a.scope,source=source.pin,source_path=str(source.root),code=pins,
        inputs=n,bytes=total,counts=counts,files=files,qualification=qualification,phases=phases,seconds=seconds,
        cpu_seconds=time.process_time()-process,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_encoder_calls=0,model_updates=0,scientific_admission=False,training_authority=False,
        number_status='measured',acceptance='author-checked')
    pin=write_committed(a.output/'report.json',report)
    write_committed(a.output/'COMMITTED',dict(report_sha256=pin))
    print(json.dumps(dict(complete=True,pin=pin,inputs=n,bytes=total,counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--scope',choices=['qualify','full'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--qualification-pin');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
