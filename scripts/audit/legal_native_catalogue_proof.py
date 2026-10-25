"""Publish and physically challenge a native authentication catalogue on aiwonder."""
import argparse
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import legal_surface_inversion as original
from audit import protected_corpus_program as producer
from claims.process_conditions import write_committed
from text import legal_native_catalogue as catalogue
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_native_batch import NativeBatchSession,same_files


def controls(session,path,pin,output):
    value=json.loads(path.read_bytes());out={}
    new=NativeBatchSession(session.root,session.pin,session.source)
    def reject(name,call,code):
        out[name]=producer.rejected(call,code)
        need(not new.descriptors,'NATIVE_CATALOGUE_PARTIAL_ADMISSION')
    reject('wrong_pin',lambda:catalogue.hydrate(new,path,'0'*64),'NATIVE_CATALOGUE_PIN')
    for name,change,code in [
        ('source',lambda v:v['binding'].update(source='0'*64),'NATIVE_CATALOGUE_BINDING'),
        ('host',lambda v:v['binding'].update(host='not-this-host'),'NATIVE_CATALOGUE_BINDING'),
        ('snapshot_guid',lambda v:v['binding']['snapshot'].update(guid='0'),'NATIVE_CATALOGUE_BINDING'),
        ('missing_shard',lambda v:v['shards'].pop(),'NATIVE_CATALOGUE_COVERAGE'),
        ('late_changed_inode',lambda v:v['shards'][-1]['identities']['vectors.npy'].__setitem__(1,-1),'NATIVE_CATALOGUE_STALE'),
        ('bytes',lambda v:v.update(authenticated_bytes=v['authenticated_bytes']+1),'NATIVE_CATALOGUE_BYTE_COUNT')]:
        bad=deepcopy(value);change(bad);p=output/(name+'.json');h=write_committed(p,bad)
        reject(name,lambda p=p,h=h:catalogue.hydrate(new,p,h),code)
    first=value['shards'][0];root=session.root/first['name']/catalogue.shard_name(
        first['start'],first['report']['end'])
    # Only the small real offset array is copied; source paths are never mutated.
    original_offsets=np.load(root/'offsets.npy',allow_pickle=False)
    p=output/'physical-offsets.npy';np.save(p,original_offsets,allow_pickle=False);before=stamp(p)
    changed=original_offsets.copy();changed[0]=1;np.save(p,changed,allow_pickle=False)
    # The v1 proof showed same-size writes can preserve all five stamp fields.
    # v2 refuses writable backing, regardless of whether this particular stamp changed.
    reject('mutable_backing',lambda:catalogue.snapshot_binding(p.parent),'NATIVE_CATALOGUE_SNAPSHOT_SCOPE')
    reject('offset_layout',lambda:catalogue.offset_values(p,first['report']),'NATIVE_CATALOGUE_OFFSETS')
    p2=output/'replacement.npy';np.save(p2,original_offsets,allow_pickle=False);before=stamp(p2)
    p2.rename(output/'retained-original.npy');np.save(p2,original_offsets,allow_pickle=False)
    reject('physical_replaced_file',lambda:same_files({p2:before},'NATIVE_CATALOGUE_STALE'),'NATIVE_CATALOGUE_STALE')
    reject('physical_missing_file',lambda:same_files({output/'absent.npy':before},'NATIVE_CATALOGUE_STALE'),
           'NATIVE_CATALOGUE_STALE')
    linkroot=output/'symlink-control';linkroot.mkdir();(linkroot/'offsets.npy').symlink_to(root/'offsets.npy')
    reject('symlink',lambda:catalogue.identities(linkroot,first),'NATIVE_CATALOGUE_SYMLINK')
    try:
        fd=os.open(root/'offsets.npy',os.O_WRONLY)
    except OSError as exc:
        need(exc.errno==errno.EROFS,'NATIVE_CATALOGUE_WRONG_WRITE_REFUSAL');out['kernel_write_refusal']='EROFS'
    else:
        os.close(fd);raise ValueError('NATIVE_CATALOGUE_SNAPSHOT_WRITABLE')
    out['atomic_refusal']=not new.descriptors
    return out


def run(a):
    began=tick=time.perf_counter();cpu=time.process_time();source=None;a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    paths=['audit/legal_native_catalogue_proof.py','text/legal_native_catalogue.py'];base=Path(__file__).resolve().parents[1]
    code={n:file_sha(base/n) for n in paths}
    try:
        source=original.sources(a.root)
        if a.mode=='publish':
            need(a.snapshot_root is not None,'NATIVE_CATALOGUE_SNAPSHOT_REQUIRED')
            session=catalogue.snapshot_session(source,a.snapshot_root);source['data'].native=session
        else:session=source['data'].native
        phases=dict(source=time.perf_counter()-tick);tick=time.perf_counter()
        if a.mode=='publish':
            def event(count,size):
                print(json.dumps(dict(authenticated_shards=count,bytes=size,elapsed=time.perf_counter()-began)),flush=True)
            pin,value=catalogue.publish(session,a.output/'catalogue.json',event=event)
            summary=dict(pin=pin,path=str(a.output/'catalogue.json'),shards=value['shard_count'],
                         authenticated_bytes=value['authenticated_bytes']);negatives={}
            phases['physical_authentication_publication']=time.perf_counter()-tick;tick=time.perf_counter()
        else:
            need(a.input and a.pin,'NATIVE_CATALOGUE_INPUT')
            summary=catalogue.attach(source,a.input,a.pin);session=source['data'].native
            phases['hydrate']=time.perf_counter()-tick;tick=time.perf_counter()
            negatives=controls(session,a.input,a.pin,a.output)
            phases['controls']=time.perf_counter()-tick;tick=time.perf_counter()
        need(not torch.cuda.is_initialized(),'NATIVE_CATALOGUE_CPU_ONLY')
        original.close(source);source=None;phases['finish']=time.perf_counter()-tick
        wall=time.perf_counter()-began
        need(.9<=sum(phases.values())/wall<=1.000001,'NATIVE_CATALOGUE_PHASE_COVERAGE')
        need(all(file_sha(base/n)==h for n,h in code.items()),'NATIVE_CATALOGUE_CODE_CHANGED')
        result=dict(complete=True,mode=a.mode,code=code,catalogue=summary,controls=negatives,phases=phases,
                    wall=wall,cpu=time.process_time()-cpu,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    gpu_used=False,promoted=False,capability_demonstrated=False,acceptance='author-checked')
        pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=pin,**result)),flush=True)
    finally:
        if source:original.close(source)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['publish','readback'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--pin');p.add_argument('--snapshot-root',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
        raise
