"""Complete native training frames under an exhaustive source-binding certificate.

The certificate is usable only with the exact, freshly authenticated shard bytes.
This avoids reparsing unchanged per-input provenance during every training epoch.
The native bank and its ordinary checked reader remain authoritative.
"""
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import torch

from text.constellation_corpus import file_sha, need
from text.legal_native_store import SHARD_INPUTS


class CertifiedNativeFrames:
    def __init__(self, data, audit_root, audit_pin):
        self.data=data;self.roster=data.roster;self.audit_root=Path(audit_root)
        path=self.audit_root/'report.json'
        need(file_sha(path)==audit_pin,'TRAIN_STREAM_AUDIT_PIN')
        audit=json.loads(path.read_bytes())
        need(audit['complete'] and audit['full_corpus_readback']
             and audit['binding']['source_pin']==data.source.pin
             and audit['binding']['bank_pin']==data.native.pin,'TRAIN_STREAM_AUDIT_BINDING')
        self.refs={};self.maps=OrderedDict();self.touched=set()
        self.stats=dict(calls=0,inputs=0,windows=0,payload_bytes=0,maximum_payload_bytes=0,
                        descriptor_seconds=0.,gather_seconds=0.,reader_seconds=0.)
        for name in self.roster:
            item=audit['scans'][name];path=self.audit_root/name/'stored/scan.json'
            need(file_sha(path)==item['pin'] and item['all_source_bindings']
                 and item['all_windows_finite_unit'],'TRAIN_STREAM_AUDIT_LANE')
            scan=json.loads(path.read_bytes());refs={};cursor=0
            for ref in scan['shards']:
                lo,hi=map(int,ref['path'].split('-'))
                need(lo==cursor and 0<hi-lo<=SHARD_INPUTS,'TRAIN_STREAM_CERTIFICATE_RANGE')
                refs[lo]=(hi,ref['pin']);cursor=hi
            need(cursor==data.native.config['inputs'],'TRAIN_STREAM_CERTIFICATE_COVERAGE')
            self.refs[name]=refs

    def descriptor(self,name,start):
        began=time.perf_counter();d=self.data.native.descriptor(name,start)
        need(start in self.refs[name] and self.refs[name][start]==(d['report']['end'],d['pin']),
             'TRAIN_STREAM_SHARD_CERTIFICATE')
        self.touched.add((name,start));self.stats['descriptor_seconds']+=time.perf_counter()-began
        return d

    def mapping(self,name,start,descriptor):
        key=(name,start)
        if key not in self.maps:
            while len(self.maps)>=len(self.roster):
                _,old=self.maps.popitem(last=False);old._mmap.close()
            value=np.load(descriptor['root']/'vectors.npy',mmap_mode='r',allow_pickle=False)
            r=descriptor['report']
            need(value.dtype==np.float32 and value.shape==(r['windows'],r['dimensions']),
                 'TRAIN_STREAM_VECTOR_LAYOUT')
            self.maps[key]=value
        self.maps.move_to_end(key);return self.maps[key]

    def frames(self,uids):
        began=time.perf_counter();self.data.fresh()
        need(isinstance(uids,(list,tuple)) and 0<len(uids)<=128
             and all(type(u) is int and 0<=u<self.data.native.config['inputs'] for u in uids),
             'TRAIN_STREAM_UID_SCOPE')
        groups={}
        for i,uid in enumerate(uids):groups.setdefault(uid//SHARD_INPUTS*SHARD_INPUTS,[]).append((i,uid))
        descriptors={};payload=0;windows=0
        for name in self.roster:
            for start,rows in sorted(groups.items()):
                d=self.descriptor(name,start);descriptors[name,start]=d
                for _,uid in rows:
                    a,b=map(int,d['offsets'][uid-start:uid-start+2])
                    need(b>a,'TRAIN_STREAM_MISSING_WINDOW');windows+=b-a
                    payload+=(b-a)*d['report']['dimensions']*4
        need(payload<=16<<20,'TRAIN_STREAM_PHYSICAL_MEMORY')
        tick=time.perf_counter();frames=[{} for _ in uids]
        for (name,start),d in descriptors.items():
            self.data.native.descriptor_fresh(d);values=self.mapping(name,start,d)
            for i,uid in groups[start]:
                a,b=map(int,d['offsets'][uid-start:uid-start+2]);v=np.array(values[a:b],copy=True)
                need(np.isfinite(v).all(),'TRAIN_STREAM_NONFINITE')
                frames[i][name]=torch.from_numpy(v)
            self.data.native.descriptor_fresh(d)
        self.data.fresh();self.stats['gather_seconds']+=time.perf_counter()-tick
        self.stats['calls']+=1;self.stats['inputs']+=len(uids);self.stats['windows']+=windows
        self.stats['payload_bytes']+=payload;self.stats['maximum_payload_bytes']=max(payload,self.stats['maximum_payload_bytes'])
        self.stats['reader_seconds']+=time.perf_counter()-began
        return tuple(frames)

    def tiles(self,uids):
        for lo in range(0,len(uids),128):
            ids=[int(x) for x in uids[lo:lo+128]];yield ids,self.frames(ids)

    def finish(self):
        self.data.fresh();self.data.source.fresh()
        for key in self.touched:self.data.native.descriptor_fresh(self.data.native.descriptors[key])

    def close(self):
        self.finish()
        for value in self.maps.values():value._mmap.close()
        self.maps.clear()

    def training_authority(self):
        raise ValueError('MNLM_TRAIN_STREAM_SCIENTIFIC_ADMISSION_REQUIRED')
