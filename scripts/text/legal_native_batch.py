"""Authenticate native shards once per read session; gather complete typed windows.

Resident descriptors contain offsets and immutable file identities, not open files.
All values/trace groups are opened only for one shard segment and closed afterward.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
import time

import numpy as np
import pyarrow.parquet as pq

from text.constellation_corpus import need
from text.constellation_input import stamp
from text.legal_native_store import NativeBank,NativeShard,SHARD_INPUTS,IO_INPUTS,shard_name


def same_files(stamps,code):
    try: unchanged=all(stamp(path)==expected for path,expected in stamps.items())
    except OSError: unchanged=False
    need(unchanged,code)


@dataclass(frozen=True)
class NativeWindowBatch:
    instrument: str
    lens_id: str
    uids: tuple
    values: np.ndarray
    offsets: np.ndarray
    traces: tuple

    def input(self,index):
        need(type(index) is int and 0<=index<len(self.uids),'NATIVE_BATCH_INDEX')
        lo,hi=map(int,self.offsets[index:index+2])
        return self.values[lo:hi],json.loads(self.traces[index])


class NativeBatchSession:
    """Read-only native data; a separate experiment must authorize any training use."""
    def __init__(self,root,pin,source):
        original=NativeBank(root,pin)
        need(original.config['source']['pin']==source.pin and
             original.config['inputs']==source.report['counts']['unique_inputs'],'NATIVE_BATCH_SOURCE')
        self.root,self.pin,self.config=original.root,original.pin,original.config
        self.source=source;self.config_stamp=original.config_stamp
        self.source_commit_stamp=stamp(source.root/'COMMITTED')
        self.descriptors={}
        self.stats=dict(authenticated_shards=0,authenticated_bytes=0,authentication_seconds=0.,
                        calls=0,row_groups_read=0,returned_windows=0,returned_bytes=0,reader_seconds=0.)

    def fresh(self):
        same_files({self.root/'configuration.json':self.config_stamp},'NATIVE_BATCH_CONFIGURATION_CHANGED')
        same_files({self.source.root/'COMMITTED':self.source_commit_stamp},'NATIVE_BATCH_SOURCE_COMMIT_CHANGED')
        self.source.fresh()

    @staticmethod
    def descriptor_fresh(descriptor):
        same_files(descriptor['stamps'],'NATIVE_BATCH_SHARD_CHANGED')

    def descriptor(self,name,start):
        key=(name,start)
        if key not in self.descriptors:
            began=time.perf_counter();end=min(start+SHARD_INPUTS,self.config['inputs'])
            saved=NativeShard(self.root/name/shard_name(start,end),self.pin,
                              self.config['instruments'][name]['lens_id'],start,end)
            try:
                saved.fresh()
                value=dict(root=saved.root,pin=saved.pin,report=saved.report,
                           offsets=np.array(saved.offsets,copy=True),stamps=dict(saved.stamps))
                self.descriptor_fresh(value)
            finally:
                saved.table.close();saved.vectors._mmap.close();saved.offsets._mmap.close()
            self.descriptors[key]=value
            self.stats['authenticated_shards']+=1
            self.stats['authenticated_bytes']+=sum(x['bytes'] for x in value['report']['files'].values())
            self.stats['authentication_seconds']+=time.perf_counter()-began
        value=self.descriptors[key];self.descriptor_fresh(value);return value

    def read_many(self,name,uids,*,memory_bytes=64<<20):
        began=time.perf_counter();self.fresh()
        need(name in self.config['roster'] and isinstance(uids,(list,tuple)) and len(uids)>0
             and all(type(i) is int and 0<=i<self.config['inputs'] for i in uids),'NATIVE_BATCH_REQUEST')
        need(type(memory_bytes) is int and memory_bytes>0,'NATIVE_BATCH_MEMORY_POLICY')
        first=self.descriptor(name,uids[0]//SHARD_INPUTS*SHARD_INPUTS)
        dimensions=first['report']['dimensions']
        need(len(uids)*(dimensions*4+8)+8<=memory_bytes,'NATIVE_BATCH_MEMORY_TILE_REQUIRED')
        requests=defaultdict(lambda:defaultdict(list));lengths=np.empty(len(uids),dtype='<i8')
        for position,uid in enumerate(uids):requests[uid//SHARD_INPUTS*SHARD_INPUTS][uid].append(position)
        descriptors={start:self.descriptor(name,start) for start in sorted(requests)}
        for start,group in requests.items():
            offsets=descriptors[start]['offsets']
            for uid,positions in group.items():lengths[positions]=offsets[uid-start+1]-offsets[uid-start]
        output_offsets=np.r_[np.int64(0),np.cumsum(lengths,dtype='<i8')]
        need(all(d['report']['dimensions']==dimensions for d in descriptors.values()),'NATIVE_BATCH_DIMENSIONS')
        output_bytes=int(output_offsets[-1])*dimensions*4+output_offsets.nbytes
        need(output_bytes<=memory_bytes,'NATIVE_BATCH_MEMORY_TILE_REQUIRED')
        values=np.empty((int(output_offsets[-1]),dimensions),dtype='<f4');traces=[None]*len(uids)
        for start,descriptor in descriptors.items():
            self.descriptor_fresh(descriptor);root=descriptor['root']
            vectors=np.load(root/'vectors.npy',mmap_mode='r',allow_pickle=False)
            table=None
            try:
                table=pq.ParquetFile(root/'traces.parquet')
                groups=defaultdict(list)
                for uid in requests[start]:groups[(uid-start)//IO_INPUTS].append(uid)
                for row_group,group_uids in sorted(groups.items()):
                    rows=table.read_row_group(row_group);self.stats['row_groups_read']+=1
                    for uid in group_uids:
                        record=rows.slice((uid-start)%IO_INPUTS,1).to_pylist()[0]
                        expected=self.source.arrays['hashes'][uid].tobytes().hex()
                        need(record['index']==uid and record['input_sha256']==expected,'NATIVE_BATCH_INPUT_BINDING')
                        trace_bytes=record['provenance'];trace=json.loads(trace_bytes)
                        lo,hi=map(int,descriptor['offsets'][uid-start:uid-start+2])
                        need(trace['surface_sha256']==expected and len(trace['windows'])==hi-lo,
                             'NATIVE_BATCH_COMPLETE_WINDOWS')
                        output_bytes+=len(trace_bytes)*len(requests[start][uid])
                        need(output_bytes<=memory_bytes,'NATIVE_BATCH_MEMORY_TILE_REQUIRED')
                        block=vectors[lo:hi]
                        need(bool(np.isfinite(block).all()),'NATIVE_BATCH_NONFINITE')
                        for position in requests[start][uid]:
                            a,b=map(int,output_offsets[position:position+2]);values[a:b]=block;traces[position]=trace_bytes
            finally:
                if table is not None:table.close()
                vectors._mmap.close()
            self.descriptor_fresh(descriptor)
        self.fresh();need(all(t is not None for t in traces),'NATIVE_BATCH_UNFILLED')
        values.flags.writeable=False;output_offsets.flags.writeable=False
        self.stats['calls']+=1;self.stats['returned_windows']+=len(values);self.stats['returned_bytes']+=output_bytes
        self.stats['reader_seconds']+=time.perf_counter()-began
        return NativeWindowBatch(name,self.config['instruments'][name]['lens_id'],tuple(uids),
                                 values,output_offsets,tuple(traces))

    def training_authority(self):
        raise ValueError('MNLM_NATIVE_BATCH_TRAINING_AUTHORITY_ABSENT')
