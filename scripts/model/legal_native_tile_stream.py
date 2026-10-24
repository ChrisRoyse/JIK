"""Consolidate overlapping descriptor checks around one certified native tile.

Every descriptor file is checked before/after use; retained vector FDs are also
checked before/after reading. Values, offsets and copying match the reference.
One thread owns this reader. A failed in-flight read closes all its handles.
"""
import os
from threading import get_ident
import time

import numpy as np
import torch
from model.legal_native_resident_stream import NativeFileView, ResidentNativeFrames
from text.constellation_corpus import need
from text.legal_native_store import SHARD_INPUTS


class TileNativeFrames(ResidentNativeFrames):
    def __init__(self,data,audit_root,audit_pin):
        super().__init__(data,audit_root,audit_pin,mode='pread',handle_limit=64)
        self.owner=get_ident();self.active=None

    def _view(self,name,start,descriptor):
        key=(name,start);path=descriptor['root']/'vectors.npy'
        need(self.active is not None and self.active.get(key) is descriptor,'TILE_NATIVE_UNCHECKED_DESCRIPTOR')
        if key in self.maps:
            self.io_stats['hits']+=1;self.maps.move_to_end(key);return self.maps[key]
        while len(self.maps)>=self.handle_limit:
            _,old=self.maps.popitem(last=False);old.close();self.io_stats['evictions']+=1
        if key not in self.layouts:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                r=descriptor['report']
                need(value.dtype==np.float32 and value.shape==(r['windows'],r['dimensions'])
                     and value.flags.c_contiguous,'TRAIN_STREAM_VECTOR_LAYOUT')
                self.layouts[key]=(value.offset,value.shape)
            finally:value._mmap.close()
        value=NativeFileView(path,descriptor,self.layouts[key]);self.maps[key]=value
        self.io_stats['opens']+=1;self.io_stats['peak_handles']=max(self.io_stats['peak_handles'],len(self.maps))
        return value

    @staticmethod
    def _fd_fresh(value,descriptor):
        need(value.fd is not None,'RESIDENT_NATIVE_CLOSED')
        s=os.fstat(value.fd)
        need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
             ==descriptor['stamps'][descriptor['root']/'vectors.npy'],'TILE_NATIVE_FD_CHANGED')

    def frames(self,uids):
        need(not self.closed,'RESIDENT_NATIVE_CLOSED')
        need(get_ident()==self.owner and self.active is None,'TILE_NATIVE_OWNER')
        need(isinstance(uids,(list,tuple)) and 0<len(uids)<=128
             and all(type(u) is int and 0<=u<self.data.native.config['inputs'] for u in uids),'TRAIN_STREAM_UID_SCOPE')
        began=time.perf_counter();self.data.fresh();groups={}
        for i,uid in enumerate(uids):groups.setdefault(uid//SHARD_INPUTS*SHARD_INPUTS,[]).append((i,uid))
        self.active={};payload=windows=0
        try:
            for name in self.roster:
                for start,rows in sorted(groups.items()):
                    # descriptor() checks every file and its exhaustive source certificate.
                    d=self.descriptor(name,start);self.active[name,start]=d
                    for _,uid in rows:
                        a,b=map(int,d['offsets'][uid-start:uid-start+2])
                        need(b>a,'TRAIN_STREAM_MISSING_WINDOW');windows+=b-a
                        payload+=(b-a)*d['report']['dimensions']*4
            need(payload<=16<<20,'TRAIN_STREAM_PHYSICAL_MEMORY')
            tick=time.perf_counter();frames=[{} for _ in uids]
            for (name,start),d in self.active.items():
                value=self._view(name,start,d);self._fd_fresh(value,d)
                for i,uid in groups[start]:
                    a,b=map(int,d['offsets'][uid-start:uid-start+2]);v=np.array(value[a:b],copy=True)
                    need(np.isfinite(v).all(),'TRAIN_STREAM_NONFINITE');frames[i][name]=torch.from_numpy(v)
                self._fd_fresh(value,d);self.data.native.descriptor_fresh(d)
            self.data.fresh();self.stats['gather_seconds']+=time.perf_counter()-tick
            self.stats['calls']+=1;self.stats['inputs']+=len(uids);self.stats['windows']+=windows
            self.stats['payload_bytes']+=payload;self.stats['maximum_payload_bytes']=max(payload,self.stats['maximum_payload_bytes'])
            self.stats['reader_seconds']+=time.perf_counter()-began
            return tuple(frames)
        except BaseException:
            for value in self.maps.values():value.close()
            self.maps.clear();self.closed=True
            raise
        finally:self.active=None
