"""Exact byte-addressed surfaces; UIDs and split metadata are joins, never features."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp

SCHEMA='legal-surface-bytes-v1'
TILE=128
OUTPUT_BYTES=16<<20


class SurfaceBytes:
    def __init__(self, root, pin, source):
        self.root=Path(root); self.source=source; self.closed=False; self.fd=None; self.offsets=None
        try:
            path=self.root/'report.json'; need(file_sha(path)==pin,'SURFACE_BYTES_REPORT_PIN')
            r=json.loads(path.read_bytes()); self.report=r; self.pin=pin
            need(r['schema']==SCHEMA and r['complete'] and r['source']==source.pin,'SURFACE_BYTES_SOURCE')
            need(json.loads((self.root/'COMMITTED').read_bytes())==dict(report_sha256=pin),'SURFACE_BYTES_COMMIT')
            self.stamps={path:stamp(path),self.root/'COMMITTED':stamp(self.root/'COMMITTED')}
            for name,expected in r['files'].items():
                p=self.root/name; need(file_sha(p)==expected,'SURFACE_BYTES_PHYSICAL')
                self.stamps[p]=stamp(p)
            self.offsets=np.load(self.root/'offsets.npy',mmap_mode='r',allow_pickle=False)
            self.size=int(r['inputs']); self.payload=int(r['bytes'])
            need(self.offsets.dtype==np.dtype('<u8') and self.offsets.shape==(self.size+1,)
                 and int(self.offsets[0])==0 and int(self.offsets[-1])==self.payload
                 and (self.root/'surfaces.bin').stat().st_size==self.payload,'SURFACE_BYTES_LAYOUT')
            for lo in range(0,self.size,16384):
                need(bool((self.offsets[lo+1:min(lo+16384,self.size)+1]>=self.offsets[lo:min(lo+16384,self.size)]).all()),
                     'SURFACE_BYTES_OFFSETS')
            self.fd=os.open(self.root/'surfaces.bin',os.O_RDONLY)
            self.stats=dict(calls=0,requests=0,bytes=0,preads=0)
            self.fresh()
        except BaseException:
            self._release(); raise

    def fresh(self):
        need(not self.closed,'SURFACE_BYTES_CLOSED'); self.source.fresh()
        need(all(stamp(p)==s for p,s in self.stamps.items()),'SURFACE_BYTES_CHANGED')
        if self.fd is not None:
            s=os.fstat(self.fd)
            need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==self.stamps[self.root/'surfaces.bin'],
                 'SURFACE_BYTES_OPEN_IDENTITY')

    def read(self, uids):
        self.fresh()
        need(isinstance(uids,(tuple,list)) and 0<len(uids)<=TILE
             and all(type(u) is int and 0<=u<self.size for u in uids),'SURFACE_BYTES_UIDS')
        intervals=[tuple(map(int,self.offsets[u:u+2])) for u in uids]
        need(sum(b-a for a,b in intervals)<=OUTPUT_BYTES,'SURFACE_BYTES_MEMORY_TILE')
        result=[]
        for uid,(lo,hi) in zip(uids,intervals):
            raw=os.pread(self.fd,hi-lo,lo); self.stats['preads']+=1
            need(len(raw)==hi-lo,'SURFACE_BYTES_SHORT_READ')
            need(hashlib.sha256(raw).digest()==self.source.arrays['hashes'][uid].tobytes(),'SURFACE_BYTES_HASH')
            raw.decode('utf-8','strict'); result.append(raw)
        self.fresh(); self.stats['calls']+=1; self.stats['requests']+=len(uids)
        self.stats['bytes']+=sum(map(len,result)); return tuple(result)

    def occurrences(self, data, indices, *, side, fit=False):
        need(data.source.pin==self.source.pin,'SURFACE_BYTES_OCCURRENCE_SOURCE')
        values=data.indices(indices,side,fit=fit)  # Split eligibility precedes surface/native access.
        return self.read([int(u) for u in data.mapping[values]])

    def _release(self):
        if self.fd is not None: os.close(self.fd); self.fd=None
        if self.offsets is not None: self.offsets._mmap.close(); self.offsets=None
        self.closed=True

    def close(self):
        try: self.fresh()
        finally: self._release()

    def training_authority(self):
        raise ValueError('MNLM_SURFACE_BYTES_SCIENTIFIC_ADMISSION_REQUIRED')
