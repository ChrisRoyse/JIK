"""Retain a finite journal suffix and check deterministic replay before commit."""
import hashlib
import json
import os
from pathlib import Path

from senses.contract import canonical,digest


def need(ok,code):
    if not ok:raise ValueError('PLACEMENT_TAIL_'+code)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def scan(stream,size,start,census,binding,*,destination=None,torn=None):
    """Exactly size bytes; appends beyond this observed boundary are not read."""
    need(type(size) is int and size>=0,'SIZE')
    cursor=start['step'];chain=start['chain'];model=start['model'];optimizer=start['optimizer']
    remaining=size;h=hashlib.sha256();nbytes=0;unfinished=b''
    while remaining:
        raw=stream.readline(remaining);need(bool(raw),'TRUNCATED_SOURCE');remaining-=len(raw)
        if not raw.endswith(b'\n'):
            need(remaining==0,'TORN_NOT_LAST');unfinished=raw
            if torn is not None:torn.write(raw)
            break
        need(cursor<len(census.documents),'EXTRA_RECORD')
        row=json.loads(raw);body={k:v for k,v in row.items() if k!='hash'}
        need(set(row)=={'parent','cursor','update','hash'} and raw==canonical(row)+b'\n'
             and row['cursor']==cursor and row['parent']==chain and row['hash']==digest(body),'CHAIN')
        update=row['update']
        need(update['step']==cursor+1 and update['document']==census.documents[cursor]
             and update['targets']==int(census.targets[cursor]) and update['before']==model
             and update['noise_seed']==binding['policy']['noise_seed']+cursor,'UPDATE')
        chain,model,optimizer=row['hash'],update['after'],update['optimizer'];cursor+=1
        h.update(raw);nbytes+=len(raw)
        if destination is not None:destination.write(raw)
    return dict(schema='placement-retained-tail-v1',start=start,
        end=dict(**census.counts(cursor),chain=chain,model=model,optimizer=optimizer),
        records=cursor-start['step'],bytes=nbytes,sha256=h.hexdigest(),
        torn=dict(bytes=len(unfinished),sha256=hashlib.sha256(unfinished).hexdigest()),
        source_bytes=start['prefix_bytes']+size)


def seal(source,out,start,census,binding,source_bytes):
    out=Path(out);need(source_bytes>=start['prefix_bytes'],'SOURCE_BEFORE_CHECKPOINT')
    with Path(source).open('rb') as src, (out/'retained-tail.jsonl').open('xb') as dst, (out/'uncommitted-tail.bin').open('xb') as torn:
        need(os.fstat(src.fileno()).st_size>=source_bytes,'SOURCE_SHRANK');src.seek(start['prefix_bytes'])
        result=scan(src,source_bytes-start['prefix_bytes'],start,census,binding,destination=dst,torn=torn)
        for stream in (dst,torn):stream.flush();os.fsync(stream.fileno())
    validate(out,result,start,census,binding);return result


def validate(root,value,start,census,binding):
    need(value['schema']=='placement-retained-tail-v1' and value['start']==start,'START')
    path=Path(root)/'retained-tail.jsonl';torn=Path(root)/'uncommitted-tail.bin'
    need(path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],'COMMITTED_BYTES')
    need(torn.stat().st_size==value['torn']['bytes'] and sha(torn)==value['torn']['sha256'],'UNCOMMITTED_BYTES')
    with torn.open('rb') as stream:
        while block:=stream.read(1<<20):need(b'\n' not in block,'UNCOMMITTED_NEWLINE')
    with path.open('rb') as stream:actual=scan(stream,value['bytes'],start,census,binding)
    need(actual['end']==value['end'] and actual['records']==value['records']
         and actual['sha256']==value['sha256'] and actual['torn']['bytes']==0
         and value['source_bytes']==start['prefix_bytes']+value['bytes']+value['torn']['bytes'],'RECOUNT')
    return actual


class Replay:
    def __init__(self,path,value,cursor):
        need(value['start']['step']==cursor,'REPLAY_CURSOR')
        self.value=value;self.cursor=cursor;self.h=hashlib.sha256();self.bytes=0;self.closed=False
        self.stream=Path(path).open('rb')
        try:
            need(os.fstat(self.stream.fileno()).st_size==value['bytes']
                 and hashlib.file_digest(self.stream,'sha256').hexdigest()==value['sha256'],'REPLAY_SOURCE')
            self.stream.seek(0)
            if self.complete:self.finish()
        except BaseException:self.stream.close();raise

    @property
    def complete(self):return self.cursor==self.value['end']['step']

    def check(self,raw,cursor):
        need(not self.closed and cursor==self.cursor and not self.complete,'REPLAY_ORDER')
        expected=self.stream.readline()
        need(expected.endswith(b'\n') and raw==expected,'REPLAY_MISMATCH')
        self.h.update(raw);self.bytes+=len(raw);self.cursor+=1
        if self.complete:self.finish()

    def finish(self):
        need(self.complete,'REPLAY_INCOMPLETE')
        if not self.closed:
            need(not self.stream.read(1) and self.bytes==self.value['bytes']
                 and self.h.hexdigest()==self.value['sha256'],'REPLAY_END')
            self.stream.close();self.closed=True
        return dict(complete=True,start=self.value['start']['step'],end=self.cursor,
                    records=self.value['records'],bytes=self.bytes,sha256=self.h.hexdigest(),byte_exact=True)

    def __enter__(self):return self
    def __exit__(self,*exc):self.stream.close()
