"""Pinned, host-local read-session authentication for immutable native shards.

Publication hashes every shard through NativeBatchSession on a read-only ZFS
snapshot. Reuse binds its GUID/creation transaction and refuses writable backing.
Timestamps alone cannot prove immutability (phase 176's retained negative).
"""
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from claims.process_conditions import write_committed
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp
from text.legal_native_batch import NativeBatchSession,same_files
from text.legal_native_store import SHARD_INPUTS,shard_name

SCHEMA='legal-native-authenticated-catalogue-v2'
FILES={'report.json','COMMITTED','vectors.npy','offsets.npy','traces.parquet'}


def snapshot_binding(root):
    root=Path(root).resolve();prefix=Path('/zfs/hot/mnlm/.zfs/snapshot')
    need(root.is_relative_to(prefix),'NATIVE_CATALOGUE_SNAPSHOT_SCOPE')
    parts=root.relative_to(prefix).parts
    need(len(parts)>1 and re.fullmatch(r'native-catalogue-[0-9]{8}-v[0-9]+',parts[0]),
         'NATIVE_CATALOGUE_SNAPSHOT_SCOPE')
    need(os.statvfs(root).f_flag & os.ST_RDONLY,'NATIVE_CATALOGUE_WRITABLE')
    name='hotpool/mnlm@'+parts[0]
    raw=subprocess.check_output(['zfs','get','-Hp','-o','property,value','type,guid,createtxg',name],text=True)
    values=dict(line.split('\t') for line in raw.splitlines())
    need(values['type']=='snapshot' and values['guid'].isdigit() and values['createtxg'].isdigit(),
         'NATIVE_CATALOGUE_SNAPSHOT_IDENTITY')
    return dict(name=name,guid=values['guid'],createtxg=values['createtxg'],relative='/'.join(parts[1:]))


def binding(session):
    session.fresh()
    return dict(host=os.uname().nodename,root=str(session.root.resolve()),bank=session.pin,
                source=session.source.pin,configuration=list(session.config_stamp),
                source_commit=list(session.source_commit_stamp),snapshot=snapshot_binding(session.root))


def keys(session):
    return [(n,s) for n in session.config['roster']
            for s in range(0,session.config['inputs'],SHARD_INPUTS)]


def publish(session,path,*,event=None):
    before=binding(session);rows=[];total=0
    for name,start in keys(session):
        d=session.descriptor(name,start);session.descriptor_fresh(d)
        need({p.name for p in d['stamps']}==FILES,'NATIVE_CATALOGUE_FILE_SET')
        rows.append(dict(name=name,start=start,pin=d['pin'],report=d['report'],
                         identities={p.name:list(v) for p,v in d['stamps'].items()}))
        total+=sum(x['bytes'] for x in d['report']['files'].values())
        if event and len(rows)%64==0:event(len(rows),total)
    need(binding(session)==before,'NATIVE_CATALOGUE_BINDING_CHANGED')
    for d in session.descriptors.values():session.descriptor_fresh(d)
    body=dict(schema=SCHEMA,binding=before,complete=True,shards=rows,
              shard_count=len(rows),authenticated_bytes=total,
              identity_policy='readonly_zfs_guid_txg_plus_device_inode_size_mtime_ns_ctime_ns',
              retention_owner='ChrisRoyse/MNLM#29',acceptance='author-checked')
    pin=write_committed(path,body)
    need(file_sha(path)==pin and json.loads(path.read_bytes())==body,'NATIVE_CATALOGUE_PUBLICATION')
    return pin,body


def validate_binding(value,session):
    need(value['schema']==SCHEMA and value['complete'] and value['binding']==binding(session)
         and value['identity_policy']=='readonly_zfs_guid_txg_plus_device_inode_size_mtime_ns_ctime_ns',
         'NATIVE_CATALOGUE_BINDING')
    need([(r['name'],r['start']) for r in value['shards']]==keys(session)
         and value['shard_count']==len(value['shards']),'NATIVE_CATALOGUE_COVERAGE')


def identities(root,row):
    need(set(row['identities'])==FILES,'NATIVE_CATALOGUE_FILE_SET')
    result={root/n:tuple(v) for n,v in row['identities'].items()}
    need(all(len(v)==5 and all(type(x) is int for x in v) for v in result.values()),
         'NATIVE_CATALOGUE_IDENTITY_SCHEMA')
    need(all(not p.is_symlink() for p in result),'NATIVE_CATALOGUE_SYMLINK')
    same_files(result,'NATIVE_CATALOGUE_STALE')
    return result


def offset_values(path,report):
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    try:
        need(values.dtype==np.int64 and values.shape==(report['end']-report['start']+1,)
             and values[0]==0 and values[-1]==report['windows'] and bool((np.diff(values)>0).all()),
             'NATIVE_CATALOGUE_OFFSETS')
        return np.array(values,copy=True)
    finally:values._mmap.close()


def hydrate(session,path,pin):
    path=Path(path);before=stamp(path)
    need(not session.descriptors,'NATIVE_CATALOGUE_NONEMPTY_SESSION')
    need(file_sha(path)==pin,'NATIVE_CATALOGUE_PIN');value=json.loads(path.read_bytes())
    need(stamp(path)==before,'NATIVE_CATALOGUE_CHANGED');validate_binding(value,session)
    pending={};total=offset_bytes=0
    for row in value['shards']:
        name,start=row['name'],row['start'];end=min(start+SHARD_INPUTS,session.config['inputs'])
        root=session.root/name/shard_name(start,end);observed=identities(root,row)
        raw=(root/'report.json').read_bytes();report=json.loads(raw)
        # Small metadata is rehashed; the heavy payload was hashed at publication.
        need(file_sha(root/'report.json')==row['pin'] and report==row['report']
             and json.loads((root/'COMMITTED').read_bytes())==dict(report_sha256=row['pin']),
             'NATIVE_CATALOGUE_REPORT')
        need(report['configuration_pin']==session.pin and report['lens_id']==session.config['instruments'][name]['lens_id']
             and (report['start'],report['end'])==(start,end) and report['complete']
             and report['inputs']==end-start,'NATIVE_CATALOGUE_SHARD_BINDING')
        offsets=offset_values(root/'offsets.npy',report);same_files(observed,'NATIVE_CATALOGUE_STALE')
        pending[name,start]=dict(root=root,pin=row['pin'],report=report,offsets=offsets,stamps=observed)
        total+=sum(x['bytes'] for x in report['files'].values());offset_bytes+=offsets.nbytes
    need(total==value['authenticated_bytes'],'NATIVE_CATALOGUE_BYTE_COUNT')
    need(stamp(path)==before and binding(session)==value['binding'],'NATIVE_CATALOGUE_CHANGED')
    # Atomic in-process admission: no partial catalogue survives a refusal.
    for d in pending.values():session.descriptor_fresh(d)
    session.descriptors.update(pending)
    return dict(path=str(path),pin=pin,shards=len(pending),authenticated_bytes=total,
                offsets_loaded_bytes=offset_bytes,rehash_payload_bytes=0,
                snapshot=value['binding']['snapshot'],reuse='pinned_readonly_snapshot_not_new_hash_audit')


def snapshot_session(source,root):
    data=source['data'];old=data.native;snapshot_binding(root)
    need(not old.descriptors and source['reader'].stats['calls']==0 and not source['reader'].maps,
         'NATIVE_CATALOGUE_ALREADY_READING')
    session=NativeBatchSession(root,old.pin,data.source)
    need(session.config==old.config,'NATIVE_CATALOGUE_SNAPSHOT_CONFIGURATION')
    return session


def attach(source,path,pin):
    path=Path(path);need(file_sha(path)==pin,'NATIVE_CATALOGUE_PIN');value=json.loads(path.read_bytes())
    session=snapshot_session(source,value['binding']['root']);receipt=hydrate(session,path,pin)
    source['data'].native=session
    return receipt
