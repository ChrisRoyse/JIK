"""Bounded, pinned tensor shards for the larger correction and its AdamW moments."""
from copy import deepcopy
import json
import os
from pathlib import Path

import torch
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.protected_surface import check
from text.constellation_corpus import file_sha,need


def save(root,model,optimizer,*,step,binding):
    check(model,optimizer,step);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    tensors=[]
    def encode(v):
        if isinstance(v,torch.Tensor):
            need(v.numel()*v.element_size()<=model.config.checkpoint_bytes//2,'PROTECTED_STATE_TENSOR_TILE')
            i=len(tensors);tensors.append(v.detach().cpu());return ['tensor',i]
        if isinstance(v,dict):return ['dict',[[encode(k),encode(x)] for k,x in v.items()]]
        if isinstance(v,(list,tuple)):return [type(v).__name__,[encode(x) for x in v]]
        need(type(v) in (int,float,str,bool,type(None)),'PROTECTED_STATE_TYPE');return ['scalar',v]
    payload=dict(correction=model.correction.state_dict(),optimizer=optimizer.state_dict())
    tree=encode(payload);refs=[];start=0
    while start<len(tensors):
        stop=start;size=0
        while stop<len(tensors) and size+tensors[stop].numel()*tensors[stop].element_size()<=model.config.checkpoint_bytes//2:
            size+=tensors[stop].numel()*tensors[stop].element_size();stop+=1
        need(stop>start,'PROTECTED_STATE_PROGRESS');path=root/f'tensors-{start:05d}-{stop:05d}.pt'
        with path.open('xb') as f:torch.save(tensors[start:stop],f);f.flush();os.fsync(f.fileno())
        need(path.stat().st_size<=model.config.checkpoint_bytes,'PROTECTED_STATE_SHARD_BYTES')
        refs.append(dict(name=path.name,start=start,stop=stop,bytes=path.stat().st_size,sha256=file_sha(path)))
        start=stop
    body=dict(schema='protected-surface-state-v1',binding=binding,step=step,parent=model.parent_pin,tree=tree,shards=refs,
              state=fingerprint(payload),tensor_count=len(tensors),max_shard_bytes=model.config.checkpoint_bytes)
    pin=write_committed(root/'manifest.json',body)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(root=str(root),pin=pin,state=body['state'],step=step)


def read(root,pin,*,binding,parent):
    root=Path(root);need((root/'manifest.json').is_file() and file_sha(root/'manifest.json')==pin,'PROTECTED_STATE_MANIFEST_PIN')
    m=json.loads((root/'manifest.json').read_bytes())
    need(m['schema']=='protected-surface-state-v1' and m['binding']==binding and m['parent']==parent,'PROTECTED_STATE_BINDING')
    need(m['max_shard_bytes']==64<<20,'PROTECTED_STATE_SHARD_POLICY')
    tensors=[]
    for r in m['shards']:
        name=r['name'];need(name==Path(name).name and not (root/name).is_symlink(),'PROTECTED_STATE_PATH')
        path=root/name;need(path.is_file() and path.stat().st_size==r['bytes']<=m['max_shard_bytes'] and file_sha(path)==r['sha256'],
                            'PROTECTED_STATE_SHARD_PIN')
        values=torch.load(path,map_location='cpu',weights_only=True)
        need(r['start']==len(tensors) and r['stop']-r['start']==len(values)
             and all(isinstance(v,torch.Tensor) and bool(v.isfinite().all()) for v in values),'PROTECTED_STATE_TENSORS')
        tensors.extend(values)
    need(len(tensors)==m['tensor_count'],'PROTECTED_STATE_COVERAGE');used=set()
    def decode(node):
        kind,v=node
        if kind=='tensor':
            need(type(v) is int and 0<=v<len(tensors) and v not in used,'PROTECTED_STATE_REFERENCE');used.add(v);return tensors[v]
        if kind=='dict':
            pairs=[(decode(k),decode(x)) for k,x in v];need(len(dict(pairs))==len(pairs),'PROTECTED_STATE_KEYS');return dict(pairs)
        if kind in ('list','tuple'):
            values=[decode(x) for x in v];return values if kind=='list' else tuple(values)
        need(kind=='scalar' and type(v) in (int,float,str,bool,type(None)),'PROTECTED_STATE_NODE');return v
    payload=decode(m['tree'])
    need(len(used)==len(tensors) and fingerprint(payload)==m['state'],'PROTECTED_STATE_PAYLOAD')
    return payload,m


def restore(ref,model,optimizer,*,binding):
    payload,m=read(ref['root'],ref['pin'],binding=binding,parent=model.parent_pin)
    need(m['state']==ref['state'] and m['step']==ref['step'],'PROTECTED_STATE_REFERENCE_BINDING')
    model.correction.load_state_dict(payload['correction'],strict=True)
    optimizer.load_state_dict(deepcopy(payload['optimizer']));check(model,optimizer,m['step'])
    need(fingerprint(dict(correction=model.correction.state_dict(),optimizer=optimizer.state_dict()))==m['state'],'PROTECTED_STATE_RESTORED')
    return m['step']
