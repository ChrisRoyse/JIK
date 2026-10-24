"""Commit sparse updater generations through a canonical serialized boundary.

USearch 2.26.5 does not serialize its insertion RNG. Every successful commit
therefore replaces the live indices with fresh loads of the committed bytes.
Warm and cold successors start at the same boundary; unconstrained in-memory
continuation of older checkpoints is deliberately not claimed.
"""
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
from usearch.index import Index,MetricKind,ScalarKind

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.chunked_route import SparseUpdates
from kernel.repository_kernel import AssociationCompanion,RepositoryKernel,code_pins,lanes
from model.tool_kernel import PLAN


def producers():
    return dict(base=code_pins(),checkpoint=sha(Path(__file__).read_bytes()))


def read(path):return json.loads(Path(path).read_bytes())


def flush_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load_index(path,dimensions):
    index=Index(ndim=dimensions,metric='cos',dtype='f32',connectivity=PLAN['ann_connectivity'],
                expansion_add=PLAN['ann_expansion_add'],expansion_search=PLAN['ann_expansion_search'])
    index.load(path)
    check_index(index,dimensions)
    return index


def check_index(index,dimensions):
    check((index.ndim,index.metric,index.dtype,index.connectivity,index.expansion_add,index.expansion_search)==
          (dimensions,MetricKind.Cos,ScalarKind.F32,PLAN['ann_connectivity'],PLAN['ann_expansion_add'],
           PLAN['ann_expansion_search']),'UPDATE_STATE_INDEX_PLAN')


def validate(updater,lane,graph):
    check_index(updater.index,lane.unit.shape[1])
    check(updater.signatures==lane.signatures and set(updater.keys)==set(lane.ids),'UPDATE_STATE_ATOMS')
    keys=[k for atom in lane.ids for k in updater.keys[atom]]
    check(len(keys)==len(set(keys))==len(lane.unit) and sorted(keys)==sorted(updater.index.keys[:].tolist()),
          'UPDATE_STATE_INDEX_KEYS')
    check(type(updater.next_key) is int and updater.next_key>max(keys) and
          all(type(k) is int and 0<=k<updater.next_key for k in updater.owners),'UPDATE_STATE_KEY_CLOCK')
    for atom in lane.ids:
        ids=updater.keys[atom]
        check(len(ids)==len(lane.windows(atom)) and all(updater.owners.get(k)==atom for k in ids),'UPDATE_STATE_OWNER')
        actual=np.stack(updater.index.get(np.asarray(ids,np.uint64),dtype='f32'))
        check(np.array_equal(actual,lane.windows(atom).astype(np.float32)),'UPDATE_STATE_NATIVE_VECTOR')
    check(set(updater.previous_edges)==set(lane.ids) and
          [e for k in lane.ids for e in updater.previous_edges[k]]==graph['edges'],'UPDATE_STATE_EDGES')
    topology=nx.DiGraph();topology.add_nodes_from(lane.ids)
    topology.add_edges_from((a,b) for a,b,_ in graph['edges'])
    expected={}
    for component in nx.weakly_connected_components(topology):
        ids=sorted(component);edges=[e for k in ids for e in updater.previous_edges[k]]
        key=digest(dict(ids=ids,edges=edges,plan=PLAN))
        check(key in updater.components,'UPDATE_STATE_COMPONENT_KEY')
        expected[key]=updater.components[key]
    check(set(expected)==set(updater.components),'UPDATE_STATE_COMPONENTS')
    for name in ('candidate_graph','members','graph_only_support','entries','groundedness_gaps'):
        check(sorted(v for part in expected.values() for v in part[name])==graph[name],'UPDATE_STATE_COMPONENT_JOIN')
    check({k:v for part in expected.values() for k,v in part['scores'].items()}==graph['scores'],
          'UPDATE_STATE_COMPONENT_SCORES')
    check(all(isinstance(k,str) and len(k)==64 and np.isfinite(v) and abs(v)<=1.+1e-12
              for k,v in updater.cache.items()),'UPDATE_STATE_CACHE')
    return dict(atoms=len(lane.ids),windows=len(keys),vectors_equal=True,components=len(expected),
                cache_entries=len(updater.cache),next_key=updater.next_key)


def save(companion,root):
    root=Path(root)
    check(not root.exists() and companion.last is not None,'UPDATE_STATE_PATH')
    value=companion.last;kernel=companion.previous
    check(value['recall_passes'] and sha((Path(kernel['path'])/'manifest.json').read_bytes())==kernel['pin'],
          'UPDATE_STATE_KERNEL')
    companion.population.fresh()
    lane_map=lanes(companion.population,value['ids'])
    checks={n:validate(companion.updates[n],lane,value['graphs'][n]) for n,lane in lane_map.items()}
    cache=read(Path(kernel['path'])/'cache.json')
    check(cache=={n:u.cache for n,u in companion.updates.items()},'UPDATE_STATE_KERNEL_CACHE')
    root.mkdir();files={};canonical={}
    for name,u in sorted(companion.updates.items()):
        binary=u.index.save()
        check(isinstance(binary,(bytes,bytearray)) and binary,'UPDATE_STATE_SERIALIZATION')
        binary=bytes(binary)
        index_name=name+'.usearch';metadata_name=name+'.json'
        with (root/index_name).open('xb') as stream:
            stream.write(binary);stream.flush();os.fsync(stream.fileno())
        index=load_index(root/index_name,lane_map[name].unit.shape[1])
        check(bytes(index.save())==binary,'UPDATE_STATE_ROUNDTRIP')
        canonical[name]=index
        metadata=dict(instrument=u.instrument,plan=u.plan,keys=u.keys,
            owners=[[k,v] for k,v in sorted(u.owners.items())],signatures=u.signatures,
            next_key=u.next_key,components=u.components,previous_edges=u.previous_edges)
        write_committed(root/metadata_name,metadata)
        for path in (root/index_name,root/metadata_name):
            files[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path.read_bytes()))
    manifest=dict(schema='repository-updater-state-v2',continuation='canonical-reload-each-commit-v1',
        kernel=kernel,source=value['source'],
        code=producers(),files=files,checks=checks,excluded=companion.excluded,queries=companion.queries,
        cache_sha256=value['cache_sha256'],scientific_admission=False,serving=False)
    pin=write_committed(root/'manifest.json',manifest)
    flush_directory(root)
    write_committed(root/'COMMITTED',dict(manifest=pin))
    flush_directory(root);flush_directory(root.parent)
    for name,index in canonical.items():companion.updates[name].index=index
    return pin,checks


def restore(population,root,pin):
    root=Path(root)
    check((root/'manifest.json').is_file() and (root/'COMMITTED').is_file(),'UPDATE_STATE_UNCOMMITTED')
    raw=(root/'manifest.json').read_bytes()
    check(sha(raw)==pin and read(root/'COMMITTED')==dict(manifest=pin),'UPDATE_STATE_PIN')
    manifest=json.loads(raw)
    check(manifest['schema']=='repository-updater-state-v2' and
          manifest['continuation']=='canonical-reload-each-commit-v1' and manifest['code']==producers()
          and not manifest['scientific_admission'] and not manifest['serving'],'UPDATE_STATE_CONTRACT')
    kernel=manifest['kernel']
    accepted=RepositoryKernel(population,kernel['path'],kernel['pin'],expected_source=manifest['source'])
    value=accepted.value
    check(manifest['excluded']==value['excluded_tasks'] and manifest['queries']==value['queries']
          and manifest['cache_sha256']==value['cache_sha256'],'UPDATE_STATE_SOURCE')
    cache=read(Path(kernel['path'])/'cache.json')
    lane_map=accepted.lanes
    expected={n+ext for n in lane_map for ext in ('.json','.usearch')}
    check(set(manifest['files'])==expected,'UPDATE_STATE_FILES')
    for name,info in manifest['files'].items():
        path=root/name
        check(path.is_file() and path.stat().st_size==info['bytes'] and sha(path.read_bytes())==info['sha256'],'UPDATE_STATE_BYTES')
    companion=AssociationCompanion(population,manifest['excluded'],manifest['queries'])
    checks={}
    for name,lane in lane_map.items():
        m=read(root/(name+'.json'))
        check(m['instrument']==population.specs[name].identity and m['plan']==PLAN,'UPDATE_STATE_INSTRUMENT')
        u=SparseUpdates(m['instrument'],PLAN,cache[name])
        u.index=load_index(root/(name+'.usearch'),lane.unit.shape[1])
        u.keys,u.signatures=m['keys'],m['signatures']
        u.owners=dict(m['owners']);u.next_key=m['next_key']
        check(len(u.owners)==len(m['owners']),'UPDATE_STATE_DUPLICATE_OWNER')
        u.components,u.previous_edges=m['components'],m['previous_edges']
        checks[name]=validate(u,lane,value['graphs'][name])
        companion.updates[name]=u
    check(checks==manifest['checks'],'UPDATE_STATE_READBACK')
    companion.last=value;companion.previous=kernel
    return companion,checks
