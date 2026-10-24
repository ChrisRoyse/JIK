"""Publish a bounded real native index while retaining its borrowed matrix through save."""
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from usearch.index import Index

from claims.process_conditions import write_committed
from kernel.native_window_index import INDEX_POLICY,POLICY,execution_policy,unit,window_keys
from text.constellation_corpus import file_sha,need


def publish(root,uids,read,binding,*,memory_bytes=16<<20):
    start=time.perf_counter();cpu=time.process_time();execution=execution_policy(1,memory_bytes)
    need(importlib.metadata.version('usearch')==POLICY['usearch_version'],'BOUNDED_INDEX_VERSION')
    need(uids.dtype==np.int64 and uids.ndim==1 and POLICY['top_k']<len(uids)<=512
         and np.all(uids>=0) and np.all(uids<2**32) and np.all(uids[1:]>uids[:-1]),'BOUNDED_INDEX_SCOPE')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    rows=read(uids.tolist());need(len(rows)==len(uids),'BOUNDED_INDEX_READ_COUNT')
    for row in rows:unit(row)
    need(len({r.shape[1] for r in rows})==1,'BOUNDED_INDEX_DIMENSIONS')
    lengths=np.asarray([len(r) for r in rows],dtype='<i8');keys=window_keys(uids.tolist(),rows)
    values=np.concatenate(rows);arrays=values.nbytes+keys.nbytes+uids.nbytes+lengths.nbytes
    capacity=1<<(len(keys)-1).bit_length()
    # Account for the measured graph arena floor as well as the retained matrix.
    projected=(8<<20)+capacity*2048+arrays+3*values.nbytes
    need(projected<=memory_bytes,'BOUNDED_INDEX_MEMORY_PROJECTION')
    index=Index(ndim=values.shape[1],metric='cos',dtype='f32',connectivity=POLICY['connectivity'],
                expansion_add=POLICY['expansion_add'],expansion_search=POLICY['expansion_search'])
    before=values.tobytes();added=index.add(keys,values,copy=False,threads=1)
    need(np.array_equal(added,keys),'BOUNDED_INDEX_ADD_KEYS')
    # `values` is intentionally retained until after save and index disposal below.
    upper=index.memory_usage+arrays+3*values.nbytes
    need(upper<=memory_bytes,'BOUNDED_INDEX_MEMORY_REQUIRED')
    restored=np.asarray(index.get(keys,dtype='f32'))
    actual=index.memory_usage+arrays+len(before)+restored.nbytes
    need(np.array_equal(restored,values),'BOUNDED_INDEX_NATIVE_EXACT')
    index.save(root/'windows.usearch')
    need(values.tobytes()==before,'BOUNDED_INDEX_SOURCE_CHANGED')
    metadata=dict(inputs=len(uids),windows=len(keys),maximum_windows=int(lengths.max()),dimensions=index.ndim,
                  index_memory_bytes=index.memory_usage,retained_build_bytes=actual,
                  physical_build_upper_bytes=upper,projected_build_bytes=projected)
    del index
    # No borrowed index survives the lifetime of its contiguous source matrix.
    np.save(root/'uids.npy',uids,allow_pickle=False);np.save(root/'lengths.npy',lengths,allow_pickle=False)
    files={f.name:dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in root.iterdir() if f.is_file()}
    seconds=time.perf_counter()-start
    result=dict(schema='typed-native-window-index-v2',complete=True,policy=INDEX_POLICY,execution=execution,
        binding=binding,files=files,**metadata,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        phases_seconds={'bounded_read_build_save':seconds},source_vectors_borrowed_during_build=True,
        source_matrix_retained_through_save=True,borrowed_index_disposed_before_source_release=True,
        scientific_admission=False,recall_qualified=False,number_status='measured',acceptance='author-checked',
        code_sha256=file_sha(Path(__file__)))
    pin=write_committed(root/'report.json',result);write_committed(root/'COMMITTED',dict(report_sha256=pin))
    return pin
