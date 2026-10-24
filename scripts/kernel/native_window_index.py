"""One immutable instrument index; every native window remains independently addressed."""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from usearch.index import Index
from claims.process_conditions import write_committed
from text.constellation_corpus import digest,file_sha,need
from text.constellation_input import stamp

POLICY=dict(usearch_version='2.26.5',connectivity=32,expansion_add=256,expansion_search=256,
            window_candidates=128,top_k=10,physical_inputs=512,threads=1,
            metric='minimum_directed_mean_maxsim',native_dtype='float32',index_dtype='f32')
INDEX_POLICY={**{k:v for k,v in POLICY.items() if k!='threads'},'query_threads':1}
INDEX_MEMORY_BYTES=48<<30


def unit(values):
    need(values.dtype==np.float32 and values.ndim==2 and len(values)>0
         and np.isfinite(values).all(),'WINDOW_INDEX_NATIVE_ARRAY')
    v=values.astype(np.float64);norm=np.sqrt(np.einsum('ij,ij->i',v,v))
    need(bool((norm>0).all()),'WINDOW_INDEX_ZERO_NATIVE')
    return v/norm[:,None]


def score_table(queries,rows):
    """Tiled complete-window scalar scores; never averages native coordinates."""
    return score_units([unit(x) for x in queries],[unit(x) for x in rows])


def score_units(q,r):
    need(q and r and len({x.shape[1] for x in q+r})==1,'WINDOW_INDEX_DIMENSIONS')
    out=np.empty((len(q),len(r)),dtype='<f8')
    single_q=[i for i,x in enumerate(q) if len(x)==1];single_r=[i for i,x in enumerate(r) if len(x)==1]
    if single_q and single_r:
        a=np.stack([q[i][0] for i in single_q]);b=np.stack([r[i][0] for i in single_r])
        out[np.ix_(single_q,single_r)]=np.clip(np.einsum('id,jd->ij',a,b,optimize=False),-1,1)
    multiple_r=[j for j,x in enumerate(r) if len(x)>1]
    for i,a in enumerate(q):
        for j in (range(len(r)) if len(a)>1 else multiple_r):
            b=r[j]
            matrix=np.clip(np.einsum('id,jd->ij',a,b,optimize=False),-1,1)
            out[i,j]=min(matrix.max(0).mean(),matrix.max(1).mean())
    return out


def window_keys(uids,rows):
    need(len(uids)==len(rows)>0 and all(type(i) is int and 0<=i<2**32 for i in uids)
         and all(0<len(x)<2**32 for x in rows),'WINDOW_INDEX_KEY_RANGE')
    return np.concatenate([(np.uint64(uid)<<np.uint64(32))+np.arange(len(x),dtype='<u8')
                           for uid,x in zip(uids,rows)])


def execution_policy(build_threads,memory_bytes):
    need(type(build_threads) is int and 1<=build_threads<=32,'WINDOW_INDEX_BUILD_THREADS')
    need(type(memory_bytes) is int and 0<memory_bytes<=INDEX_MEMORY_BYTES,'WINDOW_INDEX_MEMORY_POLICY')
    return dict(build_threads=build_threads,index_memory_bytes=memory_bytes)


def publish(root,uids,read,binding,*,build_threads=1,memory_bytes=INDEX_MEMORY_BYTES,progress=None):
    execution=execution_policy(build_threads,memory_bytes)
    need(importlib.metadata.version('usearch')==POLICY['usearch_version'],'WINDOW_INDEX_VERSION')
    need(uids.dtype==np.int64 and uids.ndim==1 and len(uids)>POLICY['top_k']
         and np.all(uids>=0) and np.all(uids<2**32) and np.all(uids[1:]>uids[:-1]),'WINDOW_INDEX_UIDS')
    need(2*uids.nbytes<=memory_bytes,'WINDOW_INDEX_MEMORY_REQUIRED')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    lengths=np.empty(len(uids),dtype='<i8');index=None;phases=dict(native_read=0.,packing=0.,ann_add=0.,progress=0.)
    for lo in range(0,len(uids),POLICY['physical_inputs']):
        tick=time.perf_counter();ids=uids[lo:lo+POLICY['physical_inputs']].tolist();rows=read(ids)
        phases['native_read']+=time.perf_counter()-tick;tick=time.perf_counter()
        for row in rows:unit(row)
        need(len(rows)==len(ids),'WINDOW_INDEX_READ_COUNT')
        if index is None:
            index=Index(ndim=rows[0].shape[1],metric='cos',dtype='f32',
                        connectivity=POLICY['connectivity'],expansion_add=POLICY['expansion_add'],
                        expansion_search=POLICY['expansion_search'])
        need(all(x.shape[1]==index.ndim for x in rows),'WINDOW_INDEX_DIMENSIONS')
        keys=window_keys(ids,rows);values=np.concatenate(rows,axis=0)
        lengths[lo:lo+len(ids)]=[len(x) for x in rows]
        # Project the next capacity tier before insertion; also check the library's actual allocation afterward.
        capacity=max(int(index.capacity),1<<(int(len(index)+len(keys))-1).bit_length())
        projected=capacity*(index.ndim*4+2048)+2*uids.nbytes+values.nbytes+keys.nbytes
        need(projected<=memory_bytes,'WINDOW_INDEX_MEMORY_PROJECTION')
        phases['packing']+=time.perf_counter()-tick;tick=time.perf_counter()
        added=index.add(keys,values,threads=build_threads)
        need(np.array_equal(added,keys),'WINDOW_INDEX_ADD_KEYS')
        need(index.memory_usage+2*uids.nbytes+values.nbytes+keys.nbytes<=memory_bytes,'WINDOW_INDEX_MEMORY_REQUIRED')
        phases['ann_add']+=time.perf_counter()-tick
        hi=lo+len(ids)
        if progress is not None and (hi%16384==0 or hi==len(uids)):
            tick=time.perf_counter();progress(dict(inputs=hi,windows=len(index),memory_bytes=index.memory_usage,
                seconds=time.perf_counter()-start,phases_seconds=dict(phases)))
            phases['progress']+=time.perf_counter()-tick
    tick=time.perf_counter();need(len(index)==int(lengths.sum()),'WINDOW_INDEX_ADD_COUNT')
    index.save(root/'windows.usearch');np.save(root/'uids.npy',uids,allow_pickle=False)
    np.save(root/'lengths.npy',lengths,allow_pickle=False)
    files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(root.iterdir()) if p.is_file()}
    phases['save_and_hash']=time.perf_counter()-tick;seconds=time.perf_counter()-start
    need(sum(phases.values())/seconds>=.9,'WINDOW_INDEX_PHASE_ATTRIBUTION')
    result=dict(schema='typed-native-window-index-v2',complete=True,policy=INDEX_POLICY,execution=execution,
        binding=binding,files=files,
        inputs=len(uids),windows=len(index),maximum_windows=int(lengths.max()),dimensions=index.ndim,
        index_memory_bytes=index.memory_usage,phases_seconds=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu,
        scientific_admission=False,recall_qualified=False,number_status='measured',acceptance='author-checked',
        code_sha256=file_sha(Path(__file__)))
    pin=write_committed(root/'report.json',result);write_committed(root/'COMMITTED',dict(report_sha256=pin))
    return pin


class NativeWindowIndex:
    def __init__(self,root,pin,binding):
        self.root=Path(root);self.pin=pin;raw=(self.root/'report.json').read_bytes()
        need(digest(raw)==pin,'WINDOW_INDEX_REPORT_PIN')
        need(json.loads((self.root/'COMMITTED').read_bytes())==dict(report_sha256=pin),'WINDOW_INDEX_COMMIT')
        self.report=r=json.loads(raw)
        legacy=r['schema']=='typed-native-window-index-v1'
        need((legacy or r['schema']=='typed-native-window-index-v2') and r['complete']
             and r['policy']==(POLICY if legacy else INDEX_POLICY) and r['binding']==binding,'WINDOW_INDEX_BINDING')
        if not legacy:
            execution=r['execution'];execution_policy(execution['build_threads'],execution['index_memory_bytes'])
            need(r['index_memory_bytes']<=execution['index_memory_bytes'],'WINDOW_INDEX_MEMORY_REQUIRED')
        need(importlib.metadata.version('usearch')==POLICY['usearch_version'],'WINDOW_INDEX_VERSION')
        need(set(r['files'])=={'uids.npy','lengths.npy','windows.usearch'},'WINDOW_INDEX_FILES')
        self.stamps={self.root/n:stamp(self.root/n) for n in ('report.json','COMMITTED')}
        for name,ref in r['files'].items():
            p=self.root/name;need(p.stat().st_size==ref['bytes'] and file_sha(p)==ref['sha256'],'WINDOW_INDEX_FILE_PIN')
            self.stamps[p]=stamp(p)
        self.uids=np.load(self.root/'uids.npy',allow_pickle=False);self.lengths=np.load(self.root/'lengths.npy',allow_pickle=False)
        need(self.uids.dtype==self.lengths.dtype==np.int64 and self.uids.shape==self.lengths.shape==(r['inputs'],)
             and np.all(self.uids>=0) and np.all(self.uids<2**32) and np.all(self.uids[1:]>self.uids[:-1])
             and np.all(self.lengths>0) and np.all(self.lengths<2**32) and int(self.lengths.sum())==r['windows'],
             'WINDOW_INDEX_LAYOUT')
        self.uids.flags.writeable=False;self.lengths.flags.writeable=False
        self.index=Index.restore(self.root/'windows.usearch',view=True,expansion_search=POLICY['expansion_search'])
        need(self.index is not None and len(self.index)==r['windows'] and self.index.ndim==r['dimensions'],
             'WINDOW_INDEX_COLD_COUNT')

    def fresh(self):
        try:unchanged=all(stamp(p)==s for p,s in self.stamps.items())
        except OSError:unchanged=False
        need(unchanged,'WINDOW_INDEX_CHANGED')

    def query(self,uid,values,read,*,lens_id):
        self.fresh();need(lens_id==self.report['binding']['lens_id'],'WINDOW_INDEX_WRONG_LENS')
        unit(values);need(values.shape[1]==self.report['dimensions'],'WINDOW_INDEX_QUERY_DIMENSIONS')
        expected=read([uid])
        need(len(expected)==1 and np.array_equal(values,expected[0]),'WINDOW_INDEX_QUERY_SOURCE')
        candidates=set();work=0
        for window in values:
            hits=self.index.search(window,count=min(POLICY['window_candidates'],len(self.index)),threads=1)
            work+=hits.computed_distances
            keys=np.asarray(hits.keys,dtype=np.uint64);owners=(keys>>np.uint64(32)).astype('<i8')
            positions=np.searchsorted(self.uids,owners)
            need(np.all(positions<len(self.uids)) and np.array_equal(self.uids[positions],owners),'WINDOW_INDEX_UNKNOWN_OWNER')
            need(np.all((keys&np.uint64(2**32-1))<self.lengths[positions]),'WINDOW_INDEX_UNKNOWN_WINDOW')
            candidates.update(map(int,owners))
        candidates.discard(uid);ids=sorted(candidates)
        need(len(ids)>=POLICY['top_k'],'WINDOW_INDEX_CANDIDATE_SCARCITY')
        scores=[]
        for lo in range(0,len(ids),POLICY['physical_inputs']):
            part=ids[lo:lo+POLICY['physical_inputs']]
            scores.extend(score_table([values],read(part))[0].tolist())
        ranked=sorted(zip(ids,scores),key=lambda x:(-x[1],x[0]))[:POLICY['top_k']];self.fresh()
        return dict(uid=uid,hits=[[u,s] for u,s in ranked],candidate_uids=ids,ann_distance_computations=int(work),
                    exhaustive=False,relation='same_instrument_native_window_neighbour')


def exact_queries(uids,queries,read):
    query_uids=list(queries);q=[unit(x) for x in queries.values()];k=POLICY['top_k']
    best_ids=np.full((len(q),k),np.iinfo(np.int64).max,dtype='<i8');best_scores=np.full((len(q),k),-np.inf,dtype='<f8')
    for lo in range(0,len(uids),POLICY['physical_inputs']):
        ids=uids[lo:lo+POLICY['physical_inputs']];r=[unit(x) for x in read(ids.tolist())]
        for first in range(0,len(q),32):
            last=min(first+32,len(q));scores=score_units(q[first:last],r)
            scores[np.array(query_uids[first:last])[:,None]==ids[None]]=-np.inf
            merged_scores=np.concatenate((best_scores[first:last],scores),axis=1)
            merged_ids=np.concatenate((best_ids[first:last],np.broadcast_to(ids,(last-first,len(ids)))),axis=1)
            selected=np.lexsort((merged_ids,-merged_scores),axis=1)[:,:k]
            best_ids[first:last]=np.take_along_axis(merged_ids,selected,axis=1)
            best_scores[first:last]=np.take_along_axis(merged_scores,selected,axis=1)
    need(np.isfinite(best_scores).all(),'WINDOW_INDEX_EXACT_SCARCITY')
    return {uid:[[int(other),float(score)] for other,score in zip(best_ids[i],best_scores[i])]
            for i,uid in enumerate(query_uids)}


def multiplex(results):
    edges={}
    for name,rows in results.items():
        for row in rows:
            for rank,(target,score) in enumerate(row['hits'],1):
                key=(row['uid'],target)
                edge=edges.setdefault(key,dict(origin=key[0],target=key[1],readings={}))
                need(name not in edge['readings'],'WINDOW_INDEX_DUPLICATE_VOTE')
                edge['readings'][name]=dict(rank=rank,score=score)
    return dict(edges=[edges[k] for k in sorted(edges)],absence_means='not_materialized_not_no_association',
                consensus_required=False,scientific_admission=False)
