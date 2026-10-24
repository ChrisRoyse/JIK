"""Publish unchanged typed graph arrays with complete query and publication clocks."""
import json
import os
from pathlib import Path
import time

import numpy as np

from claims.process_conditions import write_committed
from kernel.native_graph_accounted import PhaseClock,query_batch
from kernel.native_graph_store import ARRAY_DTYPES
from text.constellation_corpus import digest,file_sha,need

TIMING_CONTRACT='adjacent-monotonic-v1'


def publish(root,uids,lane,plan_pin,*,threads):
    clock=PhaseClock(False);cpu=time.process_time()
    need(isinstance(uids,list) and 0<len(uids)<=16384 and all(type(u) is int for u in uids)
         and uids==sorted(set(uids)),'GRAPH_STORE_SOURCE_SCOPE')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);n=len(uids)
    a={k:np.empty((n,10) if k in ('targets','scores') else (n,32) if k=='candidate_hashes' else n,
                  dtype=v) for k,v in ARRAY_DTYPES.items()}
    a['queries'][:]=uids;windows=work=0;clock.mark('initialization')
    with (root/'batches.jsonl').open('x') as journal:
        for batch,lo in enumerate(range(0,n,64)):
            result=query_batch(lane,uids[lo:lo+64],threads=threads);clock.mark('query')
            need([r['uid'] for r in result['rows']]==uids[lo:lo+64],'GRAPH_STORE_BATCH_ORDER')
            for j,row in enumerate(result['rows'],lo):
                hits=row['hits'];ids=[u for u,_ in hits];values=[s for _,s in hits]
                need(len(ids)==len(set(ids))==10 and row['uid'] not in ids
                     and all(0<=u<lane.index.report['inputs'] for u in ids),'GRAPH_STORE_EDGE_SCOPE')
                need(all(np.isfinite(s) and -1<=s<=1 for s in values)
                     and hits==sorted(hits,key=lambda pair:(-pair[1],pair[0])),'GRAPH_STORE_EDGE_SCORE')
                a['targets'][j]=ids;a['scores'][j]=values;a['batch_indices'][j]=batch
                a['candidate_counts'][j]=len(row['candidate_uids'])
                a['candidate_hashes'][j]=np.frombuffer(bytes.fromhex(digest(np.asarray(row['candidate_uids'],dtype='<i8').tobytes())),dtype='u1')
            receipt=dict(batch=batch,start=lo,stop=lo+len(result['rows']),work=result['work'],
                         query_uid_hash=digest(np.asarray(uids[lo:lo+64],dtype='<i8').tobytes()),
                         seconds=result['seconds'],cpu_seconds=result['cpu_seconds'],
                         phases_seconds=result['phases_seconds'],timing=result['timing'])
            journal.write(json.dumps(receipt,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
            windows+=result['work']['native_windows'];work+=result['work']['computed_distances'];clock.mark('journal')
    lane.fresh()
    for name,values in a.items():
        with (root/(name+'.npy')).open('xb') as stream:
            np.save(stream,values,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    files={f.name:dict(bytes=f.stat().st_size,sha256=file_sha(f)) for f in root.iterdir() if f.is_file()}
    clock.mark('validate_and_save');timing=clock.finish()
    report=dict(schema='typed-native-graph-shard-v2',complete=True,plan_pin=plan_pin,binding=lane.binding,
                queries=n,edges=10*n,native_windows=windows,distance_work=work,files=files,
                seconds=timing['seconds'],cpu_seconds=time.process_time()-cpu,phases_seconds=timing['phases_seconds'],
                timing_contract=TIMING_CONTRACT,timing=timing,all_pairs_exhaustive=False,consensus_required=False,
                absence_means='not_materialized_not_no_association',scientific_admission=False,
                implicit_training_authority=False,acceptance='author-checked',number_status='measured')
    # An independent structural/numeric readback is required before the caller seals.
    return write_committed(root/'report.json',report)
