"""Durable shards of independently typed native associations, with batch work receipts."""
import json
import os
from pathlib import Path
import time

import numpy as np

from claims.process_conditions import write_committed
from kernel.native_graph_batch import query_batch
from text.constellation_corpus import digest, file_sha, need

ARRAY_DTYPES = dict(queries='<i8', targets='<u4', scores='<f8', candidate_counts='<u4',
                    candidate_hashes='u1', batch_indices='<u4')


def publish(root, uids, lane, plan_pin, *, threads):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); cpu = time.process_time(); n = len(uids)
    need(0 < n <= 16384 and uids == sorted(set(uids)), 'GRAPH_STORE_SOURCE_SCOPE')
    a = {k: np.empty((n, 10) if k in ('targets', 'scores') else (n, 32) if k == 'candidate_hashes' else n,
                     dtype=v) for k, v in ARRAY_DTYPES.items()}
    a['queries'][:] = uids; timings = dict(query=0., journal=0., arrays_and_seal=0.)
    windows = work = 0
    with (root/'batches.jsonl').open('x') as journal:
        for batch, lo in enumerate(range(0, n, 64)):
            result = query_batch(lane, uids[lo:lo+64], threads=threads); timings['query'] += result['seconds']
            tick = time.perf_counter()
            need([r['uid'] for r in result['rows']] == uids[lo:lo+64], 'GRAPH_STORE_BATCH_ORDER')
            for j, row in enumerate(result['rows'], lo):
                hits = row['hits']; ids = [u for u, _ in hits]; values = [s for _, s in hits]
                need(len(ids) == len(set(ids)) == 10 and row['uid'] not in ids
                     and all(0 <= u < lane.index.report['inputs'] for u in ids), 'GRAPH_STORE_EDGE_SCOPE')
                need(all(np.isfinite(s) and -1 <= s <= 1 for s in values)
                     and hits == sorted(hits, key=lambda pair: (-pair[1], pair[0])), 'GRAPH_STORE_EDGE_SCORE')
                a['targets'][j] = ids; a['scores'][j] = values; a['batch_indices'][j] = batch
                a['candidate_counts'][j] = len(row['candidate_uids'])
                a['candidate_hashes'][j] = np.frombuffer(bytes.fromhex(digest(np.asarray(row['candidate_uids'], dtype='<i8').tobytes())), dtype='u1')
            receipt = dict(batch=batch, start=lo, stop=lo+len(result['rows']), work=result['work'],
                           query_uid_hash=digest(np.asarray(uids[lo:lo+64], dtype='<i8').tobytes()),
                           seconds=result['seconds'], cpu_seconds=result['cpu_seconds'], phases_seconds=result['phases_seconds'])
            journal.write(json.dumps(receipt, sort_keys=True)+'\n'); journal.flush(); os.fsync(journal.fileno())
            windows += result['work']['native_windows']; work += result['work']['computed_distances']
            timings['journal'] += time.perf_counter()-tick
    tick = time.perf_counter(); lane.fresh()
    for name, values in a.items():
        with (root/(name+'.npy')).open('xb') as stream:
            np.save(stream, values, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    files = {f.name: dict(bytes=f.stat().st_size, sha256=file_sha(f)) for f in root.iterdir() if f.is_file()}
    timings['arrays_and_seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
    need(sum(timings.values())/seconds >= .9, 'GRAPH_STORE_TIMING_COVERAGE')
    report = dict(schema='typed-native-graph-shard-v2', complete=True, plan_pin=plan_pin,
                  binding=lane.binding, queries=n, edges=10*n, native_windows=windows, distance_work=work,
                  files=files, seconds=seconds, cpu_seconds=time.process_time()-cpu, phases_seconds=timings,
                  all_pairs_exhaustive=False, consensus_required=False,
                  absence_means='not_materialized_not_no_association', scientific_admission=False,
                  implicit_training_authority=False, acceptance='author-checked', number_status='measured')
    # The independent reader must accept this receipt before COMMITTED is written.
    return write_committed(root/'report.json', report)
