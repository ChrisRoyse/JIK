"""Batch ANN queries without changing typed native-window reranking semantics."""
import time

import numpy as np

from kernel.native_window_index import POLICY, score_table, unit
from text.constellation_corpus import need


def query_batch(lane, uids, *, threads):
    start = time.perf_counter(); cpu = time.process_time(); lane.fresh()
    need(type(threads) is int and threads in (1, 4, 16, 32), 'GRAPH_BATCH_THREADS')
    need(isinstance(uids, list) and 0 < len(uids) <= 64
         and all(type(u) is int for u in uids) and len(set(uids)) == len(uids), 'GRAPH_BATCH_SCOPE')
    tick = time.perf_counter(); native = lane(uids)
    for row in native: unit(row)
    lengths = np.asarray([len(x) for x in native], dtype='<i8'); bounds = np.r_[0, np.cumsum(lengths)]
    values = np.concatenate(native)
    phases = dict(source=time.perf_counter()-tick); tick = time.perf_counter()
    count = min(POLICY['window_candidates'], len(lane.index.index))
    need(values.nbytes+len(values)*count*16 <= 64 << 20, 'GRAPH_BATCH_MEMORY')
    matches = lane.index.index.search(values, count=count, threads=threads)
    phases['search'] = time.perf_counter()-tick; tick = time.perf_counter()
    need(matches.keys.dtype == np.uint64 and matches.keys.shape == (len(values), count)
         and np.array_equal(matches.counts, np.full(len(values), count)), 'GRAPH_BATCH_SEARCH_LAYOUT')
    # Do not use matches[i].computed_distances: the library returns an average.
    work = dict(native_windows=len(values), queries=len(uids), computed_distances=int(matches.computed_distances),
                visited_members=int(matches.visited_members), scope='whole_native_window_batch', per_query=None)
    need(work['computed_distances'] > 0, 'GRAPH_BATCH_DISTANCE_WORK')
    owners = (matches.keys >> np.uint64(32)).astype('<i8')
    positions = np.searchsorted(lane.index.uids, owners)
    need(np.all(positions < len(lane.index.uids)) and np.array_equal(lane.index.uids[positions], owners),
         'GRAPH_BATCH_UNKNOWN_OWNER')
    need(np.all((matches.keys & np.uint64(2**32-1)) < lane.index.lengths[positions]), 'GRAPH_BATCH_UNKNOWN_WINDOW')
    candidates = []
    for i, uid in enumerate(uids):
        ids = np.unique(owners[bounds[i]:bounds[i+1]])
        ids = ids[ids != uid].tolist(); need(len(ids) >= POLICY['top_k'], 'GRAPH_BATCH_SCARCITY')
        candidates.append(ids)
    phases['candidate_owners'] = time.perf_counter()-tick
    phases['candidate_vectors'] = 0.; phases['rerank'] = 0.; rows = []
    for uid, query, ids in zip(uids, native, candidates):
        scored = []
        for lo in range(0, len(ids), 512):
            part = ids[lo:lo+512]; tick = time.perf_counter(); target = lane(part)
            phases['candidate_vectors'] += time.perf_counter()-tick; tick = time.perf_counter()
            scored.extend(zip(part, score_table([query], target)[0].tolist()))
            phases['rerank'] += time.perf_counter()-tick
        tick = time.perf_counter(); ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:POLICY['top_k']]
        rows.append(dict(uid=uid, hits=[[u, s] for u, s in ranked], candidate_uids=ids,
                         exhaustive=False, relation='same_instrument_native_window_neighbour'))
        phases['rerank'] += time.perf_counter()-tick
    tick = time.perf_counter(); lane.fresh(); phases['final_freshness'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'GRAPH_BATCH_PHASE_ATTRIBUTION')
    return dict(rows=rows, work=work, phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu)
