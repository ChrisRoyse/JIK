"""Independent, tiled native-vector and full-gallery ranking readback.

This path does not call the producer's normalizer, score table or top-k reducer.
It consumes the same authenticated native source and the frozen index bytes.
"""
import heapq
import time

import numpy as np

from text.constellation_corpus import need


POLICY = dict(source_tile=384, query_tile=16, top_k=10, scalar_error=5e-13,
              normalization='numpy_linalg_norm_float64',
              comparison='contiguous_pair_sum_complete_window_min_directed_mean_maxsim',
              selection='partition_threshold_then_heap_score_uid', recall_gate=.95)


def normalized(rows):
    result = []
    for row in rows:
        need(row.dtype == np.float32 and row.ndim == 2 and len(row) > 0
             and np.isfinite(row).all(), 'INDEX_REFERENCE_NATIVE')
        value = row.astype(np.float64)
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        need(bool((norm > 0).all()), 'INDEX_REFERENCE_ZERO_NATIVE')
        result.append(value/norm)
    return result


def scores(queries, gallery):
    need(queries and gallery and len({x.shape[1] for x in queries+gallery}) == 1,
         'INDEX_REFERENCE_DIMENSIONS')
    output = np.empty((len(queries), len(gallery)), dtype=np.float64)
    single_q = [i for i, x in enumerate(queries) if len(x) == 1]
    single_g = [i for i, x in enumerate(gallery) if len(x) == 1]
    if single_q and single_g:
        right = np.stack([gallery[j][0] for j in single_g])
        # Each contiguous coordinate row has the same reduction regardless of
        # other queries/candidates. BLAS GEMM/GEMV selects shape-dependent kernels
        # and can assign different scores to byte-identical candidate vectors.
        for i in single_q:
            output[i, single_g] = np.clip(np.sum(right*queries[i][0], axis=1), -1, 1)
    multi_g = [j for j, x in enumerate(gallery) if len(x) > 1]
    for i, left in enumerate(queries):
        for j in (range(len(gallery)) if len(left) > 1 else multi_g):
            right = np.ascontiguousarray(gallery[j])
            dots = np.stack([np.sum(right*row, axis=1) for row in left])
            dots = np.clip(dots, -1, 1)
            output[i, j] = min(sum(map(max, dots))/len(left),
                               sum(map(max, dots.T))/len(gallery[j]))
    return output


def retain(previous, ids, values, query):
    """Keep all candidates tied at the tile cutoff before exact UID tie breaking."""
    k = POLICY['top_k']
    eligible = np.flatnonzero(ids != query)
    if len(eligible) > k:
        threshold = np.partition(values[eligible], len(eligible)-k)[len(eligible)-k]
        eligible = eligible[values[eligible] >= threshold]
    candidates = previous+[(int(ids[j]), float(values[j])) for j in eligible]
    return heapq.nsmallest(k, candidates, key=lambda pair: (-pair[1], pair[0]))


def verify_rows(assessment, expected):
    rows = assessment['rows']; k = POLICY['top_k']
    need(rows and [r['uid'] for r in rows] == list(expected)
         and len({r['uid'] for r in rows}) == len(rows), 'INDEX_REFERENCE_QUERY_SCOPE')
    maximum = 0.; recovered = 0
    for row in rows:
        target = expected[row['uid']]
        need(len(row['exact_hits']) == len(target) == k
             and [u for u, _ in row['exact_hits']] == [u for u, _ in target],
             'INDEX_REFERENCE_EXACT_RANK')
        errors = [abs(a[1]-b[1]) for a, b in zip(row['exact_hits'], target)]
        need(all(np.isfinite(errors)), 'INDEX_REFERENCE_NONFINITE_SCORE')
        maximum = max(maximum, max(errors))
        hits = row['hits']; candidates = row['candidate_uids']
        need(len(hits) == k and len({u for u, _ in hits}) == k
             and row['uid'] not in {u for u, _ in hits}
             and candidates == sorted(set(candidates))
             and {u for u, _ in hits} <= set(candidates), 'INDEX_REFERENCE_RETURNED_SCOPE')
        count = len({u for u, _ in target} & {u for u, _ in hits})
        need(count == row['recovered'] and row['recall'] == count/k,
             'INDEX_REFERENCE_ROW_RECALL')
        recovered += count
    recall = recovered/(len(rows)*k)
    need(maximum <= POLICY['scalar_error'], 'INDEX_REFERENCE_SCORE_ERROR')
    need(assessment['recall_gate'] == POLICY['recall_gate']
         and assessment['recall'] == recall and assessment['qualified'] == (recall >= .95),
         'INDEX_REFERENCE_RECALL')
    return dict(queries=len(rows), recovered=recovered, requested=len(rows)*k,
                recall=recall, qualified=recall >= .95, maximum_scalar_error=maximum,
                every_exact_rank_reproduced=True)


def verify_candidates(row, query, read):
    ids = row['candidate_uids']
    need(ids == sorted(set(ids)) and row['uid'] not in ids, 'INDEX_REFERENCE_CANDIDATE_SCOPE')
    values = scores([query], normalized(read(ids)))[0]
    expected = heapq.nsmallest(POLICY['top_k'], zip(ids, values), key=lambda pair: (-pair[1], pair[0]))
    need([u for u, _ in row['hits']] == [u for u, _ in expected], 'INDEX_REFERENCE_CANDIDATE_RANK')
    errors = [abs(a[1]-b[1]) for a, b in zip(row['hits'], expected)]
    need(errors and all(np.isfinite(errors)) and max(errors) <= POLICY['scalar_error'],
         'INDEX_REFERENCE_CANDIDATE_SCORE')
    return float(max(errors))


def readback(index, read, assessment, progress=None):
    start = tick = time.perf_counter()
    uids = [r['uid'] for r in assessment['rows']]
    need(uids and len(uids) == len(set(uids)), 'INDEX_REFERENCE_QUERY_SCOPE')
    queries = normalized(read(uids)); best = [[] for _ in queries]
    phases = dict(query_source=time.perf_counter()-tick, source=0., vectors=0.,
                  normalization=0., exact_scores=0., ranking=0., progress=0.)
    windows = 0; last_progress = 0
    for lo in range(0, len(index.uids), POLICY['source_tile']):
        tick = time.perf_counter()
        ids = index.uids[lo:lo+POLICY['source_tile']]; native = read(ids.tolist())
        phases['source'] += time.perf_counter()-tick; tick = time.perf_counter()
        need(np.array_equal([len(x) for x in native], index.lengths[lo:lo+len(ids)]),
             'INDEX_REFERENCE_WINDOW_LOSS')
        keys = np.array([int(uid)*4294967296+j for uid, row in zip(ids, native)
                         for j in range(len(row))], dtype=np.uint64)
        saved = index.index.get(keys, dtype='f32'); expected = np.concatenate(native)
        need(len(saved) == len(expected) and all(x is not None and np.array_equal(x, y)
             for x, y in zip(saved, expected)), 'INDEX_REFERENCE_NATIVE_VECTOR')
        windows += len(keys); phases['vectors'] += time.perf_counter()-tick
        tick = time.perf_counter(); gallery = normalized(native)
        phases['normalization'] += time.perf_counter()-tick
        for first in range(0, len(queries), POLICY['query_tile']):
            last = min(first+POLICY['query_tile'], len(queries)); tick = time.perf_counter()
            table = scores(queries[first:last], gallery)
            phases['exact_scores'] += time.perf_counter()-tick; tick = time.perf_counter()
            for offset, values in enumerate(table):
                j = first+offset; best[j] = retain(best[j], ids, values, uids[j])
            phases['ranking'] += time.perf_counter()-tick
        done = lo+len(ids)
        if progress is not None and (done-last_progress >= 65536 or done == len(index.uids)):
            tick = time.perf_counter()
            progress(dict(inputs=done, windows=windows, queries=len(queries),
                          seconds=time.perf_counter()-start, phases_seconds=dict(phases)))
            last_progress = done; phases['progress'] += time.perf_counter()-tick
    tick = time.perf_counter()
    need(windows == len(index.index) == index.report['windows'], 'INDEX_REFERENCE_WINDOW_COUNT')
    expected = dict(zip(uids, best)); result = verify_rows(assessment, expected)
    phases['receipt_comparison'] = time.perf_counter()-tick; tick = time.perf_counter()
    candidate_error = max(verify_candidates(row, query, read) for row, query in zip(assessment['rows'], queries))
    phases['independent_candidate_reranking'] = time.perf_counter()-tick; tick = time.perf_counter()
    original = {r['uid']: r for r in assessment['rows']}
    for uid in list(reversed(uids))+uids[:1]:
        fresh = index.query(uid, read([uid])[0], read, lens_id=index.report['binding']['lens_id'])
        need(fresh['hits'] == original[uid]['hits']
             and fresh['candidate_uids'] == original[uid]['candidate_uids'],
             'INDEX_REFERENCE_QUERY_REPLAY')
    index.fresh(); phases['cold_ann_replay'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'INDEX_REFERENCE_PHASE_ATTRIBUTION')
    return dict(**result, inputs=len(index.uids), windows=windows, every_native_vector_exact=True,
                every_candidate_reranking_reproduced=True, maximum_candidate_scalar_error=candidate_error,
                reverse_and_repeat_queries_exact=True, exact_rows=[dict(uid=u, hits=expected[u]) for u in uids],
                seconds=seconds, phases_seconds=phases, policy=POLICY)
