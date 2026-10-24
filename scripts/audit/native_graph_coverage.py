"""Independent complete-row graph census and sampled singleton/native-score replay."""
import json
from pathlib import Path

import numpy as np

from audit.native_index_reference import normalized, verify_candidates
from text.constellation_corpus import digest, file_sha, need


def read_shard(root, pin, plan_pin, lane, expected, *, committed=True):
    root = Path(root); raw = (root/'report.json').read_bytes()
    need(digest(raw) == pin, 'GRAPH_COVERAGE_REPORT_PIN'); r = json.loads(raw)
    need(r['schema'] == 'typed-native-graph-shard-v2' and r['complete']
         and r['plan_pin'] == plan_pin and r['binding'] == lane.binding, 'GRAPH_COVERAGE_BINDING')
    if committed:
        need(json.loads((root/'COMMITTED').read_bytes()) == dict(report_sha256=pin), 'GRAPH_COVERAGE_COMMIT')
    names = {'queries.npy', 'targets.npy', 'scores.npy', 'candidate_counts.npy', 'candidate_hashes.npy',
             'batch_indices.npy', 'batches.jsonl'}
    need(set(r['files']) == names, 'GRAPH_COVERAGE_FILES'); a = {}
    for name, ref in r['files'].items():
        path = root/name
        need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'GRAPH_COVERAGE_FILE_PIN')
        if name.endswith('.npy'): a[name[:-4]] = np.load(path, allow_pickle=False)
    check_arrays(a, expected, len(lane.index.uids))
    n = len(expected); cursor = 0; windows = distances = 0; batches = 0
    with (root/'batches.jsonl').open() as f:
        for number, line in enumerate(f):
            row = json.loads(line); stop = min(cursor+64, n)
            need(row['batch'] == number and row['start'] == cursor and row['stop'] == stop
                 and np.all(a['batch_indices'][cursor:stop] == number), 'GRAPH_COVERAGE_BATCH_ORDER')
            work = row['work']; source = lane.index.lengths[np.asarray(expected[cursor:stop], dtype='<i8')]
            need(work['scope'] == 'whole_native_window_batch' and work['per_query'] is None
                 and work['queries'] == stop-cursor and work['native_windows'] == int(source.sum())
                 and work['computed_distances'] > 0, 'GRAPH_COVERAGE_NATIVE_WINDOWS')
            need(row['query_uid_hash'] == digest(a['queries'][cursor:stop].tobytes()), 'GRAPH_COVERAGE_BATCH_SOURCE')
            need(np.all(a['candidate_counts'][cursor:stop] <= source*128), 'GRAPH_COVERAGE_CANDIDATE_SCOPE')
            windows += work['native_windows']; distances += work['computed_distances']; cursor = stop; batches += 1
    need(cursor == n and batches == (n+63)//64 and r['queries'] == n and r['edges'] == n*10
         and r['native_windows'] == windows and r['distance_work'] == distances, 'GRAPH_COVERAGE_TOTALS')
    lane.fresh()
    return r, a


def check_arrays(a, expected, population):
    n = len(expected); q = a['queries']; ids = a['targets']; s = a['scores']
    need(q.dtype == np.int64 and q.shape == (n,) and np.array_equal(q, expected)
         and np.all(q[1:] > q[:-1]) and np.all(q >= 0) and np.all(q < population), 'GRAPH_COVERAGE_QUERY_SCOPE')
    need(ids.dtype == np.uint32 and s.dtype == np.float64 and ids.shape == s.shape == (n, 10), 'GRAPH_COVERAGE_EDGE_LAYOUT')
    need(np.all(ids < population) and not np.any(ids == q[:, None])
         and np.all(np.diff(np.sort(ids.astype('<i8'), axis=1), axis=1) > 0), 'GRAPH_COVERAGE_EDGE_SCOPE')
    need(np.isfinite(s).all() and np.all(np.abs(s) <= 1)
         and np.all((s[:, :-1] > s[:, 1:]) | ((s[:, :-1] == s[:, 1:]) & (ids[:, :-1] < ids[:, 1:]))),
         'GRAPH_COVERAGE_SCORE_ORDER')
    need(a['candidate_counts'].dtype == np.uint32 and a['candidate_counts'].shape == (n,)
         and np.all(a['candidate_counts'] >= 10) and a['batch_indices'].dtype == np.uint32
         and a['batch_indices'].shape == (n,) and a['candidate_hashes'].dtype == np.uint8
         and a['candidate_hashes'].shape == (n, 32), 'GRAPH_COVERAGE_AUXILIARY_LAYOUT')


def replay_samples(a, lane, selected):
    positions = {int(uid): i for i, uid in enumerate(a['queries'])}; error = 0.
    for uid in reversed(selected):
        i = positions[uid]; row = lane.query(uid)
        expected = [[int(u), float(s)] for u, s in zip(a['targets'][i], a['scores'][i])]
        need(row['hits'] == expected, 'GRAPH_COVERAGE_SAMPLE_SCORE')
        need(len(row['candidate_uids']) == int(a['candidate_counts'][i])
             and digest(np.asarray(row['candidate_uids'], dtype='<i8').tobytes()) == bytes(a['candidate_hashes'][i]).hex(),
             'GRAPH_COVERAGE_SAMPLE_CANDIDATES')
        error = max(error, verify_candidates(row, normalized(lane([uid]))[0], lane))
    return dict(samples=len(selected), sample_uids=selected, every_sample_exact=True,
                maximum_independent_score_error=error, unsampled_numeric_edges_replayed=False)


def full_coverage(records, population, windows):
    cursor = count = native = 0
    for row in records:
        need(row['start'] == cursor and row['stop'] > cursor and row['queries'] == row['stop']-row['start'],
             'GRAPH_COVERAGE_SHARD_GAP')
        cursor = row['stop']; count += row['queries']; native += row['native_windows']
    need(cursor == count == population and native == windows, 'GRAPH_COVERAGE_INCOMPLETE_FULL')
