"""Read persisted graph arrays through a separate path and recalculate native scores."""
import json
from pathlib import Path

import numpy as np

from audit.native_index_reference import normalized, verify_candidates
from text.constellation_corpus import digest, file_sha, need


def open_shard(root, pin, binding):
    root = Path(root); raw = (root/'report.json').read_bytes()
    need(digest(raw) == pin, 'GRAPH_READBACK_REPORT_PIN'); r = json.loads(raw)
    need(json.loads((root/'COMMITTED').read_bytes()) == dict(report_sha256=pin), 'GRAPH_READBACK_COMMIT')
    need(r['schema'] == 'typed-native-association-shard-v1' and r['complete'] and r['binding'] == binding,
         'GRAPH_READBACK_BINDING')
    need(set(r['files']) == {n+'.npy' for n in ['queries', 'targets', 'scores', 'candidate_counts',
                                             'distance_work', 'candidate_hashes']}, 'GRAPH_READBACK_FILES')
    a = {}
    for name, ref in r['files'].items():
        path = root/name
        need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'GRAPH_READBACK_FILE_PIN')
        a[name[:-4]] = np.load(path, allow_pickle=False)
    return r, a


def verify_scope(r, a, expected):
    q = a['queries']; n = len(q)
    need(q.dtype == np.int64 and np.array_equal(q, expected) and len(set(map(int, q))) == n
         and r['queries'] == n and r['edges'] == n*10, 'GRAPH_READBACK_COVERAGE')
    need(a['targets'].dtype == np.uint32 and a['scores'].dtype == np.float64
         and a['targets'].shape == a['scores'].shape == (n, 10), 'GRAPH_READBACK_LAYOUT')
    need(a['candidate_counts'].dtype == np.uint32 and a['distance_work'].dtype == np.uint64
         and a['candidate_counts'].shape == a['distance_work'].shape == (n,)
         and a['candidate_hashes'].dtype == np.uint8 and a['candidate_hashes'].shape == (n, 32),
         'GRAPH_READBACK_AUXILIARY_LAYOUT')
    for i, uid in enumerate(q):
        row = list(zip(map(int, a['targets'][i]), map(float, a['scores'][i])))
        need(len({u for u, _ in row}) == 10 and all(0 <= u < r['population'] and u != uid for u, _ in row),
             'GRAPH_READBACK_EDGE_SCOPE')
        need(all(np.isfinite(s) and -1 <= s <= 1 for _, s in row)
             and row == sorted(row, key=lambda pair: (-pair[1], pair[0])), 'GRAPH_READBACK_EDGE_ORDER')
        need(int(a['candidate_counts'][i]) >= 10 and int(a['distance_work'][i]) > 0,
             'GRAPH_READBACK_WORK')
    return n


def replay(root, pin, lane, expected):
    r, a = open_shard(root, pin, lane.binding); n = verify_scope(r, a, expected)
    maximum = 0.
    for i in reversed(range(n)):
        uid = int(a['queries'][i]); row = lane.query(uid)
        saved = [[int(u), float(s)] for u, s in zip(a['targets'][i], a['scores'][i])]
        need(row['hits'] == saved and len(row['candidate_uids']) == int(a['candidate_counts'][i])
             and row['ann_distance_computations'] == int(a['distance_work'][i]), 'GRAPH_READBACK_QUERY_REPLAY')
        h = bytes(a['candidate_hashes'][i]).hex()
        need(digest(np.asarray(row['candidate_uids'], dtype='<i8').tobytes()) == h, 'GRAPH_READBACK_CANDIDATES')
        maximum = max(maximum, verify_candidates(row, normalized(lane([uid]))[0], lane))
    lane.fresh()
    return dict(queries=n, edges=n*10, every_row_replayed=True, reverse_order_exact=True,
                maximum_independent_score_error=maximum, scientific_admission=False, acceptance='author-checked')
