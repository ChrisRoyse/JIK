"""Typed native neighbours from an immutable, exhaustively source-checked index.

The index is a byte-exact replica of the native readings, not a new instrument.
The certificate does not admit a sense, establish truth or authorize training.
"""
import json
from pathlib import Path
import time

import numpy as np

from claims.process_conditions import write_committed
from kernel.native_window_index import NativeWindowIndex, POLICY, score_table
from text.constellation_corpus import digest, file_sha, need
from text.constellation_input import stamp

SHARD_ROWS = 16384


def pinned(path, pin):
    raw = path.read_bytes()
    need(digest(raw) == pin, 'GRAPH_CERTIFICATE_PIN')
    return json.loads(raw)


class CertifiedIndexLane:
    def __init__(self, certificate, pin):
        self.certificate = Path(certificate); self.pin = pin
        r = pinned(self.certificate, pin)
        need(r['complete'] and r['mode'] == 'full', 'GRAPH_CERTIFICATE_INCOMPLETE')
        ref = r['baseline']; self.baseline = pinned(Path(ref['path'])/'report.json', ref['pin'])
        self.source = self.baseline['source']
        a = r['result']; ref = a['index']; self.name = a['instrument']
        need(a['complete_gallery'] and a['audit']['every_native_vector_exact']
             and a['audit']['every_candidate_reranking_reproduced'], 'GRAPH_NATIVE_CERTIFICATE')
        metadata = pinned(Path(ref['path'])/'report.json', ref['pin'])
        b = metadata['binding']
        need(b['instrument'] == self.name and b['source_pin'] == self.source['pin']
             and b['bank_pin'] == self.baseline['bank']['pin']
             and b['scope'] == 'full_legal_unique_input_gallery', 'GRAPH_CERTIFICATE_BINDING')
        receipt = a['source_receipt']; source = pinned(Path(receipt['path']), receipt['pin'])
        need(source['index'] == ref and a['census']['inputs'] == metadata['inputs']
             and a['census']['windows'] == metadata['windows'] == a['audit']['windows']
             and a['audit']['inputs'] == metadata['inputs'], 'GRAPH_CERTIFICATE_COVERAGE')
        self.index = NativeWindowIndex(Path(ref['path']), ref['pin'], b)
        self.stamps = {p: stamp(p) for p in [self.certificate, Path(receipt['path']),
                                            Path(r['baseline']['path'])/'report.json']}
        self.binding = dict(instrument=self.name, lens_id=b['lens_id'], source_pin=b['source_pin'],
                            bank_pin=b['bank_pin'], index=ref, certificate=dict(path=str(self.certificate), pin=pin),
                            scope='full_corpus_transductive_neighbour_diagnostic',
                            scientific_admission=False, implicit_training_authority=False)
        self.stats = dict(calls=0, inputs=0, windows=0, read_seconds=0., maximum_payload_bytes=0)

    def fresh(self):
        self.index.fresh()
        try: unchanged = all(stamp(p) == s for p, s in self.stamps.items())
        except OSError: unchanged = False
        need(unchanged, 'GRAPH_CERTIFICATE_CHANGED')

    def __call__(self, uids):
        began = time.perf_counter(); self.fresh()
        need(isinstance(uids, (list, tuple)) and 0 < len(uids) <= 512
             and all(type(u) is int and 0 <= u < 2**32 for u in uids), 'GRAPH_READ_SCOPE')
        ids = np.asarray(uids, dtype='<i8'); positions = np.searchsorted(self.index.uids, ids)
        need(np.all(positions < len(self.index.uids))
             and np.array_equal(self.index.uids[positions], ids), 'GRAPH_UNKNOWN_INPUT')
        lengths = self.index.lengths[positions]; count = int(lengths.sum())
        payload = count*self.index.report['dimensions']*4
        need(payload <= 64 << 20, 'GRAPH_READ_MEMORY')
        keys = np.concatenate([(np.uint64(u) << np.uint64(32))+np.arange(n, dtype='<u8')
                               for u, n in zip(uids, lengths)])
        values = self.index.index.get(keys, dtype='f32')
        need(len(values) == count and all(x is not None for x in values), 'GRAPH_MISSING_NATIVE_WINDOW')
        matrix = np.asarray(values)
        need(matrix.dtype == np.float32 and matrix.shape == (count, self.index.report['dimensions'])
             and np.isfinite(matrix).all(), 'GRAPH_NATIVE_LAYOUT')
        bounds = np.r_[0, np.cumsum(lengths)]; matrix.flags.writeable = False
        rows = [matrix[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        self.fresh(); self.stats['calls'] += 1; self.stats['inputs'] += len(uids)
        self.stats['windows'] += count; self.stats['read_seconds'] += time.perf_counter()-began
        self.stats['maximum_payload_bytes'] = max(payload, self.stats['maximum_payload_bytes'])
        return rows

    def query(self, uid):
        return self.index.query(uid, self([uid])[0], self, lens_id=self.binding['lens_id'])

    def pair(self, origin, target):
        left, right = self([origin, target])
        return dict(origin=origin, target=target, instrument=self.name, lens_id=self.binding['lens_id'],
                    score=float(score_table([left], [right])[0, 0]),
                    relation='same_instrument_native_window_similarity', scientific_admission=False)


def validate_arrays(arrays, population):
    q, ids, scores = (arrays[n] for n in ('queries', 'targets', 'scores'))
    need(q.dtype == np.int64 and q.ndim == 1 and 0 < len(q) <= SHARD_ROWS
         and np.all(q >= 0) and np.all(q < population) and np.all(q[1:] > q[:-1]), 'GRAPH_QUERY_COVERAGE')
    need(ids.dtype == np.uint32 and scores.dtype == np.float64
         and ids.shape == scores.shape == (len(q), POLICY['top_k']), 'GRAPH_EDGE_LAYOUT')
    need(np.all(ids < population) and not np.any(ids == q[:, None])
         and np.all(np.diff(np.sort(ids.astype('<i8'), axis=1), axis=1) > 0), 'GRAPH_EDGE_SCOPE')
    need(np.isfinite(scores).all() and np.all(np.abs(scores) <= 1), 'GRAPH_EDGE_SCORES')
    need(np.all((scores[:, :-1] > scores[:, 1:]) |
                ((scores[:, :-1] == scores[:, 1:]) & (ids[:, :-1] < ids[:, 1:]))), 'GRAPH_EDGE_ORDER')
    need(arrays['candidate_counts'].dtype == np.uint32 and arrays['candidate_counts'].shape == q.shape
         and np.all(arrays['candidate_counts'] >= POLICY['top_k']), 'GRAPH_CANDIDATE_COUNT')
    need(arrays['distance_work'].dtype == np.uint64 and arrays['distance_work'].shape == q.shape
         and np.all(arrays['distance_work'] > 0), 'GRAPH_DISTANCE_WORK')
    need(arrays['candidate_hashes'].dtype == np.uint8
         and arrays['candidate_hashes'].shape == (len(q), 32), 'GRAPH_CANDIDATE_DIGEST')


def publish_shard(root, uids, lane, progress=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    need(0 < len(uids) <= SHARD_ROWS, 'GRAPH_SHARD_MEMORY')
    start = time.perf_counter(); cpu = time.process_time(); n = len(uids)
    arrays = dict(queries=np.asarray(uids, dtype='<i8'), targets=np.empty((n, 10), dtype='<u4'),
                  scores=np.empty((n, 10), dtype='<f8'), candidate_counts=np.empty(n, dtype='<u4'),
                  distance_work=np.empty(n, dtype='<u8'), candidate_hashes=np.empty((n, 32), dtype='u1'))
    first_read_seconds = lane.stats['read_seconds']; tick = time.perf_counter()
    for i, uid in enumerate(uids):
        row = lane.query(int(uid)); arrays['targets'][i] = [u for u, _ in row['hits']]
        arrays['scores'][i] = [s for _, s in row['hits']]
        arrays['candidate_counts'][i] = len(row['candidate_uids'])
        arrays['distance_work'][i] = row['ann_distance_computations']
        arrays['candidate_hashes'][i] = np.frombuffer(bytes.fromhex(digest(np.asarray(row['candidate_uids'], dtype='<i8').tobytes())), dtype='u1')
        if progress is not None and ((i+1) % 512 == 0 or i+1 == n): progress(i+1)
    phases = dict(query_and_gather=time.perf_counter()-tick); tick = time.perf_counter()
    validate_arrays(arrays, lane.index.report['inputs']); lane.fresh()
    files = {}
    for name, value in arrays.items():
        path = root/(name+'.npy'); np.save(path, value, allow_pickle=False)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    phases['validate_and_publish'] = time.perf_counter()-tick
    report = dict(schema='typed-native-association-shard-v1', complete=True, binding=lane.binding,
                  files=files, queries=n, edges=n*10, population=lane.index.report['inputs'],
                  phases_seconds=phases, seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                  native_read_seconds_in_queries=lane.stats['read_seconds']-first_read_seconds,
                  relation='same_instrument_native_window_neighbour', consensus_required=False,
                  absence_means='not_materialized_not_no_association', all_pairs_exhaustive=False,
                  scientific_admission=False, implicit_training_authority=False,
                  acceptance='author-checked', number_status='measured', code_sha256=file_sha(Path(__file__)))
    pin = write_committed(root/'report.json', report)
    write_committed(root/'COMMITTED', dict(report_sha256=pin)); return pin
