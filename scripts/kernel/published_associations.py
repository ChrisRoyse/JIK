"""Resident, typed access to sealed adjacency shards; no search or model loading."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

FILES = {'queries.npy', 'targets.npy', 'scores.npy', 'candidate_counts.npy',
         'candidate_hashes.npy', 'batch_indices.npy', 'batches.jsonl'}


def need(ok, code):
    if not ok: raise ValueError('MNLM_ASSOCIATION_'+code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat(); return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def unchanged(stamps):
    try: ok = all(stamp(p) == s for p, s in stamps.items())
    except OSError: ok = False
    need(ok, 'PERSISTED_STATE_CHANGED')


def read_json(path, pin=None, stamps=None):
    path = Path(path); before = stamp(path); need(before[2] <= 16 << 20, 'METADATA_MEMORY')
    raw = path.read_bytes()
    need(pin is None or hashlib.sha256(raw).hexdigest() == pin, 'REPORT_PIN')
    need(stamp(path) == before, 'METADATA_CHANGED')
    if stamps is not None: stamps[path] = before
    return json.loads(raw)


def publication(root, pin):
    root = Path(root); seals = {}; marker = read_json(root/'COMMITTED', stamps=seals)
    need(marker['report_sha256'] == pin, 'COMMIT_PIN')
    relative = Path(marker.get('report_path', 'report.json'))
    need(relative == Path('report.json') or (len(relative.parts) == 3 and relative.parts[0] == 'runs'
        and relative.parts[1].isdigit() and len(relative.parts[1]) == 6 and relative.parts[2] == 'report.json'), 'REPORT_ADDRESS')
    r = read_json(root/relative, pin, seals)
    need(r['complete'] and r['scope'] == 'full' and r['full_graph_complete']
         and not r['scientific_admission'] and r['all_rows_structurally_read'], 'FULL_PUBLICATION_REQUIRED')
    if 'new_shards' in r:
        plan = r['contract']; imported = read_json(root/'import.json', plan['import_pin'], seals)
        rows = imported['shards']+r['new_shards']; coverage = r['coverage']
    else:
        plan = r['plan']; rows = r['shards']; coverage = dict(population=plan['population'], queries=r['queries'], edges=r['edges'])
    population = coverage['population']; binding = plan['binding']
    need(type(population) is int and population > 10 and coverage['queries'] == population
         and coverage['edges'] == 10*population, 'FULL_COVERAGE')
    need(plan['policy']['top_k'] == 10 and plan['policy']['all_pairs_exhaustive'] is False
         and not binding['scientific_admission'] and not binding['implicit_training_authority'], 'SCOPE')
    cursor = 0
    for row in rows:
        need(row['start'] == cursor and type(row['stop']) is int and cursor < row['stop'] <= population
             and row['stop']-cursor == row['queries'] <= 16384, 'SHARD_PARTITION')
        cursor = row['stop']
    need(cursor == population, 'MISSING_ORIGINS')
    return r, binding, population, rows, seals


def close_maps(arrays):
    for a in arrays.values(): a._mmap.close()


def inspect_shard(row, binding, population):
    root = Path(row['path']); stamps = {}
    need(read_json(root/'COMMITTED', stamps=stamps) == dict(report_sha256=row['pin']), 'SHARD_UNSEALED')
    r = read_json(root/'report.json', row['pin'], stamps)
    need(r['complete'] and r['schema'] == 'typed-native-graph-shard-v2' and r['binding'] == binding
         and r['queries'] == row['queries'] and r['edges'] == row['queries']*10, 'SHARD_BINDING')
    need(r['consensus_required'] is False and r['all_pairs_exhaustive'] is False
         and r['absence_means'] == 'not_materialized_not_no_association'
         and not r['scientific_admission'] and not r['implicit_training_authority'], 'SHARD_SEMANTICS')
    need(set(r['files']) == FILES, 'SHARD_FILES')
    for name, ref in r['files'].items():
        p = root/name; before = stamp(p)
        need(before[2] == ref['bytes'] and sha(p) == ref['sha256'], 'PAYLOAD_PIN')
        need(stamp(p) == before, 'PAYLOAD_CHANGED'); stamps[p] = before
    a = {}
    try:
        for name in FILES-{'batches.jsonl'}: a[name[:-4]] = np.load(root/name, mmap_mode='r', allow_pickle=False)
        n = row['queries']; q, ids, scores = (a[k] for k in ['queries', 'targets', 'scores'])
        need(q.dtype == np.dtype('<i8') and np.array_equal(q, np.arange(row['start'], row['stop'], dtype='<i8')), 'QUERY_IDENTITY')
        need(ids.dtype == np.dtype('<u4') and scores.dtype == np.dtype('<f8') and ids.shape == scores.shape == (n, 10), 'EDGE_LAYOUT')
        need(np.all(ids < population) and not np.any(ids == q[:, None])
             and np.all(np.diff(np.sort(ids.astype('<i8'), axis=1), axis=1) > 0), 'EDGE_SCOPE')
        need(np.isfinite(scores).all() and np.all(np.abs(scores) <= 1)
             and np.all((scores[:, :-1] > scores[:, 1:]) | ((scores[:, :-1] == scores[:, 1:]) & (ids[:, :-1] < ids[:, 1:]))), 'EDGE_ORDER')
        need(a['candidate_counts'].dtype == np.uint32 and a['candidate_counts'].shape == (n,)
             and np.all(a['candidate_counts'] >= 10) and a['candidate_hashes'].dtype == np.uint8
             and a['candidate_hashes'].shape == (n, 32) and a['batch_indices'].dtype == np.uint32
             and np.array_equal(a['batch_indices'], np.arange(n, dtype='<u4')//64), 'AUXILIARY_LAYOUT')
        unchanged(stamps)
    finally: close_maps(a)
    return dict(**row, stamps=stamps, files=r['files'])


class PublishedAssociations:
    """Authenticate once; retain bounded maps and return owned Python edge records."""
    def __init__(self, root, pin, *, source_pin, bank_pin, lens_id, cache_shards=8):
        began = time.perf_counter(); self.cache = OrderedDict(); self.closed = False
        need(type(cache_shards) is int and 1 <= cache_shards <= 16, 'CACHE_MEMORY')
        self.cache_shards = cache_shards; self.root = Path(root); self.pin = pin
        r, self.binding, self.population, rows, self.seals = publication(root, pin)
        need(self.binding['source_pin'] == source_pin and self.binding['bank_pin'] == bank_pin
             and self.binding['lens_id'] == lens_id, 'REQUESTED_BINDING')
        self.shards = [inspect_shard(v, self.binding, self.population) for v in rows]
        self.stops = tuple(v['stop'] for v in self.shards); unchanged(self.seals)
        self.stats = dict(authentication_seconds=time.perf_counter()-began, payload_bytes_authenticated=sum(
            ref['bytes'] for s in self.shards for ref in s['files'].values()), shards=len(self.shards), origins=self.population,
            stored_edges=10*self.population, calls=0, rows=0, cache_hits=0, cache_misses=0, maps_opened=0,
            maximum_live_maps=0, payload_bytes_returned=0, read_seconds=0.)

    def fresh(self, uid=None):
        need(not self.closed, 'READER_CLOSED'); unchanged(self.seals)
        if uid is not None:
            need(type(uid) is int and 0 <= uid < self.population, 'ORIGIN_SCOPE')
            unchanged(self.shards[bisect_right(self.stops, uid)]['stamps'])

    def _maps(self, number):
        s = self.shards[number]; unchanged(s['stamps'])
        if number in self.cache:
            self.stats['cache_hits'] += 1; self.cache.move_to_end(number); return self.cache[number]
        if len(self.cache) == self.cache_shards: close_maps(self.cache.popitem(last=False)[1])
        arrays = {}
        try:
            for n in ['queries', 'targets', 'scores']: arrays[n] = np.load(Path(s['path'])/(n+'.npy'), mmap_mode='r', allow_pickle=False)
            unchanged(s['stamps'])
        except BaseException:
            close_maps(arrays); raise
        self.cache[number] = arrays; self.stats['cache_misses'] += 1; self.stats['maps_opened'] += 3
        self.stats['maximum_live_maps'] = max(self.stats['maximum_live_maps'], 3*len(self.cache))
        return arrays

    def read(self, uids, *, lens_id):
        self.fresh(); need(lens_id == self.binding['lens_id'], 'WRONG_INSTRUMENT')
        need(isinstance(uids, (list, tuple)) and 0 < len(uids) <= 128
             and all(type(u) is int and 0 <= u < self.population for u in uids), 'ORIGIN_SCOPE')
        began = time.perf_counter(); result = [None]*len(uids); groups = {}
        # Consume each shard once per batch, keeping only owned rows after eviction.
        # Restore caller order, including repeated origins, without sharing edge dicts.
        for position, uid in enumerate(uids):
            groups.setdefault(bisect_right(self.stops, uid), []).append((position, uid))
        for number, members in groups.items():
            s = self.shards[number]; a = self._maps(number)
            for position, uid in members:
                j = uid-s['start']; need(int(a['queries'][j]) == uid, 'ROW_JOIN')
                edges = [dict(origin=uid, target=int(target), score=float(score), rank=k+1,
                    instrument=self.binding['instrument'], lens_id=lens_id, source_pin=self.binding['source_pin'],
                    bank_pin=self.binding['bank_pin'], graph_pin=self.pin, shard_pin=s['pin'],
                    relation='same_instrument_native_window_neighbour')
                    for k, (target, score) in enumerate(zip(a['targets'][j], a['scores'][j]))]
                result[position] = dict(origin=uid, edges=edges, consensus_required=False, all_pairs_exhaustive=False,
                    absence_means='not_materialized_not_no_association', scientific_admission=False, implicit_training_authority=False)
        for number in groups: unchanged(self.shards[number]['stamps'])
        self.fresh(); self.stats['calls'] += 1; self.stats['rows'] += len(uids)
        self.stats['payload_bytes_returned'] += 128*len(uids); self.stats['read_seconds'] += time.perf_counter()-began
        return result

    def training_authority(self):
        raise ValueError('MNLM_ASSOCIATION_TRANSDUCTIVE_GRAPH_NOT_TRAINING_AUTHORITY')

    def close(self):
        for arrays in self.cache.values(): close_maps(arrays)
        self.cache.clear(); self.closed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
