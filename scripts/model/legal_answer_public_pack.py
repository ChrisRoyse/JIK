"""Pinned public prefixes shared across cases; no future targets or model state.

Only byte-identical UID readings are shared. Each lane retains ragged native
windows. Case order and repeated occurrences are reconstructed without pooling.
"""
import json
from pathlib import Path
import re

import numpy as np
import torch
from model.complete_answer_cohort import validate_cases
from model.legal_answer_evaluation_plan import DTYPE
from model.native_population_training import fingerprint
from model.native_request_input import legal
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import stamp

SCHEMA = 'legal-public-native-prefix-pack-v1'
POLICY = dict(native_encoder_calls=0, model_loads=0, optimizer_updates=0,
    dimensions=768, tile_atoms=128, tile_bytes=16<<20,
    deduplication='exact_public_uid_only', future_vectors=False,
    scope='observed_source_prefixes_not_legal_QA_or_world_truth', scientific_admission=False)


def union(rows):
    values = np.unique(np.concatenate((rows['uids'].ravel(), rows['donor_uids'].ravel())))
    return values[values >= 0].astype('<i8')


def indices(uids, requested):
    at = np.searchsorted(uids, requested)
    need(bool((at < len(uids)).all()) and np.array_equal(uids[at], requested), 'PUBLIC_PACK_UNKNOWN_UID')
    return at


def value(rows, uids, offsets, vectors, roster, ordinal):
    need(type(ordinal) is int and 0 <= ordinal < len(rows), 'PUBLIC_PACK_CASE_RANGE')
    row = rows[ordinal]; n = int(row['observed']); result = {}; payload = 0
    for mode, field in [('correct', 'uids'), ('train_donor', 'donor_uids')]:
        frames = []
        for j,k in enumerate(indices(uids, row[field][:n])):
            frame = {}
            for lane,name in enumerate(roster):
                lo,hi = map(int, offsets[name][int(k):int(k)+2])
                need(hi-lo == int(row['layout'][j,lane]) > 0, 'PUBLIC_PACK_WINDOW_COUNT')
                payload += (hi-lo)*768*4
                need(payload <= POLICY['tile_bytes'], 'PUBLIC_PACK_CASE_MEMORY')
                v = np.array(vectors[name][lo:hi], copy=True)
                need(np.isfinite(v).all(), 'PUBLIC_PACK_NONFINITE')
                frame[name] = torch.from_numpy(v)
            frames.append(frame)
        result[mode] = dict(frames=tuple(frames), descriptor=legal(n))
    result['reversed'] = dict(frames=result['correct']['frames'][::-1], descriptor=legal(n))
    return result


def case(root, binding, rows, ordinal, public):
    row = rows[ordinal]; document = int(row['document']); state = fingerprint(public)
    pin = digest(canonical(dict(schema=SCHEMA, binding=binding, ordinal=ordinal,
        row=digest(row.tobytes()), state=state)))
    return dict(ordinal=ordinal, document=document,
        public=dict(path=str(Path(root)/f'public-{ordinal:06d}'), pin=pin, state=state),
        seed=20261024+document, context='legal-cohort-'+str(document),
        reversal_distinct=bool(row['reversal_distinct']))


class PublicPack:
    def __init__(self, root, pin):
        self.root = Path(root); self.maps = []; self.stamps = {}; self.closed = False
        self.pin = pin
        need(file_sha(self.root/'report.json') == pin, 'PUBLIC_PACK_REPORT_PIN')
        self.report = r = json.loads((self.root/'report.json').read_bytes())
        need(set(r) == {'schema','complete','binding','roster','files','summary','policy'}
             and r['complete'] and r['schema'] == SCHEMA and r['policy'] == POLICY, 'PUBLIC_PACK_SCHEMA')
        self.roster = tuple(r['roster'])
        need(len(self.roster) == len(set(self.roster)) == 6
             and all(re.fullmatch('[a-z0-9-]+', n) for n in self.roster), 'PUBLIC_PACK_ROSTER')
        expected = {'uids.npy','rows.npy','cases.jsonl'} | {
            n+suffix for n in self.roster for suffix in ('.offsets.npy','.windows.npy')}
        need(set(r['files']) == expected, 'PUBLIC_PACK_FILES')
        self.stamps[self.root/'report.json'] = stamp(self.root/'report.json')
        for name,ref in r['files'].items():
            path = self.root/name; before = stamp(path)
            need(set(ref) == {'sha256','bytes'} and before[2] == ref['bytes']
                 and file_sha(path) == ref['sha256'] and stamp(path) == before, 'PUBLIC_PACK_FILE_PIN')
            self.stamps[path] = before
        try:
            self.uids = self.array('uids.npy'); self.rows = self.array('rows.npy')
            need(self.rows.dtype == DTYPE and self.rows.ndim == 1 and len(self.rows) > 0,
                 'PUBLIC_PACK_ROWS')
            need(self.uids.dtype == np.dtype('<i8') and self.uids.ndim == 1
                 and np.array_equal(self.uids, union(self.rows)), 'PUBLIC_PACK_PUBLIC_UNION')
            need(bool((np.diff(self.rows['document']) > 0).all()) and bool((self.rows['status'] == 0).all())
                 and bool(np.isin(self.rows['observed'], [1,2]).all())
                 and bool(np.isin(self.rows['side'], [1,2]).all()), 'PUBLIC_PACK_SCHEDULE')
            for row in self.rows:
                n = int(row['observed'])
                need(bool((row['uids'][:n] >= 0).all()) and bool((row['donor_uids'][:n] >= 0).all())
                     and bool((row['uids'][n:] == -1).all()) and bool((row['donor_uids'][n:] == -1).all())
                     and bool((row['layout'][n:] == 0).all()), 'PUBLIC_PACK_UNUSED_PADDING')
            self.offsets = {n:self.array(n+'.offsets.npy') for n in self.roster}
            self.vectors = {n:self.array(n+'.windows.npy') for n in self.roster}
            for name in self.roster:
                o,v = self.offsets[name],self.vectors[name]
                need(o.dtype == np.dtype('<i8') and o.shape == (len(self.uids)+1,)
                     and int(o[0]) == 0 and bool((np.diff(o) > 0).all())
                     and v.dtype == np.dtype('<f4') and v.shape == (int(o[-1]),768), 'PUBLIC_PACK_LAYOUT')
            windows = [int(self.offsets[n][-1]) for n in self.roster]
            expected_summary = dict(cases=len(self.rows),unique_atoms=len(self.uids),
                observed_occurrences=2*int(self.rows['observed'].sum()),windows=windows,
                payload_bytes=sum(windows)*768*4,uid_sha256=digest(self.uids.tobytes()),
                source_shard_ranges=len(np.unique(self.uids//16384)))
            need(r['summary'] == expected_summary, 'PUBLIC_PACK_SUMMARY')
            with (self.root/'cases.jsonl').open('rb') as stream:
                self.cases = [json.loads(line) for line in stream]
            validate_cases(self.cases)
            need(len(self.cases) == len(self.rows)
                 and [c['ordinal'] for c in self.cases] == list(range(len(self.rows)))
                 and [c['document'] for c in self.cases] == self.rows['document'].tolist(), 'PUBLIC_PACK_CASES')
            self.addresses = {c['public']['path']:c for c in self.cases}
            need(len(self.addresses) == len(self.cases) and all(c['public']['path'] == str(
                self.root/f'public-{c["ordinal"]:06d}') for c in self.cases), 'PUBLIC_PACK_ADDRESS')
            self.fresh()
        except BaseException:
            for a in self.maps:a._mmap.close()
            raise

    def array(self, name):
        a = np.load(self.root/name, mmap_mode='r', allow_pickle=False); self.maps.append(a); return a

    def load(self, reference):
        self.fresh()
        need(set(reference) == {'path','pin','state'}, 'PUBLIC_PACK_REFERENCE_FIELDS')
        row = self.addresses.get(reference['path'])
        need(row is not None and row['public'] == reference, 'PUBLIC_PACK_REFERENCE')
        public = value(self.rows,self.uids,self.offsets,self.vectors,self.roster,row['ordinal'])
        need(case(self.root,self.report['binding'],self.rows,row['ordinal'],public) == row,
             'PUBLIC_PACK_CASE_STATE')
        self.fresh(); return public

    def fresh(self, physical=False):
        need(not self.closed and all(stamp(p) == s for p,s in self.stamps.items()), 'PUBLIC_PACK_CHANGED')
        if physical:
            need(file_sha(self.root/'report.json') == self.pin and all(file_sha(self.root/n) == r['sha256']
                 for n,r in self.report['files'].items()), 'PUBLIC_PACK_FINAL_PIN')

    def close(self):
        try:self.fresh(physical=True)
        finally:
            for a in self.maps:a._mmap.close()
            self.closed = True
