"""Authenticated indexed exact-shingle evidence; no training authority."""
import json
from pathlib import Path

import numpy as np

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp
from text.legal_document_shingles import SCHEMA, POLICY


def header(root, pin):
    need(file_sha(root/'report.json') == pin, 'SHINGLE_REPORT_PIN')
    need((root/'COMMITTED').exists() and json.loads((root/'COMMITTED').read_bytes()) == dict(report_sha256=pin),
         'SHINGLE_UNCOMMITTED')
    report = json.loads((root/'report.json').read_bytes())
    need(report['schema'] == SCHEMA and report['policy'] == POLICY and report['complete']
         and not report['errors'], 'SHINGLE_SCOPE')
    need(set(report['files']) == {'tuples.u32', 'offsets.npy', 'word-counts.npy',
                                 'words.jsonl', 'documents.jsonl', 'before.json'}, 'SHINGLE_FILE_SET')
    return report


def check_document(meta, offsets, word_count, values, vocabulary_size):
    need(meta['start'] == int(offsets[0]) and meta['end'] == int(offsets[1])
         and meta['words'] == int(word_count), 'SHINGLE_DOCUMENT_ADDRESS')
    need(values.dtype == np.dtype('<u4') and values.shape == (int(offsets[1]-offsets[0]), 5),
         'SHINGLE_DOCUMENT_SHAPE')
    need(len(values) <= max(0, int(word_count)-4) and (not len(values) or int(values.max()) < vocabulary_size),
         'SHINGLE_DOCUMENT_WORD_ID')
    packed = values.view('V20').reshape(-1)
    need(np.array_equal(packed, np.unique(packed)), 'SHINGLE_DOCUMENT_UNIQUENESS')


class Shingles:
    def __init__(self, root, pin):
        self.root = Path(root); self.pin = pin; self.report = r = header(self.root, pin)
        self.stamps = {self.root/n: stamp(self.root/n) for n in ('report.json', 'COMMITTED')}
        for name, ref in r['files'].items():
            path = self.root/name
            need(path.stat().st_size == ref['bytes'] and file_sha(path) == ref['sha256'], 'SHINGLE_FILE_PIN')
            self.stamps[path] = stamp(path)
        self.offsets = np.load(self.root/'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.counts = np.load(self.root/'word-counts.npy', mmap_mode='r', allow_pickle=False)
        n = r['counts']['documents']; h = r['counts']['unique_per_document_memberships']
        need(self.offsets.dtype == np.dtype('<i8') and self.offsets.shape == (n+1,)
             and self.counts.dtype == np.dtype('<u4') and self.counts.shape == (n,), 'SHINGLE_ARRAY_SHAPE')
        need(self.offsets[0] == 0 and self.offsets[-1] == h and np.all(np.diff(self.offsets) >= 0)
             and (self.root/'tuples.u32').stat().st_size == h*20, 'SHINGLE_ARRAY_OFFSETS')
        self.values = np.memmap(self.root/'tuples.u32', mode='r', dtype='<u4', shape=(h, 5))
        self.documents = [json.loads(line) for line in (self.root/'documents.jsonl').open('rb')]
        need(len(self.documents) == n and all(d['index'] == i for i, d in enumerate(self.documents)),
             'SHINGLE_DOCUMENT_ORDER')

    def fresh(self):
        need(all(stamp(p) == value for p, value in self.stamps.items()), 'SHINGLE_CHANGED')

    def document(self, index, record):
        need(type(index) is int and 0 <= index < len(self.documents), 'SHINGLE_INDEX')
        self.fresh(); meta = self.documents[index]
        need(meta['record'] == record, 'SHINGLE_CONSUMER_RECORD')
        lo, hi = map(int, self.offsets[index:index+2]); values = self.values[lo:hi]
        check_document(meta, (lo, hi), self.counts[index], values, self.report['counts']['vocabulary'])
        self.fresh(); return values

    def training_authority(self):
        raise ValueError('MNLM_SHINGLE_TRAINING_AUTHORITY_ABSENT')
