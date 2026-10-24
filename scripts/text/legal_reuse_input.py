"""Indexed native inputs and every contextual occurrence from a legal reuse bank."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from text.constellation_corpus import file_sha, need
from text.constellation_input import stamp
from text.legal_reuse import ARRAYS, SCHEMA, bind


class ReuseInputs:
    def __init__(self, root, pin):
        self.root = Path(root); self.pin = pin
        need(file_sha(self.root/'report.json') == pin, 'REUSE_REPORT_PIN')
        self.report = r = json.loads((self.root/'report.json').read_bytes())
        need(r['schema'] == SCHEMA and r['complete'] and not r['errors'], 'REUSE_SCOPE')
        need(json.loads((self.root/'COMMITTED').read_bytes()) == dict(report_sha256=pin), 'REUSE_COMMITTED')
        self.stamps = {self.root/'report.json': stamp(self.root/'report.json')}
        for name, item in r['files'].items():
            path = self.root/name
            need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'REUSE_FILE')
            self.stamps[path] = stamp(path)
        self.arrays = {name: np.load(self.root/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in ARRAYS}
        self.table = pq.ParquetFile(self.root/'unique.parquet'); self.group = None; self.cached = None
        n, u = r['counts']['occurrences'], r['counts']['unique_inputs']
        need(len(self.arrays['documents']) == r['counts']['documents'], 'REUSE_DOCUMENT_COUNT')
        for name in ARRAYS:
            expected = (len(self.arrays['documents']), 2) if name == 'documents' else ({
                'mapping': n, 'hashes': u, 'first': u, 'counts': u, 'offsets': u+1,
                'occurrences': n, 'successors': n-len(self.arrays['documents'])}[name],)
            need(self.arrays[name].shape == expected, 'REUSE_ARRAY_SHAPE')
        need(self.table.metadata.num_rows == u, 'REUSE_UNIQUE_ROWS')
        need(self.table.num_row_groups == (u+r['group_rows']-1)//r['group_rows'], 'REUSE_GROUP_COUNT')
        for i in range(self.table.num_row_groups):
            need(self.table.metadata.row_group(i).num_rows == min(r['group_rows'], u-i*r['group_rows']),
                 'REUSE_GROUP_ROWS')

    def fresh(self):
        need(all(stamp(p) == value for p, value in self.stamps.items()), 'REUSE_CHANGED')

    def input(self, uid):
        need(type(uid) is int and 0 <= uid < self.report['counts']['unique_inputs'], 'REUSE_INPUT_INDEX')
        self.fresh(); group, row = divmod(uid, self.report['group_rows'])
        if self.group != group:
            self.cached = self.table.read_row_group(group); self.group = group
        result = self.cached.slice(row, 1).to_pylist()[0]
        need(result['index'] == uid and result['text_sha256'] == self.arrays['hashes'][uid].tobytes().hex(),
             'REUSE_INPUT_ADDRESS')
        bind(result['text'].encode(), result['text_sha256']); self.fresh()
        return result

    def occurrences(self, uid):
        need(type(uid) is int and 0 <= uid < self.report['counts']['unique_inputs'], 'REUSE_INPUT_INDEX')
        self.fresh(); lo, hi = self.arrays['offsets'][uid:uid+2]
        values = self.arrays['occurrences'][int(lo):int(hi)]
        need(len(values) == self.arrays['counts'][uid] and np.all(self.arrays['mapping'][values] == uid),
             'REUSE_POSTING_BINDING')
        return values

    def training_authority(self):
        raise ValueError('MNLM_TEXT_CORPUS_ADMISSION_REQUIRED')
