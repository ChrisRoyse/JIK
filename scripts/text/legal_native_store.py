"""Immutable, range-indexed legal native window shards; explicit partial coverage."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from text.constellation_corpus import digest, file_sha, need
from text.constellation_input import stamp

SCHEMA = 'legal-native-complete-window-bank-v1'
SHARD_INPUTS = 16384
IO_INPUTS = 1024


def durable_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def shard_name(start, end):
    return f'{start:012d}-{end:012d}'


def ranges(n):
    return [(i, min(i+SHARD_INPUTS, n)) for i in range(0, n, SHARD_INPUTS)]


class NativeShard:
    def __init__(self, root, configuration_pin, lens_id, start, end):
        self.root = Path(root)
        need(self.root.is_dir() and (self.root/'COMMITTED').is_file(), 'NATIVE_SHARD_UNCOMMITTED')
        raw = (self.root/'report.json').read_bytes(); self.pin = digest(raw); r = json.loads(raw)
        need(json.loads((self.root/'COMMITTED').read_bytes()) == dict(report_sha256=self.pin), 'NATIVE_SHARD_SEAL')
        need(r['schema'] == SCHEMA and r['configuration_pin'] == configuration_pin
             and r['lens_id'] == lens_id and (r['start'], r['end']) == (start, end)
             and r['complete'] and r['inputs'] == end-start, 'NATIVE_SHARD_SCOPE')
        self.report = r
        self.stamps = {self.root/'report.json': stamp(self.root/'report.json'),
                       self.root/'COMMITTED': stamp(self.root/'COMMITTED')}
        need(set(r['files']) == {'vectors.npy', 'offsets.npy', 'traces.parquet'}, 'NATIVE_SHARD_FILES')
        for name, item in r['files'].items():
            p = self.root/name
            need(p.stat().st_size == item['bytes'] and file_sha(p) == item['sha256'], 'NATIVE_SHARD_FILE_PIN')
            self.stamps[p] = stamp(p)
        self.vectors = np.load(self.root/'vectors.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(self.root/'offsets.npy', mmap_mode='r', allow_pickle=False)
        need(self.vectors.dtype == np.float32 and self.vectors.shape == (r['windows'], r['dimensions'])
             and self.offsets.dtype == np.int64 and self.offsets.shape == (end-start+1,)
             and self.offsets[0] == 0 and self.offsets[-1] == len(self.vectors)
             and (np.diff(self.offsets) > 0).all(), 'NATIVE_SHARD_ARRAYS')
        self.table = pq.ParquetFile(self.root/'traces.parquet')
        need(self.table.metadata.num_rows == end-start, 'NATIVE_SHARD_TRACE_ROWS')
        expected_groups = (end-start+IO_INPUTS-1)//IO_INPUTS
        need(self.table.num_row_groups == expected_groups, 'NATIVE_SHARD_TRACE_GROUPS')
        for i in range(expected_groups):
            need(self.table.metadata.row_group(i).num_rows == min(IO_INPUTS, end-start-i*IO_INPUTS),
                 'NATIVE_SHARD_TRACE_GROUP_SIZE')
        self.cached_group = None; self.cached = None

    def fresh(self):
        need(all(stamp(p) == value for p, value in self.stamps.items()), 'NATIVE_SHARD_CHANGED')

    def input(self, uid, expected_sha256):
        need(type(uid) is int and self.report['start'] <= uid < self.report['end'], 'NATIVE_SHARD_INPUT_RANGE')
        self.fresh(); index = uid-self.report['start']; group, row = divmod(index, IO_INPUTS)
        if group != self.cached_group:
            self.cached = self.table.read_row_group(group); self.cached_group = group
        record = self.cached.slice(row, 1).to_pylist()[0]
        need(record['index'] == uid and record['input_sha256'] == expected_sha256, 'NATIVE_SHARD_INPUT_BINDING')
        trace = json.loads(record['provenance']); lo, hi = map(int, self.offsets[index:index+2])
        need(trace['surface_sha256'] == expected_sha256 and len(trace['windows']) == hi-lo, 'NATIVE_SHARD_WINDOW_BINDING')
        self.fresh(); return self.vectors[lo:hi], trace


class NativeBank:
    def __init__(self, root, pin):
        self.root = Path(root)
        raw = (self.root/'configuration.json').read_bytes()
        need(digest(raw) == pin, 'NATIVE_BANK_CONFIG_PIN')
        self.pin = pin; self.config = json.loads(raw)
        need(self.config['schema'] == SCHEMA and self.config['shard_inputs'] == SHARD_INPUTS
             and self.config['io_inputs'] == IO_INPUTS, 'NATIVE_BANK_CONFIG')
        self.cached = {}; self.config_stamp = stamp(self.root/'configuration.json')

    def shard(self, name, uid):
        need(name in self.config['instruments'] and type(uid) is int and 0 <= uid < self.config['inputs'],
             'NATIVE_BANK_INPUT_SCOPE')
        need(stamp(self.root/'configuration.json') == self.config_stamp, 'NATIVE_BANK_CHANGED')
        start = uid//SHARD_INPUTS*SHARD_INPUTS; end = min(start+SHARD_INPUTS, self.config['inputs'])
        key = (name, start)
        if key not in self.cached:
            # One shard per lane is resident; consumers can batch by range.
            for old in list(self.cached):
                if old[0] == name: del self.cached[old]
            self.cached[key] = NativeShard(self.root/name/shard_name(start, end), self.pin,
                                          self.config['instruments'][name]['lens_id'], start, end)
        return self.cached[key]

    def input(self, name, uid, expected_sha256):
        return self.shard(name, uid).input(uid, expected_sha256)

    def training_authority(self):
        raise ValueError('MNLM_TEXT_CORPUS_ADMISSION_REQUIRED')
