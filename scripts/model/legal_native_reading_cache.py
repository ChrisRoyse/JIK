"""Bounded immutable reading cache under the unchanged certified frame checks.

Only bytes from an authenticated native shard enter this process-local cache.
The inherited frames method still checks source identities before and after every
batch and copies every returned array. No tensor or model state is cached.
"""
from collections import OrderedDict
import sys

import numpy as np

from model.legal_native_resident_stream import ResidentNativeFrames
from text.constellation_corpus import need


class ReadingView:
    def __init__(self, owner, name, start, descriptor):
        self.owner, self.name, self.start = owner, name, start
        self.descriptor = descriptor

    def __getitem__(self, selected):
        return self.owner.reading(self.name, self.start, self.descriptor, selected)


class ReadingNativeFrames(ResidentNativeFrames):
    """LRU of exact native ranges; use one instance per source/consumer process.

Accounting includes bytes, keys and their referents, an allocator allowance per
entry, and three times OrderedDict storage. Temporary miss/output copies remain
bounded by the inherited 16 MiB batch limit. RSS is measured separately: this is
an object-storage bound, not a claim to control the Python allocator or OS cache.
"""
    def __init__(self, data, audit_root, audit_pin, *, cache_bytes=1 << 30,
                 handle_limit=64):
        need(type(cache_bytes) is int and 1 << 20 <= cache_bytes <= 32 << 30,
             'READING_CACHE_POLICY')
        super().__init__(data, audit_root, audit_pin, handle_limit=handle_limit)
        self.cache_limit = cache_bytes
        self.readings = OrderedDict()
        self.owned_bytes = 0
        self.cache_stats = dict(hits=0, misses=0, evictions=0, bypasses=0,
                                source_bytes=0, reused_bytes=0, peak_entries=0,
                                peak_accounted_bytes=0)

    @staticmethod
    def charge(key, raw):
        return sys.getsizeof(raw)+sys.getsizeof(key)+sum(map(sys.getsizeof, key))+512

    def accounted_bytes(self):
        return self.owned_bytes+3*sys.getsizeof(self.readings)

    def evict(self):
        key, raw = self.readings.popitem(last=False)
        self.owned_bytes -= self.charge(key, raw)
        self.cache_stats['evictions'] += 1
        if not self.readings: self.readings = OrderedDict()

    def mapping(self, name, start, descriptor):
        need(not self.closed, 'RESIDENT_NATIVE_CLOSED')
        return ReadingView(self, name, start, descriptor)

    def reading(self, name, start, descriptor, selected):
        need(not self.closed, 'RESIDENT_NATIVE_CLOSED')
        r = descriptor['report']
        need(isinstance(selected, slice) and selected.step is None
             and type(selected.start) is int and type(selected.stop) is int
             and 0 <= selected.start < selected.stop <= r['windows'],
             'RESIDENT_NATIVE_SLICE')
        size = (selected.stop-selected.start)*r['dimensions']*4
        need(size <= 16 << 20, 'TRAIN_STREAM_PHYSICAL_MEMORY')
        key = (name, start, selected.start, selected.stop)
        if key in self.readings:
            raw = self.readings[key]
            self.readings.move_to_end(key)
            self.cache_stats['hits'] += 1
            self.cache_stats['reused_bytes'] += len(raw)
        else:
            # Only misses open a native file. The caller's source checks also
            # surround hits, so cached data cannot bypass a changed source.
            view = super().mapping(name, start, descriptor)
            raw = view[selected].tobytes(order='C')
            need(len(raw) == size, 'READING_CACHE_LAYOUT')
            self.cache_stats['misses'] += 1
            self.cache_stats['source_bytes'] += len(raw)
            cost = self.charge(key, raw)
            if cost+3*sys.getsizeof(OrderedDict()) > self.cache_limit:
                self.cache_stats['bypasses'] += 1
            else:
                while self.readings and self.accounted_bytes()+cost > self.cache_limit:
                    self.evict()
                self.readings[key] = raw
                self.owned_bytes += cost
                # OrderedDict can resize on insertion. Its new allocation is
                # charged immediately; excess entries are retired before return.
                while self.readings and self.accounted_bytes() > self.cache_limit:
                    self.evict()
        need(self.accounted_bytes() <= self.cache_limit, 'READING_CACHE_MEMORY')
        self.cache_stats['peak_entries'] = max(self.cache_stats['peak_entries'], len(self.readings))
        self.cache_stats['peak_accounted_bytes'] = max(
            self.cache_stats['peak_accounted_bytes'], self.accounted_bytes())
        return np.frombuffer(raw, dtype=np.float32).reshape(
            selected.stop-selected.start, r['dimensions'])

    def close(self):
        try:
            super().close()
        finally:
            self.readings.clear()
            self.readings = OrderedDict()
            self.owned_bytes = 0
