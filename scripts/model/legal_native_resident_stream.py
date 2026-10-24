"""Bounded native I/O siblings; retain the certified reader's exact checks.

File handles cache addresses, not readings. Every returned native window is still
copied and checked by CertifiedNativeFrames under before/after file identities.
No arithmetic, source ordering, provenance or per-instrument layout changes.
"""
import os

import numpy as np

from model.legal_native_training_stream import CertifiedNativeFrames
from text.constellation_corpus import need


class NativeFileView:
    def __init__(self, path, descriptor, layout):
        self.fd = None
        self.offset, self.shape = layout
        self._mmap = self  # Same deterministic close protocol as the mmap reader.
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            s = os.fstat(fd)
            need((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                 == descriptor['stamps'][path], 'RESIDENT_NATIVE_OPEN_IDENTITY')
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def __getitem__(self, selected):
        need(isinstance(selected, slice) and selected.step is None
             and type(selected.start) is int and type(selected.stop) is int
             and 0 <= selected.start < selected.stop <= self.shape[0],
             'RESIDENT_NATIVE_SLICE')
        need(self.fd is not None, 'RESIDENT_NATIVE_CLOSED')
        rows = selected.stop-selected.start
        size = rows*self.shape[1]*4
        need(size <= 16 << 20, 'TRAIN_STREAM_PHYSICAL_MEMORY')
        raw = os.pread(self.fd, size, self.offset+selected.start*self.shape[1]*4)
        need(len(raw) == size, 'RESIDENT_NATIVE_SHORT_READ')
        return np.frombuffer(raw, dtype=np.float32).reshape(rows, self.shape[1])

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ResidentNativeFrames(CertifiedNativeFrames):
    """Use explicit positioned reads or a byte-bounded map cache.

An independent instance owns its handles; readers must not be used concurrently.
The original six-map implementation remains the numerical/source reference.
"""
    def __init__(self, data, audit_root, audit_pin, *, mode='pread',
                 handle_limit=64, mapped_bytes=1 << 30):
        need(mode in ('pread', 'mmap') and type(handle_limit) is int
             and 0 < handle_limit <= 256 and type(mapped_bytes) is int
             and 0 < mapped_bytes <= 1 << 30, 'RESIDENT_NATIVE_POLICY')
        super().__init__(data, audit_root, audit_pin)
        self.mode, self.handle_limit, self.mapped_limit = mode, handle_limit, mapped_bytes
        self.layouts = {}
        self.mapped_bytes = 0
        self.closed = False
        self.io_stats = dict(opens=0, hits=0, evictions=0, peak_handles=0,
                             peak_mapped_bytes=0)

    def mapping(self, name, start, descriptor):
        need(not self.closed, 'RESIDENT_NATIVE_CLOSED')
        key = (name, start)
        path = descriptor['root']/'vectors.npy'
        if key in self.maps:
            self.io_stats['hits'] += 1
            self.maps.move_to_end(key)
            return self.maps[key]
        size = descriptor['report']['windows']*descriptor['report']['dimensions']*4
        need(self.mode == 'pread' or size <= self.mapped_limit, 'RESIDENT_NATIVE_MAP_MEMORY')
        while self.maps and (len(self.maps) >= self.handle_limit or
                (self.mode == 'mmap' and self.mapped_bytes+size > self.mapped_limit)):
            _, old = self.maps.popitem(last=False)
            if self.mode == 'mmap': self.mapped_bytes -= old.nbytes
            old._mmap.close()
            self.io_stats['evictions'] += 1
        self.data.native.descriptor_fresh(descriptor)
        if self.mode == 'mmap' or key not in self.layouts:
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                r = descriptor['report']
                need(value.dtype == np.float32 and value.shape == (r['windows'], r['dimensions'])
                     and value.flags.c_contiguous, 'TRAIN_STREAM_VECTOR_LAYOUT')
                self.layouts[key] = (value.offset, value.shape)
            except BaseException:
                value._mmap.close()
                raise
            if self.mode == 'pread': value._mmap.close()
        if self.mode == 'pread':
            value = NativeFileView(path, descriptor, self.layouts[key])
        try:
            self.data.native.descriptor_fresh(descriptor)
        except BaseException:
            value._mmap.close()
            raise
        self.maps[key] = value
        self.mapped_bytes += size if self.mode == 'mmap' else 0
        self.io_stats['opens'] += 1
        self.io_stats['peak_handles'] = max(self.io_stats['peak_handles'], len(self.maps))
        self.io_stats['peak_mapped_bytes'] = max(self.io_stats['peak_mapped_bytes'], self.mapped_bytes)
        return value

    def frames(self, uids):
        need(not self.closed, 'RESIDENT_NATIVE_CLOSED')
        return super().frames(uids)

    def close(self):
        if self.closed: return
        try:
            self.finish()
        finally:
            for value in self.maps.values(): value._mmap.close()
            self.maps.clear()
            self.mapped_bytes = 0
            self.closed = True
