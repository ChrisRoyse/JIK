"""Read adjacent certified native windows once, preserving every typed output byte."""
import time

import numpy as np
import torch

from model.legal_native_resident_stream import ResidentNativeFrames
from text.constellation_corpus import need
from text.legal_native_store import SHARD_INPUTS


class CoalescedNativeFrames(ResidentNativeFrames):
    """Bounded pread runs, independent output arrays, unchanged source checks.

    No holes are read, and no values or provenance are cached. Source intervals
    are sorted only for I/O; results retain caller order and duplicate entries.
    An instance owns its handles and must not be shared between threads.
    """
    def __init__(self, *args, handle_limit=64):
        super().__init__(*args, mode='pread', handle_limit=handle_limit)
        self.range_stats = dict(requests=0, unique_ranges=0, read_runs=0,
                                read_bytes=0, peak_slab_bytes=0)

    def frames(self, uids):
        need(not self.closed, 'RESIDENT_NATIVE_CLOSED')
        began = time.perf_counter(); self.data.fresh()
        need(isinstance(uids, (list, tuple)) and 0 < len(uids) <= 128
             and all(type(u) is int and 0 <= u < self.data.native.config['inputs'] for u in uids),
             'TRAIN_STREAM_UID_SCOPE')
        groups = {}
        for i, uid in enumerate(uids):
            groups.setdefault(uid//SHARD_INPUTS*SHARD_INPUTS, []).append((i, uid))
        descriptors = {}; payload = windows = 0
        for name in self.roster:
            for start, rows in sorted(groups.items()):
                d = self.descriptor(name, start); intervals = {}
                for i, uid in rows:
                    a, b = map(int, d['offsets'][uid-start:uid-start+2])
                    need(0 <= a < b <= d['report']['windows'], 'TRAIN_STREAM_MISSING_WINDOW')
                    intervals.setdefault((a, b), []).append(i)
                    windows += b-a; payload += (b-a)*d['report']['dimensions']*4
                descriptors[name, start] = (d, intervals)
        need(payload <= 16 << 20, 'TRAIN_STREAM_PHYSICAL_MEMORY')
        tick = time.perf_counter(); frames = [{} for _ in uids]
        for (name, start), (d, intervals) in descriptors.items():
            self.data.native.descriptor_fresh(d); values = self.mapping(name, start, d)
            ranges = sorted(intervals); runs = []
            row_bytes = d['report']['dimensions']*4
            for a, b in ranges:
                if runs and runs[-1][1] == a and (b-runs[-1][0])*row_bytes <= 16 << 20:
                    runs[-1][1] = b; runs[-1][2].append((a, b))
                else:
                    runs.append([a, b, [(a, b)]])
            for lo, hi, members in runs:
                slab = values[lo:hi]
                for a, b in members:
                    for i in intervals[a, b]:
                        v = np.array(slab[a-lo:b-lo], copy=True)
                        need(np.isfinite(v).all(), 'TRAIN_STREAM_NONFINITE')
                        frames[i][name] = torch.from_numpy(v)
                self.range_stats['read_runs'] += 1
                self.range_stats['read_bytes'] += slab.nbytes
                self.range_stats['peak_slab_bytes'] = max(self.range_stats['peak_slab_bytes'], slab.nbytes)
                del slab
            self.range_stats['requests'] += sum(map(len, intervals.values()))
            self.range_stats['unique_ranges'] += len(intervals)
            self.data.native.descriptor_fresh(d)
        self.data.fresh(); self.stats['gather_seconds'] += time.perf_counter()-tick
        self.stats['calls'] += 1; self.stats['inputs'] += len(uids); self.stats['windows'] += windows
        self.stats['payload_bytes'] += payload
        self.stats['maximum_payload_bytes'] = max(payload, self.stats['maximum_payload_bytes'])
        self.stats['reader_seconds'] += time.perf_counter()-began
        return tuple(frames)
