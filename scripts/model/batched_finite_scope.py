"""Optional bounded device/dtype batches for exact Boolean tensor validation."""
from contextlib import contextmanager

import torch

from model import native_device_transport as transport
from model import native_placement_efficient as efficient
from text.constellation_corpus import need

ORIGINAL = transport.finite_tensors
ACTIVE = False
JOIN_BYTES = 1 << 20


class FiniteBatches:
    def __init__(self, join_bytes=JOIN_BYTES):
        need(type(join_bytes) is int and 16 <= join_bytes <= JOIN_BYTES, 'FINITE_BATCH_ALLOWANCE')
        self.join_bytes = join_bytes
        self.closed = False
        self.counts = dict(calls=0, tensors=0, elements=0, batches=0, direct_checks=0,
                           peak_joined_bytes=0, peak_batch_bytes=0)

    @torch.no_grad()
    def __call__(self, values):
        need(not self.closed, 'FINITE_BATCH_CLOSED')
        self.counts['calls'] += 1
        pending, sizes, flags = {}, {}, {}

        def flush(key):
            parts = pending[key]
            if not parts:
                return
            # This buffer supplies only a Boolean predicate, never a learned slot.
            joined = torch.cat(parts) if len(parts) > 1 else parts[0]
            size = joined.numel()*joined.element_size()
            need(size <= self.join_bytes, 'FINITE_BATCH_JOIN_EXCEEDED')
            self.counts['peak_batch_bytes'] = max(self.counts['peak_batch_bytes'], size)
            if len(parts) > 1:
                self.counts['peak_joined_bytes'] = max(self.counts['peak_joined_bytes'], size)
            flags[key[0]].append(torch.isfinite(joined).all())
            self.counts['batches'] += 1
            parts.clear()
            sizes[key] = 0

        for value in values:
            device = value.device
            flags.setdefault(device, [])
            self.counts['tensors'] += 1
            self.counts['elements'] += value.numel()
            if value.layout != torch.strided or not value.is_contiguous():
                # Retain the original layout-specific operation on its own device.
                flags[device].append(torch.isfinite(value).all())
                self.counts['direct_checks'] += 1
                continue
            key = (device, value.dtype)
            pending.setdefault(key, [])
            sizes.setdefault(key, 0)
            width = value.element_size()
            flat = value.detach().view(-1)
            offset = 0
            while offset < flat.numel():
                available = (self.join_bytes-sizes[key])//width
                if available == 0:
                    flush(key)
                    continue
                count = min(available, flat.numel()-offset)
                pending[key].append(flat[offset:offset+count])
                sizes[key] += count*width
                offset += count
                if self.join_bytes-sizes[key] < width:
                    flush(key)
        for key in pending:
            flush(key)
        return all(bool(torch.stack(group).all()) for group in flags.values() if group)

    def report(self):
        return dict(self.counts, join_bytes=self.join_bytes, closed=self.closed)


@contextmanager
def batched_finiteness():
    global ACTIVE
    need(not ACTIVE and transport.finite_tensors is ORIGINAL
         and efficient.finite_tensors is ORIGINAL, 'FINITE_BATCH_ALREADY_INSTALLED')
    checker = FiniteBatches()
    ACTIVE = True
    transport.finite_tensors = efficient.finite_tensors = checker
    try:
        yield checker
    finally:
        intact = (transport.finite_tensors is checker and efficient.finite_tensors is checker)
        transport.finite_tensors = efficient.finite_tensors = ORIGINAL
        ACTIVE = False
        checker.closed = True
        need(intact, 'FINITE_BATCH_OWNERSHIP_CHANGED')
