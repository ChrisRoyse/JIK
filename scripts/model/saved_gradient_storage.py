"""Bound saved autograd storage without changing its value, dtype or device.

Counts referenced backing storages once, including views and parameters. This is
not total process RSS or an allocator peak. No CPU offload or reduced precision.
"""
from pathlib import Path

import torch
from text.constellation_corpus import need


class SavedTensor:
    def __init__(self, tensor, owner, key):
        # Holding the graph output itself here would form an autograd cycle.
        self.tensor = tensor.detach()
        self.version = tensor._version
        self.owner, self.key = owner, key

    def __del__(self):
        self.owner.release(self.key)


class SavedStorage:
    def __init__(self, allowance, device):
        need(type(allowance) is int and allowance > 0, 'GRADIENT_STORAGE_ALLOWANCE')
        device = torch.device(device)
        if device.type == 'cuda':
            available = torch.cuda.mem_get_info(device)[0]
        else:
            need(device.type == 'cpu', 'GRADIENT_STORAGE_DEVICE')
            row = next(x for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
            available = int(row.split()[1])*1024
        self.limit = min(allowance, available)
        self.current = self.peak = self.packs = self.unpacks = 0
        self.storages = {}

    def pack(self, tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        extra = key[2] if key not in self.storages else 0
        need(self.current+extra <= self.limit, 'GRADIENT_STORAGE_MEMORY')
        self.storages[key] = self.storages.get(key, 0)+1
        self.current += extra
        self.peak = max(self.peak, self.current)
        self.packs += 1
        return SavedTensor(tensor, self, key)

    def unpack(self, saved):
        need(saved.owner is self and saved.tensor._version == saved.version,
             'GRADIENT_STORAGE_MUTATED')
        self.unpacks += 1
        return saved.tensor

    def release(self, key):
        self.storages[key] -= 1
        if self.storages[key] == 0:
            del self.storages[key]
            self.current -= key[2]

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def report(self):
        return dict(limit=self.limit, peak_saved_storage_bytes=self.peak,
                    remaining_saved_storage_bytes=self.current, packs=self.packs,
                    unpacks=self.unpacks, remaining_storages=len(self.storages))
