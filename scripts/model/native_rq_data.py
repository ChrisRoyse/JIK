"""Bounded native-window reads and TRAIN statistics for a derived semantic index."""
from collections import OrderedDict
import hashlib

import numpy as np
import torch

from kernel.legal_native_associations import ProofLane
from text.constellation_corpus import need


class NativeRQData:
    def __init__(self, data, proof_root, proof_pin):
        self.data = data; self.roster = data.roster
        self.proofs = {n: ProofLane(proof_root, proof_pin, data.source, n,
                                   data.native.config['instruments'][n]['lens_id']) for n in self.roster}
        self.cache = OrderedDict(); self.cache_bytes = 0; self.peak_cache_bytes = 0
        self.maximum_read_bytes = 0; self.cache_limit = 32 << 20; self.read_limit = 16 << 20

    def frames(self, uids):
        need(uids and len(uids) <= 128 and all(type(u) is int for u in uids), 'RQ_DATA_READ_SCOPE')
        missing = sorted({u for u in uids if u not in self.cache})
        for uid in uids:
            if uid in self.cache: self.cache.move_to_end(uid)
        collected = {u: {} for u in missing}; used = 0
        for name in self.roster:
            ordinary = [u for u in missing if u not in self.proofs[name].rows]
            if ordinary:
                batch = self.data.native.read_many(name, ordinary, memory_bytes=self.read_limit-used)
                for i, uid in enumerate(ordinary):
                    value = batch.values[batch.offsets[i]:batch.offsets[i+1]].copy()
                    collected[uid][name] = torch.from_numpy(value); used += value.nbytes
                del batch
            for uid in missing:
                if uid in self.proofs[name].rows:
                    value = self.proofs[name]([uid])[0].copy()
                    collected[uid][name] = torch.from_numpy(value); used += value.nbytes
            need(used <= self.read_limit, 'RQ_DATA_READ_MEMORY')
        self.maximum_read_bytes = max(self.maximum_read_bytes, used)
        for uid, frame in collected.items():
            size = sum(x.numel()*x.element_size() for x in frame.values())
            need(size <= self.cache_limit, 'RQ_DATA_ATOM_MEMORY')
            while self.cache_bytes+size > self.cache_limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= sum(x.numel()*x.element_size() for x in old.values())
            self.cache[uid] = frame; self.cache_bytes += size
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        # A read tile is smaller than the cache; it must never evict its own rows.
        need(all(u in self.cache for u in uids), 'RQ_DATA_TILE_EVICTED')
        for uid in uids: self.cache.move_to_end(uid)
        return tuple(self.cache[u] for u in uids)

    def tiles(self, uids):
        for lo in range(0, len(uids), 128):
            ids = uids[lo:lo+128]; yield ids, self.frames(ids)


def fingerprint_frames(hasher, ids, frames, roster):
    for uid, frame in zip(ids, frames):
        hasher.update(int(uid).to_bytes(8, 'little'))
        for name in roster:
            value = frame[name].numpy()
            hasher.update(name.encode()+b'\0'); hasher.update(np.asarray(value.shape, dtype='<i8').tobytes())
            hasher.update(value.tobytes())


def training_components(data, uids, presence):
    need(uids and len(set(uids)) == len(uids) and all(presence[u]&1 for u in uids), 'RQ_DATA_NON_TRAIN')
    source = data.source; offsets = source.arrays['offsets']; locations = source.arrays['occurrences']
    selected = []
    for uid in uids:
        occurrences = locations[int(offsets[uid]):int(offsets[uid+1])]
        eligible = occurrences[data.sides[occurrences] == 0]
        need(len(eligible) > 0, 'RQ_DATA_TRAIN_OCCURRENCE'); selected.append(int(eligible[0]))
    documents = np.searchsorted(data.documents[:, 1], selected, side='right')
    return selected, data.components[documents].astype('<i8').tolist()


@torch.no_grad()
def statistics(reader, uids, presence):
    need(all(presence[u]&1 for u in uids) and len(set(uids)) == len(uids), 'RQ_DATA_NON_TRAIN')
    sums = {}; squares = {}; positions = {}; weights = {}; counts = []; native = hashlib.sha256()
    for ids, frames in reader.tiles(uids):
        fingerprint_frames(native, ids, frames, reader.roster)
        for frame in frames:
            counts.append([len(frame[n]) for n in reader.roster])
            for name in reader.roster:
                x = frame[name].double(); w = len(x)
                if name not in sums:
                    sums[name] = torch.zeros(x.shape[1], dtype=torch.float64)
                    squares[name] = torch.zeros_like(sums[name]); positions[name] = []; weights[name] = []
                sums[name] += x.mean(0); squares[name] += x.square().mean(0)
                for j in range(w):
                    if j == len(positions[name]): positions[name].append(torch.zeros_like(sums[name])); weights[name].append(0.)
                    positions[name][j] += x[j]/w; weights[name][j] += 1/w
    centers = {n: sums[n]/len(uids) for n in reader.roster}
    variance = {n: (squares[n]/len(uids)-centers[n].square()).mean() for n in reader.roster}
    need(all(bool(torch.isfinite(v) & (v > 0)) for v in variance.values()), 'RQ_DATA_CONSTANT_INSTRUMENT')
    return dict(centers=centers, variance=variance,
                positions={n: torch.stack([x/w for x,w in zip(positions[n],weights[n])]) for n in reader.roster},
                position_weights=weights, window_counts=torch.tensor(counts, dtype=torch.int64),
                uids=uids, native_sha256=native.hexdigest())


def matched_donors(frames, ids, components, roster):
    signatures = [tuple(len(f[n]) for n in roster) for f in frames]
    result = list(range(len(frames))); eligible = [False]*len(frames)
    for i in range(len(frames)):
        for offset in range(1, len(frames)):
            j = (i+offset) % len(frames)
            if (ids[i] != ids[j] and components[i] != components[j] and signatures[i] == signatures[j]
                    and any(not torch.equal(frames[i][n], frames[j][n]) for n in roster)):
                result[i] = j; eligible[i] = True; break
    return torch.tensor(result, dtype=torch.int64), torch.tensor(eligible, dtype=torch.bool)
