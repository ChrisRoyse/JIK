"""Immutable labels from a frozen student codebook; native slots stay authoritative."""
import json
import os
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import digest, file_sha, need

SHARD = 16384
TILE = 128


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def validate_uids(ids, population):
    need(isinstance(ids, np.ndarray) and ids.dtype == np.dtype('<i8') and ids.ndim == 1
         and 0 < len(ids) <= SHARD and bool((np.diff(ids) > 0).all())
         and 0 <= int(ids[0]) <= int(ids[-1]) < population, 'TARGET_CACHE_UID_SCOPE')


def validate_arrays(codes, counts, rows, sizes, roster):
    need(codes.dtype == counts.dtype == np.uint8 and codes.shape == (rows, len(sizes))
         and counts.shape == (rows, len(roster)), 'TARGET_CACHE_ARRAY_LAYOUT')
    need(all(bool((codes[:, i] < n).all()) for i, n in enumerate(sizes)), 'TARGET_CACHE_CODE_RANGE')
    need(bool((counts > 0).all()), 'TARGET_CACHE_WINDOW_COUNT')


class FrozenLabels:
    def __init__(self, model):
        need(not model.training and not any(p.requires_grad for p in model.parameters())
             and model.encoder.query.device.type == 'cpu', 'TARGET_CACHE_MODEL_FROZEN_CPU')
        self.model = model
        self.state = fingerprint(model.state_dict())
        # The parameter bytes remain the trained fp32 authority. This separate,
        # hash-bound runtime computes labels without changing the trained model.
        self.runtime = deepcopy(model).double()
        self.runtime_state = fingerprint(self.runtime.state_dict())
        self.versions = self.identity()
        self.sizes, self.roster = tuple(model.sizes), tuple(model.roster)

    def identity(self):
        return tuple((id(x), x._version) for m in (self.model,self.runtime)
                     for x in m.state_dict(keep_vars=True).values())

    def fresh(self):
        need(self.versions == self.identity() and all(not m.training and not any(p.requires_grad for p in m.parameters())
             for m in (self.model,self.runtime)), 'TARGET_CACHE_MODEL_CHANGED')

    @torch.no_grad()
    def encode(self, frames, *, reference=False, canonical=False):
        self.fresh()
        need(0 < len(frames) <= TILE, 'TARGET_CACHE_PHYSICAL_TILE')
        # Weight identity is invariant across this bounded loop. Individual
        # singleton forward/quantizer calls define the canonical label runtime.
        if canonical:
            parts = [self._measure((frame,), reference=reference) for frame in frames]
            result = tuple(np.concatenate([p[i] for p in parts]) for i in range(2))
        else: result = self._measure(frames, reference=reference)
        self.fresh()
        return result

    def _measure(self, frames, *, reference):
        latent = self.runtime.latent(frames)
        need(bool(torch.isfinite(latent).all()), 'TARGET_CACHE_NONFINITE_LATENT')
        codes = self.runtime.quantize(latent)[0].numpy()
        counts = np.asarray([[len(f[n]) for n in self.roster] for f in frames], dtype=np.int64)
        need(bool((counts <= 255).all()), 'TARGET_CACHE_COUNT_STORAGE')
        if reference:
            # Separate NumPy distance and selection path; no call to quantize.
            residual = latent.numpy().astype(np.float64); chosen = []
            for book in self.runtime.books:
                b = book.detach().numpy().astype(np.float64)
                distances = np.square(residual[:, None] - b[None]).sum(-1)
                selected = np.argmin(distances, axis=1); chosen.append(selected)
                residual = residual - b[selected]
            need(np.array_equal(codes, np.stack(chosen, -1)), 'TARGET_CACHE_REFERENCE_CODES')
        result = codes.astype(np.uint8), counts.astype(np.uint8)
        validate_arrays(*result, len(frames), self.sizes, self.roster)
        return result


def save_array(path, values):
    with Path(path).open('xb') as f:
        np.save(f, values, allow_pickle=False); f.flush(); os.fsync(f.fileno())


def shard_data(root, pin, binding, ids, *, committed=True):
    root = Path(root)
    need((root/'report.json').is_file() and file_sha(root/'report.json') == pin, 'TARGET_CACHE_REPORT_PIN')
    if committed:
        need((root/'COMMITTED').is_file() and (root/'COMMITTED').read_text().strip() == pin,
             'TARGET_CACHE_UNCOMMITTED')
    r = json.loads((root/'report.json').read_bytes())
    validate_uids(ids, binding['population'])
    need(r['complete'] and r['binding'] == binding and r['rows'] == len(ids)
         and r['uid_digest'] == digest(ids.tobytes()), 'TARGET_CACHE_SHARD_BINDING')
    arrays = []
    for name in ['codes', 'counts']:
        f = root/(name+'.npy'); item = r['files'][f.name]
        need(f.is_file() and f.stat().st_size == item['bytes'] and file_sha(f) == item['sha256'],
             'TARGET_CACHE_ARRAY_PIN')
        arrays.append(np.load(f, allow_pickle=False))
    validate_arrays(*arrays, len(ids), binding['sizes'], binding['roster'])
    return r, tuple(arrays)


def publish(root, reader, labels, binding, ids):
    root = Path(root); validate_uids(ids, binding['population'])
    need(binding['model_state'] == labels.state, 'TARGET_CACHE_MODEL_BINDING')
    root.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter(); cpu = time.process_time()
    phases = dict(reading=0., encoding=0., persistence=0.)
    codes = np.empty((len(ids), len(labels.sizes)), dtype=np.uint8)
    counts = np.empty((len(ids), len(labels.roster)), dtype=np.uint8)
    for lo in range(0, len(ids), TILE):
        tick = time.perf_counter(); frames = reader.frames(ids[lo:lo+TILE].tolist())
        phases['reading'] += time.perf_counter()-tick; tick = time.perf_counter()
        c, n = labels.encode(frames); codes[lo:lo+len(c)] = c; counts[lo:lo+len(n)] = n
        phases['encoding'] += time.perf_counter()-tick
    tick = time.perf_counter(); reader.finish(); labels.fresh()
    save_array(root/'codes.npy', codes); save_array(root/'counts.npy', counts)
    files = {f.name: dict(bytes=f.stat().st_size, sha256=file_sha(f)) for f in root.glob('*.npy')}
    phases['persistence'] += time.perf_counter()-tick
    report = dict(complete=True, binding=binding, rows=len(ids), first=int(ids[0]), last=int(ids[-1]),
                  uid_digest=digest(ids.tobytes()), files=files, phases_seconds=phases,
                  seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu,
                  native_windows=counts.astype(np.int64).sum(0).tolist())
    pin = write_committed(root/'report.json', report)
    _, saved = shard_data(root, pin, binding, ids, committed=False)
    need(all(np.array_equal(a,b) for a,b in zip((codes,counts),saved)), 'TARGET_CACHE_WRITE_READBACK')
    with (root/'COMMITTED').open('x') as f:
        f.write(pin+'\n'); f.flush(); os.fsync(f.fileno())
    sync_directory(root); sync_directory(root.parent)
    return pin


def ranges(plan):
    for start in range(0, plan['rows'], SHARD):
        stop = min(start+SHARD, plan['rows'])
        ids = (np.arange(start, stop, dtype='<i8') if plan['scope'] == 'full'
               else np.asarray(plan['uids'][start:stop], dtype='<i8'))
        yield start, stop, ids


def export(root, shards, plan, binding):
    root = Path(root); out = root/'export'; out.mkdir()
    values = [np.lib.format.open_memmap(out/(name+'.npy'), mode='w+', dtype='u1',
              shape=(plan['rows'], width)) for name,width in [('codes',len(binding['sizes'])),
                                                            ('counts',len(binding['roster']))]]
    need(len(shards) == (plan['rows']+SHARD-1)//SHARD, 'TARGET_CACHE_EXPORT_SHARDS')
    for ref,(lo,hi,ids) in zip(shards,ranges(plan)):
        need(ref['start'] == lo and ref['stop'] == hi, 'TARGET_CACHE_EXPORT_ORDER')
        _, arrays = shard_data(root/ref['path'], ref['pin'], binding, ids)
        for dest,src in zip(values,arrays): dest[lo:hi] = src
    for dest in values: dest.flush(); dest._mmap.close()
    for f in out.glob('*.npy'):
        with f.open('rb') as stream: os.fsync(stream.fileno())
    sync_directory(out)
    return {str(f.relative_to(root)): file_sha(f) for f in out.glob('*.npy')}
