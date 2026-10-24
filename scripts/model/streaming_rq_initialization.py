"""Out-of-core canonical CPU farthest-point initialization of a derived RQ index.

Native slots are never concatenated here: the input is the quantizer's own latent.
Every TRAIN row participates; work is tiled, with deterministic first-row ties.
"""
from pathlib import Path

import numpy as np
import torch

from text.constellation_corpus import file_sha, need

TILE = 4096
MEMORY = 64 << 20


def memory_check(rows, width, sizes, tile):
    need(type(rows) is int and type(width) is int and rows > 0 and width > 0,
         'STREAM_RQ_LAYOUT')
    need(sizes and all(type(s) is int and s > 1 for s in sizes) and rows >= max(sizes),
         'STREAM_RQ_SUPPORT')
    need(type(tile) is int and tile > 0, 'STREAM_RQ_TILE')
    # Conservative live workspace, including Torch elementwise intermediates.
    working = 8*tile*width*4 + 8*128*max(sizes)*width*4 + 8*tile
    need(working <= MEMORY, 'STREAM_RQ_PHYSICAL_MEMORY')
    return working


def open_latents(path, pin, sizes, tile):
    need(file_sha(path) == pin, 'STREAM_RQ_LATENT_PIN')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    need(values.dtype == np.float32 and values.ndim == 2, 'STREAM_RQ_LAYOUT')
    memory_check(*values.shape, sizes, tile)
    return values


@torch.no_grad()
def initialize(path, pin, output, sizes, *, tile=TILE, progress=None):
    path, output = Path(path), Path(output)
    values = open_latents(path, pin, sizes, tile); n, width = values.shape
    output.mkdir(exist_ok=False)
    residual = np.lib.format.open_memmap(output/'residual.npy', mode='w+', dtype='<f4', shape=values.shape)
    distance = np.lib.format.open_memmap(output/'distance.npy', mode='w+', dtype='<f4', shape=(n,))
    codes = np.lib.format.open_memmap(output/'codes.npy', mode='w+', dtype='<i8', shape=(n,len(sizes)))
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    for lo in range(0,n,tile):
        x = np.array(values[lo:lo+tile], copy=True)
        need(np.isfinite(x).all(), 'STREAM_RQ_NONFINITE'); residual[lo:lo+len(x)] = x
    residual.flush(); books = []; choices = []; phases = []
    for level, size in enumerate(sizes):
        selected = [0]
        for lo in range(0,n,tile):
            x = torch.from_numpy(np.array(residual[lo:lo+tile], copy=True))
            d = (x-torch.from_numpy(residual[0].copy())).square().sum(-1)
            need(bool(torch.isfinite(d).all()), 'STREAM_RQ_DISTANCE_NONFINITE')
            distance[lo:lo+len(x)] = d.numpy()
        for _ in range(1,size):
            distance[selected] = -np.inf; best = -np.inf; winner = None
            for lo in range(0,n,tile):
                d = np.array(distance[lo:lo+tile], copy=True); j = int(d.argmax())
                if d[j] > best: best, winner = float(d[j]), lo+j
            need(winner is not None and np.isfinite(best) and best > 0, 'STREAM_RQ_COLLAPSE')
            selected.append(winner); center = torch.from_numpy(residual[winner].copy())
            for lo in range(0,n,tile):
                x = torch.from_numpy(np.array(residual[lo:lo+tile], copy=True))
                d = (x-center).square().sum(-1)
                need(bool(torch.isfinite(d).all()), 'STREAM_RQ_DISTANCE_NONFINITE')
                old = torch.from_numpy(np.array(distance[lo:lo+len(x)], copy=True))
                distance[lo:lo+len(x)] = torch.minimum(old,d).numpy()
        book = torch.from_numpy(np.array(residual[selected], copy=True)); cursor = 0
        # The existing canonical implementation uses 128 rows for this reduction.
        for lo in range(0,n,128):
            x = torch.from_numpy(np.array(residual[lo:lo+128], copy=True))
            d = (x[:,None]-book[None]).square().sum(-1)
            need(bool(torch.isfinite(d).all()), 'STREAM_RQ_DISTANCE_NONFINITE')
            ids = d.argmin(-1); remainder = x-book[ids]
            codes[lo:lo+len(x),level] = ids.numpy(); residual[lo:lo+len(x)] = remainder.numpy(); cursor += len(x)
        need(cursor == n, 'STREAM_RQ_COVERAGE'); residual.flush(); codes.flush(); distance.flush()
        books.append(book); choices.append(selected)
        item = dict(level=level, rows=cursor, selected=selected, residual_sha256=file_sha(output/'residual.npy'))
        phases.append(item)
        if progress: progress(item)
    need((path.stat().st_size,path.stat().st_mtime_ns) == before and file_sha(path) == pin,
         'STREAM_RQ_SOURCE_CHANGED')
    for mapped in [values,residual,distance,codes]: mapped._mmap.close()
    return dict(books=books, choices=choices, phases=phases,
                working_bytes=memory_check(n,width,sizes,tile), rows=n)
