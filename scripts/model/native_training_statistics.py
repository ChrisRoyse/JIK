"""Bounded equal-atom statistics over complete, typed native windows.

These are student normalization/baseline statistics, never substitute readings.
The independent position-wise reducer intentionally uses different arithmetic.
"""
import numpy as np

from text.constellation_corpus import need

TILE = 1024
MEMORY = 64 << 20
ERROR = 2e-12


def empty(dimensions, maximum):
    need(type(dimensions) is int and dimensions > 0 and type(maximum) is int and maximum > 0,
         'TRAIN_STATS_DIMENSIONS')
    return dict(sum=np.zeros(dimensions, np.float64), square=np.zeros(dimensions, np.float64),
                positions=np.zeros((maximum, dimensions), np.float64),
                weights=np.zeros(maximum, np.float64), position_atoms=np.zeros(maximum, np.int64),
                histogram=np.zeros(maximum, np.int64), inputs=np.zeros((), np.int64),
                windows=np.zeros((), np.int64))


def validate(vectors, offsets, selected, maximum):
    need(vectors.ndim == 2 and vectors.dtype == np.float32 and offsets.dtype == np.int64
         and offsets.ndim == 1 and len(offsets) > 1 and offsets[0] == 0
         and int(offsets[-1]) == len(vectors) and bool((np.diff(offsets) > 0).all()),
         'TRAIN_STATS_WINDOW_LAYOUT')
    need(selected.ndim == 1 and selected.dtype == np.int64 and len(selected) > 0
         and bool(((selected >= 0) & (selected < len(offsets)-1)).all())
         and bool((np.diff(selected) > 0).all()), 'TRAIN_STATS_SELECTED')
    counts = offsets[selected+1]-offsets[selected]
    need(int(counts.max()) <= maximum, 'TRAIN_STATS_UNSEEN_POSITION')
    return counts


def reduce(vectors, offsets, selected, maximum, *, reference=False):
    counts = validate(vectors, offsets, selected, maximum)
    dimensions = vectors.shape[1]; out = empty(dimensions, maximum); peak = 0
    for lo in range(0, len(selected), TILE):
        ids = selected[lo:lo+TILE]; sizes = counts[lo:lo+TILE]
        if reference:
            # One window position at a time, with per-atom weights. No grouped
            # window averaging or producer accumulator is used here.
            for pos in range(maximum):
                mask = sizes > pos
                if not mask.any(): continue
                at = offsets[ids[mask]]+pos; weight = 1./sizes[mask]
                need(len(at)*dimensions*32 <= MEMORY, 'TRAIN_STATS_PHYSICAL_MEMORY')
                x = vectors[at].astype(np.float64)
                need(bool(np.isfinite(x).all()), 'TRAIN_STATS_NONFINITE')
                weighted = (x*weight[:, None]).sum(0)
                out['sum'] += weighted; out['square'] += (x*x*weight[:, None]).sum(0)
                out['positions'][pos] += weighted; out['weights'][pos] += weight.sum()
                out['position_atoms'][pos] += len(at)
                peak = max(peak, len(at)*dimensions*32)
        else:
            for size0 in np.unique(sizes):
                size = int(size0); group = ids[sizes == size]
                per_atom = size*(dimensions*32+16)
                take = min(TILE, MEMORY//per_atom)
                need(take > 0, 'TRAIN_STATS_PHYSICAL_MEMORY')
                for start in range(0, len(group), take):
                    chunk = group[start:start+take]
                    at = offsets[chunk, None]+np.arange(size)[None]
                    x = vectors[at].astype(np.float64)
                    need(bool(np.isfinite(x).all()), 'TRAIN_STATS_NONFINITE')
                    out['sum'] += x.mean(1).sum(0)
                    out['square'] += np.square(x).mean(1).sum(0)
                    out['positions'][:size] += (x/size).sum(0)
                    out['weights'][:size] += len(chunk)/size
                    out['position_atoms'][:size] += len(chunk)
                    peak = max(peak, len(chunk)*per_atom)
        out['histogram'] += np.bincount(sizes, minlength=maximum+1)[1:]
    out['inputs'][...] = len(selected); out['windows'][...] = int(counts.sum())
    return out, counts, peak


def add(total, partial):
    need(total.keys() == partial.keys(), 'TRAIN_STATS_FIELDS')
    for key in total:
        need(total[key].shape == partial[key].shape and total[key].dtype == partial[key].dtype,
             'TRAIN_STATS_PARTIAL_SHAPE')
        total[key] += partial[key]


def finish(total):
    n = int(total['inputs'])
    need(n > 0 and bool((total['weights'] > 0).all()), 'TRAIN_STATS_EMPTY_POSITION')
    center = total['sum']/n
    variance = np.asarray((total['square']/n-center*center).mean())
    need(bool(np.isfinite(variance) and variance > 0), 'TRAIN_STATS_CONSTANT')
    return dict(center=center, variance=variance,
                positions=total['positions']/total['weights'][:, None],
                weights=total['weights'].copy(), position_atoms=total['position_atoms'].copy(),
                histogram=total['histogram'].copy(), inputs=total['inputs'].copy(),
                windows=total['windows'].copy())


def compare(actual, expected):
    need(actual.keys() == expected.keys(), 'TRAIN_STATS_COMPARE_FIELDS')
    errors = {}
    for key, value in actual.items():
        other = expected[key]
        need(value.shape == other.shape and value.dtype == other.dtype, 'TRAIN_STATS_COMPARE_SHAPE')
        if value.dtype.kind in 'iu':
            need(np.array_equal(value, other), 'TRAIN_STATS_COMPARE_COUNTS')
        else:
            scale = max(1., float(np.max(np.abs(other), initial=0.)))
            errors[key] = float(np.max(np.abs(value-other), initial=0.))/scale
            need(np.isfinite(value).all() and errors[key] <= ERROR, 'TRAIN_STATS_COMPARE_ARITHMETIC')
    return errors
