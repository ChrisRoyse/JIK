"""Complete task windows without corpus-size ceilings; finite tiled scoring.

Tile size bounds temporary arithmetic, never the number of observations.
Historical ChunkedLane remains unchanged for pinned journal replay.
"""
import numpy as np

from claims.capability_population import sha
from kernel.chunked_route import ChunkedLane
from kernel.native_route import check


class TaskLane(ChunkedLane):
    def __init__(self, atoms):
        self.ids = sorted(atoms)
        check(self.ids and all(isinstance(k, str) and k for k in self.ids), 'TASK_LANE_IDS')
        pieces = [atoms[k] for k in self.ids]
        check(all(a.dtype == np.float32 and a.ndim == 2 and len(a) > 0
                  and a.shape[1] > 0 and np.isfinite(a).all() for a in pieces), 'TASK_LANE_VALUES')
        dims = pieces[0].shape[1]
        check(all(a.shape[1] == dims for a in pieces), 'TASK_LANE_DIMENSIONS')
        self.offsets = np.asarray([0] + list(np.cumsum([len(a) for a in pieces])), np.int64)
        self.original = np.concatenate(pieces)
        self.original.flags.writeable = False
        self.unit = self.normalize(self.original)
        self.unit.flags.writeable = False
        self.rows = {key: i for i, key in enumerate(self.ids)}
        self.owners = np.repeat(np.arange(len(self.ids)), np.diff(self.offsets))
        self.signatures = {k: sha(atoms[k].astype('<f4', copy=False).tobytes()) for k in self.ids}

    @staticmethod
    def normalize(values):
        check(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), 'TASK_LANE_QUERY')
        v = np.asarray(values, np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        check((norms > 1e-10).all(), 'ZERO_NORM')
        return v / norms

    def query(self, values):
        check(values.ndim == 2 and values.shape[1] == self.unit.shape[1]
              and len(values) > 0, 'TASK_LANE_QUERY_SHAPE')
        return self.normalize(values)


def lanes(population, ids):
    return {name: TaskLane({k: population.values[k][name] for k in ids})
            for name in sorted(population.specs)}
