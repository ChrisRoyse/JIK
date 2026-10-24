"""Preserve native ANN results while restoring the documented batch axis."""
import numpy as np
from usearch.index import BatchMatches, Matches

from text.constellation_corpus import need


def batch_layout(matches, rows, count):
    """USearch 2.26.5 returns Matches whenever there is exactly one query.

    No search, score, padding or count substitution occurs here. Both supported
    layouts must contain exactly the requested number of real results.
    """
    need(type(rows) is int and rows > 0 and type(count) is int and count > 0,
         'NATIVE_SEARCH_REQUEST')
    need(type(matches) in (Matches, BatchMatches), 'NATIVE_SEARCH_RESULT_TYPE')
    singleton = type(matches) is Matches
    shape = (count,) if singleton else (rows, count)
    need(not singleton or rows == 1, 'NATIVE_SEARCH_SINGLETON_CARDINALITY')
    need(isinstance(matches.keys, np.ndarray) and matches.keys.dtype == np.uint64
         and matches.keys.shape == shape, 'NATIVE_SEARCH_KEYS_LAYOUT')
    need(isinstance(matches.distances, np.ndarray) and matches.distances.shape == shape
         and matches.distances.dtype == np.float32 and np.isfinite(matches.distances).all(),
         'NATIVE_SEARCH_DISTANCES_LAYOUT')
    for name in ('visited_members', 'computed_distances'):
        value = getattr(matches, name)
        need(type(value) is int and value > 0, 'NATIVE_SEARCH_WORK_COUNTER')
    if singleton:
        return BatchMatches(keys=matches.keys[None, :], distances=matches.distances[None, :],
                            counts=np.asarray([count], dtype=np.int64),
                            visited_members=matches.visited_members,
                            computed_distances=matches.computed_distances)
    need(isinstance(matches.counts, np.ndarray) and matches.counts.shape == (rows,)
         and np.issubdtype(matches.counts.dtype, np.integer)
         and np.array_equal(matches.counts, np.full(rows, count)), 'NATIVE_SEARCH_COUNTS')
    return matches
