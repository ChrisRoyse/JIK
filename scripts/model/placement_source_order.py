"""Order-only placement experiments with document-bound stochastic samples.

This is an additive experiment API, not a replacement for a running continuation.
One update still means one complete document with its original internal sequence.
"""
from collections import defaultdict
from fractions import Fraction
import heapq
from typing import NamedTuple


def need(ok, code):
    if not ok:
        raise ValueError('MNLM_PLACEMENT_ORDER_' + code)


class Document(NamedTuple):
    document: int
    source: str
    start: int
    stop: int
    targets: int
    original_step: int
    side: str


class Schedule:
    def __init__(self, rows, mode, noise_seed):
        need(mode in ('original', 'interleaved'), 'MODE')
        need(type(noise_seed) is int and noise_seed >= 0, 'SEED')
        original = tuple(Document(**row) for row in rows)
        need(bool(original) and len({r.document for r in original}) == len(original), 'DOCUMENTS')
        groups = defaultdict(list)
        for step, row in enumerate(original, 1):
            need(all(type(v) is int for v in (row.document, row.start, row.stop, row.targets, row.original_step))
                 and row.document >= 0 and row.start >= 0 and row.targets == row.stop-row.start-1 > 0,
                 'BOUNDARIES')
            need(isinstance(row.source, str) and row.source and row.side == 'train', 'SOURCE_OR_SIDE')
            need(row.original_step == step, 'ORIGINAL_ORDER')
            groups[row.source].append(row)
        if mode == 'original':
            ordered = original
        else:
            heap = [(Fraction(1, 2*len(group)), name, 0) for name, group in groups.items()]
            heapq.heapify(heap); ordered = []
            while heap:
                _, name, position = heapq.heappop(heap)
                ordered.append(groups[name][position]); position += 1
                if position < len(groups[name]):
                    heapq.heappush(heap, (Fraction(2*position+1, 2*len(groups[name])), name, position))
            ordered = tuple(ordered)
        # Tuple members are immutable; no full-plan hash/serialization inside the update loop.
        self.original = original; self.documents = ordered
        self.mode = mode; self.noise_seed = noise_seed

    def entry(self, step):
        need(type(step) is int and 0 <= step < len(self.documents), 'CURSOR')
        row = self.documents[step]
        return dict(step=step+1, **row._asdict(), noise_seed=self.noise_seed+row.original_step-1)

    def entries(self):
        return [self.entry(step) for step in range(len(self.documents))]


def update(model, optimizer, item, step, policy, *, target, schedule):
    """Reuse the existing arithmetic, changing only order and the explicit RNG address.

The adjusted seed base is a fresh call-local dictionary. The canonical policy,
optimizer cursor, initialization, and frozen instruments are never mutated.
"""
    from model.native_placement_batched_checks import update as numerical_update
    row = schedule.entry(step)
    need(policy['noise_seed'] == schedule.noise_seed, 'POLICY_SEED')
    need(item['side'] == 'train' and item['document'] == row['document'], 'NEXT_DOCUMENT')
    need(item['start'] == row['start'] and item['stop'] == row['stop']
         and len(item['codes']) == row['targets'], 'ITEM_BOUNDARIES')
    call_policy = dict(policy, noise_seed=row['noise_seed']-step)
    result = numerical_update(model, optimizer, item, step, call_policy, target=target)
    need(result[0]['noise_seed'] == row['noise_seed'], 'NUMERICAL_SEED')
    return result
