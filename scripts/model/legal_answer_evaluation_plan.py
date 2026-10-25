"""Complete held-out schedule; controls depend only on observed native layout.

Source IDs route data, never become predictor features. No future target is read
to select a control. Buckets are built once, not searched once per candidate.
"""
from bisect import bisect_left, bisect_right
import hashlib
from pathlib import Path

import numpy as np
from model import legal_complete_answer_plan as source_plan
from text.constellation_corpus import need

PLAN = ('legal-complete-answer-plan-v2-20261024',
        '892de320b5b122235976ba4066ec4d9b4a1668a6261fd6ed1342e84e3cfae960')
COLD = ('legal-complete-answer-plan-cold-v2-20261024',
        'ec9a122b7fb7f0ddbaddc45d9d529e5ff3180f56d24b0ccd49a0aee7dd54b740')
POLICY = dict(schema='legal-answer-evaluation-plan-v1',
    cases='all_development_and_query_documents_in_source_order',
    donor='TRAIN_same_collection_observed_count_ordered_native_layout_different_prefix',
    donor_order='public_uid_tuple_then_document_id',
    distinctness='atom_addresses_only_native_vector_equality_checked_at_generation',
    selection='sha256_of_bound_source_and_public_row_modulo_eligible_count',
    unmatched='CONTROL_UNAVAILABLE_retained_in_denominator',
    contexts=['correct', 'reversed', 'train_donor'],
    proposals={'shared': 1, 'own_slot': 1, 'whole_proposals': 6},
    future_used_for_control=False, untouched_test=False, identifiers_are_features=False)
DTYPE = np.dtype([(n, '<i8') for n in
    ('document', 'side', 'component', 'collection', 'observed', 'donor', 'eligible', 'status', 'reversal_distinct')] +
    [('uids', '<i8', (2,)), ('donor_uids', '<i8', (2,)), ('layout', 'u1', (2, 6))])
PREFIX = np.dtype([('collection', '<i8'), ('observed', '<i8'),
                   ('uids', '<i8', (2,)), ('layout', 'u1', (2, 6))])
STATUS = {0: 'READY', 1: 'CONTROL_UNAVAILABLE_LAYOUT', 2: 'CONTROL_UNAVAILABLE_DISTINCT_PREFIX'}


def authenticate(root):
    source = source_plan.Source(root)
    try:
        r, c = [source.report(Path(root)/n, p) for n, p in (PLAN, COLD)]
        need(r['mode'] == 'publish' and c['mode'] == 'cold' and c['previous_pin'] == PLAN[1]
             and r['binding'] == c['binding'] == source.binding
             and r['summary'] == c['summary'] and r['controls'] == c['controls'],
             'ANSWER_EVAL_SOURCE_AUTHORITY')
        source.bind(Path(root)/PLAN[0]/'plan.npy', r['files']['plan.npy'])
        return source
    except BaseException:
        source.close()
        raise


def prefixes(source):
    rows = np.zeros(len(source.documents), dtype=PREFIX)
    rows['uids'] = -1
    collections = {n: i for i, n in enumerate(source.collections)}
    for i, (lo, hi) in enumerate(source.documents):
        # This is the already-frozen public prefix policy, not a generation limit.
        n = min(2, int(hi-lo)-1); ids = source.mapping[int(lo):int(lo)+n]
        rows[i]['collection'] = collections[source.metadata[i]['source_path']]
        rows[i]['observed'] = n; rows[i]['uids'][:n] = ids
        rows[i]['layout'][:n] = source.counts[ids]
    return rows


def key(row):
    return int(row['collection']), int(row['observed']), row['layout'].tobytes()


def buckets(source, public):
    result = {}
    for i in np.flatnonzero(source.sides == 0):
        result.setdefault(key(public[i]), []).append((tuple(map(int, public[i]['uids'])), int(i)))
    for values in result.values():
        values.sort()
    return result


def selector(source, number, public):
    h = hashlib.sha256(b'legal-answer-evaluation-donor-v1\0')
    h.update(source.binding['corpus']['pin'].encode('ascii'))
    h.update(int(number).to_bytes(8, 'little'))
    h.update(public.tobytes())
    return int.from_bytes(h.digest(), 'big')


def select(source, number, public, index):
    pool = index.get(key(public), ())
    uids = tuple(map(int, public['uids']))
    low = bisect_left(pool, (uids, -1))
    high = bisect_right(pool, (uids, len(source.documents)))
    eligible = len(pool)-(high-low)
    if not eligible:
        return -1, 0, 1 if not pool else 2
    rank = selector(source, number, public) % eligible
    rank += (high-low) if rank >= low else 0
    donor = pool[rank][1]
    need(int(source.components[donor]) != int(source.components[number]), 'ANSWER_EVAL_COMPONENT_LEAK')
    return donor, eligible, 0


def expected(source, number, public, index):
    value = np.zeros((), dtype=DTYPE); p = public[number]
    donor, eligible, status = select(source, number, p, index)
    for name, x in dict(document=number, side=int(source.sides[number]),
        component=int(source.components[number]), collection=int(p['collection']),
        observed=int(p['observed']), donor=donor, eligible=eligible, status=status,
        reversal_distinct=int(p['observed'] == 2 and p['uids'][0] != p['uids'][1])).items():
        value[name] = x
    value['uids'] = p['uids']; value['layout'] = p['layout']
    value['donor_uids'] = -1 if donor < 0 else public[donor]['uids']
    return value


def build(source, public, index, path):
    cases = np.flatnonzero(source.sides != 0)
    rows = np.lib.format.open_memmap(path, mode='w+', dtype=DTYPE, shape=(len(cases),))
    try:
        for j, number in enumerate(cases):
            rows[j] = expected(source, int(number), public, index)
        rows.flush()
    finally:
        rows._mmap.close()


class Schedule:
    def __init__(self, source, rows, public, index):
        self.source, self.rows, self.public, self.index = source, rows, public, index
        cases = np.flatnonzero(source.sides != 0)
        need(rows.dtype == DTYPE and rows.shape == cases.shape
             and np.array_equal(rows['document'], cases), 'ANSWER_EVAL_COMPLETE_CASES')

    def validate(self, row):
        number = int(row['document'])
        need(0 <= number < len(self.public) and self.source.sides[number] != 0,
             'ANSWER_EVAL_HELD_OUT_ONLY')
        need(row.dtype == DTYPE and row.tobytes() == expected(
            self.source, number, self.public, self.index).tobytes(), 'ANSWER_EVAL_ROW_AUTHORITY')

    def request(self, ordinal):
        need(type(ordinal) is int and 0 <= ordinal < len(self.rows), 'ANSWER_EVAL_ORDINAL')
        row = self.rows[ordinal]; self.validate(row)
        n = int(row['observed'])
        return dict(operation='continue_observed_source', position_kind='attested_document_order',
            observed_uids=tuple(map(int, row['uids'][:n])))

    def donor_request(self, ordinal):
        self.request(ordinal); row = self.rows[ordinal]
        need(int(row['status']) == 0, STATUS[int(row['status'])])
        n = int(row['observed'])
        return dict(operation='continue_observed_source', position_kind='attested_document_order',
            observed_uids=tuple(map(int, row['donor_uids'][:n])))
