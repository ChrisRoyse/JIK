"""Real-source regression checks for shape-independent association arithmetic."""
from copy import deepcopy
from decimal import Decimal, localcontext
import heapq
import time

import numpy as np

from audit.native_index_reference import POLICY, normalized, scores, verify_candidates
from audit.legal_stock_readback import rejected
from text.constellation_corpus import need


def tied_scores(values, positions):
    need(len(positions) > 1 and len({float(values[j]) for j in positions}) == 1,
         'INDEX_ARITHMETIC_IDENTICAL_INPUT_TIE')


def shape_checks(query, raw):
    """Use actual records as padding/donors; no synthetic feature vectors."""
    q = normalized([query])[0]; gallery = normalized(raw)
    expected = scores([q], gallery)
    variants = dict(
        reversed_gallery=scores([q], gallery[::-1])[:, ::-1],
        mixed_query=scores([gallery[0], q, gallery[-1], q], gallery)[[1]],
        duplicate_query=scores([q, q], gallery)[[1]],
        duplicate_gallery=scores([q], gallery+gallery)[:, len(gallery):],
        single_pair=np.array([[scores([q], [x])[0, 0] for x in gallery]]),
        tiled_gallery=np.concatenate([scores([q], gallery[j:j+17])
                                      for j in range(0, len(gallery), 17)], axis=1))
    for name, value in variants.items():
        need(expected.shape == value.shape and expected.dtype == value.dtype
             and expected.tobytes() == value.tobytes(), 'INDEX_ARITHMETIC_SHAPE_'+name.upper())
    return dict(variants=list(variants), byte_exact=True, query_windows=len(query),
                candidate_windows=sum(len(x) for x in raw), candidates=len(raw))


def decimal_witness(query, raw, ids):
    with localcontext() as context:
        context.prec = 80
        def unit(rows):
            out = []
            for row in rows:
                values = [Decimal.from_float(float(x)) for x in row]
                norm = sum(x*x for x in values).sqrt()
                out.append([x/norm for x in values])
            return out
        left = unit(query); values = {}
        for uid, row in zip(ids, raw):
            right = unit(row)
            dots = [[sum(a*b for a, b in zip(x, y)) for y in right] for x in left]
            values[uid] = min(sum(map(max, dots))/len(left),
                              sum(map(max, zip(*dots)))/len(right))
        order = sorted(ids, key=lambda uid: (-values[uid], uid))
        return dict(order=order, values={str(u): str(values[u]) for u in ids})


def candidate_regression(assessment, read, witness, saved):
    start = time.perf_counter(); fail = witness['first_failure']; case = None
    maximum = 0.; shapes = {}; records = []
    for row in assessment['rows']:
        ids = row['candidate_uids']; query = read([row['uid']])[0]; raw = read(ids)
        by_uid = dict(zip(ids, raw))
        error = verify_candidates(row, normalized([query])[0],
                                  lambda wanted: [by_uid[u] for u in wanted])
        maximum = max(maximum, error)
        records.append(dict(uid=row['uid'], candidates=len(ids), maximum_scalar_error=error))
        if row['uid'] == fail['uid'] or len(query) > 1:
            shapes[str(row['uid'])] = shape_checks(query, raw)
        if row['uid'] == fail['uid']: case = (row, query, raw)
    need(case is not None and maximum <= POLICY['scalar_error'], 'INDEX_ARITHMETIC_MISSING_WITNESS')
    row, query, raw = case; ids = row['candidate_uids']
    need(np.array_equal(query, saved['query']), 'INDEX_ARITHMETIC_WITNESS_QUERY')
    disputed = fail['disputed']; selected = [x['uid'] for x in disputed]
    actual = [raw[ids.index(u)] for u in selected]
    need(all(np.array_equal(x, saved['uid_'+str(u)]) for u, x in zip(selected, actual)),
         'INDEX_ARITHMETIC_WITNESS_NATIVE')
    precision = decimal_witness(query, actual, selected)
    need(precision['order'] == fail['decimal80_order'] == [u for u, _ in row['hits']]
         and all(precision['values'][str(x['uid'])] == x['decimal80'] for x in disputed),
         'INDEX_ARITHMETIC_DECIMAL_WITNESS')
    groups = {}
    for j, value in enumerate(raw): groups.setdefault((value.shape, value.tobytes()), []).append(j)
    aliases = [positions for positions in groups.values() if len(positions) > 1]
    need(aliases, 'INDEX_ARITHMETIC_NO_ACTUAL_ALIASES')
    q = normalized([query])[0]; gallery = normalized(raw); value = scores([q], gallery)[0]
    for positions in aliases: tied_scores(value, positions)
    # Reproduce the old defect on the very same physical source arrays.
    need(len(q) == 1 and all(len(x) == 1 for x in gallery), 'INDEX_ARITHMETIC_OLD_WITNESS_SHAPE')
    old = np.clip(np.stack([q[0]]) @ np.stack([x[0] for x in gallery]).T, -1, 1)[0]
    broken = [positions for positions in aliases if len({float(old[j]) for j in positions}) > 1]
    need(broken, 'INDEX_ARITHMETIC_OLD_FAILURE_NOT_REPRODUCED')
    old_order = [u for u, _ in heapq.nsmallest(POLICY['top_k'], zip(ids, old), key=lambda pair: (-pair[1], pair[0]))]
    need(old_order != [u for u, _ in row['hits']], 'INDEX_ARITHMETIC_OLD_RANK_NOT_REPRODUCED')
    wrong = deepcopy(row); positions = [j for j, (u, _) in enumerate(wrong['hits']) if ids.index(u) in broken[0]]
    need(len(positions) > 1, 'INDEX_ARITHMETIC_ALIAS_HITS')
    a, b = positions[:2]; wrong['hits'][a], wrong['hits'][b] = wrong['hits'][b], wrong['hits'][a]
    controls = dict(
        old_shape_dependent_ties=rejected(lambda: tied_scores(old, broken[0])),
        swapped_exact_tie_order=rejected(lambda: verify_candidates(wrong, q, read)))
    return dict(queries=len(records), candidate_checks=records, maximum_scalar_error=maximum,
                shape_checks=shapes, witness_uid=row['uid'],
                alias_groups=[dict(uids=[ids[j] for j in positions], scores=[float(value[j]) for j in positions],
                                   old_scores=[float(old[j]) for j in positions]) for positions in aliases],
                old_order=old_order, canonical_order=[u for u, _ in row['hits']], decimal80=precision,
                controls=controls, seconds=time.perf_counter()-start)
