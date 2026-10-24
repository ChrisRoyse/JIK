"""Predict a successor through one instrument's TRAIN-filtered native neighbours."""
import numpy as np

from audit.native_index_reference import normalized, scores as independent_scores
from kernel.native_window_index import score_table
from text.constellation_corpus import need


class NativeSuccessor:
    def __init__(self, lane, continuation, *, candidates, expansion):
        need((candidates, expansion) in [(128, 256), (512, 1024)], 'NATIVE_SUCCESSOR_QUERY_POLICY')
        self.lane, self.continuation = lane, continuation
        self.candidates, self.expansion = candidates, expansion
        self.initial_expansion = lane.index.index.expansion_search
        lane.index.index.expansion_search = expansion

    def donor(self, uid, number, successor=None):
        c = self.continuation
        need(type(uid) is int and 0 <= uid < c.u and type(number) is int and 0 <= number < c.u
             and number != uid and int(c.presence[number]) & 1, 'NATIVE_SUCCESSOR_TRAIN_DONOR')
        actual = int(c.modal[number])
        need(0 <= actual < c.u and int(c.presence[actual]) & 1, 'NATIVE_SUCCESSOR_TRAIN_TARGET')
        need(successor is None or successor == actual, 'NATIVE_SUCCESSOR_MAPPING')
        return actual

    def query(self, uid):
        c = self.continuation; lane = self.lane
        need(type(uid) is int and 0 <= uid < c.u, 'NATIVE_SUCCESSOR_QUERY_UID')
        c.fresh(); lane.fresh()
        need(lane.index.index.expansion_search == self.expansion, 'NATIVE_SUCCESSOR_RUNTIME_CHANGED')
        native = lane([uid])[0]; owners = set(); work = visited = 0
        for window in native:
            hit = lane.index.index.search(window, count=self.candidates, threads=1)
            keys = np.asarray(hit.keys, dtype=np.uint64); ids = (keys >> np.uint64(32)).astype(np.int64)
            need(len(keys) == self.candidates and bool(((ids >= 0) & (ids < c.u)).all())
                 and bool(((keys & np.uint64(2**32-1)) < lane.index.lengths[ids]).all()), 'NATIVE_SUCCESSOR_SEARCH_KEYS')
            owners.update(map(int, ids)); work += int(hit.computed_distances); visited += int(hit.visited_members)
        owners.discard(uid); candidate_uids = sorted(owners)
        eligible = [u for u in candidate_uids if int(c.presence[u]) & 1 and int(c.modal[u]) >= 0]
        ranked = []; maximum_error = 0.
        for lo in range(0, len(eligible), 512):
            part = eligible[lo:lo+512]; values = lane(part)
            actual = score_table([native], values)[0]
            independent = independent_scores(normalized([native]), normalized(values))[0]
            maximum_error = max(maximum_error, float(np.max(np.abs(actual-independent))))
            need(maximum_error <= 5e-13, 'NATIVE_SUCCESSOR_INDEPENDENT_SCORE')
            ranked.extend(zip(part, actual.tolist()))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        number = int(ranked[0][0]) if ranked else None
        successor = self.donor(uid, number) if number is not None else None
        c.fresh(); lane.fresh()
        return dict(uid=uid, candidate_uids=candidate_uids, train_candidate_scores=[[u,s] for u,s in ranked],
                    donor_uid=number, successor_uid=successor, status='prediction' if number is not None else 'abstain_no_eligible_candidate',
                    native_windows=len(native), distance_work=work, visited_members=visited, maximum_score_error=maximum_error)

    def close(self):
        self.lane.index.index.expansion_search = self.initial_expansion
        self.lane.fresh(); self.continuation.fresh()


def scope(rows, uids):
    need([r['uid'] for r in rows] == sorted(set(uids)) and len(rows) == len(set(uids)), 'NATIVE_SUCCESSOR_QUERY_COVERAGE')
    for row in rows:
        candidates = row['candidate_uids']; ranked = row['train_candidate_scores']
        need(candidates == sorted(set(candidates)) and row['uid'] not in candidates,
             'NATIVE_SUCCESSOR_CANDIDATE_ORDER')
        need(len({u for u,s in ranked}) == len(ranked) and {u for u,s in ranked} <= set(candidates)
             and ranked == sorted(ranked, key=lambda pair: (-pair[1], pair[0])), 'NATIVE_SUCCESSOR_RANK_ORDER')
        need(row['donor_uid'] == (ranked[0][0] if ranked else None), 'NATIVE_SUCCESSOR_SELECTED_DONOR')
