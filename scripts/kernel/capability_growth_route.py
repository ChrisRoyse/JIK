"""Query each native window and all kernel entries before whole-context graph routing."""
import copy
import hashlib
import heapq

import numpy as np

from kernel.capability_route import TaskRouter as HistoricalRouter
from kernel.native_route import check


class TaskRouter(HistoricalRouter):
    def search(self, query_values):
        lane, plan = self.lane, self.plan
        identity = hashlib.sha256(query_values.astype('<f4', copy=False).tobytes()).hexdigest()
        if identity in self.cache: return copy.deepcopy(self.cache[identity])
        q = lane.query(query_values)
        scores, signatures = {}, {}
        def score(key):
            if key not in scores:
                signature = lane.signatures[key]
                if signature not in signatures: signatures[signature] = lane.score(q, key)
                scores[key] = signatures[signature]
            return scores[key]
        window_hits, ann_seeds, ann_distances = {}, set(), 0
        for window in q:
            native = window.astype(np.float32); key = native.tobytes()
            if key not in window_hits:
                hits = self.index.search(native, min(len(lane.unit), plan['ann_candidates']), threads=1)
                window_hits[key] = tuple(int(k) for k in hits.keys)
                ann_distances += hits.computed_distances
            ann_seeds.update(self.owners[k] for k in window_hits[key])
        entries = sorted(self.entries | ann_seeds)
        check(entries and set(entries) <= set(lane.ids), 'GROWTH_ENTRY_SCOPE')
        visited, retained = set(entries), set(entries)
        frontier = [(-score(k), k) for k in entries]; heapq.heapify(frontier)
        while frontier:
            neg, current = heapq.heappop(frontier)
            ranked = sorted(retained, key=lambda k: (-scores[k], k))
            if len(ranked) >= plan['ef_search'] and -neg < scores[ranked[-1]]: break
            for key in self.adjacency[current]:
                if key in visited: continue
                visited.add(key); value = score(key)
                ranked = sorted(retained, key=lambda k: (-scores[k], k))
                if len(ranked) < plan['ef_search'] or (-value, key) < (-scores[ranked[-1]], ranked[-1]):
                    retained.add(key); heapq.heappush(frontier, (-value, key))
                    if len(retained) > plan['ef_search']: retained.remove(ranked[-1])
        hits = sorted(retained, key=lambda k: (-scores[k], k))[:plan['top_k']]
        check(len(hits) == plan['top_k'], 'GROWTH_ROUTE_SCARCITY')
        result = dict(hits=[[k, scores[k]] for k in hits], entries=entries, visited=len(visited),
            distance_computations=len(signatures), logical_scores=len(scores),
            duplicate_scores_reused=len(scores)-len(signatures), work_limit=None,
            method='kernel-and-native-window-entries/weak-graph-best-first',
            stop='frontier_exhausted_or_below_retained_beam', kernel_entries=sorted(self.entries),
            native_window_entries=sorted(ann_seeds), native_windows=len(q),
            unique_window_queries=len(window_hits), ann_distances=int(ann_distances))
        self.cache[identity] = copy.deepcopy(result)
        return result
