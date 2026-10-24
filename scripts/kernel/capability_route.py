"""Convergent full-window graph routing; no operator-imposed comparison ceiling."""
import copy
import hashlib
import heapq

import numpy as np

from kernel.capability_graph import validate_graph
from kernel.native_route import check


class TaskRouter:
    def __init__(self, lane, graph, plan, index, owners):
        validate_graph(lane.ids, graph)
        self.lane, self.plan, self.index, self.owners = lane, plan, index, owners
        self.entries = set(graph['entries'])
        self.cache = {}
        self.adjacency = {k: [] for k in lane.ids}
        adjacency = {k: set() for k in lane.ids}
        for a, b, _ in graph['edges']: adjacency[a].add(b); adjacency[b].add(a)
        self.adjacency = {k: tuple(sorted(v)) for k, v in adjacency.items()}

    def search(self, query_values):
        lane, plan = self.lane, self.plan
        identity = hashlib.sha256(query_values.astype('<f4', copy=False).tobytes()).hexdigest()
        if identity in self.cache: return copy.deepcopy(self.cache[identity])
        q = lane.query(query_values)
        scores, signatures = {}, {}
        def score(key):
            if key not in scores:
                signature = lane.signatures[key]
                if signature not in signatures:
                    signatures[signature] = lane.score(q, key)
                scores[key] = signatures[signature]
            return scores[key]
        entries = sorted(self.entries, key=lambda k: (-score(k), k))[:plan['entry_points']]
        check(len(entries) == plan['entry_points'], 'ENTRY_SCARCITY')
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
        check(len(hits) == plan['top_k'], 'ROUTE_SCARCITY')
        result = dict(hits=[[k, scores[k]] for k in hits], entries=entries, visited=len(visited),
                      distance_computations=len(signatures), logical_scores=len(scores),
                      duplicate_scores_reused=len(scores)-len(signatures), work_limit=None,
                      method='query-nearest-kernel-entry/weak-graph-best-first',
                      stop='frontier_exhausted_or_below_retained_beam')
        self.cache[identity] = copy.deepcopy(result)
        return result


def assess(population, lane_map, graphs, indices, owners, plan, routers=None):
    result = {}
    for name, lane in lane_map.items():
        router = routers[name] if routers is not None else TaskRouter(lane, graphs[name], plan, indices[name], owners[name])
        rows, normalized, oracle = [], {}, {}
        for other in lane.ids:
            signature = lane.signatures[other]
            if signature not in normalized:
                raw = population.values[other][name].astype(np.float64)
                raw /= np.sqrt(np.einsum('ij,ij->i', raw, raw))[:, None]
                normalized[signature] = raw
        for key in population.queries():
            q = population.values[key][name].astype(np.float64)
            q /= np.sqrt(np.einsum('ij,ij->i', q, q))[:, None]
            identity = hashlib.sha256(q.tobytes()).hexdigest()
            if identity not in oracle:
                oracle[identity] = {}
                for signature, raw in normalized.items():
                    matrix = np.einsum('ik,jk->ij', q, raw)
                    oracle[identity][signature] = float(min(matrix.max(axis=0).mean(), matrix.max(axis=1).mean()))
            scores = {other: oracle[identity][lane.signatures[other]] for other in lane.ids}
            exact = sorted(scores, key=lambda k: (-scores[k], k))[:plan['top_k']]
            trace = router.search(population.values[key][name])
            check(all(abs(value-scores[k]) <= 2e-12 for k, value in trace['hits']), 'TASK_SCORE_READBACK')
            recall = len(set(exact) & {k for k, _ in trace['hits']}) / plan['top_k']
            rows.append(dict(query=key, exact=exact, route=trace, recall=recall))
        recall = sum(r['recall'] for r in rows)/len(rows)
        result[name] = dict(rows=rows, recall=recall, passes=recall >= plan['recall_gate'],
                           exact_reference_pairs=len(oracle)*len(normalized),
                           exact_logical_pairs=len(rows)*len(lane.ids))
    return result
