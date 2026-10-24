"""Bounded native-window graph routing; never pools or joins instrument vectors."""
import heapq
import importlib.metadata

import numpy as np
from usearch.index import Index


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_KERNEL_" + code)


class NativeLane:
    def __init__(self, ids, vectors, offsets):
        check(isinstance(ids, list) and 0 < len(ids) <= 2000
              and all(isinstance(x, str) and x for x in ids)
              and len(ids) == len(set(ids)), "NATIVE_IDS")
        check(vectors.dtype == np.float32 and vectors.ndim == 2
              and 0 < vectors.shape[1] <= 4096 and len(vectors) <= 32000
              and np.isfinite(vectors).all(), "NATIVE_VALUES")
        check(offsets.ndim == 1 and offsets.dtype == np.int64
              and len(offsets) == len(ids) + 1 and offsets[0] == 0
              and offsets[-1] == len(vectors) and (np.diff(offsets) > 0).all()
              and (np.diff(offsets) <= 16).all(), "NATIVE_OFFSETS")
        self.ids, self.original, self.offsets = ids, vectors.copy(), offsets.copy()
        self.rows = {key: i for i, key in enumerate(ids)}
        self.unit = self.normalize(vectors)
        self.owners = np.repeat(np.arange(len(ids)), np.diff(offsets))

    @staticmethod
    def normalize(values):
        check(values.ndim == 2 and 0 < len(values) <= 32000
              and np.isfinite(values).all(), "QUERY_VALUES")
        v = np.asarray(values, np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        check((norms > 1e-10).all(), "ZERO_NORM")
        return v / norms

    def windows(self, key):
        check(key in self.rows, "UNKNOWN_ATOM")
        i = self.rows[key]
        return self.unit[self.offsets[i]:self.offsets[i + 1]]

    def query(self, values):
        check(values.ndim == 2 and values.shape[1] == self.unit.shape[1]
              and 0 < len(values) <= 16, "QUERY_LAYOUT")
        return self.normalize(values)

    def score(self, query, key):
        matrix = query @ self.windows(key).T
        return float(min(matrix.max(axis=0).mean(), matrix.max(axis=1).mean()))

    def weave(self, plan):
        check(importlib.metadata.version("usearch") == plan["usearch_version"],
              "ANN_VERSION")
        index = Index(ndim=self.unit.shape[1], metric="cos", dtype="f32",
                      connectivity=plan["ann_connectivity"],
                      expansion_add=plan["ann_expansion_add"],
                      expansion_search=plan["ann_expansion_search"])
        keys = np.arange(len(self.unit), dtype=np.uint64)
        index.add(keys, self.unit.astype(np.float32), threads=1)
        edges, work = [], 0
        for key in self.ids:
            candidates = set()
            for window in self.windows(key):
                hits = index.search(window.astype(np.float32),
                                    min(len(keys), plan["ann_candidates"]), threads=1)
                work += hits.computed_distances
                candidates.update(self.ids[int(self.owners[int(k)])] for k in hits.keys)
            candidates.discard(key)
            check(len(candidates) >= plan["neighbors"], "ANN_SCARCITY")
            ranked = sorted(((self.score(self.windows(key), other), other)
                             for other in candidates), key=lambda x: (-x[0], x[1]))
            for score, other in ranked[:plan["neighbors"]]:
                check(score > 0, "NONPOSITIVE_EDGE")
                edges.append([key, other, score])
        return edges, dict(window_nodes=len(keys), search_distance_computations=int(work))


class Router:
    """Validate and compile immutable topology once per loaded artifact."""
    def __init__(self, lane, graph, plan):
        from kernel.derive import validate_graph
        validate_graph(lane.ids, graph)
        self.lane, self.plan = lane, dict(plan)
        self.entries = tuple(graph["entries"])
        adjacency = {key: set() for key in lane.ids}
        for a, b, _ in graph["edges"]:
            adjacency[a].add(b); adjacency[b].add(a)
        self.adjacency = {key: tuple(sorted(v)) for key, v in adjacency.items()}

    def search(self, query_values, *, budget=None):
        return _route(self, query_values, budget)


def _route(router, query_values, budget):
    """Calyx query-nearest entry / weak-graph best-first, extended to full MaxSim."""
    lane, plan = router.lane, router.plan
    q = lane.query(query_values)
    limit = plan["max_distance_computations"] if budget is None else budget
    check(type(limit) is int and 0 < limit <= plan["max_distance_computations"], "BUDGET")
    scores = {}

    def score(key):
        if key not in scores:
            check(len(scores) < limit, "ROUTE_BUDGET")
            scores[key] = lane.score(q, key)
        return scores[key]

    entries = sorted(router.entries, key=lambda key: (-score(key), key))
    check(len(entries) >= plan["entry_points"], "ENTRY_SCARCITY")
    entries = entries[:plan["entry_points"]]
    visited = set(entries)
    frontier = [(-score(key), key) for key in entries]
    heapq.heapify(frontier)
    retained = set(entries)
    while frontier:
        neg, current = heapq.heappop(frontier)
        ranked = sorted(retained, key=lambda key: (-scores[key], key))
        if len(ranked) >= plan["ef_search"] and -neg < scores[ranked[-1]]:
            break
        for key in router.adjacency[current]:
            if key in visited:
                continue
            visited.add(key)
            value = score(key)
            ranked = sorted(retained, key=lambda key: (-scores[key], key))
            if len(ranked) < plan["ef_search"] or (-value, key) < (-scores[ranked[-1]], ranked[-1]):
                retained.add(key)
                heapq.heappush(frontier, (-value, key))
                if len(retained) > plan["ef_search"]:
                    retained.remove(ranked[-1])
    hits = sorted(retained, key=lambda key: (-scores[key], key))[:plan["top_k"]]
    check(len(hits) == plan["top_k"], "ROUTE_SCARCITY")
    return dict(hits=[[key, scores[key]] for key in hits], entries=entries,
                visited=len(visited), distance_computations=len(scores))
