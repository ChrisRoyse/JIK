"""Incremental task graph updates, sharing repeated native-window searches per generation."""
import importlib.metadata
import networkx as nx
import numpy as np
from usearch.index import Index
from claims.episodic_memory import digest
from kernel.capability_graph import topology, derive, validate_graph
from kernel.native_route import check
from kernel.chunked_route import TILE

class TaskUpdates:
    """An instrument owns its index, score cache and component derivations."""
    def __init__(self, instrument, plan, cache=None):
        check(importlib.metadata.version('usearch') == plan['usearch_version'], 'ANN_VERSION')
        self.instrument, self.plan = instrument, dict(plan)
        self.index = None
        self.keys, self.owners, self.signatures = {}, {}, {}
        self.next_key = 0
        self.cache = dict(cache or {})
        self.components, self.previous_edges = {}, {}

    def update(self, lane):
        plan = self.plan
        if self.index is None:
            self.index = Index(ndim=lane.unit.shape[1], metric='cos', dtype='f32',
                connectivity=plan['ann_connectivity'], expansion_add=plan['ann_expansion_add'],
                expansion_search=plan['ann_expansion_search'])
        changed = {k for k,pin in lane.signatures.items() if self.signatures.get(k) != pin}
        live_ids = set(lane.ids)
        removed = set(self.signatures)-set(lane.ids)
        old_ids = set(self.signatures)
        for k in sorted(removed | (changed & old_ids)):
            self.index.remove(np.asarray(self.keys.pop(k), np.uint64), threads=1)
        added_rows = 0
        for k in sorted(changed):
            rows = lane.windows(k).astype(np.float32)
            keys = np.arange(self.next_key, self.next_key+len(rows), dtype=np.uint64)
            self.index.add(keys, rows, threads=1)
            self.keys[k] = keys.tolist()
            self.owners.update({int(v):k for v in keys})
            self.next_key += len(rows); added_rows += len(rows)
        self.signatures = dict(lane.signatures)
        edges, exact, reused, dot_products, ann_work = [], 0, 0, 0, 0
        neighbors, candidate_sets = {}, {}
        ann_cache, ann_requests = {}, 0
        for key in lane.ids:
            candidates = set()
            for window in lane.windows(key):
                value = window.astype(np.float32)
                signature = value.tobytes()
                ann_requests += 1
                if signature not in ann_cache:
                    hits = self.index.search(value,
                        min(len(lane.unit), plan['ann_candidates']), threads=1)
                    ann_work += hits.computed_distances
                    ann_cache[signature] = tuple(int(k) for k in hits.keys)
                candidates.update(self.owners[k] for k in ann_cache[signature])
            candidates.discard(key)
            check(candidates <= live_ids, 'ANN_OWNER')
            candidate_sets[key] = candidates
        # Window ANN supplies directed links. Close them in both directions before
        # scoring logical atoms: a long atom can otherwise hide short neighbors.
        undirected = {k:set(v) for k,v in candidate_sets.items()}
        for key, candidates in candidate_sets.items():
            for other in candidates:
                undirected[other].add(key)
        expanded = 0
        for key in lane.ids:
            candidates = set(undirected[key])
            if len(candidates) < plan['neighbors']:
                frontier = set(candidates)
                while frontier and len(candidates) < plan['neighbors']:
                    new = set().union(*(undirected[k] for k in sorted(frontier))) - candidates - {key}
                    candidates.update(new); frontier = new
                    expanded += 1
            if len(candidates) < plan['neighbors']:
                raise ValueError('MNLM_KERNEL_ANN_SCARCITY:'+str(dict(atom=key,
                    windows=len(lane.windows(key)),direct=len(candidate_sets[key]),
                    reverse_closed=len(undirected[key]),reachable=len(candidates))))
            ranked = []
            for other in sorted(candidates):
                a,b = sorted((lane.signatures[key], lane.signatures[other]))
                signature = digest(dict(instrument=self.instrument, a=a, b=b, metric='min-mean-MaxSim-v1'))
                if signature not in self.cache:
                    self.cache[signature] = lane.score(lane.windows(key), other)
                    exact += 1
                    dot_products += len(lane.windows(key))*len(lane.windows(other))
                else:
                    reused += 1
                ranked.append((self.cache[signature], other))
            ranked.sort(key=lambda x:(-x[0],x[1]))
            neighbors[key] = [[key, other, score] for score,other in ranked[:plan['neighbors']]]
            check(all(e[2] > 0 for e in neighbors[key]), 'NONPOSITIVE_EDGE')
            edges.extend(neighbors[key])
        graph = topology(lane.ids, edges)
        parts, new_components, reused_components = [], {}, 0
        for component in sorted(nx.weakly_connected_components(graph), key=lambda c:sorted(c)):
            ids = sorted(component)
            local_edges = [e for k in ids for e in neighbors[k]]
            pin = digest(dict(ids=ids, edges=local_edges, plan=plan))
            if pin in self.components:
                part = self.components[pin]; reused_components += 1
            else:
                part = derive(ids, local_edges, plan)
            parts.append(part); new_components[pin] = part
        result = dict(edges=edges, scores={}, world_grounded_fraction=0.0,
            production_admission=False, recall_assessed=False, method='dirty-weak-component/'+parts[0]['method'])
        for name in ('candidate_graph','members','graph_only_support','entries','groundedness_gaps'):
            result[name] = sorted(v for p in parts for v in p[name])
        for part in parts:
            result['scores'].update(part['scores'])
        validate_graph(lane.ids, result)
        work = dict(atoms=len(lane.ids), windows=len(lane.unit), added_windows=added_rows,
            added_or_changed_atoms=len(changed), withdrawn_atoms=len(removed),
            exact_pairs=exact, reused_pair_requests=reused, window_dot_products=dot_products,
            candidate_policy='symmetric_window_ann_then_sparse_reachability',
            direct_scarce_atoms=sum(len(v) < plan['neighbors'] for v in candidate_sets.values()),
            reachability_expansions=expanded,
            ann_distances=int(ann_work), components=len(parts), reused_components=reused_components,
            ann_requests=ann_requests, ann_unique_queries=len(ann_cache),
            ann_duplicate_queries_reused=ann_requests-len(ann_cache),
            changed_old_neighbors=sum(neighbors[k] != self.previous_edges.get(k) for k in old_ids & set(lane.ids)),
            retained_edges=len(edges), tile_elements=TILE*TILE, score_cache_entries=len(self.cache))
        self.components, self.previous_edges = new_components, neighbors
        return result, work
