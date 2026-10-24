"""Compare query seeding on the retained failed generation, without re-embedding or gate changes."""
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.capability_lane import TaskLane
from kernel.capability_route import TaskRouter
from kernel.repository_update_state import load_index
from model.capability_input import CapabilityNative
from model.capability_delta_input import DeltaStore, DeltaNative


def read(path): return json.loads(Path(path).read_bytes())


def sources(config, old_config):
    old = [CapabilityNative(**s) for s in read(old_config)['populations']]
    store = DeltaStore(**config['measurements'])
    new = [DeltaNative(str(p.root), pin, store=store) for pin, p in store.populations.items()]
    return {p.population.pin: p for p in old+new}


def search(router, values, seeds, score, *, quotient=False):
    lane, plan = router.lane, router.plan
    # A signature beam keeps distinct readings, while all logical tasks remain addressable.
    groups = defaultdict(list)
    for key in lane.ids: groups[lane.signatures[key]].append(key)
    representative = {k: min(groups[lane.signatures[k]]) if quotient else k for k in lane.ids}
    adjacency = defaultdict(set)
    for key, neighbors in router.adjacency.items():
        adjacency[representative[key]].update(representative[n] for n in neighbors)
    retained = {representative[k] for k in seeds}; visited = set(retained)
    frontier = [(-score(k), k) for k in retained]; heapq.heapify(frontier)
    while frontier:
        neg, current = heapq.heappop(frontier)
        ranked = sorted(retained, key=lambda k: (-score(k), k))
        if len(ranked) >= plan['ef_search'] and -neg < score(ranked[-1]): break
        for key in sorted(adjacency[current]):
            if key in visited: continue
            visited.add(key); value = score(key)
            ranked = sorted(retained, key=lambda k: (-score(k), k))
            if len(ranked) < plan['ef_search'] or (-value, key) < (-score(ranked[-1]), ranked[-1]):
                retained.add(key); heapq.heappush(frontier, (-value, key))
                if len(retained) > plan['ef_search']: retained.remove(ranked[-1])
    candidates = {k for k in lane.ids if representative[k] in retained}
    hits = sorted(candidates, key=lambda k: (-score(k), k))[:plan['top_k']]
    return hits, len(visited)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('config', 'old-config', 'kernel', 'output'): p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'ROUTE_DIAG_EXISTS')
    started = time.perf_counter(); config = read(a.config); kernel = read(a.kernel / 'manifest.json')
    need(not kernel['recall_passes'], 'ROUTE_DIAG_EXPECTED_FAILURE')
    native = sources(config, a.old_config)
    values, families = {}, {}
    for provider in native.values():
        pop = provider.population
        for key, task in pop.tasks.items():
            families[key] = pop.family
            values[key] = {s.name: np.concatenate([provider.field(f, s.name)[0] for f in task['context']])
                           for s in provider.specs}
    arms = ('all_entries', 'per_component', 'window_ann', 'entries_plus_ann', 'signature_beam')
    results, shapes = {}, {}
    for name, graph in kernel['graphs'].items():
        lane = TaskLane({k: values[k][name] for k in kernel['ids']})
        need(all(sha(values[k][name].astype('<f4').tobytes()) == kernel['slots'][k][name]['sha256']
                 for k in lane.ids), 'ROUTE_DIAG_NATIVE_BYTES')
        index = load_index(a.kernel / (name+'.usearch'), lane.unit.shape[1])
        need(sha((a.kernel / (name+'.usearch')).read_bytes()) == kernel['routing'][name]['sha256'],
             'ROUTE_DIAG_INDEX_PIN')
        owners = dict(kernel['routing'][name]['owners'])
        router = TaskRouter(lane, graph, kernel['plan'], index, owners)
        topology = nx.Graph(); topology.add_nodes_from(lane.ids)
        topology.add_edges_from((a, b) for a, b, _ in graph['edges'])
        components = list(nx.connected_components(topology))
        component_id = {k: i for i, part in enumerate(components) for k in part}
        rows = []; tick = time.perf_counter()
        for recorded in kernel['calibration'][name]['rows']:
            key = recorded['query']; q = lane.query(values[key][name])
            before = router.search(values[key][name])
            need(before == recorded['route'], 'ROUTE_DIAG_BASELINE_REPLAY')
            cache, logical = {}, set()
            def score(k):
                signature = lane.signatures[k]; logical.add(k)
                if signature not in cache: cache[signature] = lane.score(q, k)
                return cache[signature]
            ann_seeds = set()
            ann_calls = {}
            for window in q:
                raw = window.astype(np.float32)
                if raw.tobytes() not in ann_calls:
                    hits = index.search(raw, min(len(lane.unit), kernel['plan']['ann_candidates']), threads=1)
                    ann_calls[raw.tobytes()] = [int(x) for x in hits.keys]
                ann_seeds.update(owners[x] for x in ann_calls[raw.tobytes()])
            component_seeds = {min(set(part) & router.entries, key=lambda k: (-score(k), k)) for part in components}
            seeds = dict(all_entries=router.entries, per_component=component_seeds, window_ann=ann_seeds,
                         entries_plus_ann=router.entries | ann_seeds, signature_beam=router.entries | ann_seeds)
            exact = set(recorded['exact']); trials = {}
            for arm in arms:
                hits, visits = search(router, values[key][name], seeds[arm], score, quotient=arm=='signature_beam')
                trials[arm] = dict(recall=len(exact & set(hits))/kernel['plan']['top_k'], visits=visits,
                                   hits=hits, seeds=len(seeds[arm]))
            accessible = {component_id[k] for k in before['entries']}
            rows.append(dict(query=key, family=families[key], baseline=recorded['recall'],
                exact_outside_seed_components=sum(component_id[k] not in accessible for k in exact), trials=trials,
                exact_scores_shared_across_arms=len(cache), logical_scores=len(logical)))
        summary = {}
        for arm in arms:
            summary[arm] = {f: float(np.mean([r['trials'][arm]['recall'] for r in rows if r['family']==f]))
                            for f in ('coding', 'conversation', 'reasoning')}
        results[name] = dict(rows=rows, summary=summary, seconds=time.perf_counter()-tick)
        shapes[name] = dict(atoms=len(lane.ids), signatures=len(set(lane.signatures.values())),
            windows=len(lane.unit), components=len(components), entries=len(router.entries), queries=len(rows),
            baseline_missed_neighbors_in_unvisited_components=sum(r['exact_outside_seed_components'] for r in rows))
        print(json.dumps(dict(instrument=name, shapes=shapes[name], summary=summary)), flush=True)
    report = dict(schema='capability-route-diagnosis-v1', kernel=str(a.kernel),
        kernel_pin=sha((a.kernel/'manifest.json').read_bytes()), results=results, shapes=shapes,
        wall_seconds=time.perf_counter()-started, calibrated_threshold_changed=False, reembedded=False,
        production_changed=False, acceptance='author-checked', number_status='measured',
        exposure='development_exposed_diagnostic', code_sha256=sha(Path(__file__).read_bytes()))
    pin = write_committed(a.output, report)
    print(json.dumps(dict(report=pin, seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__': main()
