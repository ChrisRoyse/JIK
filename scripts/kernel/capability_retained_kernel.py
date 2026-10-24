"""Recall-gated task-context kernels with complete, separately addressed target closures."""
import json
import os
from pathlib import Path
import time

from claims.capability_population import need, sha
from claims.process_conditions import write_committed
from kernel.capability_retained_updates import TaskUpdates
from kernel.capability_lane import lanes
from kernel.capability_associations import weave
from kernel.capability_route import assess
from kernel.capability_growth_route import TaskRouter
from kernel.repository_kernel import code_pins as reference_pins
from kernel.repository_update_state import load_index
from model.tool_kernel import PLAN


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return dict(reference=reference_pins(), capability={n: sha((root/n).read_bytes()) for n in (
        'kernel/capability_retained_kernel.py', 'kernel/capability_growth_route.py',
        'kernel/capability_split.py', 'kernel/capability_population.py', 'kernel/capability_kernel.py',
        'kernel/capability_associations.py', 'kernel/capability_route.py',
        'kernel/capability_lane.py', 'kernel/capability_retained_updates.py', 'kernel/capability_graph.py')})


def target_ids(population, keys):
    return {population.links[k]['task']['population'] + ':' + f['id'] for k in keys
            for f in population.target(k)['targets']['fields']}


def measure(population, lane_map, graphs, indices, owners, routers=None):
    if routers is None:
        routers = {n: TaskRouter(lane, graphs[n], PLAN, indices[n], owners[n]) for n, lane in lane_map.items()}
    result = assess(population, lane_map, graphs, indices, owners, PLAN, routers)
    for value in result.values():
        for row in value['rows']:
            hits = [k for k, _ in row['route']['hits']]
            expected, actual = target_ids(population, row['exact']), target_ids(population, hits)
            row.update(target_recall=len(expected & actual)/len(expected),
                       expected_targets=sorted(expected), returned_targets=sorted(actual))
        value['target_recall'] = sum(r['target_recall'] for r in value['rows']) / len(value['rows'])
        families = sorted({population.records[r['query']]['family'] for r in value['rows']})
        value['families'] = {}
        for family in families:
            rows = [r for r in value['rows'] if population.records[r['query']]['family'] == family]
            v = dict(queries=len(rows), recall=sum(r['recall'] for r in rows)/len(rows),
                     target_recall=sum(r['target_recall'] for r in rows)/len(rows))
            v['passes'] = min(v['recall'], v['target_recall']) >= PLAN['recall_gate']
            value['families'][family] = v
        value['passes'] = value['passes'] and value['target_recall'] >= PLAN['recall_gate'] \
            and all(v['passes'] for v in value['families'].values())
    return result


class TaskCompanion:
    def __init__(self, population):
        self.population = population
        self.updates = {n: TaskUpdates(s.identity, PLAN) for n, s in population.specs.items()}
        self.previous = self.last = None

    def sync(self, groups, output):
        start = time.perf_counter()
        p, output = self.population, Path(output)
        need(not output.exists(), 'TASK_ARTIFACT_EXISTS')
        p.fresh(); source = p.source(groups)
        if self.last is not None and source == self.last['source']:
            return self.last, self.previous['pin'], dict(reused=True)
        checked = time.perf_counter()
        lane_map = lanes(p, source['ids'])
        staged = time.perf_counter()
        graphs, work = {}, {}
        for name, lane in lane_map.items():
            graphs[name], work[name] = self.updates[name].update(lane)
        woven = time.perf_counter()
        output.mkdir()
        routing = {}
        for name, u in self.updates.items():
            binary = bytes(u.index.save())
            with (output/(name+'.usearch')).open('xb') as stream:
                stream.write(binary); stream.flush(); os.fsync(stream.fileno())
            keys = [k for ids in u.keys.values() for k in ids]
            routing[name] = dict(sha256=sha(binary), owners=[[k, u.owners[k]] for k in sorted(keys)])
        write_committed(output/'candidate-graphs.json', dict(graphs=graphs, work=work, routing=routing))
        calibration = measure(p, lane_map, graphs, {n: u.index for n, u in self.updates.items()},
                              {n: dict(r['owners']) for n, r in routing.items()})
        assessed = time.perf_counter()
        cache_pin = write_committed(output/'cache.json', {n: u.cache for n, u in self.updates.items()})
        associations = weave(p, graphs)
        association_pin = write_committed(output/'associations.json', associations)
        value = dict(schema='connected-task-retained-kernel-v1', source=source, plan=PLAN,
            routing_policy=dict(max_distance_computations=None, historical_plan_limit_applies=False,
                seeding='all_kernel_entries_plus_native_window_ann'),
            parent=self.previous, code=code_pins(), split=p.split, ids=source['ids'],
            instruments={n: s.identity for n, s in p.specs.items()},
            slots={k: p.records[k]['receipt']['slots'] for k in source['ids']},
            connections={k: p.links[k] for k in source['ids']},
            graphs=graphs, work=work, calibration=calibration, cache_sha256=cache_pin,
            routing=routing,
            associations_sha256=association_pin, cross_domain_edges=associations['cross_domain_edges'],
            recall_passes=all(v['passes'] for v in calibration.values()),
            scientific_admission=False, learned_capability=False, serving=False,
            exposure='development_exposed_group_holdout', number_status='measured',
            acceptance='author-checked', scope='context_similarity_and_attested_target_closure')
        pin = write_committed(output/'manifest.json', value)
        write_committed(output/'COMMITTED', dict(manifest=pin))
        end = time.perf_counter()
        timings = dict(wall_seconds=end-start, phases=dict(source=checked-start, lanes=staged-checked,
            weave=woven-staged, recall=assessed-woven, persist=end-assessed))
        write_committed(output/'timings.json', timings)
        self.last, self.previous = value, dict(path=str(output), pin=pin)
        return value, pin, timings


class TaskKernel:
    def __init__(self, population, path, pin, *, source):
        self.population, self.path = population, Path(path)
        raw = (self.path/'manifest.json').read_bytes()
        need(sha(raw) == pin and json.loads((self.path/'COMMITTED').read_bytes()) == dict(manifest=pin),
             'TASK_ARTIFACT_PIN')
        v = json.loads(raw)
        need(v['source'] == source == population.source(source['groups']), 'TASK_STALE_SOURCE')
        population.fresh()
        need(v['schema'] == 'connected-task-retained-kernel-v1' and v['code'] == code_pins()
             and v['plan'] == PLAN and v['split'] == population.split
             and v['routing_policy'] == dict(max_distance_computations=None, historical_plan_limit_applies=False,
                seeding='all_kernel_entries_plus_native_window_ann')
             and v['instruments'] == {n: s.identity for n, s in population.specs.items()},
             'TASK_ARTIFACT_CONTRACT')
        ids = source['ids']
        need(v['ids'] == ids and not set(ids) & set(population.queries())
             and v['slots'] == {k: population.records[k]['receipt']['slots'] for k in ids},
             'TASK_ARTIFACT_COVERAGE')
        need(v['connections'] == {k: population.links[k] for k in ids}, 'TASK_ARTIFACT_CONNECTIONS')
        need(sha((self.path/'cache.json').read_bytes()) == v['cache_sha256'], 'TASK_CACHE_PIN')
        raw = (self.path/'associations.json').read_bytes()
        need(sha(raw) == v['associations_sha256'] and json.loads(raw) == weave(population, v['graphs']),
             'TASK_ASSOCIATION_READBACK')
        need(all(v[k] is False for k in ('scientific_admission', 'learned_capability', 'serving')),
             'TASK_FALSE_PROMOTION')
        need(v['recall_passes'], 'TASK_RECALL_GATE')
        self.lanes = lanes(population, ids)
        indices, owners = {}, {}
        for name, lane in self.lanes.items():
            path = self.path/(name+'.usearch')
            r = v['routing'][name]
            need(sha(path.read_bytes()) == r['sha256'], 'TASK_ROUTING_BYTES')
            index = load_index(path, lane.unit.shape[1])
            mapping = dict(r['owners'])
            need(len(mapping) == len(r['owners']) == len(lane.unit)
                 and set(mapping) == set(index.keys[:].tolist()) and set(mapping.values()) == set(ids),
                 'TASK_ROUTING_OWNERS')
            import numpy as np
            for key in ids:
                keys = sorted(k for k, owner in mapping.items() if owner == key)
                values = np.stack(index.get(np.asarray(keys, np.uint64), dtype='f32'))
                need(np.array_equal(values, lane.windows(key).astype(np.float32)), 'TASK_ROUTING_NATIVE')
            indices[name], owners[name] = index, mapping
        self.routers = {n: TaskRouter(lane, v['graphs'][n], PLAN, indices[n], owners[n])
                        for n, lane in self.lanes.items()}
        actual = measure(population, self.lanes, v['graphs'], indices, owners, self.routers)
        need(actual == v['calibration'] and all(r['passes'] for r in actual.values()), 'TASK_RECALL_READBACK')
        self.value, self.pin = v, pin

    def association(self, origin, target, *, source):
        """An unmaterialized pair remains measurable from all original typed windows."""
        need(source == self.value['source'], 'TASK_STALE_SOURCE')
        need(origin in self.population.eligible and target in self.value['ids'], 'TASK_PAIR_SCOPE')
        return dict(origin=origin, target=target, kernel=self.pin,
            readings={n: lane.score(lane.query(self.population.values[origin][n]), target)
                      for n, lane in self.lanes.items()}, scientific_admission=False)

    def route(self, key, *, source):
        need(source == self.value['source'], 'TASK_STALE_SOURCE')
        need(key in self.population.eligible, 'TASK_QUERY_INELIGIBLE')
        traces = {n: r.search(self.population.values[key][n]) for n, r in self.routers.items()}
        hits = {k for t in traces.values() for k, _ in t['hits']}
        need(not hits & set(self.population.queries()), 'TASK_QUERY_TARGET_LEAK')
        return dict(kernel=self.pin, query=key, traces=traces,
            targets={k: self.population.target(k) for k in sorted(hits)},
            source=source, scientific_admission=False, claim='measured_context_similarity_only')
