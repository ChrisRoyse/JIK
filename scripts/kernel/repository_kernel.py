"""Journal-bound association companion; retrieval admission remains recall-gated."""
import json
from pathlib import Path
import time

import numpy as np

from claims.codata_source import sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.chunked_route import ChunkedLane, SparseUpdates
from kernel.native_route import Router, check
from model.repository_input import utc_ns
from model.tool_kernel import PLAN


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {name:sha((root/name).read_bytes()) for name in (
        'kernel/chunked_route.py','kernel/repository_population.py','kernel/repository_kernel.py',
        'kernel/native_route.py','kernel/derive.py','kernel/component_kernel.py','model/tool_kernel.py')}


def lanes(population, ids):
    return {name:ChunkedLane({k:population.values[k][name] for k in ids
                             if len(population.values[k][name])}) for name in sorted(population.specs)}


def assess(population, lane_map, graphs, queries):
    """Queries never build the graph. Exact reference uses original full windows."""
    result = {}
    for name,lane in lane_map.items():
        router = Router(lane,graphs[name],PLAN)
        rows = []
        for key in queries:
            q = population.values[key][name].astype(np.float64)
            q /= np.sqrt(np.einsum('ij,ij->i',q,q))[:,None]
            scores = {}
            for other in lane.ids:
                original = population.values[other][name].astype(np.float64)
                original /= np.sqrt(np.einsum('ij,ij->i',original,original))[:,None]
                matrix = np.einsum('ik,jk->ij',q,original)
                scores[other] = float(min(matrix.max(axis=0).mean(),matrix.max(axis=1).mean()))
            exact = sorted(scores,key=lambda k:(-scores[k],k))[:PLAN['top_k']]
            trace = router.search(population.values[key][name])
            check(all(abs(score-scores[k]) <= 2e-12 for k,score in trace['hits']), 'REPOSITORY_SCORE_READBACK')
            recall = len(set(exact) & {k for k,_ in trace['hits']})/PLAN['top_k']
            # Report ties diagnostically; the admission gate still uses strict recall.
            floor = scores[exact[-1]]
            tied = sum(scores[k] >= floor-2e-12 for k,_ in trace['hits'])/PLAN['top_k']
            rows.append(dict(query=key,exact=exact,route=trace,recall=recall,tie_aware_recall=tied))
        check(rows, 'REPOSITORY_RECALL_EMPTY')
        recall = sum(r['recall'] for r in rows)/len(rows)
        result[name] = dict(rows=rows,recall=recall,passes=recall >= PLAN['recall_gate'])
    return result


class AssociationCompanion:
    def __init__(self, population, excluded_tasks, queries):
        self.population, self.excluded, self.queries = population, list(excluded_tasks), list(queries)
        check(all(population.records[k]['task'] in self.excluded for k in queries), 'REPOSITORY_QUERY_SPLIT')
        self.updates = {n:SparseUpdates(s.identity,PLAN) for n,s in population.specs.items()}
        self.previous = None
        self.last = None

    def sync(self, groups, output, *, withdrawn=()):
        start = time.perf_counter()
        output = Path(output)
        check(not output.exists(), 'REPOSITORY_ARTIFACT_EXISTS')
        p = self.population
        p.fresh()
        source_checked = time.perf_counter()
        ids = p.prefix(groups,self.excluded,withdrawn)
        if self.last is not None and self.last['source'] == p.source(groups) and self.last['ids'] == ids:
            return self.last,self.previous['pin'],dict(wall_seconds=time.perf_counter()-start,
                phases=dict(source=time.perf_counter()-start),reused_artifact=True)
        lane_map = lanes(p,ids)
        staged = time.perf_counter()
        graphs,work = {},{}
        for name,lane in lane_map.items():
            graphs[name],work[name] = self.updates[name].update(lane)
        woven = time.perf_counter()
        calibration = assess(p,lane_map,graphs,self.queries)
        assessed = time.perf_counter()
        output.mkdir()
        cache = {n:u.cache for n,u in self.updates.items()}
        cache_pin = write_committed(output/'cache.json',cache)
        value = dict(schema='repository-association-kernel-v1',source=p.source(groups),plan=PLAN,
            parent=self.previous, code=code_pins(), excluded_tasks=self.excluded,queries=self.queries,
            withdrawn=sorted(withdrawn), ids=ids, absent=[k for k in ids if p.records[k]['absence']],
            instruments={n:s.identity for n,s in p.specs.items()},
            slots={k:p.records[k]['receipt']['slots'] for k in ids},
            graphs=graphs,work=work,calibration=calibration,cache_sha256=cache_pin,
            source_files=p.files, recall_passes=all(r['passes'] for r in calibration.values()),
            scientific_admission=False,world_truth_established=False,learned_capability=False,
            serving=False,number_status='measured',acceptance='author-checked',
            scope='retained_repository_native_association_retrieval',
            integration='journal_driven_companion_not_installed_as_model_active_kernel')
        pin = write_committed(output/'manifest.json',value)
        write_committed(output/'COMMITTED',dict(manifest=pin))
        end = time.perf_counter()
        timings = dict(wall_seconds=end-start, phases=dict(source=source_checked-start,
            lanes=staged-source_checked,weave=woven-staged,recall=assessed-woven,persist=end-assessed))
        write_committed(output/'timings.json',timings)
        self.previous = dict(path=str(output),pin=pin)
        self.last = value
        return value,pin,timings


class RepositoryKernel:
    def __init__(self, population, path, pin, *, expected_source):
        self.population,self.path = population,Path(path)
        raw = (self.path/'manifest.json').read_bytes()
        check(sha(raw) == pin and json.loads((self.path/'COMMITTED').read_bytes()) == dict(manifest=pin),
              'REPOSITORY_ARTIFACT_PIN')
        v = json.loads(raw)
        check(v['source'] == expected_source == population.source(v['source']['supported_intakes']),
              'REPOSITORY_STALE_HEAD')
        population.fresh()
        check(v['code'] == code_pins() and v['plan'] == PLAN and v['source_files'] == population.files
            and v['instruments'] == {n:s.identity for n,s in population.specs.items()}
            and all(v[k] is False for k in ('scientific_admission','world_truth_established','learned_capability','serving')),
              'REPOSITORY_ARTIFACT_CONTRACT')
        ids = population.prefix(v['source']['supported_intakes'],v['excluded_tasks'],v['withdrawn'])
        check(v['ids'] == ids and v['absent'] == [k for k in ids if population.records[k]['absence']]
            and v['slots'] == {k:population.records[k]['receipt']['slots'] for k in ids}
            and not set(v['queries']) & set(ids), 'REPOSITORY_ARTIFACT_COVERAGE')
        check(sha((self.path/'cache.json').read_bytes()) == v['cache_sha256'], 'REPOSITORY_CACHE_PIN')
        check(v['recall_passes'] and set(v['calibration']) == set(population.specs)
              and all(r['passes'] and r['recall'] >= PLAN['recall_gate'] for r in v['calibration'].values()),
              'REPOSITORY_RECALL_GATE')
        self.value, self.pin = v,pin
        self.lanes = lanes(population,ids)
        actual = assess(population,self.lanes,v['graphs'],v['queries'])
        check(actual == v['calibration'],'REPOSITORY_RECALL_REPLAY')
        self.routers = {n:Router(lane,v['graphs'][n],PLAN) for n,lane in self.lanes.items()}
        self.available = max(utc_ns(population.records[k]['available']) for k in ids)

    def route(self, key, *, source, cutoff):
        check(source == self.value['source'], 'REPOSITORY_STALE_HEAD')
        check(key in self.population.values and utc_ns(cutoff) >= self.available
              and utc_ns(cutoff) >= utc_ns(self.population.records[key]['available']), 'REPOSITORY_ROUTE_CUTOFF')
        check(not self.population.records[key]['absence'], 'REPOSITORY_ABSENT_QUERY')
        traces = {n:r.search(self.population.values[key][n]) for n,r in self.routers.items()}
        return dict(kernel=self.pin,query=key,traces=traces,source=source,
            sources={k:self.population.records[k]['episode'] for t in traces.values() for k,_ in t['hits']},
            scientific_admission=False,claim='measured_native_similarity_only')
