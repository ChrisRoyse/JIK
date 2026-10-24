"""Real native corpus -> typed kernel graph -> cold independent recall readback."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel import artifact, capsule
from kernel.derive import validate_graph
from kernel.native_route import NativeLane, Router, check


def independent_score(query, corpus):
    # Reconstruct from persisted original f32 windows, not router's normalized cache.
    q, c = query.astype(np.float64), corpus.astype(np.float64)
    q /= np.sqrt(np.einsum('ij,ij->i', q, q))[:, None]
    c /= np.sqrt(np.einsum('ij,ij->i', c, c))[:, None]
    pairs = np.einsum('ik,jk->ij', q, c)
    return float(min(sum(max(row) for row in pairs)/len(q),
                     sum(max(row) for row in pairs.T)/len(c)))


def independently_acyclic(ids, edges, removed):
    # Kahn readback independent of networkx SCC/acyclic implementation.
    nodes = set(ids)-set(removed)
    incoming = {key: 0 for key in nodes}
    adjacent = {key: [] for key in nodes}
    for a,b,_ in edges:
        if a in nodes and b in nodes:
            incoming[b] += 1; adjacent[a].append(b)
    ready = sorted(key for key,count in incoming.items() if count == 0)
    visited = 0
    while ready:
        a = ready.pop(); visited += 1
        for b in adjacent[a]:
            incoming[b] -= 1
            if incoming[b] == 0:
                ready.append(b)
    return visited == len(nodes)


def audit(loaded):
    meta, lanes, queries, graphs, routers = loaded
    output = {}
    for name in capsule.LANES:
        lane, graph, rows = lanes[name], graphs[name], []
        check(set(queries[name]) == set(meta['plan']['query_ids']) and len(queries[name]) == 24,
              'AUDIT_QUERIES')
        def original(key):
            i = lane.rows[key]
            return lane.original[lane.offsets[i]:lane.offsets[i+1]]
        edge_diff = max(abs(independent_score(original(a), original(b))-score)
                        for a,b,score in graph['edges'])
        check(edge_diff < 2e-12, 'EDGE_READBACK')
        check(independently_acyclic(graph['candidate_graph'], graph['edges'], graph['members']),
              'ACYCLIC_READBACK')
        max_diff = 0.
        for key, query in queries[name].items():
            direct = {x: independent_score(query, original(x)) for x in lane.ids}
            ranked = sorted(direct, key=lambda x: (-direct[x], x))[:meta['plan']['top_k']]
            trace = routers[name].search(query)
            difference = max(abs(score-direct[x]) for x,score in trace['hits'])
            max_diff = max(max_diff, difference)
            check(difference < 2e-12, 'SCORE_READBACK')
            recall = len(set(ranked) & {x for x,_ in trace['hits']}) / len(ranked)
            rows.append(dict(query=key, exact=ranked, route=trace, recall_at_10=recall))
        recall = sum(x['recall_at_10'] for x in rows)/len(rows)
        output[name] = dict(queries=rows, mean_recall_at_10=recall,
                           recall_gate_pass=recall >= meta['plan']['recall_gate'],
                           max_edge_score_error=edge_diff, max_routed_score_error=max_diff,
                           max_scores=max(x['route']['distance_computations'] for x in rows),
                           mean_scores=sum(x['route']['distance_computations'] for x in rows)/len(rows))
    return output


def controls(args, run, loaded):
    meta, lanes, queries, graphs, routers = loaded
    output, name = [], capsule.LANES[0]
    q = next(iter(queries[name].values()))
    def refuse(label, code, fn):
        try:
            fn()
        except ValueError as exc:
            check(str(exc) == 'MNLM_KERNEL_'+code, 'WRONG_REFUSAL')
            output.append(dict(name=label, code=code))
        else:
            raise AssertionError(label+' did not refuse')
    refuse('foreign_capsule_pin', 'CAPSULE_PIN',
           lambda: capsule.load(args.output/'capsule','0'*64))
    refuse('foreign_graph_pin', 'ARTIFACT_PIN',
           lambda: artifact.load(args.output/'capsule',run['capsule'],args.output/'graph','0'*64))
    refuse('overwrite_graph', 'ARTIFACT_EXISTS',
           lambda: artifact.build(args.output/'capsule',run['capsule'],args.output/'graph'))
    refuse('route_budget', 'ROUTE_BUDGET', lambda: routers[name].search(q,budget=1))
    refuse('query_layout', 'QUERY_LAYOUT', lambda: routers[name].search(q[:,:-1]))
    refuse('zero_query', 'ZERO_NORM', lambda: routers[name].search(np.zeros_like(q)))
    refuse('nan_query', 'QUERY_VALUES', lambda: routers[name].search(np.full_like(q,np.nan)))
    graph = copy.deepcopy(graphs[name]);graph['edges'][0][1] = 'foreign-atom'
    refuse('foreign_edge', 'GRAPH_EDGE', lambda: validate_graph(lanes[name].ids,graph))
    graph = copy.deepcopy(graphs[name]);graph['entries'] = []
    refuse('missing_entries', 'GRAPH_SUPPORT', lambda: validate_graph(lanes[name].ids,graph))
    graph = copy.deepcopy(graphs[name]);graph['world_grounded_fraction'] = 1.
    refuse('grounding_inflation', 'GROUNDING_SCOPE', lambda: validate_graph(lanes[name].ids,graph))
    qc = copy.deepcopy(queries);qc[name] = {}
    refuse('empty_query_recall', 'AUDIT_QUERIES', lambda: audit((meta,lanes,qc,graphs,routers)))
    qc = copy.deepcopy(queries);qc[name].pop(next(iter(qc[name])))
    refuse('missing_query', 'AUDIT_QUERIES', lambda: audit((meta,lanes,qc,graphs,routers)))
    bad = args.output/'corrupt-capsule'
    shutil.copytree(args.output/'capsule',bad)
    p = bad/(name+'.npz');raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    refuse('changed_native_bytes', 'CAPSULE_NATIVE_PIN', lambda: capsule.load(bad,run['capsule']))
    badg = args.output/'corrupt-graph';shutil.copytree(args.output/'graph',badg)
    (badg/(name+'.json')).write_bytes(b'{}')
    refuse('changed_graph_bytes', 'GRAPH_PIN',
           lambda: artifact.load(args.output/'capsule',run['capsule'],badg,run['graph']))
    return output


def main():
    parser = argparse.ArgumentParser()
    for key in ('population','native','plan','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--cold',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    check(capsule.sha(args.plan) == args.plan_sha256,'PLAN_PIN')
    plan=capsule.read(args.plan)
    if args.cold:
        run=capsule.read(args.output/'run.json')
        loaded=artifact.load(args.output/'capsule',run['capsule'],args.output/'graph',run['graph'])
        measured=audit(loaded);before=capsule.read(args.output/'recall.json')
        check(measured == before,'COLD_RECALL_CHANGED')
        capsule.write(args.output/'cold-readback.json',dict(complete=True,acceptance='author-checked',
                      capsule=run['capsule'],graph=run['graph'],recall_exact=True,
                      recall_sha256=capsule.sha(args.output/'recall.json'),
                      wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(cold=True,complete=True)));return
    check(not args.output.exists(),'OUTPUT_EXISTS');args.output.mkdir()
    phases={};stage=start
    pin=capsule.create(args.population,args.native,plan,args.output/'capsule')
    now=time.perf_counter();phases['source_slice']=now-stage;stage=now
    gp=artifact.build(args.output/'capsule',pin,args.output/'graph')
    now=time.perf_counter();phases['weave_kernel_publish']=now-stage;stage=now
    run=dict(capsule=pin,graph=gp,plan_sha256=args.plan_sha256)
    capsule.write(args.output/'run.json',run)
    loaded=artifact.load(args.output/'capsule',pin,args.output/'graph',gp)
    measured=audit(loaded);capsule.write(args.output/'recall.json',measured)
    now=time.perf_counter();phases['independent_readback']=now-stage;stage=now
    tested=controls(args,run,loaded)
    now=time.perf_counter();phases['controls']=now-stage
    report=dict(complete=True,acceptance='author-checked',production_admission=False,
                meaning_asserted=False,world_truth_established=False,
                useful_capability_gain_demonstrated=False,
                measured=dict(corpus_atoms=len(plan['corpus_ids']),queries=len(plan['query_ids']),
                              instruments=len(capsule.LANES),controls=len(tested)),
                recall={name:{k:v for k,v in row.items() if k!='queries'} for name,row in measured.items()},
                controls=tested,phase_seconds=phases,wall_seconds=time.perf_counter()-start,
                files={n:capsule.sha(args.output/n) for n in ('run.json','recall.json')})
    capsule.write(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('controls','files')}))


if __name__=='__main__':
    main()
