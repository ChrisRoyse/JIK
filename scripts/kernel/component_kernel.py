"""Calyx-compatible component derivation with an exact bitset DFVS small path."""
import itertools
import math

import networkx as nx

from kernel.derive import feedback_vertices,topology,validate_graph
from kernel.native_route import check


def exact_feedback(graph):
    nodes = sorted(graph)
    check(len(nodes) <= 20,'EXACT_COMPONENT_BUDGET')
    index = {node:i for i,node in enumerate(nodes)}
    outgoing = [sum(1 << index[v] for v in graph.successors(node)) for node in nodes]
    full = (1 << len(nodes))-1
    def acyclic(alive):
        while alive:
            candidates = alive
            while candidates:
                bit = candidates & -candidates
                if not outgoing[bit.bit_length()-1] & alive:
                    alive ^= bit
                    break
                candidates ^= bit
            else:
                return False
        return True
    # Exactly the original cardinality / lexicographic enumeration, with one
    # immutable adjacency compilation instead of a fresh subgraph per trial.
    for count in range(len(nodes)+1):
        for removed in itertools.combinations(range(len(nodes)),count):
            if acyclic(full ^ sum(1 << i for i in removed)):
                return [nodes[i] for i in removed],'bounded-exact'
    raise AssertionError('MNLM_KERNEL_EXACT_NO_RESULT')


def component_derive(ids, edges, plan):
    graph = topology(ids,edges)
    between = nx.betweenness_centrality(graph,normalized=True,weight='distance')
    degree = dict(graph.degree()); max_degree = max(1,max(degree.values()))
    scores = {key:.4*degree[key]/max_degree+.4*between[key] for key in ids}
    ranked = sorted(ids,key=lambda k:(-scores[k],k))
    selected = ranked[:max(1,math.ceil(len(ids)*plan['kernel_fraction']))]
    subgraph = graph.subgraph(selected)
    members,method = exact_feedback(subgraph) if len(selected) <= 20 else feedback_vertices(subgraph)
    entries,support = set(members),[]
    for component in nx.weakly_connected_components(graph):
        if not component & entries:
            key = min(component,key=lambda k:(-scores[k],k))
            support.append(key);entries.add(key)
    result = dict(edges=edges,candidate_graph=sorted(selected),members=members,
        graph_only_support=sorted(support),entries=sorted(entries),method=method,scores=scores,
        world_grounded_fraction=0.0,groundedness_gaps=sorted(entries),
        production_admission=False,recall_assessed=False)
    validate_graph(ids,result)
    return result


def minimum_readback(graph, removed):
    """Independent source-peeling proof: no one-larger retained DAG exists."""
    nodes = sorted(graph);index = {n:i for i,n in enumerate(nodes)}
    check(len(nodes) <= 20,'EXACT_COMPONENT_BUDGET')
    incoming = [sum(1 << index[v] for v in graph.predecessors(node)) for node in nodes]
    for kept in itertools.combinations(range(len(nodes)),len(nodes)-len(removed)+1):
        alive = sum(1 << i for i in kept)
        while alive:
            sources = sum(1 << i for i in kept if (alive >> i) & 1 and not incoming[i] & alive)
            if not sources:
                break
            alive ^= sources
        check(bool(alive),'EXACT_NOT_MINIMUM')
