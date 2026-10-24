"""Bounded port of Calyx kernel_graph/dfvs; graph proof is never recall admission."""
import itertools
import math

import networkx as nx

from kernel.native_route import check


def topology(ids, edges):
    check(len(ids) == len(set(ids)) and 0 < len(ids) <= 2000, "GRAPH_IDS")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(ids))
    pairs = set()
    for a, b, score in edges:
        check(a in graph and b in graph and a != b and (a, b) not in pairs
              and isinstance(score, (int, float)) and math.isfinite(score)
              and 0 < score <= 1.00000001, "GRAPH_EDGE")
        pairs.add((a, b))
        graph.add_edge(a, b, distance=1.0 / score)
    return graph


def remainder_acyclic(graph, removed):
    return nx.is_directed_acyclic_graph(graph.subgraph(set(graph) - set(removed)))


def feedback_vertices(graph):
    nodes = sorted(graph)
    if len(nodes) <= 20:
        for count in range(len(nodes) + 1):
            for trial in itertools.combinations(nodes, count):
                if remainder_acyclic(graph, trial):
                    return list(trial), "bounded-exact"
    removed = set()
    degree = dict(graph.degree())
    while True:
        remaining = graph.subgraph(set(graph) - removed)
        cyclic = [c for c in nx.strongly_connected_components(remaining) if len(c) > 1]
        if not cyclic:
            break
        for component in cyclic:
            removed.add(max(component, key=lambda x: (degree[x], x)))
    check(len(removed) <= 512, "LOCAL_SEARCH_BUDGET")
    for key in sorted(removed):
        if remainder_acyclic(graph, removed - {key}):
            removed.remove(key)
    check(remainder_acyclic(graph, removed), "DFVS_REMAINDER")
    return sorted(removed), "original-degree-greedy-scc-local-shrink"


def derive(ids, edges, plan):
    graph = topology(ids, edges)
    between = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    degree = dict(graph.degree())
    max_degree = max(1, max(degree.values()))
    # Source attribution is not a grounded world outcome. No world anchor is invented.
    scores = {key: .4 * degree[key] / max_degree + .4 * between[key] for key in ids}
    ranked = sorted(ids, key=lambda key: (-scores[key], key))
    selected = ranked[:max(1, math.ceil(len(ids) * plan["kernel_fraction"]))]
    members, method = feedback_vertices(graph.subgraph(selected))
    entries, support = set(members), []
    for component in nx.weakly_connected_components(graph):
        if not component & entries:
            key = min(component, key=lambda x: (-scores[x], x))
            support.append(key); entries.add(key)
    result = dict(edges=edges, candidate_graph=sorted(selected), members=members,
                  graph_only_support=sorted(support), entries=sorted(entries), method=method,
                  world_grounded_fraction=0.0, groundedness_gaps=sorted(entries),
                  scores=scores, production_admission=False, recall_assessed=False)
    validate_graph(ids, result)
    return result


def validate_graph(ids, value):
    graph = topology(ids, value["edges"])
    selected, members = value["candidate_graph"], value["members"]
    support, entries = value["graph_only_support"], value["entries"]
    for group in (selected, members, support, entries):
        check(isinstance(group, list) and len(group) == len(set(group))
              and set(group) <= set(ids), "GRAPH_MEMBERS")
    check(set(members) <= set(selected) and not set(members) & set(support)
          and set(entries) == set(members) | set(support) and entries, "GRAPH_SUPPORT")
    check(remainder_acyclic(graph.subgraph(selected), members), "DFVS_REMAINDER")
    check(all(c & set(entries) for c in nx.weakly_connected_components(graph)), "COMPONENT_ENTRY")
    check(value["world_grounded_fraction"] == 0.0 and value["production_admission"] is False
          and value["groundedness_gaps"] == sorted(entries), "GROUNDING_SCOPE")
