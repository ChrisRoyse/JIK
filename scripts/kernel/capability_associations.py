"""Multiplex relationships: every retained instrument vote stays separately addressable."""
from itertools import combinations

from claims.episodic_memory import digest


def weave(population, graphs):
    edges, ranks = {}, {}
    for name, graph in sorted(graphs.items()):
        counts = {}
        for origin, target, score in graph['edges']:
            counts[origin] = counts.get(origin, 0) + 1
            key = digest([origin, target])
            edge = edges.setdefault(key, dict(origin=origin, target=target, readings={}))
            edge['readings'][name] = dict(score=score, rank=counts[origin])
            ranks[(name, origin, target)] = counts[origin]
    cross_domain = 0
    for edge in edges.values():
        a, b = edge['origin'], edge['target']
        edge['families'] = [population.records[k]['family'] for k in (a, b)]
        cross_domain += int(edge['families'][0] != edge['families'][1])
        # Relate the same two objects under different coordinate systems.
        # Never take a cosine between coordinates produced by different encoders.
        edge['cross_lens'] = [dict(instruments=[i, j],
            ranks=[ranks[(i, a, b)], ranks[(j, a, b)]], relation='co_measured_neighbor')
            for i, j in combinations(sorted(edge['readings']), 2)]
    return dict(schema='typed-task-associations-v1', edges=edges, cross_domain_edges=cross_domain,
        absence_means='not_materialized_not_no_association',
        meaning_claim='instrument_relative_neighborhood_only', scientific_admission=False)
