"""Freeze exposed query groups before measuring any connected-task retrieval."""
import math
import re
from itertools import combinations

import networkx as nx

from claims.capability_population import need
from claims.episodic_memory import digest


def normalized(text):
    return re.sub(r'\d+(?:[.,]\d+)*', '#', ' '.join(text.casefold().split()))


def freeze(population):
    p = population
    groups = sorted({t['split_group'] for t in p.tasks.values()})
    graph = nx.Graph(); graph.add_nodes_from(groups)
    exact, templates, shingles, pairs = {}, {}, {}, []
    for key, t in sorted(p.tasks.items()):
        context = [p.fields[k]['text'] for k in t['context']]
        pin = digest(context)
        group = t['split_group']
        if pin in exact and exact[pin] != group:
            graph.add_edge(group, exact[pin]); pairs.append([group, exact[pin], 'exact_context'])
        exact[pin] = group
        # Split metadata only. These strings never enter a sensing branch.
        first = normalized('\n'.join(context) if p.family == 'conversation' else context[0])
        if first in templates and templates[first] != group:
            graph.add_edge(group, templates[first]); pairs.append([group, templates[first], 'template'])
        templates[first] = group
        words = first.split()
        if len(words) >= 20:
            shingles[(group, first)] = {tuple(words[i:i+5]) for i in range(len(words)-4)}
    for (a, x), (b, y) in combinations(shingles.items(), 2):
        if a[0] == b[0]: continue
        jaccard = len(x & y) / len(x | y)
        if jaccard >= .8:
            graph.add_edge(a[0], b[0]); pairs.append([a[0], b[0], 'shingle', jaccard])
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=digest)
    goal = max(2 if p.family == 'coding' else 1, math.floor(len(groups) * .1))
    excluded = []
    for component in components:
        if len(excluded) >= goal: break
        excluded.extend(component)
    excluded = sorted(excluded)
    train = [k for k, t in sorted(p.tasks.items()) if t['split_group'] not in excluded]
    queries = [k for k, t in sorted(p.tasks.items()) if t['split_group'] in excluded]
    need(len(train) >= 17 and queries, 'TASK_SPLIT_INSUFFICIENT')
    return dict(schema='capability-split-v1', population=p.pin, family=p.family,
        exposure='development_exposed_not_sealed_evaluation', excluded=excluded,
        train=train, queries=queries, components=components, duplicate_pairs=pairs,
        policy=dict(component_order='sha256_canonical_sorted_groups', fraction=.1,
                    minimum_groups=2 if p.family == 'coding' else 1,
                    template='complete_conversation_context_else_first_intent;casefold_whitespace_numbers', near_jaccard=.8,
                    shingles=5, minimum_words=20, semantic_dedup_established=False),
        groups=groups, scientific_admission=False)
