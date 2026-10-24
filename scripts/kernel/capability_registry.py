"""One exposed split registry across source arrivals and capability families."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import CapabilityPopulation, need
from claims.capability_revision_population import RevisionPopulation
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from kernel.capability_split import normalized
from text.evaluation_multiscale import file_sha


def freeze_together(populations, history):
    """Inverted shingle postings generate candidates; no dense task-pair matrix."""
    start = time.perf_counter()
    graph = nx.Graph()
    exact, templates, patterns, edges = {}, {}, {}, set()
    memberships = defaultdict(set)

    def link(a, b, kind):
        if a != b:
            graph.add_edge(a, b)
            edges.add((*sorted((a, b)), kind))

    for pin, pop in sorted(populations.items()):
        for key, task in sorted(pop.tasks.items()):
            group = task['split_group']
            graph.add_node(group)
            memberships[group].add(pin)
            texts = [pop.fields[f]['text'] for f in task['context']]
            signature = digest(texts)
            if signature in exact:
                link(group, exact[signature], 'exact_context')
            exact[signature] = group
            template = normalized('\n'.join(texts) if pop.family == 'conversation' else texts[0])
            if template in templates:
                link(group, templates[template], 'template')
            templates[template] = group
            words = template.split()
            if len(words) >= 20:
                identity = digest(template)
                if identity not in patterns:
                    patterns[identity] = dict(shingles={tuple(words[i:i+5]) for i in range(len(words)-4)},
                                              groups=set())
                patterns[identity]['groups'].add(group)
    postings, previous = defaultdict(list), {}
    overlap_candidates = exact_checks = 0
    for identity, value in sorted(patterns.items()):
        x, groups = value['shingles'], value['groups']
        overlap = set()
        for shingle in x:
            overlap.update(postings[shingle])
        overlap_candidates += len(overlap)
        for other in sorted(overlap):
            item = previous[other]
            if len(groups | item['groups']) < 2:
                continue
            y = item['shingles']
            if min(len(x), len(y)) < .8 * max(len(x), len(y)):
                continue
            exact_checks += 1
            if len(x & y) / len(x | y) >= .8:
                all_groups = sorted(groups | item['groups'])
                for group in all_groups[1:]:
                    link(all_groups[0], group, 'shingle')
        previous[identity] = value
        for shingle in x:
            postings[shingle].append(identity)
    known = {}
    for pin, split in sorted(history.items()):
        need(pin in populations and split['population'] == pin, 'REGISTRY_HISTORY_POPULATION')
        p = populations[pin]
        need(set(split['groups']) == {t['split_group'] for t in p.tasks.values()}, 'REGISTRY_HISTORY_GROUPS')
        for group in split['groups']:
            side = 'query' if group in split['excluded'] else 'train'
            need(group not in known or known[group] == side, 'REGISTRY_HISTORY_CONTRADICTION')
            known[group] = side
        need(set(split['train']) == {k for k, t in p.tasks.items() if known[t['split_group']] == 'train'}
             and set(split['queries']) == {k for k, t in p.tasks.items() if known[t['split_group']] == 'query'},
             'REGISTRY_HISTORY_TASKS')
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=digest)
    assignment, novel = {}, []
    for component in components:
        states = {known[g] for g in component if g in known}
        need(len(states) <= 1, 'REGISTRY_HISTORICAL_SPLIT_COLLISION')
        if states:
            assignment.update({g: next(iter(states)) for g in component})
        else:
            novel.append(component)
    query_goal = max(1, math.floor(sum(map(len, novel)) * .1)) if novel else 0
    assigned_queries = 0
    for component in novel:
        side = 'query' if assigned_queries < query_goal else 'train'
        assignment.update({g: side for g in component})
        if side == 'query':
            assigned_queries += len(component)
    need(all(assignment[g] == side for g, side in known.items()), 'REGISTRY_HISTORY_MOVED')
    splits = {}
    for pin, pop in sorted(populations.items()):
        groups = sorted({t['split_group'] for t in pop.tasks.values()})
        split = dict(schema='capability-registered-split-v1', population=pin, family=pop.family,
            exposure='development_exposed_not_sealed_evaluation', groups=groups,
            excluded=[g for g in groups if assignment[g] == 'query'],
            train=[k for k, t in sorted(pop.tasks.items()) if assignment[t['split_group']] == 'train'],
            queries=[k for k, t in sorted(pop.tasks.items()) if assignment[t['split_group']] == 'query'],
            components=[c for c in components if set(c) & set(groups)],
            duplicate_pairs=[list(e) for e in sorted(edges) if e[0] in groups or e[1] in groups],
            scientific_admission=False)
        need(len(split['train']) >= 17 and split['queries'], 'REGISTRY_SOURCE_SPLIT_INSUFFICIENT')
        splits[pin] = split
    return dict(schema='capability-global-split-registry-v1', complete=True, splits=splits,
        assignments=assignment, memberships={g: sorted(pins) for g, pins in sorted(memberships.items())},
        components=components, duplicate_pairs=[list(e) for e in sorted(edges)],
        history_sha256=digest(history), historical_membership_unchanged=True,
        counts=dict(populations=len(populations), tasks=sum(len(p.tasks) for p in populations.values()),
                    groups=len(assignment), templates=len(templates), shingle_patterns=len(patterns),
                    posting_memberships=sum(map(len, postings.values())),
                    overlap_candidates=overlap_candidates, exact_jaccard_checks=exact_checks),
        policy=dict(shingles=5, minimum_words=20, near_jaccard=.8, fraction=.1,
                    historical_split_conflict='refuse', component_order='canonical_sha256',
                    semantic_dedup_established=False, sealed_evaluation=False),
        wall_seconds=time.perf_counter()-start)


def load_sources(old_config, delta_config, historical_splits):
    old = json.loads(old_config.read_bytes())['populations']
    delta = json.loads(delta_config.read_bytes())
    populations = {s['population_pin']: CapabilityPopulation(s['population'], s['population_pin']) for s in old}
    populations.update({s['population_pin']: RevisionPopulation(s['population'], s['population_pin'])
                        for s in delta.values()})
    by_family = json.loads(historical_splits.read_bytes())
    history = {s['population_pin']: by_family[populations[s['population_pin']].family] for s in old}
    return populations, history


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-config', 'delta-config', 'historical-splits', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    populations, history = load_sources(a.old_config, a.delta_config, a.historical_splits)
    loaded = time.perf_counter()
    result = freeze_together(populations, history)
    result.update(inputs={str(path): file_sha(path) for path in (a.old_config, a.delta_config, a.historical_splits)},
                  population_paths={pin: str(pop.root) for pin, pop in populations.items()},
                  producer_sha256=file_sha(Path(__file__)), source_load_seconds=loaded-start)
    pin = write_committed(a.output, result)
    print(json.dumps(dict(registry_sha256=pin, counts=result['counts'], wall_seconds=time.perf_counter()-start)))
