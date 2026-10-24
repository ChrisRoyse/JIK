"""Journal-state views over one or more immutable sources per capability family."""
from claims.capability_population import need
from kernel.capability_population import TaskPopulation, SharedTasks


class JoinedTasks(SharedTasks):
    """Reuse the typed union operations; sources, not families, own split contracts."""
    def __init__(self, populations):
        self.populations = dict(populations)
        need({p.population.family for p in populations.values()} ==
             {'coding', 'conversation', 'reasoning'}, 'TASK_LIVE_FAMILY_COVERAGE')
        self.specs = next(iter(populations.values())).specs
        need(all(p.specs == self.specs for p in populations.values()), 'TASK_LIVE_INSTRUMENTS')
        self.records, self.values, self.links, self.owners, self.statuses = {}, {}, {}, {}, {}
        self.eligible = []
        self.split = dict(schema='growing-capability-split-v1', sources={k: p.split for k, p in populations.items()})
        for key in ('groups', 'excluded', 'train', 'queries'):
            self.split[key] = sorted({x for p in populations.values() for x in p.split[key]})
        for p in populations.values():
            need(not set(p.records) & set(self.records), 'TASK_LIVE_ID_ALIAS')
            for name in ('records', 'values', 'links', 'statuses'): getattr(self, name).update(getattr(p, name))
            self.owners.update({k: p for k in p.records}); self.eligible.extend(p.eligible)
        self.eligible.sort()
        self.files = {k: p.files for k, p in populations.items()}


def population(session):
    populations = {}
    for evidence in session.capability_sources.values():
        native = evidence.native
        need(native.pin in session.task_splits, 'TASK_LIVE_UNREGISTERED_SOURCE')
        p = TaskPopulation(native, session.capability_tasks, session.capability_fields,
                           session.capability_relations, session.memory.state, session.task_splits[native.pin])
        if p.records: populations[native.pin] = p
    families = {p.population.family: p for p in populations.values()}
    # Preserve the accepted three-source generation exactly; no migration rewrite.
    if len(populations) == len(families) == 3: return SharedTasks(families)
    return JoinedTasks(populations)
