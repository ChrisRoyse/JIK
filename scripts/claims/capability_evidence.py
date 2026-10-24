"""Attested fields and task connections in the existing source-evidence memory."""
from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest
from text.evaluation_multiscale import file_sha


class CapabilityEvidence:
    def __init__(self, base, native, bridges):
        self.base, self.native, self.bridges = base, native, bridges
        self.population = native.population
        self.field_episodes, self.task_episodes, self.receipts = {}, {}, {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def fresh(self):
        self.native.fresh()
        # A cache is not permission to ignore modified retained source bytes.
        for name, reference in self.population.report['files'].items():
            need(file_sha(self.population.root / name) == reference['sha256'], 'MEMORY_STALE_SOURCE_BYTES')

    def field_episode(self, key):
        if key not in self.field_episodes:
            p = self.population
            f = p.fields[key]
            body = dict(claim_kind='source_statement', content=f['text'],
                source=dict(capability_population=p.pin, field=key, source=f['source']),
                evidence=[], event_time=f['clock'], role=f['role'], family=p.family)
            self.field_episodes[key] = dict(id=digest(body), **body)
        return self.field_episodes[key]

    def task_episode(self, key):
        if key not in self.task_episodes:
            p = self.population
            t = p.tasks[key]
            fields = [f for side in ('context', 'targets', 'consequences') for f in t[side]]
            related = [self.field_episode(f)['id'] for f in fields]
            bridge = self.bridges.get(key, [])
            need(len(set(related + bridge)) == len(related + bridge), 'MEMORY_DEPENDENCY_ALIAS')
            body = dict(claim_kind='source_statement', content=canonical(t).decode(),
                source=dict(capability_population=p.pin, task=key),
                evidence=related + bridge, event_time=None, family=p.family,
                role='attested_task_connections', clock_claim='source_fields_only')
            self.task_episodes[key] = dict(id=digest(body), **body)
        return self.task_episodes[key]

    def validate_episode(self, episode):
        source = episode.get('source', {})
        if source.get('capability_population') != self.population.pin:
            return self.base.validate_episode(episode)
        if 'field' in source:
            need(source['field'] in self.population.fields, 'MEMORY_FIELD')
            expected = self.field_episode(source['field'])
        else:
            need(source.get('task') in self.population.tasks, 'MEMORY_TASK')
            expected = self.task_episode(source['task'])
        need(episode == expected, 'MEMORY_EPISODE_BINDING')

    def assess(self, episode):
        if episode.get('source', {}).get('capability_population') != self.population.pin:
            return self.base.assess(episode)
        self.validate_episode(episode)
        body = dict(checker='attested-capability-source-v1', episode=episode['id'],
            scope='pinned_artifact_contents_only', claim_kind='source_statement',
            source=episode['source'], world_validity='unassessed',
            native_binding=self.native.pin, scientific_admission=False,
            learned_capability=False, serving=False)
        return dict(id=digest(body), **body)

    def field_receipt(self, key):
        if key not in self.receipts:
            slots = {}
            for spec in self.native.specs:
                values, description = self.native.field(key, spec.name)
                slots[spec.name] = dict(instrument=spec.identity, shape=list(values.shape),
                    dtype='float32-le', sha256=sha(values.astype('<f4').tobytes()),
                    description=description)
            self.receipts[key] = dict(episode=self.field_episode(key)['id'],
                binding=self.native.pin, slots=slots)
        return self.receipts[key]


def execution_bridges(session, population):
    """Join to original observed execution; never infer an outcome from code text."""
    if population.family != 'coding': return {}
    existing = {}
    for name in ('repository_cohort_intakes', 'repository_depth_intakes',
                 'repository_demonstration_intakes'):
        for intake in getattr(session, name).values():
            for condition, episode in intake['episodes'].items():
                need(condition not in existing or existing[condition] == episode, 'MEMORY_EXECUTION_ALIAS')
                existing[condition] = episode
    bridges = {}
    for key, task in population.tasks.items():
        provenance = task['provenance']
        conditions = ([provenance['condition']] if task['kind'] == 'reproduce_recorded_patch'
                      else [provenance[s]['condition'] for s in ('failure', 'repair')])
        need(all(c in existing for c in conditions), 'MEMORY_EXECUTION_MISSING')
        bridges[key] = [existing[c] for c in conditions]
        for condition, ep in zip(conditions, bridges[key], strict=True):
            record = session.memory.episodes[ep]
            need(record['claim_kind'] == 'event_occurrence'
                 and record['what']['condition'] == condition, 'MEMORY_EXECUTION_JOIN')
    return bridges
