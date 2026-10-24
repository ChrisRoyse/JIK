"""Connected capability sources are successors of the existing intrinsic-loop journal."""
import copy
from pathlib import Path

from claims.capability_evidence import CapabilityEvidence, execution_bridges
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.capability_input import CapabilityNative
from model.repository_live_memory import LiveRepositoryMemoryLoop


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n: sha((root / n).read_bytes()) for n in (
        'model/capability_memory.py', 'model/capability_input.py',
        'claims/capability_evidence.py', 'claims/capability_population.py')}


class CapabilityMemoryLoop(LiveRepositoryMemoryLoop):
    def __init__(self, *args, **kwargs):
        self.capability_intakes, self.capability_sources, self.capability_tasks = {}, {}, {}
        self.capability_fields, self.capability_relations = {}, {}
        super().__init__(*args, **kwargs)

    def _capability_source(self, spec):
        need(set(spec) == {'population', 'population_pin', 'plan', 'plan_pin', 'native', 'native_pin'},
             'MEMORY_SOURCE_SPEC')
        identity = digest(spec)
        if identity in self.capability_sources:
            evidence = self.capability_sources[identity]
            evidence.fresh()
            return evidence
        native = CapabilityNative(**spec)
        native.fresh()
        need(tuple(native.specs) == tuple(sorted(self.model.specs, key=lambda s: s.name)),
             'MEMORY_INSTRUMENT_ROSTER')
        return CapabilityEvidence(self.memory.checker, native, execution_bridges(self, native.population))

    def _capability_support(self, episode):
        record = self.memory.episodes.get(episode)
        if record is not None and record['claim_kind'] == 'event_occurrence':
            return self._repository_checked(episode)
        return self._support_checked(episode)

    def observe_capability(self, spec, groups, *, expected=None):
        self._fresh()
        need(type(groups) is list and groups and groups == sorted(set(groups)), 'MEMORY_GROUPS')
        need(self.live_intent is None, 'MEMORY_REPOSITORY_UPDATE_PENDING')
        evidence = self._capability_source(spec)
        p = evidence.population
        available = {t['split_group'] for t in p.tasks.values()}
        need(set(groups) <= available, 'MEMORY_UNKNOWN_GROUP')
        selected = {k: t for k, t in p.tasks.items() if t['split_group'] in groups}
        need(selected and not set(selected) & self.capability_tasks.keys(), 'MEMORY_DUPLICATE_TASK')
        fields = sorted({f for t in selected.values() for side in ('context', 'targets', 'consequences')
                         for f in t[side]})
        field_set = set(fields)
        links = [e for e in p.edges if e['origin'] in field_set and e['target'] in field_set]
        # Original data/source objects are immutable. Copy only mutable memory state for one transaction.
        old, staged = self.memory, copy.copy(self.memory)
        for name in ('episodes', 'state', 'children', 'events'):
            setattr(staged, name, copy.deepcopy(getattr(old, name)))
        if digest(spec) not in self.capability_sources: staged.checker = evidence
        self.memory = staged
        try:
            records, receipts, tasks = [], {}, {}
            for field in fields:
                records.extend(self._intake(evidence.field_episode(field)))
                receipts[field] = evidence.field_receipt(field)
            for key, task in selected.items():
                for ep in evidence.bridges.get(key, []): self._capability_support(ep)
                ep = evidence.task_episode(key)
                records.extend(self._intake(ep))
                tasks[key] = dict(episode=ep['id'], family=p.family, kind=task['kind'],
                    split_group=task['split_group'], population=p.pin, binding=evidence.native.pin,
                    selections={s: p.selection(key, side=s) for s in ('context', 'targets', 'consequences')},
                    related_executions=evidence.bridges.get(key, []), partition='development_exposed',
                    scientific_admission=False, training_ready=False, serving=False)
            payload = dict(tasks=tasks, fields=receipts, relationships=links, memory_records=records,
                native_binding=evidence.native.binding, source_spec=spec, groups=groups,
                scientific_admission=False, learned_capability=False, serving=False)
            body = dict(kind='capability_intake', spec=spec, groups=groups, prior_head=self.head,
                        payload=digest(payload), code=code_pins())
            self._echo_publish(body, payload, expected)
        except BaseException:
            self.memory = old
            raise
        self.capability_sources[digest(spec)] = evidence
        self.capability_intakes[digest(body)] = dict(body=body, tasks=list(tasks), fields=fields)
        self.capability_tasks.update(tasks)
        self.capability_fields.update({p.pin + ':' + k: v for k, v in receipts.items()})
        self.capability_relations.update({digest(e): e for e in links})
        self.journal.cache.clear()
        return body

    def capability_selection(self, key, *, side, binding):
        self._fresh()
        need(key in self.capability_tasks, 'MEMORY_UNKNOWN_TASK')
        task = self.capability_tasks[key]
        need(binding == task['binding'], 'MEMORY_STALE_BINDING')
        need(side in ('context', 'targets', 'consequences'), 'MEMORY_SELECTION_SIDE')
        self._support_checked(task['episode'])
        # Checking the task alone is insufficient if a source was retired separately.
        for ep in self.memory.episodes[task['episode']]['evidence']: self._capability_support(ep)
        return copy.deepcopy(task['selections'][side])

    def review_capability(self, episode, *, reason, expected=None):
        """Recheck retained source bytes; this never repairs an incorrect source claim."""
        self._fresh()
        ep = self.memory.episodes.get(episode)
        need(ep is not None and ep.get('source', {}).get('capability_population') is not None,
             'MEMORY_REVIEW_EPISODE')
        need(isinstance(reason, str) and reason.strip(), 'MEMORY_REVIEW_REASON')
        need(self.memory.state[episode]['active'] and self.memory.state[episode]['grounded_scope'] is None,
             'MEMORY_REVIEW_STATE')
        candidates = [e for e in self.capability_sources.values()
                      if e.population.pin == ep['source']['capability_population']]
        need(len(candidates) == 1, 'MEMORY_REVIEW_SOURCE')
        candidates[0].fresh()
        for parent in ep['evidence']: self._capability_support(parent)
        old, staged = self.memory, copy.copy(self.memory)
        for name in ('episodes', 'state', 'children', 'events'):
            setattr(staged, name, copy.deepcopy(getattr(old, name)))
        self.memory = staged
        try:
            records = [staged.apply(dict(op='check_source', episode=episode))]
            receipt = staged.state[episode]['receipt']['id']
            records.append(staged.apply(dict(op='promote', episode=episode, receipt=receipt)))
            payload = dict(records=records, reason=reason, source_claim_only=True)
            body = dict(kind='capability_source_review', episode=episode, reason=reason,
                        prior_head=self.head, payload=digest(payload), code=code_pins())
            self._echo_publish(body, payload, expected)
        except BaseException:
            self.memory = old
            raise
        return body

    def _replay(self, body):
        if body.get('kind', '').startswith('capability_'):
            need(body['code'] == code_pins(), 'MEMORY_PRODUCER_CODE')
        if body.get('kind') == 'capability_intake':
            self.observe_capability(body['spec'], body['groups'], expected=body)
        elif body.get('kind') == 'capability_source_review':
            self.review_capability(body['episode'], reason=body['reason'], expected=body)
        else:
            super()._replay(body)

    def snapshot(self):
        result = super().snapshot()
        if self.capability_intakes:
            eligible = [k for k, t in self.capability_tasks.items()
                if self.memory.state[t['episode']]['active']
                and self.memory.state[t['episode']]['grounded_scope'] is not None]
            result['capability_memory'] = dict(intakes=copy.deepcopy(self.capability_intakes),
                tasks=copy.deepcopy(self.capability_tasks), fields=copy.deepcopy(self.capability_fields),
                relationships=copy.deepcopy(self.capability_relations), eligible_tasks=sorted(eligible),
                scientific_admission=False, training_ready=False, learned_capability=False, serving=False)
        return result
