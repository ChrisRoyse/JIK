"""Versioned continual intake of new capability sources without rewriting historical producers."""
import copy
import json
from pathlib import Path

from claims.capability_evidence import CapabilityEvidence
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.capability_delta_input import DeltaNative, DeltaStore
from model.capability_live_memory import LiveCapabilityMemoryLoop


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {name: sha((root / name).read_bytes()) for name in (
        'model/capability_delta_memory.py', 'model/capability_delta_input.py',
        'claims/capability_revision_population.py', 'text/capability_delta_plan.py',
        'kernel/capability_registry.py', 'kernel/capability_split.py')}


class DeltaCapabilityMemoryLoop(LiveCapabilityMemoryLoop):
    def __init__(self, *args, **kwargs):
        self.delta_stores, self.delta_registries = {}, {}
        super().__init__(*args, **kwargs)

    def _delta_contract(self, spec):
        need(set(spec) == {'population', 'population_pin', 'plan', 'plan_pin', 'native', 'native_pin', 'adapter'},
             'DELTA_MEMORY_SPEC')
        adapter = spec['adapter']
        need(set(adapter) == {'schema', 'code', 'registry', 'native_proof'}
             and adapter['schema'] == 'capability-delta-native-v1'
             and adapter['code'] == code_pins(), 'DELTA_MEMORY_CODE')
        registry = adapter['registry']
        raw = Path(registry['path']).read_bytes()
        need(sha(raw) == registry['pin'], 'DELTA_MEMORY_REGISTRY_PIN')
        if registry['pin'] not in self.delta_registries:
            value = json.loads(raw)
            need(value['schema'] == 'capability-global-split-registry-v1' and value['complete']
                 and value['historical_membership_unchanged'], 'DELTA_MEMORY_REGISTRY')
            self.delta_registries[registry['pin']] = value
        value = self.delta_registries[registry['pin']]
        need(spec['population_pin'] in value['splits']
             and value['population_paths'][spec['population_pin']] == spec['population'],
             'DELTA_MEMORY_REGISTRY_SOURCE')
        for evidence in self.capability_sources.values():
            pin = evidence.population.pin
            need(pin in value['splits'] and evidence.native.pin in self.task_splits,
                 'DELTA_MEMORY_UNREGISTERED_HISTORY')
            old, proposed = self.task_splits[evidence.native.pin], value['splits'][pin]
            need(all(old[k] == proposed[k] for k in ('groups', 'excluded', 'train', 'queries')),
                 'DELTA_MEMORY_SPLIT_MOVED')
        proof = adapter['native_proof']
        root = Path(proof['path'])
        raw = (root / 'report.json').read_bytes()
        need(sha(raw) == proof['pin'], 'DELTA_MEMORY_PROOF_PIN')
        report = json.loads(raw)
        need(report['complete'] and report['all_native_windows_preserved']
             and report['plan'] == spec['plan_pin'] and report['native'] == spec['native_pin']
             and report['registry'] == registry['pin'], 'DELTA_MEMORY_PROOF_JOIN')
        return value, report['populations'][spec['population_pin']]['binding']

    def _capability_source(self, spec):
        if 'adapter' not in spec:
            return super()._capability_source(spec)
        _, binding = self._delta_contract(spec)
        identity = digest(spec)
        if identity in self.capability_sources:
            evidence = self.capability_sources[identity]
            evidence.fresh()
            return evidence
        args = {k: spec[k] for k in ('plan', 'plan_pin', 'native', 'native_pin')}
        key = digest(args)
        if key not in self.delta_stores:
            self.delta_stores[key] = DeltaStore(**args)
        native = DeltaNative(spec['population'], spec['population_pin'], store=self.delta_stores[key])
        need(native.binding == binding and tuple(native.specs) == tuple(sorted(self.model.specs, key=lambda s: s.name)),
             'DELTA_MEMORY_NATIVE_BINDING')
        native.fresh()
        return CapabilityEvidence(self.memory.checker, native, {})

    def register_task_source(self, spec, split, *, expected=None):
        if 'adapter' not in spec:
            return super().register_task_source(spec, split, expected=expected)
        self._fresh()
        need(self.task_spec is not None, 'DELTA_MEMORY_UPDATES_DISABLED')
        registry, _ = self._delta_contract(spec)
        evidence = self._capability_source(spec)
        pin = evidence.native.pin
        need(pin not in self.task_splits and split == registry['splits'][evidence.population.pin],
             'DELTA_MEMORY_REGISTER')
        body = self._task_event('register', dict(binding=pin, spec=spec, split=split), expected)
        self.task_splits[pin] = copy.deepcopy(split)
        return body

    def relocate_task_outputs(self, root, *, expected=None):
        """Journal the successor's writable directory; existing checkpoints retain their addresses."""
        self._fresh()
        root = Path(root)
        need(root.is_absolute() and root.parent.is_dir() and self.task_spec is not None
             and self.task_intent is None and str(root) != self.task_spec['root'], 'DELTA_MEMORY_OUTPUT_ROOT')
        payload = dict(previous=self.task_spec['root'], root=str(root), memory=self.memory.head)
        body = dict(kind='capability_delta_output_root', prior_head=self.head,
                    payload=digest(payload), code=code_pins())
        self._echo_publish(body, payload, expected)
        self.task_spec = dict(self.task_spec, root=str(root))
        return body

    def _replay(self, body):
        if body.get('kind') == 'capability_delta_output_root':
            need(body['code'] == code_pins(), 'DELTA_MEMORY_REPLAY_CODE')
            payload = self.journal.payload(body['payload'])
            self.relocate_task_outputs(payload['root'], expected=body)
        else:
            super()._replay(body)
