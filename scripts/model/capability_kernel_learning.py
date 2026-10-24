"""Request-aware learning from a pinned, previously replayed capability kernel.

The operation is an explicit caller control, not a new sense or a target-derived
feature. Native fields, source surfaces and fixed identity references stay separate.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from model.capability_input import CapabilityNative
from model.capability_delta_input import DeltaNative, DeltaStore
from model.capability_context_input import ContextNative, ContextStore
from model.constellation_world import ConstellationWorld
from model.native import from_arrays
from model.tool_context_regions import attach


OPERATIONS = ('continue_attested_dialogue', 'continue_recorded_steps',
              'repository_context_to_recorded_patch', 'respond_to_attested_act',
              'respond_to_source_question', 'revise_with_source_evidence',
              'solve_recorded_problem')


def pinned(ref):
    raw = Path(ref['path']).read_bytes()
    need(sha(raw) == ref['pin'], 'LEARNING_SOURCE_PIN:' + ref['path'])
    return json.loads(raw)


class KernelLearningCorpus:
    def __init__(self, config):
        self.config = config
        s = pinned(config['snapshot']); k = pinned(config['kernel'])
        audit = pinned(config['state_readback']); census = pinned(config['census'])
        need(sha(Path(config['journal']['path']).read_bytes()) == config['journal']['pin']
             == census['journal'], 'LEARNING_JOURNAL')
        need(census['snapshot_sha256'] == config['snapshot']['pin']
             and s['head'] == audit['head'] and audit['active'] == s['task_kernel']['checkpoint']
             and k['source'] == audit['active']['source'] and k['recall_passes']
             and not s['task_kernel']['pending'], 'LEARNING_CHECKED_STATE')
        self.snapshot, self.kernel = s, k
        self.tasks = s['capability_memory']['tasks']
        self.train = tuple(sorted({key for g in k['graphs'].values() for key in g['entries']}))
        self.query = tuple(k['split']['queries'])
        need(set(self.train) <= set(k['ids']) and not set(self.train) & set(self.query),
             'LEARNING_SPLIT')
        need(not {self.tasks[x]['split_group'] for x in k['ids']} &
             {self.tasks[x]['split_group'] for x in self.query}, 'LEARNING_GROUP_LEAK')
        need(all(len(self.tasks[x]['selections']['targets']['fields']) == 1
                 for x in (*self.train, *self.query)), 'LEARNING_SINGLE_TARGET_FIELD')
        self.sources, stores = {}, {}
        wanted = {self.tasks[x]['population'] for x in (*self.train, *self.query)}
        for intake in s['capability_memory']['intakes'].values():
            spec = intake['body']['spec']; pin = spec['population_pin']
            if pin not in wanted or pin in self.sources: continue
            args = {n: spec[n] for n in ('plan', 'plan_pin', 'native', 'native_pin')}
            schema = spec.get('adapter', {}).get('schema')
            if schema is None:
                native = CapabilityNative(spec['population'], pin, **args)
            else:
                need(schema in ('capability-context-native-v1', 'capability-delta-native-v1'),
                     'LEARNING_SOURCE_ADAPTER')
                store_type, native_type = ((ContextStore, ContextNative) if 'context' in schema
                                           else (DeltaStore, DeltaNative))
                key = digest(args)
                if key not in stores: stores[key] = store_type(**args)
                native = native_type(spec['population'], pin, store=stores[key])
            self.sources[pin] = native
        need(set(self.sources) == wanted, 'LEARNING_SOURCE_ROSTER')
        self.specs = next(iter(self.sources.values())).specs
        need(all(src.specs == self.specs for src in self.sources.values()), 'LEARNING_INSTRUMENTS')
        self.checked_fields = set()
        for key in (*self.train, *self.query):
            t = self.tasks[key]; source = self.sources[t['population']]
            need(source.pin == t['binding'] and t['kind'] in OPERATIONS
                 and t['kind'] == source.population.tasks[key]['kind'], 'LEARNING_TASK_BINDING')
            for side in ('context', 'targets', 'consequences'):
                need(t['selections'][side] == source.population.selection(key, side=side),
                     'LEARNING_SELECTION')
            episodes = [t['episode'], *t['related_executions']]
            for side in t['selections'].values():
                for field in side['fields']:
                    receipt = s['capability_memory']['fields'][t['population'] + ':' + field['id']]
                    episodes.append(receipt['episode'])
            need(all(s['memory']['states'][ep]['active'] and
                     s['memory']['states'][ep]['grounded_scope'] is not None for ep in episodes),
                 'LEARNING_WITHDRAWN_TASK')
        self.regions = tuple(sorted({(self.tasks[x]['population'], f['id']) for x in self.train
                                     for f in self.tasks[x]['selections']['targets']['fields']}))

    def field(self, pin, key, name):
        values, _ = self.sources[pin].field(key, name)
        index = pin, key, name
        if index not in self.checked_fields:
            receipt = self.snapshot['capability_memory']['fields'][pin + ':' + key]['slots'][name]
            need(list(values.shape) == receipt['shape'] and
                 sha(values.astype('<f4', copy=False).tobytes()) == receipt['sha256'],
                 'LEARNING_NATIVE_FIELD')
            self.checked_fields.add(index)
        return values

    def batch(self, key, side, config, device):
        need(key in self.train or key in self.query, 'LEARNING_TASK_SCOPE')
        need(side in ('context', 'targets'), 'LEARNING_SIDE')
        t = self.tasks[key]; fields = t['selections'][side]['fields']
        result = {}
        for spec in self.specs:
            values = np.concatenate([self.field(t['population'], f['id'], spec.name) for f in fields])
            result[spec.name] = from_arrays(spec, dict(tokens=values,
                offsets=np.array([0, len(values)], np.int64), absent=np.array([False])),
                (digest(dict(task=key, side=side)),), config).to(device)
        return result

    def target(self, key, config, device, *, fitting):
        need(key in (self.train if fitting else self.query), 'LEARNING_TARGET_SPLIT')
        t = self.tasks[key]; f = t['selections']['targets']['fields'][0]
        field = self.sources[t['population']].population.fields[f['id']]
        raw = field['text'].encode()
        need(sha(raw) == field['text_sha256'], 'LEARNING_SURFACE_HASH')
        return raw, self.batch(key, 'targets', config, device)

    def world(self, parent):
        references = {}
        rows = tuple(digest(dict(population=p, field=f)) for p, f in self.regions)
        for spec in self.specs:
            pieces = [self.field(p, f, spec.name) for p, f in self.regions]
            references[spec.name] = from_arrays(spec, dict(tokens=np.concatenate(pieces),
                offsets=np.array([0, *np.cumsum([len(x) for x in pieces])], np.int64),
                absent=np.zeros(len(rows), bool)), rows, parent.config).to('cuda')
        with torch.random.fork_rng(devices=[]):
            view = ConstellationWorld(parent.core, references, world_source_id=digest(
                dict(kernel=self.config['kernel'], regions=self.regions)))
        view.to('cuda').eval(); attach(view, parent); view.check_reference()
        return view


class RequestedOperation(nn.Module):
    def __init__(self, specs, width):
        super().__init__()
        self.names = tuple(s.name for s in specs)
        self.value = nn.Parameter(torch.zeros(len(OPERATIONS), len(specs), width))

    def forward(self, state, *, operation):
        need(operation in OPERATIONS, 'LEARNING_OPERATION_REQUIRED')
        need(tuple(state.hidden.shape) == tuple(self.value.shape[1:]), 'LEARNING_OPERATION_SHAPE')
        return replace(state, hidden=state.hidden + self.value[OPERATIONS.index(operation)])


def predict(world, request, native, *, key, operation):
    state = request(world.start('capability:' + key), operation=operation)
    return world(native, (state,), allowed=torch.ones(len(world.reference_rows[world.specs[0].name]),
        dtype=torch.bool, device='cuda'), phase=0, event_ids=[digest(dict(key=key, operation=operation))],
        evidence_origin=['source_observation'], advance_phase=False)
