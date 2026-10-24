"""Independent delta journal, native-array, kernel and restart readback; no live producer."""
import argparse
import copy
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_journal_reader import JournalReader
from audit.capability_live_readback import LiveReader
from audit.repository_memory_readback import chain
from claims.capability_population import need, sha
from claims.capability_revision_population import RevisionPopulation
from claims.episodic_memory import digest
from claims.process_conditions import write_committed


def read(path): return json.loads(Path(path).read_bytes())


def pinned(path, pin):
    raw = Path(path).read_bytes()
    need(sha(raw) == pin, 'DELTA_READER_PIN')
    return json.loads(raw)


class DeltaReader(LiveReader):
    def __init__(self, config, proofs):
        self.delta_config = config
        parent = Path(config['source'])
        need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'],
             'DELTA_READER_PARENT')
        # Author-check the complete historical state using its existing independent reader.
        super().__init__(read(parent / 'config.json'), proofs)
        historical = super().run(parent / 'experience.sqlite3')
        saved = read(parent / 'snapshot-final.json')
        need(historical['head'] == saved['head'] and historical['memory_head'] == saved['memory']['head']
             and self.states == saved['memory']['states'] and self.active == saved['task_kernel']['checkpoint'],
             'DELTA_READER_HISTORICAL_STATE')
        self.contract, self.prefix = chain(parent / 'experience.sqlite3')
        self.task_splits = copy.deepcopy(saved['task_kernel']['splits'])
        self.tasks, self.fields = copy.deepcopy(self.base['tasks']), copy.deepcopy(self.base['fields'])
        self.relationships = copy.deepcopy(self.base['relationships'])
        self.slots = copy.deepcopy(self.origin['slots'])
        self.links = copy.deepcopy(self.origin['connections'])
        self.source_templates = {}
        for value in self.origin['source']['families'].values():
            self.source_templates[value['files']['population']] = copy.deepcopy(value)
        self.publications, self.transitions, self.intakes = [], [], []
        self.association_readbacks = []
        self.registrations = {}
        registry = config['registry']
        self.registry = pinned(registry['path'], registry['pin'])
        proof = config['native_proof']; root = Path(proof['path'])
        self.proof = pinned(root / 'report.json', proof['pin'])
        need(self.proof['complete'] and self.proof['registry'] == registry['pin'], 'DELTA_READER_NATIVE_PROOF')
        for name, info in self.proof['files'].items():
            raw = (root / name).read_bytes()
            need(info == dict(bytes=len(raw), sha256=sha(raw)), 'DELTA_READER_PROOF_FILE')
        self.receipts, self.bindings = read(root / 'field-receipts.json'), read(root / 'bindings.json')
        definitions = read(config['delta_config'])
        self.populations = {s['population_pin']: RevisionPopulation(s['population'], s['population_pin'])
                            for s in definitions.values()}
        self.expected_specs = definitions
        self.delta_intake_reader = object.__new__(JournalReader)
        self.delta_intake_reader.populations = self.populations
        self.delta_intake_reader.proofs = {p: (self.receipts[p], self.bindings[p]) for p in self.populations}
        self.delta_intake_reader.bridges = {}
        self.arrays, self.field_arrays = {}, {}
        self.native_root = Path(config['measurements']['native'])
        self.code_root = Path(__file__).resolve().parents[1]
        self.expected_code = {n: sha((self.code_root / n).read_bytes()) for n in (
            'model/capability_delta_memory.py', 'model/capability_delta_input.py',
            'claims/capability_revision_population.py', 'text/capability_delta_plan.py',
            'kernel/capability_registry.py', 'kernel/capability_split.py')}

    def register(self, payload):
        spec, split = payload['spec'], payload['split']; pin = spec['population_pin']
        adapter = dict(schema='capability-delta-native-v1', code=self.expected_code,
                       registry=self.delta_config['registry'], native_proof=self.delta_config['native_proof'])
        need(any(spec == dict(v, **self.delta_config['measurements'], adapter=adapter)
                 for v in self.expected_specs.values()), 'DELTA_READER_REGISTER_SPEC')
        binding = digest(self.bindings[pin])
        need(payload['binding'] == binding and binding not in self.task_splits
             and split == self.registry['splits'][pin], 'DELTA_READER_SPLIT')
        self.registrations[pin] = spec
        self.task_splits[binding] = split

    def field_array(self, pin, field, name):
        key = (pin, field, name)
        if key in self.field_arrays: return self.field_arrays[key]
        receipt = self.receipts[pin][field][name]
        parts = []
        for part in receipt['description']['parts']:
            origin = self.native_root if part['origin'] == 'fresh' else Path(part['origin'])
            bank = (str(origin), name)
            if bank not in self.arrays:
                with np.load(origin / name / 'slots.npz', allow_pickle=False) as z:
                    self.arrays[bank] = z['vectors'].copy(), z['offsets'].copy()
            vectors, offsets = self.arrays[bank]
            lo, hi = map(int, offsets[part['row']:part['row']+2])
            need([lo, hi] == part['source_window_span'], 'DELTA_READER_NATIVE_SPAN')
            raw = self.populations[pin].fields[field]['text'].encode()[slice(*part['source_byte_span'])]
            need(sha(raw) == part['source_sha256'], 'DELTA_READER_SOURCE_BYTES')
            parts.append(vectors[lo:hi])
        array = np.concatenate(parts)
        need(list(array.shape) == [receipt['windows'], receipt['dimensions']]
             and sha(array.astype('<f4').tobytes()) == receipt['sha256'], 'DELTA_READER_FIELD_ARRAY')
        self.field_arrays[key] = array
        return array

    def add_slots(self, pin, key):
        pop = self.populations[pin]; original = pop.tasks[key]
        slots = {}
        for name, instrument in self.bindings[pin]['instruments'].items():
            pieces, descriptions, cursor = [], [], 0
            for field in original['context']:
                array = self.field_array(pin, field, name)
                descriptions.append(dict(self.receipts[pin][field][name]['description'],
                                         start=cursor, end=cursor+len(array)))
                cursor += len(array); pieces.append(array)
            array = np.concatenate(pieces)
            slots[name] = dict(instrument=instrument, shape=list(array.shape),
                               sha256=sha(array.astype('<f4').tobytes()), fields=descriptions)
        self.slots[key] = slots
        selected = {f for side in ('context', 'targets', 'consequences') for f in original[side]}
        links = dict(task=self.tasks[key], fields={f: self.fields[pin+':'+f] for f in selected},
                     relationships=[e for e in pop.edges if e['origin'] in selected and e['target'] in selected])
        need(all(self.relationships.get(digest(e)) == e for e in links['relationships']),
             'DELTA_READER_RELATION_MISSING')
        self.links[key] = links

    def memory_record(self, record):
        if not hasattr(self, 'source_templates'): return super().memory_record(record)
        need(record['sequence'] == self.memory_sequence and record['previous'] == self.memory_head
             and record['sha256'] == digest({k: v for k, v in record.items() if k != 'sha256'}),
             'DELTA_READER_MEMORY_CHAIN')
        event = record['event']; key = event['episode']
        need(key in self.episodes, 'DELTA_READER_REVIEW_SCOPE')
        state = self.states[key]
        if event['op'] == 'check_source':
            receipt = event['result']
            need(receipt['id'] == digest({k: v for k, v in receipt.items() if k != 'id'})
                 and receipt['episode'] == key and receipt['source'] == self.episodes[key]['source']
                 and receipt['scope'] == 'pinned_artifact_contents_only'
                 and receipt['world_validity'] == 'unassessed', 'DELTA_READER_SOURCE_RECEIPT')
            need(all(self.states[k]['active'] and self.states[k]['grounded_scope'] is not None
                     for k in self.episodes[key]['evidence']), 'DELTA_READER_PREMATURE_PROMOTION')
            state.update(receipt=receipt, review='source_checked')
        else:
            need(event['op'] == 'promote' and state['receipt'] is not None
                 and event['receipt'] == state['receipt']['id'], 'DELTA_READER_PROMOTION')
            state.update(grounded_scope=state['receipt']['scope'], review='source_checked')
        self.memory_head, self.memory_sequence = record['sha256'], self.memory_sequence+1

    def intake(self, body, payload):
        pin = body['spec']['population_pin']
        need(body['spec'] == self.registrations[pin], 'DELTA_READER_UNREGISTERED_INTAKE')
        self.delta_intake_reader.config = dict(populations=list(self.registrations.values()))
        expected = self.delta_intake_reader.intake(body, payload)
        need(not set(payload['tasks']) & set(self.tasks), 'DELTA_READER_DUPLICATE_TASK')
        actual = set()
        for record in payload['memory_records']:
            event = record['event']
            if event['op'] != 'intake':
                self.memory_record(record)
                continue
            need(record['sequence'] == self.memory_sequence and record['previous'] == self.memory_head
                 and record['sha256'] == digest({k: v for k, v in record.items() if k != 'sha256'}),
                 'DELTA_READER_MEMORY_CHAIN')
            ep = event['episode']; key = ep['id']
            need(key not in self.states and ep == expected[key]
                 and all(k in self.states for k in ep['evidence']), 'DELTA_READER_EPISODE')
            self.episodes[key] = ep
            self.states[key] = dict(active=True, grounded_scope=None, review='unassessed', receipt=None)
            for dependency in ep['evidence']: self.children.setdefault(dependency, set()).add(key)
            self.memory_head, self.memory_sequence = record['sha256'], self.memory_sequence+1
            actual.add(key)
        need(actual == set(expected), 'DELTA_READER_MISSING_EPISODE')
        self.tasks.update(payload['tasks'])
        self.fields.update({pin+':'+k: v for k, v in payload['fields'].items()})
        self.relationships.update({digest(e): e for e in payload['relationships']})
        for key in payload['tasks']: self.add_slots(pin, key)
        binding = self.bindings[pin]; split = self.task_splits[digest(binding)]
        self.source_templates[pin] = dict(family=self.populations[pin].family,
            files=dict(population=pin, native=binding, split=digest(split)),
            groups=sorted(set(split['groups']) - set(split['excluded'])),
            ids=[], statuses={}, connections={},
            query_contexts={k: digest(dict(slots=self.slots[k])) for k in split['queries']})
        self.intakes.append(dict(population=pin, tasks=len(payload['tasks']), fields=len(payload['fields']),
                                 relationships=len(payload['relationships']), memory_head=self.memory_head))

    def source(self):
        # The historical reader calls this before delta initialization.
        if not hasattr(self, 'source_templates'): return super().source()
        sources = {}
        for pin, template in self.source_templates.items():
            value = copy.deepcopy(template); statuses, eligible = {}, []
            for key, task in sorted(self.tasks.items()):
                if task['population'] != pin: continue
                fields = {f['id'] for side in task['selections'].values() for f in side['fields']}
                eps = [task['episode']] + [self.fields[pin+':'+f]['episode'] for f in fields] + task['related_executions']
                states = {ep: self.states[ep] for ep in eps}
                statuses[key] = digest(states)
                if task['split_group'] in value['groups'] and all(
                        s['active'] and s['grounded_scope'] is not None for s in states.values()): eligible.append(key)
            value.update(statuses=statuses, ids=eligible, connections={k: digest(self.links[k]) for k in eligible})
            sources[digest(value['files']['native'])] = value
        need(len(sources) > 3, 'DELTA_READER_GROWTH_MISSING')
        return dict(family='shared', files={k: v['files'] for k, v in sources.items()},
            groups=sorted({g for v in sources.values() for g in v['groups']}),
            ids=sorted(k for v in sources.values() for k in v['ids']), families=sources)

    def publication(self, payload):
        super().publication(payload)
        if not hasattr(self, 'source_templates'): return
        root = Path(payload['kernel']['path'])
        value = read(root / 'manifest.json')
        associations = pinned(root / 'associations.json', value['associations_sha256'])
        expected = {}
        for name, graph in sorted(value['graphs'].items()):
            counts = {}
            for origin, target, score in graph['edges']:
                need(origin in value['ids'] and target in value['ids'], 'DELTA_READER_EDGE_SCOPE')
                counts[origin] = counts.get(origin, 0)+1
                entry = expected.setdefault(digest([origin, target]), dict(origin=origin, target=target, readings={}))
                entry['readings'][name] = dict(score=score, rank=counts[origin])
        for entry in expected.values():
            entry['families'] = [self.tasks[k]['family'] for k in (entry['origin'], entry['target'])]
            entry['cross_lens'] = [dict(instruments=[a, b],
                ranks=[entry['readings'][n]['rank'] for n in (a, b)], relation='co_measured_neighbor')
                for a, b in combinations(sorted(entry['readings']), 2)]
        need(associations['edges'] == expected and associations['meaning_claim'] == 'instrument_relative_neighborhood_only'
             and associations['absence_means'] == 'not_materialized_not_no_association',
             'DELTA_READER_TYPED_ASSOCIATIONS')
        cross = sum(e['families'][0] != e['families'][1] for e in expected.values())
        need(cross == associations['cross_domain_edges'] == value['cross_domain_edges'], 'DELTA_READER_CROSS_DOMAIN')
        self.association_readbacks.append(dict(kernel=payload['kernel']['pin'], edges=len(expected),
            single_instrument_edges=sum(len(e['readings']) == 1 for e in expected.values()), cross_domain_edges=cross))

    def run_delta(self, journal):
        contract, rows = chain(journal)
        need(contract == self.contract and rows[:len(self.prefix)] == self.prefix, 'DELTA_READER_PREFIX')
        previous = self.prefix[-1][2]
        for body, payload, event in rows[len(self.prefix):]:
            kind = body['kind']
            need(body['prior_head'] == previous, 'DELTA_READER_PRIOR_HEAD')
            if kind == 'capability_delta_output_root':
                need(body['code'] == self.expected_code and payload['previous'] == self.spec['root']
                     and payload['memory'] == self.memory_head
                     and payload['root'] == str(Path(self.delta_config['root']) / 'task-kernels'),
                     'DELTA_READER_OUTPUT_ROOT')
                self.spec = dict(self.spec, root=payload['root'])
            elif kind == 'task_kernel_register': self.register(payload)
            elif kind == 'capability_intake': self.intake(body, payload)
            elif kind == 'task_kernel_begin':
                need(self.intent is None and payload['source'] == self.source()
                     and payload['previous'] == self.active and payload['memory'] == self.memory_head,
                     'DELTA_READER_BEGIN')
                self.intent = dict(id=digest(body), **payload)
            else:
                need(kind == 'task_kernel_publish', 'DELTA_READER_UNKNOWN_EVENT')
                self.origin = dict(slots=self.slots, connections=self.links)
                self.publication(payload)
            previous = event
        return dict(head=previous, events=len(rows), memory_head=self.memory_head,
                    active=self.active, pending=self.intent, publications=self.publications, intakes=self.intakes,
                    association_readbacks=self.association_readbacks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'proofs', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); need(not a.output.exists(), 'DELTA_READER_EXISTS')
    started = time.perf_counter(); config = read(a.config); root = Path(config['root'])
    reader = DeltaReader(config, read(a.proofs)); result = reader.run_delta(root / 'experience.sqlite3')
    saved = read(root / 'snapshot-final.json')
    need(result['head'] == saved['head'] and result['events'] == saved['events']
         and result['memory_head'] == saved['memory']['head'] and reader.states == saved['memory']['states']
         and result['active'] == saved['task_kernel']['checkpoint'] and result['pending'] is None,
         'DELTA_READER_FINAL_STATE')
    need(len(result['intakes']) == 3 and len(result['publications']) == 2, 'DELTA_READER_BOUNDARIES')
    for stage in ('update', 'cold'):
        report = read(root / (stage+'-report.json'))
        need(report['complete'] and report['head'] == result['head'] and report['events'] == result['events']
             and report['parameters_unchanged'] and report['optimizer_steps'] == 0, 'DELTA_READER_STAGE')
    report = dict(**result, journal=sha((root / 'experience.sqlite3').read_bytes()),
        native_fields_read=len(reader.field_arrays), source_eligibility_independently_reconstructed=True,
        native_context_arrays_independently_read=True, cold_state_readback=True,
        wall_seconds=time.perf_counter()-started, number_status='measured', acceptance='author-checked',
        scientific_admission=False, learned_capability=False, generated_surface_guard_complete=False,
        metric_mapping=['T2/spec27 kernel recall', 'T5/spec27 continual consistency'])
    pin = write_committed(a.output, report)
    print(json.dumps(dict(pin=pin, events=result['events'], intakes=len(result['intakes']),
                          publications=len(result['publications']))), flush=True)


if __name__ == '__main__': main()
