"""Independent SQL payload/source joins; no capability memory producer is called."""
import copy
import json
from pathlib import Path

from audit.repository_memory_readback import chain
from claims.capability_population import CapabilityPopulation, need, sha
from claims.episodic_memory import canonical, digest


def read(path): return json.loads(Path(path).read_bytes())


class JournalReader:
    def __init__(self, config, proofs):
        self.config = config
        self.populations, self.proofs = {}, {}
        for spec in config['populations']:
            p = CapabilityPopulation(spec['population'], spec['population_pin'])
            self.populations[p.pin] = p
            proof = proofs[p.family]
            root = Path(proof['path'])
            need(sha((root / 'report.json').read_bytes()) == proof['pin'], 'READER_NATIVE_PROOF')
            report = read(root / 'report.json')
            for name in ('field-receipts.json', 'binding.json'):
                raw = (root / name).read_bytes()
                need(dict(bytes=len(raw), sha256=sha(raw)) == report['files'][name], 'READER_PROOF_FILE')
            binding = read(root / 'binding.json')
            need(report['population'] == p.pin and binding['population'] == p.pin
                 and binding['native'] == spec['native_pin'] and binding['plan'] == spec['plan_pin'],
                 'READER_PROOF_BINDING')
            self.proofs[p.pin] = (read(root / 'field-receipts.json'), binding)
        parent = Path(config['source'])
        need(sha((parent / 'experience.sqlite3').read_bytes()) == config['source_pin'], 'READER_PARENT')
        self.contract, self.prefix = chain(parent / 'experience.sqlite3')
        self.old = read(parent / 'snapshot-final.json')
        self.bridges = {}
        for name in ('repository_cohort_intakes', 'repository_depth_intakes', 'repository_demonstration_intakes'):
            for intake in self.old[name].values(): self.bridges.update(intake['episodes'])

    def intake(self, body, payload):
        spec = body['spec']
        need(spec in self.config['populations'] and payload['source_spec'] == spec, 'READER_SPEC')
        p = self.populations[spec['population_pin']]
        original_receipts, binding = self.proofs[p.pin]
        groups = body['groups']
        need(groups == sorted(set(groups)) == payload['groups'], 'READER_GROUPS')
        selected = {k: t for k, t in p.tasks.items() if t['split_group'] in groups}
        fields = {f for t in selected.values() for s in ('context', 'targets', 'consequences') for f in t[s]}
        need(set(payload['tasks']) == set(selected) and set(payload['fields']) == fields, 'READER_ROSTER')
        need(payload['native_binding'] == binding, 'READER_BINDING')
        expected = {}
        for field in sorted(fields):
            f = p.fields[field]
            item = dict(claim_kind='source_statement', content=f['text'],
                source=dict(capability_population=p.pin, field=field, source=f['source']),
                evidence=[], event_time=f['clock'], role=f['role'], family=p.family)
            pin = digest(item)
            expected[pin] = dict(id=pin, **item)
            receipt = payload['fields'][field]
            need(receipt['episode'] == pin and receipt['binding'] == digest(binding), 'READER_FIELD_EPISODE')
            need(set(receipt['slots']) == set(original_receipts[field]), 'READER_SLOT_ROSTER')
            for name, slot in receipt['slots'].items():
                actual = original_receipts[field][name]
                need(slot == dict(instrument=binding['instruments'][name],
                    shape=[actual['windows'], actual['dimensions']], dtype='float32-le',
                    sha256=actual['sha256'], description=actual['description']), 'READER_NATIVE_BYTES')
        for key, t in selected.items():
            expected_selections = {side: dict(population=p.pin, task=key, side=side,
                fields=[dict(id=f, role=p.fields[f]['role'], parts=p.fields[f]['parts']) for f in t[side]],
                scientific_admission=False, training_ready=False) for side in ('context', 'targets', 'consequences')}
            actual = payload['tasks'][key]
            need(actual['selections'] == expected_selections, 'READER_INPUT_TARGET_MASK')
            related = []
            if p.family == 'coding':
                provenance = t['provenance']
                conditions = ([provenance['condition']] if t['kind'] == 'reproduce_recorded_patch'
                              else [provenance[s]['condition'] for s in ('failure', 'repair')])
                related = [self.bridges[c] for c in conditions]
            parents = [payload['fields'][f]['episode']
                       for side in ('context', 'targets', 'consequences') for f in t[side]] + related
            item = dict(claim_kind='source_statement', content=canonical(t).decode(),
                source=dict(capability_population=p.pin, task=key), evidence=parents, event_time=None,
                family=p.family, role='attested_task_connections', clock_claim='source_fields_only')
            pin = digest(item)
            expected[pin] = dict(id=pin, **item)
            need(actual == dict(episode=pin, family=p.family, kind=t['kind'], split_group=t['split_group'],
                population=p.pin, binding=digest(binding), selections=expected_selections, related_executions=related,
                partition='development_exposed', scientific_admission=False, training_ready=False, serving=False),
                'READER_TASK_EPISODE')
        links = [e for e in p.edges if e['origin'] in fields and e['target'] in fields]
        need(payload['relationships'] == links, 'READER_LOST_RELATIONSHIP')
        return expected

    def read(self, journal):
        contract, rows = chain(Path(journal))
        need(contract == self.contract and rows[:len(self.prefix)] == self.prefix, 'READER_PREFIX')
        states = copy.deepcopy(self.old['memory']['states'])
        expected, episodes, children, tasks, fields, relations = {}, {}, {}, {}, {}, {}
        head, sequence = self.old['memory']['head'], None
        intakes, invalidations, reviews = [], [], []
        for body, payload, event in rows[len(self.prefix):]:
            kind = body['kind']
            if kind == 'capability_intake':
                expected.update(self.intake(body, payload))
                need(not set(payload['tasks']) & tasks.keys(), 'READER_DUPLICATE_INTAKE')
                tasks.update(payload['tasks'])
                pop = body['spec']['population_pin']
                fields.update({pop + ':' + k: v for k, v in payload['fields'].items()})
                relations.update({digest(e): e for e in payload['relationships']})
                records = payload['memory_records']
                intakes.append(dict(event=event, groups=body['groups'], tasks=len(payload['tasks']),
                    fields=len(payload['fields']), family=self.populations[pop].family))
            elif kind == 'invalidate':
                records = [body['record']]
                invalidations.append(body['record']['event'])
            else:
                need(kind == 'capability_source_review', 'READER_TAIL_KIND')
                records = payload['records']
                reviews.append(body['episode'])
            for r in records:
                need(r['previous'] == head and r['sha256'] == digest({k: v for k, v in r.items() if k != 'sha256'}),
                     'READER_MEMORY_CHAIN')
                if sequence is not None: need(r['sequence'] == sequence, 'READER_MEMORY_ORDER')
                head, sequence = r['sha256'], r['sequence'] + 1
                e = r['event']
                op = e['op']
                if op == 'intake':
                    ep = e['episode']
                    key = ep['id']
                    need(key not in states and ep == expected[key], 'READER_SOURCE_EPISODE')
                    need(all(k in states for k in ep['evidence']), 'READER_DEPENDENCY_MISSING')
                    episodes[key] = ep
                    states[key] = dict(active=True, grounded_scope=None, review='unassessed', receipt=None)
                    for parent in ep['evidence']: children.setdefault(parent, set()).add(key)
                else:
                    key = e['episode']
                    state = states[key]
                    if op == 'check_source':
                        receipt = e['result']
                        need(receipt['id'] == digest({k: v for k, v in receipt.items() if k != 'id'})
                             and receipt['episode'] == key and receipt['source'] == episodes[key]['source']
                             and receipt['scope'] == 'pinned_artifact_contents_only'
                             and receipt['world_validity'] == 'unassessed', 'READER_ATTESTATION')
                        state.update(receipt=receipt, review='source_checked')
                    elif op == 'promote':
                        need(e['receipt'] == state['receipt']['id'], 'READER_PROMOTE')
                        state.update(grounded_scope=state['receipt']['scope'], review='source_checked')
                    else:
                        need(op == 'invalidate', 'READER_MEMORY_OP')
                        affected, queue = set(), [key]
                        while queue:
                            item = queue.pop()
                            if item in affected: continue
                            affected.add(item)
                            queue.extend(children.get(item, ()))
                        need(sorted(affected) == e['affected'], 'READER_INVALIDATION_CLOSURE')
                        for item in affected:
                            states[item].update(grounded_scope=None, review='needs_review', receipt=None)
        need(set(episodes) == set(expected), 'READER_MISSING_EPISODE')
        return dict(head=rows[-1][2], events=len(rows), memory_head=head, states=states,
            tasks=tasks, fields=fields, relationships=relations,
            intakes=intakes, invalidations=invalidations, reviews=reviews)
