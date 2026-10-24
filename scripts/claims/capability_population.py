"""Strict consumer of connected capability tasks with separate input/target access."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from claims.episodic_memory import canonical, digest
from text.evaluation_multiscale import file_sha


def need(ok, code):
    if not ok:
        raise ValueError('MNLM_CAPABILITY_POPULATION_' + code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CapabilityPopulation:
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        raw = (self.root / 'report.json').read_bytes()
        need(sha(raw) == pin == (self.root / 'COMMITTED').read_text(), 'MANIFEST_PIN')
        self.report = json.loads(raw)
        r = self.report
        need(raw == canonical(r) and r['schema'] == 'connected-capability-tasks-v1'
             and r['complete'] is True and r['training_ready'] is False
             and r['scientific_admission'] is False and r['serving'] is False, 'MANIFEST_SCOPE')
        self.family = r['family']
        need(self.family in ('coding', 'conversation', 'reasoning'), 'FAMILY')
        expected = {self.family + '/chunks/atoms.parquet', self.family + '/summary.json',
                    'fields.jsonl', 'tasks.jsonl', 'edges.jsonl', 'gaps.jsonl', 'source.json'}
        need(set(r['files']) == expected, 'FILE_ROSTER')
        self.files = {}
        for name, ref in r['files'].items():
            data = (self.root / name).read_bytes()
            need(dict(bytes=len(data), sha256=sha(data)) == ref, 'FILE_PIN')
            self.files[name] = data
        need(json.loads(self.files['source.json']) == r['source'], 'SOURCE_MANIFEST')
        self.units = pq.read_table(self.root / self.family / 'chunks/atoms.parquet').to_pylist()
        self.fields = self._rows('fields')
        self.tasks = self._rows('tasks')
        self.edges = [json.loads(line) for line in self.files['edges.jsonl'].splitlines()]
        need(len(self.edges) == len({digest(e) for e in self.edges}), 'DUPLICATE_EDGE')
        self.links = {(e['origin'], e['target'], e['kind']) for e in self.edges}
        self.steps = {}
        if self.family == 'reasoning':
            for f in self.fields.values():
                if f['role'] in ('recorded_step', 'recorded_conclusion'):
                    self.steps.setdefault(f['source']['row'], []).append(f)
            for steps in self.steps.values():
                steps.sort(key=lambda f: f['source']['codepoint_span'])
        for i, u in enumerate(self.units):
            need(u['row'] == i and u['chunk'] == 0 and u['entity'] == sha(u['text'].encode())
                 and u['start'] == 0 and u['end'] == len(u['text']), 'NATIVE_UNIT')
        for f in self.fields.values(): self.field(f)
        for e in self.edges:
            need(e['origin'] in self.fields and e['target'] in self.fields
                 and e['origin'] != e['target'] and e['evidence'], 'EDGE')
        for task in self.tasks.values(): self.task(task)
        need(r['counts']['tasks'] == len(self.tasks) and r['counts']['fields'] == len(self.fields)
             and r['counts']['edges'] == len(self.edges) and r['counts']['unique_fragments'] == len(self.units),
             'COUNTS')
        self.seal = digest(dict(fields=self.fields, tasks=self.tasks, edges=self.edges))

    def _rows(self, name):
        values = [json.loads(line) for line in self.files[name + '.jsonl'].splitlines()]
        indexed = {r['id']: r for r in values}
        need(len(indexed) == len(values) and values, 'DUPLICATE_OR_EMPTY_' + name.upper())
        return indexed

    def field(self, f):
        need(set(f) == {'id', 'family', 'role', 'text', 'text_sha256', 'source', 'clock',
                        'parts', 'fragments_are_independent_atoms'}, 'FIELD_SCHEMA')
        body = {k: v for k, v in f.items() if k not in ('id', 'parts', 'fragments_are_independent_atoms')}
        need(digest(body) == f['id'] and f['family'] == self.family and f['text'].strip()
             and sha(f['text'].encode()) == f['text_sha256'] and f['source']
             and f['fragments_are_independent_atoms'] is False, 'FIELD_BINDING')
        pieces, end = [], 0
        for p in f['parts']:
            need(type(p['unit']) is int and 0 <= p['unit'] < len(self.units), 'UNIT_INDEX')
            raw = self.units[p['unit']]['text'].encode()
            need(p['start'] == end and p['end'] == end + len(raw) and p['sha256'] == sha(raw), 'PART')
            pieces.append(raw)
            end = p['end']
        need(b''.join(pieces) == f['text'].encode(), 'FIELD_CONSERVATION')

    def edge(self, a, b, kind):
        need((a, b, kind) in self.links, 'MISSING_RELATIONSHIP')

    def task(self, t):
        need(set(t) == {'id', 'family', 'kind', 'context', 'targets', 'consequences', 'split_group',
                        'partition', 'provenance', 'checks', 'target_origin', 'scientific_admission',
                        'serving', 'sealed_evaluation'}, 'TASK_SCHEMA')
        need(t['id'] == digest({k: v for k, v in t.items() if k != 'id'}) and t['family'] == self.family,
             'TASK_BINDING')
        c, y, a = t['context'], t['targets'], t['consequences']
        need(c and y, 'MISSING_TARGET')
        need(all(len(xs) == len(set(xs)) and all(x in self.fields for x in xs) for xs in (c, y, a)),
             'TASK_REFERENCES')
        need(not (set(c) & (set(y) | set(a))) and not set(y) & set(a), 'FUTURE_LEAKAGE')
        need(not t['scientific_admission'] and not t['serving'] and not t['sealed_evaluation']
             and t['partition'] == 'development_exposed', 'TASK_SCOPE')
        if self.family == 'reasoning': self.reasoning(t)
        elif self.family == 'conversation': self.conversation(t)
        else: self.coding(t)

    def reasoning(self, t):
        c, y = [self.fields[i] for i in t['context']], [self.fields[i] for i in t['targets']]
        need(len(y) == 1 and not t['consequences'] and c[0]['role'] == 'problem', 'MATH_ROLES')
        row = c[0]['source']['row']
        need(all(f['source']['row'] == row for f in c + y), 'MATH_SOURCE_JOIN')
        target = y[0]
        if t['kind'] == 'solve_recorded_problem':
            need(len(c) == 1 and target['role'] == 'recorded_solution', 'MATH_SOLUTION_ROLE')
            self.edge(c[0]['id'], target['id'], 'provider_solution')
        else:
            need(t['kind'] == 'continue_recorded_steps'
                 and target['role'] in ('recorded_step', 'recorded_conclusion'), 'MATH_STEP_ROLE')
            start = target['source']['codepoint_span'][0]
            earlier = [f for f in self.steps[row] if f['source']['codepoint_span'][1] <= start]
            need([f['id'] for f in c[1:]] == [f['id'] for f in earlier], 'MATH_PREFIX')
            self.edge(c[-1]['id'], target['id'], 'source_sequence')

    def conversation(self, t):
        p = t['provenance']
        c, y = [self.fields[i] for i in t['context']], [self.fields[i] for i in t['targets']]
        need(t['kind'] == 'respond_to_attested_act' and not t['consequences'], 'DIALOGUE_KIND')
        need(all(f['role'] == 'attested_speaker_act' for f in c + y), 'DIALOGUE_ROLE')
        cutoff = Fraction(p['cutoff'])
        need(cutoff == min(Fraction(f['clock']['start']) for f in y)
             and all(Fraction(f['clock']['end']) <= cutoff for f in c)
             and len({f['clock']['id'] for f in c + y}) == 1, 'DIALOGUE_FUTURE')
        need(set(p['target_acts']) == {f['source']['element'] for f in y}, 'DIALOGUE_TARGET')
        lookup = {f['source']['element']: f['id'] for f in c}
        need(set(p['source_acts']) <= lookup.keys(), 'DIALOGUE_MISSING_REQUEST')
        for a in p['source_acts']:
            for b in t['targets']: self.edge(lookup[a], b, 'annotated_response')

    def coding(self, t):
        c, y, a = ([self.fields[i] for i in t[k]] for k in ('context', 'targets', 'consequences'))
        need(len(y) == 1 and y[0]['role'] == 'recorded_patch', 'PATCH_TARGET_ROLE')
        p = t['provenance']
        if t['kind'] == 'reproduce_recorded_patch':
            need({f['role'] for f in c} == {'task_report', 'original_source'}, 'PATCH_INPUT_ROLES')
            need(all(f['source']['condition'] == p['condition'] for f in c + y + a), 'PATCH_CONDITION')
            need(all(f['role'] in ('candidate_source', 'stdout', 'stderr') for f in a), 'PATCH_AFTER_ROLES')
            for f in c: self.edge(f['id'], y[0]['id'], 'recorded_task_to_patch')
            for f in a: self.edge(y[0]['id'], f['id'], 'recorded_patch_consequence')
        else:
            need(t['kind'] == 'contrast_failure_with_supplied_repair', 'PATCH_KIND')
            need(p['failure']['observed_outcome'] == 0 and p['repair']['observed_outcome'] == 1
                 and p['failure']['task'] == p['repair']['task'] and not p['autonomous_correction'],
                 'REPAIR_CONTRAST')
            need(all(f['source']['condition'] == p['failure']['condition'] for f in c)
                 and all(f['source']['condition'] == p['repair']['condition'] for f in y + a),
                 'REPAIR_ROLE_JOIN')
            failed = [f for f in c if f['role'] == 'recorded_patch']
            need(len(failed) == 1, 'REPAIR_MISSING_FAILURE')
            self.edge(failed[0]['id'], y[0]['id'], 'supplied_passing_repair_contrast')

    def selection(self, key, *, side):
        """Only selected fields leave this boundary; target annotations never accompany inputs."""
        need(side in ('context', 'targets', 'consequences') and key in self.tasks, 'SELECTION')
        task = self.tasks[key]
        self.task(task)
        return dict(population=self.pin, task=key, side=side,
                    fields=[dict(id=k, role=self.fields[k]['role'], parts=self.fields[k]['parts'])
                            for k in task[side]], scientific_admission=False, training_ready=False)

    def fresh(self):
        need(digest(dict(fields=self.fields, tasks=self.tasks, edges=self.edges)) == self.seal, 'MUTATED_VIEW')
        need(file_sha(self.root / 'report.json') == self.pin, 'STALE_MANIFEST')
