"""Read scoped answer sequences; inference requests never include private endpoints."""
from copy import deepcopy
import json
from pathlib import Path
import re

from claims.capability_population import need, sha
from claims.episodic_memory import digest
from text.capability_answer_source import AnswerSources, stamp


class AnswerTargets:
    def __init__(self, root, pin):
        self.root = Path(root)
        raw = (self.root/'report.json').read_bytes()
        need(sha(raw) == pin == (self.root/'COMMITTED').read_text(), 'ANSWER_PUBLICATION_PIN')
        self.report = r = json.loads(raw)
        need(r['schema'] == 'source-span-answer-sequences-v1' and r['complete']
             and not any(r[n] for n in ('training_ready', 'native_readings_bound', 'scientific_admission',
                                       'serving', 'correctness_established')), 'ANSWER_PUBLICATION_SCOPE')
        need(set(r['files']) == {'before.json', 'answers.jsonl', 'measurement-inputs.jsonl'}, 'ANSWER_FILES')
        rows = {}
        for name, ref in r['files'].items():
            data = (self.root/name).read_bytes()
            need(dict(sha256=sha(data), bytes=len(data)) == ref, 'ANSWER_PAYLOAD_PIN')
            if name.endswith('.jsonl'): rows[name] = [json.loads(line) for line in data.splitlines()]
        self.source = AnswerSources(r['source'])
        self.rows = {row['task']:row for row in rows['answers.jsonl']}
        self.inputs = {row['id']:row for row in rows['measurement-inputs.jsonl']}
        need(len(self.rows) == len(rows['answers.jsonl']) == r['counts']['tasks']
             and set(self.rows) == set(self.source.tasks), 'ANSWER_TASK_ROSTER')
        need(len(self.inputs) == len(rows['measurement-inputs.jsonl']) == r['counts']['measurement_inputs'],
             'ANSWER_INPUT_ROSTER')
        for identity, row in self.inputs.items():
            need(sha(row['text'].encode()) == identity and not row['native_readings_bound'], 'ANSWER_INPUT_PIN')
            for candidate in row['reuse_candidates']:
                unit = self.source.populations[candidate['population']].units[candidate['unit']]
                need(unit['entity'] == identity and unit['text'] == row['text'], 'ANSWER_REUSE_CANDIDATE')
        self.solutions = {}; self.steps = {}
        for pin, pop in self.source.populations.items():
            for f in pop.fields.values():
                if f['role'] == 'recorded_solution':
                    self.solutions[(pin,f['source']['artifact_sha256'],f['source']['row'])] = f
                elif f['role'] in ('recorded_step', 'recorded_conclusion'):
                    address = pin,f['source']['artifact_sha256'],f['source']['row']
                    self.steps.setdefault(address, []).append(f)
        for values in self.steps.values(): values.sort(key=lambda f:f['source']['codepoint_span'])
        for row in self.rows.values(): self.validate(row)
        need({a['measurement'] for r in self.rows.values() if r['status'] == 'ready'
              for a in r['answer']['atoms']} == set(self.inputs), 'ANSWER_UNUSED_INPUT')
        self.seals = {key:digest(row) for key,row in self.rows.items()}
        self.stamps = {str(self.root/n):stamp(self.root/n) for n in (*r['files'], 'report.json', 'COMMITTED')}

    def validate(self, row):
        pop, task, split = self.source.selected(row['task'])
        need(all(row[n] == value for n,value in dict(population=pop.pin, family=pop.family, split=split,
             split_group=task['split_group'], operation=task['kind'], original_targets=task['targets'],
             context=task['context'], consequences=task['consequences'], source_task=digest(task),
             scientific_admission=False, serving=False, original_exposure=task['partition']).items()),
             'ANSWER_TASK_SELECTION')
        need(row['status'] in ('ready', 'gap'), 'ANSWER_STATUS')
        if row['status'] == 'gap':
            need(row.get('reason') and 'answer' not in row, 'ANSWER_GAP_SCOPE'); return
        need(len(task['targets']) == 1, 'ANSWER_UNORDERED_TARGETS')
        target = pop.fields[task['targets'][0]]; answer = row['answer']; role = target['role']; offset = 0
        if role in ('recorded_solution', 'recorded_step', 'recorded_conclusion'):
            address = pop.pin,target['source']['artifact_sha256'],target['source']['row']
            full = self.solutions[address]; selected = self.steps[address]
            if role != 'recorded_solution':
                selected = selected[[f['id'] for f in selected].index(target['id']):]
            offset = len(full['text'][:selected[0]['source']['codepoint_span'][0]].encode())
            scope = 'recorded_solution_end'
            need(selected[-1]['role'] == 'recorded_conclusion', 'ANSWER_FINAL_CONCLUSION')
        else:
            full = target; selected = None
            scope = {'recorded_patch':'recorded_patch_end', 'attested_speaker_act':'attested_act_end',
                     'source_answer':'source_answer_field_end'}[role]
        raw = full['text'].encode(); atoms = answer['atoms']
        need(answer['source_field'] == full['id'] and answer['source_text_sha256'] == sha(raw)
             and answer['reference_span'] == [offset,len(raw)] and answer['reference_sha256'] == sha(raw[offset:])
             and answer['clock'] == full['clock'] and answer['provenance'] == full['source']
             and answer['original_checks'] == task['checks'] and answer['new_measurements_required'],
             'ANSWER_REFERENCE')
        need(answer['end'] == dict(kind='end', scope=scope, source_field=full['id'], byte_offset=len(raw),
                                  observed=True, establishes_correctness=False), 'ANSWER_ENDPOINT')
        need(atoms and (selected is None or len(atoms) == len(selected)), 'ANSWER_ATOM_COUNT')
        for i, atom in enumerate(atoms):
            lo, hi = atom['surface_span']
            need(atom['index'] == i and lo == offset and lo < hi <= len(raw)
                 and sha(raw[lo:hi]) == atom['surface_sha256'], 'ANSWER_ATOM_ORDER')
            spans = atom['measurement_spans']
            need(spans and all(type(a) is int and type(b) is int and 0 <= a < b <= len(raw) for a,b in spans),
                 'ANSWER_MEASUREMENT_SPAN')
            measured = b''.join(raw[a:b] for a,b in spans)
            need(sha(measured) == atom['measurement'] and
                 self.inputs[atom['measurement']]['text'].encode() == measured, 'ANSWER_MEASUREMENT_BYTES')
            if selected is not None:
                step = selected[i]; a,b = step['source']['codepoint_span']
                begin = len(full['text'][:a].encode()); stop = len(full['text'][:b].encode())
                next_begin = (len(full['text'][:selected[i+1]['source']['codepoint_span'][0]].encode())
                              if i+1 < len(selected) else len(raw))
                need(atom['source_step'] == step['id'] and atom['role'] == step['role'] and lo == begin
                     and hi == next_begin and spans == [[begin,stop]] and measured == step['text'].encode(),
                     'ANSWER_RECORDED_STEP')
            elif role != 'recorded_patch':
                need(len(atoms) == 1 and atom['source_step'] == full['id'] and atom['role'] == role
                     and spans == [[0,len(raw)]], 'ANSWER_WHOLE_ACT')
            else:
                need(atom['role'] == 'source_patch_hunk' and atom['source_step'] is None and len(spans) == 2,
                     'ANSWER_PATCH_ATOM')
                (hs,he),(bs,be) = spans
                need(bs < be == hi and lo in (hs,bs) and he <= bs and raw[bs:be].startswith(b'@@ ')
                     and raw[hs:he].lstrip().startswith(b'diff --git ') and b'\n@@ ' not in raw[hs:he],
                     'ANSWER_PATCH_HEADER')
                need(not re.search(rb'^(?:@@ |diff --git )', raw[bs:be].split(b'\n',1)[1], re.M),
                     'ANSWER_PATCH_HUNK_BOUNDARY')
            offset = hi
        need(offset == len(raw), 'ANSWER_INCOMPLETE')

    def fresh(self, key):
        self.source.fresh()
        need(all(stamp(p) == value for p,value in self.stamps.items()), 'ANSWER_PUBLICATION_CHANGED')
        need(key in self.rows and digest(self.rows[key]) == self.seals[key], 'ANSWER_MUTATED_VIEW')

    def request(self, key):
        self.fresh(key); pop, task, _ = self.source.selected(key)
        # This is a source descriptor, not an encoder feature vector. No target, length or endpoint.
        return dict(operation=task['kind'], context=[deepcopy({n:pop.fields[f][n]
                    for n in ('id','text','role','clock')}) for f in task['context']])

    def target(self, key, *, fitting):
        self.fresh(key); need(type(fitting) is bool, 'ANSWER_FIT_MODE')
        need(key in (self.source.train if fitting else self.source.query), 'ANSWER_FIT_SPLIT')
        row = self.rows[key]; need(row['status'] == 'ready', 'ANSWER_TASK_GAP')
        return deepcopy(row['answer'])

    def native_target(self, key, *, fitting):
        self.target(key, fitting=fitting)
        raise ValueError('MNLM_CAPABILITY_ANSWER_MISSING_PINNED_NATIVE_INPUTS')
