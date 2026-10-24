"""Publish ordered source-span answer targets, with scoped observed endpoints."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from production.publication import sync_dir
from text.capability_answer_source import AnswerSources
from text.evaluation_multiscale import file_sha

POLICY = 'source-span-answer-sequences-v1'


def patch_parts(raw):
    """Retain exact patch serialization; a hunk is measured with its file header."""
    files = list(re.finditer(rb'^diff --git .*$', raw, re.M))
    need(files and not raw[:files[0].start()].strip(), 'ANSWER_UNSUPPORTED_PATCH_PREAMBLE')
    result = []
    for i, match in enumerate(files):
        start = 0 if i == 0 else match.start()
        end = files[i+1].start() if i+1 < len(files) else len(raw)
        hunks = list(re.finditer(rb'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\r\n]*(?:\r?\n|$)',
                                 raw[start:end], re.M))
        need(hunks, 'ANSWER_UNSUPPORTED_PATCH_NO_HUNKS')
        header_end = start+hunks[0].start()
        header = raw[start:header_end]
        need(b'\n--- ' in header and b'\n+++ ' in header, 'ANSWER_UNSUPPORTED_PATCH_HEADER')
        for j, h in enumerate(hunks):
            lo = start+h.start(); hi = start+hunks[j+1].start() if j+1 < len(hunks) else end
            counts = re.match(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', raw[lo:hi])
            old = new = 0
            for line in raw[lo+len(h.group()):hi].splitlines():
                if line.startswith(b'\\ No newline at end of file'): continue
                need(line[:1] in (b' ', b'+', b'-'), 'ANSWER_UNSUPPORTED_PATCH_BODY')
                old += line[:1] in (b' ', b'-'); new += line[:1] in (b' ', b'+')
            need((old, new) == (int(counts[2] or 1), int(counts[4] or 1)),
                 'ANSWER_UNSUPPORTED_PATCH_COUNTS')
            result.append(dict(surface_span=[start if j == 0 else lo, hi],
                               measurement_spans=[[start, header_end], [lo, hi]],
                               role='source_patch_hunk', source_step=None))
    return result


class Builder:
    def __init__(self, source):
        self.source = source; self.steps = {}; self.solutions = {}; self.inputs = {}; self.reuse = defaultdict(list)
        self.cache = {}; self.solution_parts = {}
        for pin, pop in source.populations.items():
            for unit in pop.units:
                self.reuse[unit['entity']].append(dict(population=pin, unit=unit['row']))
            for f in pop.fields.values():
                if f['role'] not in ('recorded_solution', 'recorded_step', 'recorded_conclusion'): continue
                address = (pin, f['source']['artifact_sha256'], f['source']['row'])
                if f['role'] == 'recorded_solution':
                    need(address not in self.solutions, 'ANSWER_DUPLICATE_SOLUTION')
                    self.solutions[address] = f
                else: self.steps.setdefault(address, []).append(f)
        for values in self.steps.values(): values.sort(key=lambda f:f['source']['codepoint_span'])

    def structure(self, pop, target):
        cache_key = pop.pin, target['id']
        if cache_key in self.cache: return self.cache[cache_key]
        role = target['role']; full = target; start = 0
        if role in ('recorded_solution', 'recorded_step', 'recorded_conclusion'):
            address = (pop.pin, target['source']['artifact_sha256'], target['source']['row'])
            full = self.solutions[address]; text = full['text']; raw = text.encode()
            steps = self.steps[address]
            if address not in self.solution_parts:
                parts = []; positions = [0]
                for c in text: positions.append(positions[-1]+len(c.encode()))
                for i, f in enumerate(steps):
                    lo, hi = f['source']['codepoint_span']
                    stop = steps[i+1]['source']['codepoint_span'][0] if i+1 < len(steps) else len(text)
                    need(text[lo:hi] == f['text'] and not text[hi:stop].strip(), 'ANSWER_STEP_SPAN')
                    need(lo == 0 if i == 0 else lo > steps[i-1]['source']['codepoint_span'][1],
                         'ANSWER_STEP_ORDER')
                    parts.append(dict(surface_span=[positions[lo], positions[stop]],
                                      measurement_spans=[[positions[lo], positions[hi]]],
                                      role=f['role'], source_step=f['id']))
                self.solution_parts[address] = parts
            parts = self.solution_parts[address]
            need(parts and steps[-1]['role'] == 'recorded_conclusion', 'ANSWER_SOLUTION_END')
            if role != 'recorded_solution':
                index = [f['id'] for f in steps].index(target['id']); parts = parts[index:]
            start = parts[0]['surface_span'][0]; end_scope = 'recorded_solution_end'
        else:
            raw = full['text'].encode()
            if role == 'recorded_patch':
                parts = patch_parts(raw); end_scope = 'recorded_patch_end'
            else:
                need(role in ('attested_speaker_act', 'source_answer'), 'ANSWER_UNSUPPORTED_TARGET_ROLE')
                parts = [dict(surface_span=[0, len(raw)], measurement_spans=[[0, len(raw)]],
                              role=role, source_step=target['id'])]
                end_scope = 'attested_act_end' if role == 'attested_speaker_act' else 'source_answer_field_end'
        result = full, raw, start, parts, end_scope
        self.cache[cache_key] = result
        return result

    def task(self, key):
        pop, task, split = self.source.selected(key)
        base = dict(task=key, population=pop.pin, family=pop.family, split=split,
                    split_group=task['split_group'], operation=task['kind'],
                    original_targets=task['targets'], context=task['context'], consequences=task['consequences'],
                    source_task=digest(task), scientific_admission=False, serving=False,
                    original_exposure=task['partition'])
        if len(task['targets']) != 1:
            return dict(base, status='gap', reason='ANSWER_PARTIAL_ORDER_REQUIRES_BRANCHES')
        try: full, raw, start, parts, end_scope = self.structure(pop, pop.fields[task['targets'][0]])
        except ValueError as exc:
            if 'ANSWER_UNSUPPORTED_' not in str(exc): raise
            return dict(base, status='gap', reason=str(exc))
        atoms = []; cursor = start
        for index, part in enumerate(parts):
            lo, hi = part['surface_span']; need(lo == cursor and lo < hi, 'ANSWER_CONSERVATION')
            measured = b''.join(raw[a:b] for a,b in part['measurement_spans']); identity = sha(measured)
            need(measured.strip(), 'ANSWER_EMPTY_ATOM')
            value = dict(id=identity, text=measured.decode(), reuse_candidates=self.reuse.get(identity, []),
                         native_readings_bound=False)
            need(identity not in self.inputs or self.inputs[identity] == value, 'ANSWER_INPUT_COLLISION')
            self.inputs[identity] = value
            atoms.append(dict(index=index, **part, measurement=identity, surface_sha256=sha(raw[lo:hi])))
            cursor = hi
        need(cursor == len(raw), 'ANSWER_END_CONSERVATION')
        answer = dict(source_field=full['id'], source_text_sha256=sha(raw), reference_span=[start,len(raw)],
                      reference_sha256=sha(raw[start:]), atoms=atoms,
                      end=dict(kind='end', scope=end_scope, source_field=full['id'], byte_offset=len(raw),
                               observed=True, establishes_correctness=False),
                      clock=full['clock'], provenance=full['source'], original_checks=task['checks'],
                      new_measurements_required=True)
        return dict(base, status='ready', answer=answer)


def write_rows(path, rows):
    with path.open('xb') as f:
        for row in rows: f.write(canonical(row)+b'\n')
        f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path); p.add_argument('--pin', required=True)
    p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    a.output.mkdir(exist_ok=False); started = tick = time.perf_counter(); phases = {}
    try:
        source = AnswerSources(dict(path=str(a.config), pin=a.pin))
        phases['authenticate'] = time.perf_counter()-tick; tick = time.perf_counter()
        write_committed(a.output/'before.json', dict(source=source.ref, stamps=source.stamps))
        builder = Builder(source); records = [builder.task(key) for key in sorted(source.tasks)]
        phases['derive'] = time.perf_counter()-tick; tick = time.perf_counter()
        ready = [r for r in records if r['status'] == 'ready']
        write_rows(a.output/'answers.jsonl', records)
        write_rows(a.output/'measurement-inputs.jsonl', [builder.inputs[k] for k in sorted(builder.inputs)])
        source.fresh()
        phases['publish'] = time.perf_counter()-tick
        counts = dict(tasks=len(records), ready=len(ready), gaps=len(records)-len(ready),
            by_family=dict(Counter(r['family'] for r in records)), by_split=dict(Counter(r['split'] for r in records)),
            answer_atoms=sum(len(r['answer']['atoms']) for r in ready),
            multiple_atom_answers=sum(len(r['answer']['atoms']) > 1 for r in ready),
            endpoint_scopes=dict(Counter(r['answer']['end']['scope'] for r in ready)),
            reference_bytes=sum(r['answer']['reference_span'][1]-r['answer']['reference_span'][0] for r in ready),
            measurement_inputs=len(builder.inputs), measurement_bytes=sum(len(r['text'].encode()) for r in builder.inputs.values()),
            reuse_candidate_inputs=sum(bool(r['reuse_candidates']) for r in builder.inputs.values()))
        files = {n:dict(sha256=file_sha(a.output/n), bytes=(a.output/n).stat().st_size)
                 for n in ('before.json', 'answers.jsonl', 'measurement-inputs.jsonl')}
        code = {n:file_sha(Path(__file__).resolve().parents[1]/n) for n in
                ('text/capability_answer_source.py', 'text/capability_answers.py',
                 'model/capability_answer_targets.py', 'audit/capability_answers_readback.py')}
        report = dict(schema=POLICY, complete=True, source=source.ref, counts=counts, files=files, code=code,
                      wall_seconds=time.perf_counter()-started, phases=phases,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      training_ready=False, native_readings_bound=False, scientific_admission=False,
                      serving=False, correctness_established=False, author_checked=True)
        pin = write_committed(a.output/'report.json', report)
        with (a.output/'COMMITTED').open('x') as f: f.write(pin); f.flush(); os.fsync(f.fileno())
        sync_dir(a.output); print(json.dumps(dict(pin=pin, **counts)))
    except BaseException as exc:
        write_committed(a.output/'failure.json', dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__': main()
