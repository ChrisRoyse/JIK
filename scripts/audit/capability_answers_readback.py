"""Cold source-span, exact reconstruction and private-target boundary readback."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.episodic_memory import canonical
from claims.process_conditions import write_committed
from model.capability_answer_targets import AnswerTargets
from text.evaluation_multiscale import file_sha


def refused(name, fn):
    try: fn()
    except ValueError as e:
        need(name in str(e), 'ANSWER_CONTROL_WRONG_REFUSAL:'+str(e))
        return dict(expected=name, observed=str(e), refused=True)
    raise ValueError('ANSWER_CONTROL_ACCEPTED:'+name)


def patch_hunks(raw):
    """Independent line walk of actual hunk records and declared line counts."""
    lines = raw.splitlines(keepends=True); offset = 0; records = []; header_start = 0; header_end = None
    current = None; old = new = 0
    def finish(stop):
        if current is not None:
            need((old,new) == current['counts'], 'READBACK_PATCH_COUNTS')
            records.append(dict(header=[header_start,header_end], body=[current['start'],stop]))
    for line in lines:
        if line.startswith(b'diff --git '):
            finish(offset); current = None; header_start = offset if records else 0; header_end = None
        elif line.startswith(b'@@ '):
            finish(offset)
            match = re.fullmatch(rb'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\r\n]*(?:\r?\n)?',line)
            need(match is not None, 'READBACK_PATCH_HEADER')
            current = dict(start=offset, counts=(int(match[2] or 1),int(match[4] or 1)))
            if header_end is None: header_end = offset
            old = new = 0
        elif current is not None:
            if not line.startswith(b'\\ No newline at end of file'):
                need(line[:1] in (b' ',b'+',b'-'), 'READBACK_PATCH_BODY')
                old += line[:1] in (b' ',b'-'); new += line[:1] in (b' ',b'+')
        offset += len(line)
    finish(offset); need(records and offset == len(raw), 'READBACK_PATCH_END')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'): p.add_argument('--'+n, required=True, type=Path)
    p.add_argument('--pin', required=True); a = p.parse_args(); a.output.mkdir(exist_ok=False)
    started = tick = time.perf_counter(); phases = {}
    try:
        view = AnswerTargets(a.source,a.pin); phases['authenticate'] = time.perf_counter()-tick
        tick = time.perf_counter(); rows = []; controls = []; patch_cache = {}
        for key in sorted(view.rows):
            row = view.rows[key]; pop,task,split = view.source.selected(key)
            request = view.request(key)
            need(set(request) == {'operation','context'} and request['operation'] == task['kind'], 'READBACK_REQUEST')
            need([f['id'] for f in request['context']] == task['context'], 'READBACK_PRIVATE_TARGET')
            if row['status'] == 'gap':
                rows.append(dict(task=key,status='gap',reason=row['reason'])); continue
            answer = view.target(key,fitting=split=='train'); field = pop.fields[answer['source_field']]
            raw = field['text'].encode(); surface = b''.join(raw[x['surface_span'][0]:x['surface_span'][1]]
                                                           for x in answer['atoms'])
            begin,end = answer['reference_span']
            need(surface == raw[begin:end] and sha(surface) == answer['reference_sha256'], 'READBACK_BYTE_CONSERVATION')
            if field['role'] == 'recorded_patch':
                if field['id'] not in patch_cache: patch_cache[field['id']] = patch_hunks(raw)
                hunks = patch_cache[field['id']]
                need(len(hunks) == len(answer['atoms']), 'READBACK_HUNK_COUNT')
                for atom,hunk in zip(answer['atoms'],hunks):
                    need(atom['measurement_spans'] == [hunk['header'],hunk['body']], 'READBACK_HUNK_SOURCE')
            elif field['role'] == 'recorded_solution':
                original = pop.fields[task['targets'][0]]
                need(raw[begin:].decode().splitlines()[-1].startswith('#### '), 'READBACK_SOLUTION_END')
                need(answer['atoms'][0]['source_step'] == original['id'] or original['role'] == 'recorded_solution',
                     'READBACK_SUFFIX_START')
                need(answer['atoms'][-1]['role'] == 'recorded_conclusion', 'READBACK_CONCLUSION')
            rows.append(dict(task=key,status='ready',family=pop.family,split=split,atoms=len(answer['atoms']),
                             bytes=len(surface),sha256=sha(surface),end=answer['end']))
        phases['source_reconstruction_and_requests'] = time.perf_counter()-tick; tick = time.perf_counter()
        multi = next(r for r in view.rows.values() if r['status']=='ready' and len(r['answer']['atoms'])>1)
        for label,modify in (
            ('ANSWER_ENDPOINT', lambda r:r['answer']['end'].update(byte_offset=0)),
            ('ANSWER_ATOM_ORDER', lambda r:r['answer']['atoms'].reverse()),
            ('ANSWER_TASK_SELECTION', lambda r:r['context'].append(r['original_targets'][0])),
            ('ANSWER_REFERENCE', lambda r:r['answer'].update(source_text_sha256='0'*64))):
            bad = deepcopy(multi); modify(bad); controls.append(refused(label,lambda:view.validate(bad)))
        bad = deepcopy(multi); bad['answer']['atoms'].pop()
        controls.append(refused('ANSWER_',lambda:view.validate(bad)))
        q = next(k for k in view.source.query if view.rows[k]['status']=='ready')
        controls.append(refused('ANSWER_FIT_SPLIT',lambda:view.target(q,fitting=True)))
        controls.append(refused('MISSING_PINNED_NATIVE_INPUTS',lambda:view.native_target(q,fitting=False)))
        original = view.rows[q]; view.rows[q] = deepcopy(original); view.rows[q]['answer']['end']['byte_offset'] = 0
        controls.append(refused('ANSWER_MUTATED_VIEW',lambda:view.target(q,fitting=False))); view.rows[q] = original
        owned = view.target(q,fitting=False); owned['atoms'].clear()
        need(view.target(q,fitting=False)['atoms'], 'READBACK_OWNED_TARGET')
        request = view.request(q); request['context'].clear()
        need(view.request(q)['context'], 'READBACK_OWNED_REQUEST')
        corrupted = a.output/'corrupted-publication'; shutil.copytree(a.source,corrupted)
        with (corrupted/'answers.jsonl').open('ab') as f: f.write(b' ')
        controls.append(refused('ANSWER_PAYLOAD_PIN',lambda:AnswerTargets(corrupted,a.pin)))
        controls.append(refused('ANSWER_PUBLICATION_PIN',lambda:AnswerTargets(a.source,'0'*64)))
        # Alter a copied source publication rather than any retained authority.
        selected = next(iter(view.source.populations.values()))
        copied = a.output/'corrupted-source'; shutil.copytree(selected.root,copied)
        with (copied/'fields.jsonl').open('ab') as f: f.write(b' ')
        if type(selected).__name__ == 'ContextPopulation':
            cls = type(selected)
        else: cls = type(selected)
        controls.append(refused('FILE_PIN',lambda:cls(copied,selected.pin)))
        view.source.fresh(); phases['negative_controls'] = time.perf_counter()-tick
        write_committed(a.output/'rows.json',rows); write_committed(a.output/'controls.json',controls)
        counts = Counter(r['status'] for r in rows)
        ready = [r for r in rows if r['status']=='ready']; reported = view.report['counts']
        need(len(rows) == reported['tasks'] and len(ready) == reported['ready']
             and sum(r['atoms'] for r in ready) == reported['answer_atoms']
             and sum(r['bytes'] for r in ready) == reported['reference_bytes']
             and dict(Counter(r['end']['scope'] for r in ready)) == reported['endpoint_scopes'], 'READBACK_COUNTS')
        report = dict(schema='source-span-answer-readback-v1',complete=True,source=dict(path=str(a.source),pin=a.pin),
                      counts=dict(counts),answer_atoms=sum(r['atoms'] for r in ready),
                      reference_bytes=sum(r['bytes'] for r in ready),controls=len(controls),
                      source_unchanged=True,private_request_boundary=True,native_readings_bound=False,
                      ordered_answer_quality_measured=False,scientific_admission=False,author_checked=True,
                      phases=phases,wall_seconds=time.perf_counter()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      producer_sha256=file_sha(Path(__file__)),
                      files={n:file_sha(a.output/n) for n in ('rows.json','controls.json')})
        pin = write_committed(a.output/'report.json',report); print(json.dumps(dict(pin=pin,**report)))
    except BaseException as exc:
        write_committed(a.output/'failure.json',dict(type=type(exc).__name__,error=str(exc))); raise


if __name__ == '__main__': main()
