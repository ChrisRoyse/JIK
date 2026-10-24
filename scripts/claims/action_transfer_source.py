"""Outcome-blind selection of an unexposed real action (PC-07/13/37/38)."""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import scope
from senses.trajectory_population import open_bound, publish, select_metadata

POLICY = dict(schema='action-transfer-selection-policy-v1',
    seed='mnlm-action-transfer-source-v1', turns=list(range(2, 16, 2)),
    required_import='marshmallow', allowed_imports=['marshmallow', 'datetime', 'decimal',
        'json', 'enum', 'typing', 'collections', 're', 'itertools', 'functools', 'math', 'uuid'],
    program_forms=['python_stdin_heredoc', 'python_file_heredoc_then_same_file_execution'],
    exposure='exclude_all_retained_population_tasks_and_exact_program_substrings',
    rank='sha256(canonical([seed,task,turn,program_sha256,utf8_span]))_then_source_address',
    outcome_fields=[], executed=False, scientific_admission=False)
HEADER = re.compile(r'''(?m)^(?P<command>[^\n]*?)<<[ \t]*(?P<quote>['"]?)'''
                    r'''(?P<tag>[A-Za-z_]\w*)(?P=quote)(?P<tail>[^\n]*)\n''')
PYTHON = r'python(?:3(?:\.\d+)?)?'
FILENAME = r'[A-Za-z0-9_./-]+\.py'


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n: sha((root/n).read_bytes()) for n in (
        'claims/action_transfer_source.py', 'claims/tool_temporal_source.py',
        'senses/trajectory_population.py')}


def exposure(root, pin):
    root = Path(root)
    raw = (root/'report.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(), 'TRANSFER_EXPOSURE_PIN')
    report = json.loads(raw)
    check(report['schema'] == 'tool-temporal-population-v2' and report['complete']
          and not report['errors'] and report['source_pin'] == digest(scope()), 'TRANSFER_EXPOSURE_SCOPE')
    values = {}
    for name in ('selection.json', 'messages.jsonl'):
        raw = (root/name).read_bytes()
        check(len(raw) == report['files'][name]['bytes']
              and sha(raw) == report['files'][name]['sha256'], 'TRANSFER_EXPOSURE_BYTES')
        values[name] = raw
    selected = json.loads(values['selection.json'])
    check(selected['source'] == scope() and len(selected['selection']) == report['episodes'],
          'TRANSFER_EXPOSURE_SELECTION')
    tasks = {r['task'] for r in selected['selection']}
    check(len(tasks) == report['episodes'] == 64, 'TRANSFER_EXPOSURE_TASKS')
    messages = [json.loads(r)['text'].encode() for r in values['messages.jsonl'].splitlines()]
    check(len(messages) == report['units'], 'TRANSFER_EXPOSURE_MESSAGES')
    return tasks, messages, dict(root=str(root), pin=pin, files={
        n: report['files'][n] for n in values}, tasks=sorted(tasks), message_units=len(messages),
        message_utf8_bytes=sum(map(len, messages)))


def programs(text):
    """Extract only literal programs; do not evaluate shell or Python expressions."""
    for header in HEADER.finditer(text):
        command, tail = header['command'].strip(), header['tail'].strip()
        form, filename = None, None
        if re.fullmatch(PYTHON+r'\s+(?:-u\s+)?-', command) and not tail:
            form = POLICY['program_forms'][0]
        else:
            before = re.fullmatch(r'cat\s*>\s*('+FILENAME+r')', command)
            after = re.fullmatch(r'>\s*('+FILENAME+r')', tail)
            if before and not tail:
                filename = before[1]
            elif command == 'cat' and after:
                filename = after[1]
            if filename:
                form = POLICY['program_forms'][1]
        if form is None:
            continue
        end = re.search(r'(?m)^'+re.escape(header['tag'])+r'[ \t]*\r?$', text[header.end():])
        if end is None:
            continue
        right = header.end()+end.start()
        if filename and not re.search(r'(?m)^\s*'+PYTHON+r'\s+'+re.escape(filename)+r'\s*$',
                                      text[header.end()+end.end():]):
            continue
        program = text[header.end():right].encode()
        try:
            tree = ast.parse(program)
        except (SyntaxError, ValueError):
            continue
        imports, relative = set(), False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(n.name.split('.')[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                relative |= bool(node.level)
                imports.add((node.module or '').split('.')[0])
        if relative or POLICY['required_import'] not in imports or not imports <= set(POLICY['allowed_imports']):
            continue
        yield dict(form=form, imports=sorted(imports), program_sha256=sha(program),
            utf8_span=[len(text[:header.end()].encode()), len(text[:right].encode())],
            program_bytes=len(program)), program


def rank(row):
    return (digest([POLICY['seed'], row['task'], row['turn'], row['program_sha256'], row['utf8_span']]),
            row['shard'], row['row'], row['turn'], row['utf8_span'])


def choose(candidates, messages):
    checked = []
    for row, action, program in sorted(candidates, key=lambda v: rank(v[0])):
        seen = [i for i, text in enumerate(messages) if program in text]
        checked.append(dict(rank=rank(row)[0], source=row, exposed_message_units=seen))
        if not seen:
            return (row, action, program), checked
    return None, checked


def collect(opened, population, excluded):
    """Read only even AI action fields; no later response enters Python strings."""
    candidates, counts = [], Counter()
    for shard in population['scope']['shards']:
        chosen = {r['row']: r['task'] for r in population['selection']
                  if r['shard'] == shard['path'] and r['task'] not in excluded}
        offset = 0
        for batch in opened[shard['path']].iter_batches(batch_size=128, columns=['instance_id', 'trajectory']):
            indices = sorted(r for r in chosen if offset <= r < offset+batch.num_rows)
            if indices:
                rows = batch.take(pa.array([r-offset for r in indices]))
                actions = pc.list_slice(rows.column('trajectory'), 2, 16, 2)
                for i, source_row in enumerate(indices):
                    task = rows.column('instance_id')[i].as_py()
                    check(task == chosen[source_row], 'TRANSFER_SOURCE_ROW')
                    counts['tasks_read'] += 1
                    for turn, value in zip(POLICY['turns'], actions[i].values):
                        counts['action_positions'] += 1
                        if value['role'].as_py() != 'ai':
                            counts['non_ai_positions'] += 1
                            continue
                        text = value['text'].as_py()
                        if not isinstance(text, str) or not text:
                            counts['empty_actions'] += 1
                            continue
                        counts['action_utf8_bytes'] += len(text.encode())
                        found = list(programs(text))
                        counts['actions_without_eligible_program'] += not bool(found)
                        for item, program in found:
                            row = dict(item, task=task, shard=shard['path'], row=source_row,
                                turn=turn, source_sha256=shard['sha256'], revision=population['scope']['revision'],
                                action_sha256=sha(text.encode()), action_bytes=len(text.encode()),
                                role='generated_proposal', historical_occurrence_established=False)
                            candidates.append((row, text.encode(), program))
            offset += batch.num_rows
            if chosen and offset > max(chosen):
                break
    return candidates, dict(counts)


def write_bytes(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build(raw_root, exposure_root, exposure_pin, root):
    started = time.perf_counter()
    root = Path(root)
    check(not root.exists(), 'TRANSFER_OUTPUT_EXISTS')
    excluded, messages, old = exposure(exposure_root, exposure_pin)
    read_exposure = time.perf_counter()
    declared = scope()
    with ExitStack() as stack:
        opened = {s['path']: pq.ParquetFile(stack.enter_context(open_bound(raw_root, s)))
                  for s in declared['shards']}
        for shard in declared['shards']:
            check(opened[shard['path']].metadata.num_rows == shard['rows'], 'TRANSFER_RAW_ROWS')
        hashed = time.perf_counter()
        def batches(shard):
            for batch in opened[shard['path']].iter_batches(batch_size=1024, columns=['instance_id']):
                yield batch.column(0).to_pylist()
        population = select_metadata(declared, batches)
        check(excluded <= {r['task'] for r in population['selection']}, 'TRANSFER_EXPOSURE_MEMBERSHIP')
        selected_metadata = time.perf_counter()
        candidates, counts = collect(opened, population, excluded)
    collected = time.perf_counter()
    selected, comparisons = choose(candidates, messages)
    chosen = time.perf_counter()
    root.mkdir()
    publish(root/'policy.json', POLICY)
    publish(root/'metadata.json', population)
    publish(root/'exposure.json', old)
    publish(root/'candidates.json', sorted([c[0] for c in candidates], key=rank))
    publish(root/'comparisons.json', comparisons)
    if selected:
        row, action, program = selected
        publish(root/'selected.json', row)
        write_bytes(root/'action.txt', action)
        write_bytes(root/'program.py', program)
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
             for p in sorted(root.iterdir()) if p.is_file()}
    persisted = time.perf_counter()
    phases = dict(exposure_read=read_exposure-started, hash_open=hashed-read_exposure,
        metadata=selected_metadata-hashed, action_scan=collected-selected_metadata,
        exposure_comparison=chosen-collected, publication=persisted-chosen)
    report = dict(schema='action-transfer-source-v1', policy=digest(POLICY), source_pin=digest(declared),
        source_root=str(raw_root), exposure=old, counts=counts,
        raw_bytes=sum(s['bytes'] for s in declared['shards']), source_rows=population['source_rows'],
        source_tasks=population['count'], excluded_tasks=len(excluded),
        eligible_programs=len(candidates), exposure_comparisons=len(comparisons),
        status='selected' if selected else 'NO_ELIGIBLE_UNEXPOSED_ACTION',
        selected=selected[0] if selected else None, files=files, code=code_pins(),
        world_truth_established=False, executed=False, forecast_present=False,
        scientific_admission=False, serving=False, phases=phases, wall_seconds=persisted-started,
        phase_coverage=sum(phases.values())/(persisted-started),
        exposure_scope='retained_tool_population_only_broader_historical_exposure_not_audited')
    pin = publish(root/'report.json', report)
    write_bytes(root/'COMMITTED', pin.encode())
    return report, pin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'exposure', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--exposure-pin', required=True)
    args = parser.parse_args()
    result, pin = build(args.raw, args.exposure, args.exposure_pin, args.output)
    print(json.dumps(dict(pin=pin, status=result['status'], counts=result['counts'],
        eligible_programs=result['eligible_programs'], selected=result['selected'],
        wall_seconds=result['wall_seconds'])), flush=True)
