"""Select literal proposed programs, excluding retained syntax as well as bytes."""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.action_transfer_source_readback import commitment, program_check
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.action_transfer_source import POLICY as OLD_POLICY, code_pins as prior_code
from claims.action_transfer_source import exposure, programs, rank, write_bytes
from senses.trajectory_population import open_bound, publish

POLICY = dict(OLD_POLICY, schema='action-transfer-edit-policy-v1',
    added_program_form='literal_edit_from_line_one_body_standalone_python',
    syntax_exclusion='sha256(ast.dump(ast.parse(program),include_attributes=False))',
    historical_file_state_established=False, historical_edit_acceptance_established=False,
    historical_evaluation_untouched=False, author_format_inspection_exposed=True,
    scope='retained_proposed_bytes_for_new_independent_execution')
EDIT = re.compile(r'(?m)^edit[ \t]+1:(?P<last>[1-9][0-9]*)[ \t]*\r?\n'
                  r'(?P<body>[\s\S]*?)(?m:^end_of_edit[ \t]*\r?$)')


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return dict(prior_code(), **{n:sha((root/n).read_bytes()) for n in (
        'claims/action_transfer_edit_source.py', 'audit/action_transfer_source_readback.py')})


def syntax(program):
    return sha(ast.dump(ast.parse(program), include_attributes=False).encode())


def edit_bodies(text, eligible=True):
    for match in EDIT.finditer(text):
        program = match['body'].encode()
        try:
            tree = ast.parse(program)
        except (SyntaxError, ValueError):
            continue
        names, relative = set(), False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(n.name.split('.')[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or '').split('.')[0])
                relative |= bool(node.level)
        if eligible and (relative or 'marshmallow' not in names
                         or not names <= set(POLICY['allowed_imports'])):
            continue
        yield dict(form='literal_edit_body', imports=sorted(names),
            program_sha256=sha(program), program_bytes=len(program), syntax_sha256=syntax(program),
            utf8_span=[len(text[:match.start('body')].encode()), len(text[:match.end('body')].encode())]), program


def novel(row, program, excluded_syntax):
    check(syntax(program) == row['syntax_sha256'], 'TRANSFER_EDIT_SYNTAX_PIN')
    check(row['syntax_sha256'] not in excluded_syntax, 'TRANSFER_EDIT_EXPOSED_SYNTAX')


def parent(root, pin):
    root = Path(root)
    raw = (root/'report.json').read_bytes()
    report = commitment(raw, pin)
    check((root/'COMMITTED').read_text() == pin, 'TRANSFER_EDIT_PARENT_PIN')
    data = (root/'metadata.json').read_bytes()
    check(sha(data) == report['files']['metadata.json']['sha256'], 'TRANSFER_EDIT_METADATA_PIN')
    return report, json.loads(data)


def action_rows(parquet, addresses, by_group):
    """Two source read paths: batch selection and independent row-group selection."""
    if by_group:
        batches = (parquet.read_row_group(i, columns=['instance_id', 'trajectory'])
                   for i in range(parquet.num_row_groups))
    else:
        batches = parquet.iter_batches(batch_size=128, columns=['instance_id', 'trajectory'])
    offset = 0
    for batch in batches:
        keys = sorted(k for k in addresses if offset <= k < offset+batch.num_rows)
        if keys:
            selected = batch.take(pa.array([k-offset for k in keys]))
            for i, row in enumerate(keys):
                task = selected.column('instance_id')[i].as_py()
                check(task == addresses[row], 'TRANSFER_EDIT_RAW_ADDRESS')
                turns = selected.column('trajectory')[i].values
                for turn in POLICY['turns']:
                    if turn < len(turns):
                        value = turns[turn]
                        check(value['role'].as_py() == 'ai', 'TRANSFER_EDIT_ROLE')
                        yield task, row, turn, value['text'].as_py()
        offset += batch.num_rows


def discover(prior, metadata, by_group=False):
    start = time.perf_counter()
    excluded, messages, old = exposure(prior['exposure']['root'], prior['exposure']['pin'])
    old_syntax = {}
    for i, message in enumerate(messages):
        for item, program in edit_bodies(message.decode(), eligible=False):
            old_syntax.setdefault(item['syntax_sha256'], []).append(i)
        for item, program in programs(message.decode()):
            old_syntax.setdefault(syntax(program), []).append(i)
    for key in old_syntax:
        old_syntax[key] = sorted(set(old_syntax[key]))
    exposed_at = time.perf_counter()
    rows, counts = [], Counter()
    with ExitStack() as stack:
        opened = {s['path']: pq.ParquetFile(stack.enter_context(open_bound(Path(prior['source_root']), s)))
                  for s in metadata['scope']['shards']}
        hashed = time.perf_counter()
        for shard in metadata['scope']['shards']:
            addresses = {v['row']: v['task'] for v in metadata['selection']
                         if v['shard'] == shard['path'] and v['task'] not in excluded}
            for task, address, turn, text in action_rows(opened[shard['path']], addresses, by_group):
                counts['action_positions'] += 1
                counts['action_utf8_bytes'] += len(text.encode())
                options = [*edit_bodies(text), *[(dict(v, syntax_sha256=syntax(p)), p)
                                               for v, p in programs(text)]]
                for item, program in options:
                    row = dict(item, task=task, row=address, turn=turn, shard=shard['path'],
                        source_sha256=shard['sha256'], revision=metadata['scope']['revision'],
                        action_sha256=sha(text.encode()), action_bytes=len(text.encode()),
                        role='generated_proposal', historical_occurrence_established=False,
                        historical_file_state_established=False)
                    rows.append((row, text.encode(), program))
    scanned = time.perf_counter()
    check(counts['action_positions'] == prior['counts']['action_positions']
          and counts['action_utf8_bytes'] == prior['counts']['action_utf8_bytes'], 'TRANSFER_EDIT_SCAN_COVERAGE')
    rows.sort(key=lambda v:rank(v[0]))
    selected, comparisons = None, []
    for row, action, program in rows:
        seen = [i for i, text in enumerate(messages) if program in text]
        equivalents = old_syntax.get(row['syntax_sha256'], [])
        comparisons.append(dict(source=row, rank=rank(row)[0], exact_exposure=seen, syntax_exposure=equivalents))
        if not seen and not equivalents:
            selected = row, action, program
            break
    compared = time.perf_counter()
    phases = dict(exposure_and_syntax=exposed_at-start, raw_hash=hashed-exposed_at,
                  action_scan=scanned-hashed, comparison=compared-scanned)
    return dict(rows=rows, comparisons=comparisons, selected=selected, old=old,
                old_syntax=old_syntax, messages=messages, excluded=excluded,
                counts=dict(counts), phases=phases)


def build(prior_root, prior_pin, root):
    started = time.perf_counter()
    check(not root.exists(), 'TRANSFER_EDIT_OUTPUT_EXISTS')
    prior, metadata = parent(prior_root, prior_pin)
    authenticated = time.perf_counter()
    data = discover(prior, metadata)
    selected = data['selected']
    root.mkdir()
    publish(root/'policy.json', POLICY)
    publish(root/'syntax-exclusions.json', data['old_syntax'])
    publish(root/'candidates.json', [r[0] for r in data['rows']])
    publish(root/'comparisons.json', data['comparisons'])
    if selected:
        row, action, program = selected
        publish(root/'selected.json', row)
        write_bytes(root/'action.txt', action)
        write_bytes(root/'program.py', program)
    files = {p.name:dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in sorted(root.iterdir())}
    now = time.perf_counter()
    phases = dict(parent=authenticated-started, **data['phases'])
    phases['publication_and_overhead'] = now-started-sum(phases.values())
    report = dict(schema='action-transfer-edit-source-v1', parent=dict(root=str(prior_root), pin=prior_pin),
        policy=digest(POLICY), code=code_pins(), files=files, source_root=prior['source_root'],
        exposure=data['old'], counts=data['counts'], eligible_programs=len(data['rows']),
        syntax_exclusions=len(data['old_syntax']), exposure_comparisons=len(data['comparisons']),
        selected=selected[0] if selected else None,
        status='selected' if selected else 'NO_ELIGIBLE_UNEXPOSED_PROPOSAL',
        scientific_admission=False, serving=False, executed=False, forecast_present=False,
        dependency_closure_exposure_complete=False, historical_evaluation_untouched=False,
        phases=phases, wall_seconds=now-started, phase_coverage=sum(phases.values())/(now-started))
    pin = publish(root/'report.json', report)
    write_bytes(root/'COMMITTED', pin.encode())
    return report, pin


def read(root, pin):
    start = time.perf_counter()
    raw = (root/'report.json').read_bytes()
    def authenticate(value):
        check(sha(value) == pin, 'TRANSFER_EDIT_COMMITMENT')
        return json.loads(value)
    report = authenticate(raw)
    check((root/'COMMITTED').read_text() == pin and report['code'] == code_pins()
          and report['policy'] == digest(POLICY), 'TRANSFER_EDIT_PIN')
    for name, item in report['files'].items():
        check(Path(name).name == name, 'TRANSFER_EDIT_PATH')
        value = (root/name).read_bytes()
        check(len(value) == item['bytes'] and sha(value) == item['sha256'], 'TRANSFER_EDIT_BYTES')
    prior, metadata = parent(report['parent']['root'], report['parent']['pin'])
    authenticated = time.perf_counter()
    data = discover(prior, metadata, by_group=True)
    rebuilt = time.perf_counter()
    get = lambda n:json.loads((root/n).read_bytes())
    check(get('policy.json') == POLICY and get('syntax-exclusions.json') == data['old_syntax']
          and get('candidates.json') == [r[0] for r in data['rows']]
          and get('comparisons.json') == data['comparisons'] and report['counts'] == data['counts'],
          'TRANSFER_EDIT_SOURCE_RECOUNT')
    check(len(data['rows']) == report['eligible_programs']
          and len(data['old_syntax']) == report['syntax_exclusions']
          and len(data['comparisons']) == report['exposure_comparisons'], 'TRANSFER_EDIT_COUNTS')
    controls = {}
    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'TRANSFER_EDIT_CONTROL_CODE')
        else:
            raise AssertionError('TRANSFER_EDIT_CONTROL_ACCEPTED_'+name)
        controls[name] = code
    refuse('changed_commitment', 'TRANSFER_EDIT_COMMITMENT', lambda:authenticate(raw+b' '))
    if data['selected']:
        row, action, program = data['selected']
        check(row == report['selected'] == get('selected.json') and report['status'] == 'selected'
              and (root/'action.txt').read_bytes() == action and (root/'program.py').read_bytes() == program,
              'TRANSFER_EDIT_SELECTED')
        args = row, action, program, data['excluded'], data['messages']
        program_check(*args)
        novel(row, program, data['old_syntax'])
        changed = copy.deepcopy(row)
        changed['utf8_span'][0] += 1
        refuse('changed_span', 'TRANSFER_READER_PROGRAM_SPAN',
               lambda:program_check(changed, *args[1:]))
        refuse('changed_program', 'TRANSFER_READER_PROGRAM_SPAN',
               lambda:program_check(row, action, program+b' ', *args[3:]))
        refuse('exposed_task', 'TRANSFER_READER_EXPOSED_TASK',
               lambda:program_check(row, action, program, data['excluded']|{row['task']}, data['messages']))
        refuse('exposed_program', 'TRANSFER_READER_EXPOSED_PROGRAM',
               lambda:program_check(*args[:4], data['messages']+[program]))
        refuse('exposed_syntax', 'TRANSFER_EDIT_EXPOSED_SYNTAX',
               lambda:novel(row, program, {**data['old_syntax'], row['syntax_sha256']:[-1]}))
    else:
        check(report['selected'] is None and report['status'] == 'NO_ELIGIBLE_UNEXPOSED_PROPOSAL',
              'TRANSFER_EDIT_EMPTY')
    for name, item in report['files'].items():
        check(sha((root/name).read_bytes()) == item['sha256'], 'TRANSFER_EDIT_MUTATION')
    check((root/'report.json').read_bytes() == raw, 'TRANSFER_EDIT_MUTATION')
    wall = time.perf_counter()-start
    phases = dict(authentication=authenticated-start, independent_row_group_rebuild=rebuilt-authenticated,
                  comparisons_controls_and_readback=wall-(rebuilt-start))
    return dict(schema='action-transfer-edit-readback-v1', acceptance='author-checked', source_pin=pin,
        selected=report['selected'], controls=controls, status=report['status'],
        eligible_programs=report['eligible_programs'], syntax_exclusions=report['syntax_exclusions'],
        exposure_comparisons=report['exposure_comparisons'], counts=report['counts'],
        separate_row_group_readback=True, extraction_grammar_shared=True,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        scientific_admission=False, serving=False, executed=False, forecast_present=False,
        code_sha256=sha(Path(__file__).read_bytes()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('select', 'read'), required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'select':
        result, pin = build(a.source, a.pin, a.output)
    else:
        result = read(a.source, a.pin)
        pin = publish(a.output, result)
    print(json.dumps(dict(pin=pin, **{k:v for k,v in result.items() if k not in ('code','files','exposure')})), flush=True)
