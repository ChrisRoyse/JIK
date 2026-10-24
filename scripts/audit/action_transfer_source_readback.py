"""Re-read source selection by row group; check raw span, rank and exposure."""
import argparse
import ast
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.action_transfer_source import POLICY, code_pins, exposure, programs
from claims.tool_temporal_source import scope
from senses.trajectory_population import open_bound, publish, select_metadata


def commitment(raw, pin):
    check(sha(raw) == pin, 'TRANSFER_READER_COMMITMENT')
    report = json.loads(raw)
    check(raw == canonical(report)+b'\n' and report['schema'] == 'action-transfer-source-v1'
          and report['policy'] == digest(POLICY) and report['code'] == code_pins(),
          'TRANSFER_READER_SCOPE')
    return report


def order(row):
    value = [POLICY['seed'], row['task'], row['turn'], row['program_sha256'], row['utf8_span']]
    return (hashlib.sha256(canonical(value)).hexdigest(), row['shard'], row['row'],
            row['turn'], row['utf8_span'])


def program_check(row, action, program, excluded, messages):
    check(row['task'] not in excluded, 'TRANSFER_READER_EXPOSED_TASK')
    left, right = row['utf8_span']
    check(type(left) is type(right) is int and 0 <= left < right <= len(action)
          and action[left:right] == program and sha(program) == row['program_sha256']
          and len(program) == row['program_bytes'], 'TRANSFER_READER_PROGRAM_SPAN')
    check(sha(action) == row['action_sha256'] and len(action) == row['action_bytes'],
          'TRANSFER_READER_ACTION_BYTES')
    check(not any(program in text for text in messages), 'TRANSFER_READER_EXPOSED_PROGRAM')
    tree = ast.parse(program)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(n.name.split('.')[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            check(node.level == 0, 'TRANSFER_READER_RELATIVE_IMPORT')
            names.add((node.module or '').split('.')[0])
    check(sorted(names) == row['imports'] and 'marshmallow' in names
          and names <= set(POLICY['allowed_imports']), 'TRANSFER_READER_IMPORTS')


def reconstruct(opened, population, excluded):
    """Separate Parquet row-group path; materialize only nominated action fields."""
    result = []
    for shard in population['scope']['shards']:
        wanted = {r['row']: r['task'] for r in population['selection']
                  if r['shard'] == shard['path'] and r['task'] not in excluded}
        parquet, offset = opened[shard['path']], 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            keys = sorted(k for k in wanted if offset <= k < offset+size)
            if keys:
                table = parquet.read_row_group(group, columns=['instance_id', 'trajectory'])
                selected = table.take(pa.array([k-offset for k in keys]))
                for i, address in enumerate(keys):
                    task = selected['instance_id'][i].as_py()
                    check(task == wanted[address], 'TRANSFER_READER_SOURCE_ROW')
                    turns = selected['trajectory'][i].values
                    for turn in POLICY['turns']:
                        if turn >= len(turns) or turns[turn]['role'].as_py() != 'ai':
                            continue
                        text = turns[turn]['text'].as_py()
                        if not isinstance(text, str) or not text:
                            continue
                        for item, program in programs(text):
                            row = dict(item, task=task, shard=shard['path'], row=address,
                                turn=turn, source_sha256=shard['sha256'], revision=population['scope']['revision'],
                                action_sha256=sha(text.encode()), action_bytes=len(text.encode()),
                                role='generated_proposal', historical_occurrence_established=False)
                            result.append((row, text.encode(), program))
            offset += size
    return sorted(result, key=lambda item: order(item[0]))


def read(root, pin):
    started = time.perf_counter()
    root = Path(root)
    raw = (root/'report.json').read_bytes()
    check((root/'COMMITTED').read_text() == pin, 'TRANSFER_READER_COMMITMENT')
    report = commitment(raw, pin)
    for name, entry in report['files'].items():
        check(Path(name).name == name, 'TRANSFER_READER_PATH')
        value = (root/name).read_bytes()
        check(len(value) == entry['bytes'] and sha(value) == entry['sha256'], 'TRANSFER_READER_FILE_BYTES')
    get = lambda name: json.loads((root/name).read_bytes())
    check(get('policy.json') == POLICY and not report['executed'] and not report['forecast_present']
          and not report['scientific_admission'] and not report['serving'], 'TRANSFER_READER_STATUS')
    excluded, messages, old = exposure(report['exposure']['root'], report['exposure']['pin'])
    check(get('exposure.json') == report['exposure'] == old, 'TRANSFER_READER_EXPOSURE')
    validated = time.perf_counter()
    declared = scope()
    with ExitStack() as stack:
        opened = {s['path']: pq.ParquetFile(stack.enter_context(open_bound(Path(report['source_root']), s)))
                  for s in declared['shards']}
        hashed = time.perf_counter()
        def batches(shard):
            p = opened[shard['path']]
            for i in range(p.num_row_groups):
                yield p.read_row_group(i, columns=['instance_id'])['instance_id'].to_pylist()
        population = select_metadata(declared, batches)
        check(population == get('metadata.json'), 'TRANSFER_READER_METADATA')
        candidates = reconstruct(opened, population, excluded)
    reread = time.perf_counter()
    check([v[0] for v in candidates] == get('candidates.json')
          and len(candidates) == report['eligible_programs'], 'TRANSFER_READER_CANDIDATES')
    selected, comparisons = None, []
    for row, action, program in candidates:
        exposed = [i for i, message in enumerate(messages) if program in message]
        comparisons.append(dict(rank=order(row)[0], source=row, exposed_message_units=exposed))
        if not exposed:
            selected = row, action, program
            break
    check(comparisons == get('comparisons.json') and len(comparisons) == report['exposure_comparisons'],
          'TRANSFER_READER_SELECTION_ORDER')
    controls = {}
    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'TRANSFER_READER_NEGATIVE_CODE')
        else:
            raise AssertionError('TRANSFER_READER_NEGATIVE_ACCEPTED_'+name)
        controls[name] = code
    refuse('changed_commitment', 'TRANSFER_READER_COMMITMENT', lambda:commitment(raw+b' ', pin))
    if selected:
        row, action, program = selected
        check(row == get('selected.json') == report['selected'] and report['status'] == 'selected',
              'TRANSFER_READER_SELECTED')
        check((root/'action.txt').read_bytes() == action and (root/'program.py').read_bytes() == program,
              'TRANSFER_READER_RAW_BYTES')
        program_check(row, action, program, excluded, messages)
        changed = copy.deepcopy(row)
        changed['utf8_span'][0] += 1
        refuse('changed_span', 'TRANSFER_READER_PROGRAM_SPAN',
               lambda:program_check(changed, action, program, excluded, messages))
        refuse('changed_program', 'TRANSFER_READER_PROGRAM_SPAN',
               lambda:program_check(row, action, program+b' ', excluded, messages))
        refuse('exposed_task', 'TRANSFER_READER_EXPOSED_TASK',
               lambda:program_check(row, action, program, excluded|{row['task']}, messages))
        refuse('exposed_program', 'TRANSFER_READER_EXPOSED_PROGRAM',
               lambda:program_check(row, action, program, excluded, messages+[program]))
    else:
        check(report['selected'] is None and report['status'] == 'NO_ELIGIBLE_UNEXPOSED_ACTION',
              'TRANSFER_READER_EMPTY_SELECTION')
    checked = time.perf_counter()
    for name, entry in report['files'].items():
        check(sha((root/name).read_bytes()) == entry['sha256'], 'TRANSFER_READER_MUTATION')
    check((root/'report.json').read_bytes() == raw, 'TRANSFER_READER_MUTATION')
    phases = dict(artifact_and_exposure=validated-started, raw_hash=hashed-validated,
        source_reconstruction=reread-hashed, selection_and_controls=checked-reread,
        final_readback=time.perf_counter()-checked)
    wall = time.perf_counter()-started
    return dict(schema='action-transfer-source-readback-v1', acceptance='author-checked',
        source_root=str(root), source_pin=pin, status=report['status'], selected=report['selected'],
        raw_source_rows=population['source_rows'], raw_source_tasks=population['count'],
        excluded_tasks=len(excluded), eligible_programs=len(candidates), controls=controls,
        separate_row_group_readback=True, extraction_grammar_shared=True,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        scientific_admission=False, serving=False, forecast_present=False, executed=False,
        code_sha256=sha(Path(__file__).read_bytes()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = read(args.root, args.pin)
    pin = publish(args.output, result)
    print(json.dumps(dict(pin=pin, **result)), flush=True)
