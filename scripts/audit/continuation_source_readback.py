"""Reconstruct suffix selection from raw Parquet row groups and exercise refusals."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.action_transfer_edit_source import edit_bodies, syntax
from claims.action_transfer_source import programs, rank
from claims.codata_source import check, sha
from claims.continuation_source import ContinuationSource, POLICY, build, code_pins, inputs
from claims.episodic_memory import canonical, digest
from claims.proposed_program import validate
from senses.trajectory_population import open_bound, publish
from senses.contract import SenseError


def reconstruct(data):
    _, report, metadata, tasks, messages, excluded, _ = data
    found, counts = [], Counter()
    with ExitStack() as stack:
        for shard in metadata['scope']['shards']:
            addresses = {v['row']:v['task'] for v in metadata['selection']
                         if v['shard'] == shard['path'] and v['task'] not in tasks}
            stream = stack.enter_context(open_bound(Path(report['source_root']), shard))
            parquet = pq.ParquetFile(stream)
            check(parquet.metadata.num_rows == shard['rows'], 'CONTINUATION_READBACK_ROWS')
            offset = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                wanted = sorted(k for k in addresses if offset <= k < offset+size)
                if wanted:
                    table = parquet.read_row_group(group, columns=['instance_id', 'trajectory'])
                    selected = table.take(pa.array([k-offset for k in wanted]))
                    for i,address in enumerate(wanted):
                        task = selected.column('instance_id')[i].as_py()
                        check(task == addresses[address], 'CONTINUATION_READBACK_ADDRESS')
                        trajectory = selected.column('trajectory')[i].values
                        counts['tasks_read'] += 1
                        counts['recorded_turn_positions'] += len(trajectory)
                        counts['tasks_with_suffix'] += len(trajectory) > 16
                        for turn in range(16, len(trajectory), 2):
                            record = trajectory[turn]
                            check(record['role'].as_py() == 'ai', 'CONTINUATION_READBACK_ROLE')
                            text = record['text'].as_py()
                            check(isinstance(text, str) and bool(text), 'CONTINUATION_READBACK_TEXT')
                            action = text.encode()
                            counts['action_positions'] += 1
                            counts['action_utf8_bytes'] += len(action)
                            options = list(edit_bodies(text))
                            options.extend((dict(v, syntax_sha256=syntax(p)), p)
                                           for v,p in programs(text))
                            counts['actions_without_eligible_program'] += not bool(options)
                            for item,program in options:
                                row = dict(item, task=task, shard=shard['path'], row=address,
                                    turn=turn, source_sha256=shard['sha256'],
                                    revision=metadata['scope']['revision'],
                                    action_sha256=sha(action), action_bytes=len(action),
                                    role='generated_proposal', historical_occurrence_established=False,
                                    historical_file_state_established=False)
                                validate(row, action, program)
                                found.append((row, action, program))
                offset += size
    comparisons, first = [], None
    for row,action,program in sorted(found, key=lambda r:rank(r[0])):
        seen = [i for i,m in enumerate(messages) if program in m]
        reasons = (['EXPOSED_TASK'] if row['task'] in tasks else [])
        reasons += ['EXPOSED_PROGRAM'] if seen else []
        reasons += ['EXPOSED_SYNTAX'] if syntax(program) in excluded else []
        comparisons.append(dict(source=row, rank=rank(row)[0], reasons=reasons,
                                exposed_messages=seen))
        if first is None and not reasons:
            first = row, action, program
    return first, comparisons, dict(counts)


def inspect(source, expected):
    selected, comparisons, counts = expected
    check(selected is not None, 'CONTINUATION_READBACK_EXHAUSTED')
    row, action, program = selected
    check(json.loads((source.root/'comparisons.json').read_bytes()) == comparisons,
          'CONTINUATION_READBACK_COMPARISONS')
    check(source.report['counts'] == counts and source.report['candidates'] == len(comparisons),
          'CONTINUATION_READBACK_COUNTS')
    check(source.row == row and source.action == action and source.program == program,
          'CONTINUATION_READBACK_SELECTED_BYTES')
    return dict(selected=digest(row), action=sha(action), program=sha(program),
                counts=counts, comparisons=digest(comparisons))


def files(root):
    return {p.name:sha(p.read_bytes()) for p in root.iterdir() if p.is_file()}


def exhausted(root, pin, expected=None):
    raw = (root/'exhaustion.json').read_bytes()
    check(sha(raw) == pin, 'CONTINUATION_EXHAUSTION_PIN')
    value = json.loads(raw)
    check(raw == canonical(value)+b'\n' and value['policy'] == POLICY
          and value['code'] == code_pins()
          and value['status'] == 'CONTINUATION_NO_ELIGIBLE_PROPOSAL',
          'CONTINUATION_EXHAUSTION_CONTRACT')
    data = inputs(value['binding']['parent']['root'], value['binding']['parent']['pin'])
    check(value['binding'] == data[-1], 'CONTINUATION_EXHAUSTION_BINDING')
    if expected is not None:
        selected, comparisons, counts = expected
        check(selected is None, 'CONTINUATION_EXHAUSTION_FALSE')
        check(value['counts'] == counts, 'CONTINUATION_EXHAUSTION_COUNTS')
        check(value['comparisons'] == comparisons, 'CONTINUATION_EXHAUSTION_COMPARISONS')
    return value, data


def exhaustion_readback(a, started):
    value, data = exhausted(a.source, a.pin)
    before = files(a.source)
    authenticated = time.perf_counter()
    expected = reconstruct(data)
    exhausted(a.source, a.pin, expected)
    scanned = time.perf_counter()
    controls = []

    def refuse(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'CONTINUATION_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('CONTINUATION_CONTROL_ACCEPTED:'+name)
        check(files(a.source) == before, 'CONTINUATION_CONTROL_MUTATED_ORIGINAL')
        controls.append(dict(name=name, code=code))

    refuse('wrong_census_pin', 'CONTINUATION_EXHAUSTION_PIN',
           lambda:exhausted(a.source, '0'*64))
    refuse('duplicate_publication', 'CONTINUATION_OUTPUT_EXISTS',
           lambda:build(data[0].root, data[0].pin, a.source))
    with tempfile.TemporaryDirectory(prefix='continuation-control-', dir=a.output.parent) as tmp:
        clone = Path(tmp)
        changed = json.loads(canonical(value))
        changed['counts']['action_positions'] += 1
        raw = canonical(changed)+b'\n'
        (clone/'exhaustion.json').write_bytes(raw)
        refuse('resigned_wrong_counts', 'CONTINUATION_EXHAUSTION_COUNTS',
               lambda:exhausted(clone, sha(raw), expected))
        changed = dict(value, comparisons=[])
        raw = canonical(changed)+b'\n'
        (clone/'exhaustion.json').write_bytes(raw)
        refuse('resigned_erased_exclusions', 'CONTINUATION_EXHAUSTION_COMPARISONS',
               lambda:exhausted(clone, sha(raw), expected))
    ended = time.perf_counter()
    a.output.mkdir()
    report = dict(schema='continuation-exhaustion-readback-v1', source=a.pin,
        counts=expected[2], comparisons=digest(expected[1]), controls=controls,
        selected=None, immutable_before=before, immutable_after=files(a.source),
        code=sha(Path(__file__).read_bytes()),
        phases=dict(authentication=authenticated-started, row_group_reconstruction=scanned-authenticated,
                    controls=ended-scanned), wall_seconds=ended-started,
        acceptance='author-checked', number_status='measured',
        scientific_admission=False, serving=False, forecast_present=False, execution_present=False)
    pin = publish(a.output/'readback.json', report)
    print(json.dumps(dict(pin=pin, counts=report['counts'], controls=controls,
                         wall_seconds=report['wall_seconds'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exhausted', action='store_true')
    a = parser.parse_args()
    started = time.perf_counter()
    check(not a.output.exists(), 'CONTINUATION_READBACK_OUTPUT_EXISTS')
    if a.exhausted:
        exhaustion_readback(a, started)
        return
    source = ContinuationSource(a.source, a.pin)
    data = inputs(source.report['binding']['parent']['root'],
                  source.report['binding']['parent']['pin'])
    before = files(a.source)
    authenticated = time.perf_counter()
    expected = reconstruct(data)
    independent = inspect(source, expected)
    scanned = time.perf_counter()
    controls = []

    def refuse(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as error:
            check(code in str(error), 'CONTINUATION_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('CONTINUATION_CONTROL_ACCEPTED:'+name)
        check(files(a.source) == before, 'CONTINUATION_CONTROL_MUTATED_ORIGINAL')
        controls.append(dict(name=name, code=code))

    refuse('wrong_source_pin', 'CONTINUATION_PIN', lambda:ContinuationSource(a.source, '0'*64))
    refuse('duplicate_publication', 'CONTINUATION_OUTPUT_EXISTS',
           lambda:build(data[0].root, data[0].pin, a.source))
    with tempfile.TemporaryDirectory(prefix='continuation-control-', dir=a.output.parent) as tmp:
        clone = Path(tmp)/'source'
        shutil.copytree(a.source, clone)
        (clone/'program.py').write_bytes(source.program+b'\n')
        refuse('changed_program_bytes', 'CONTINUATION_BYTES',
               lambda:ContinuationSource(clone, a.pin))
        (clone/'program.py').write_bytes(source.program)
        row = dict(source.row, row=source.row['row']+1)
        changed = json.loads((clone/'comparisons.json').read_bytes())
        for item in changed:
            if item['source'] == source.row:
                item['source'] = row
        (clone/'selected.json').write_bytes(canonical(row)+b'\n')
        (clone/'comparisons.json').write_bytes(canonical(changed)+b'\n')
        report = dict(source.report, selected=row)
        report['files'] = {n:dict(bytes=(clone/n).stat().st_size, sha256=sha((clone/n).read_bytes()))
                           for n in report['files']}
        raw = canonical(report)+b'\n'
        (clone/'report.json').write_bytes(raw)
        (clone/'COMMITTED').write_text(sha(raw))
        refuse('resigned_wrong_raw_address', 'CONTINUATION_READBACK_COMPARISONS',
               lambda:inspect(ContinuationSource(clone, sha(raw)), expected))
    check(files(a.source) == before, 'CONTINUATION_READBACK_SOURCE_CHANGED')
    ended = time.perf_counter()
    a.output.mkdir()
    report = dict(schema='continuation-source-readback-v1', source=a.pin,
        independent=independent, controls=controls, immutable_before=before,
        immutable_after=files(a.source), source_reference=source.reference(),
        code=sha(Path(__file__).read_bytes()),
        phases=dict(authentication=authenticated-started, row_group_reconstruction=scanned-authenticated,
                    controls=ended-scanned), wall_seconds=ended-started,
        acceptance='author-checked', number_status='measured',
        scientific_admission=False, serving=False, forecast_present=False, execution_present=False)
    pin = publish(a.output/'readback.json', report)
    print(json.dumps(dict(pin=pin, counts=independent['counts'], controls=controls,
                         wall_seconds=report['wall_seconds'])))


if __name__ == '__main__':
    main()
