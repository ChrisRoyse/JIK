"""Outcome-blind selection from other attempts after the original attempt catalogue is exhausted."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.action_transfer_edit_source import edit_bodies, syntax
from claims.action_transfer_source import programs, rank, write_bytes
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.next_proposal import choices
from claims.continuation_source import inputs as prior_inputs, POLICY as PRIOR_POLICY
from claims.continuation_source import code_pins as prior_code
from claims.proposed_program import ProposedProgram, validate
from senses.trajectory_population import open_bound, publish


POLICY = dict(schema='remaining-attempt-source-policy-v1', first_turn=2, stride=2,
    order='inherited_catalogue_rank',
    exclude='retained_tasks_program_substrings_syntax_and_three_completed_proposals',
    attempt_exclusion='all_original_identifier_selected_addresses',
    outcome_fields=[], historical_occurrence_established=False,
    dependency_closure_exposure_complete=False, executed=False, forecast_present=False,
    scientific_admission=False, serving=False)
FILES = {'policy.json', 'comparisons.json', 'selected.json', 'action.txt', 'program.py'}


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n:sha((root/n).read_bytes()) for n in (
        'claims/remaining_attempt_source.py', 'claims/next_proposal.py',
        'claims/continuation_source.py',
        'claims/action_transfer_edit_source.py', 'claims/action_transfer_source.py',
        'senses/trajectory_population.py')}


def inputs(root, pin):
    root = Path(root)
    raw = (root/'exhaustion.json').read_bytes()
    check(sha(raw) == pin, 'ATTEMPT_PRIOR_EXHAUSTION_PIN')
    value = json.loads(raw)
    check(raw == canonical(value)+b'\n' and value['policy'] == PRIOR_POLICY
          and value['code'] == prior_code()
          and value['status'] == 'CONTINUATION_NO_ELIGIBLE_PROPOSAL'
          and all(r['reasons'] for r in value['comparisons']),
          'ATTEMPT_PRIOR_EXHAUSTION_CONTRACT')
    parent = value['binding']['parent']
    previous,report,metadata,tasks,messages,excluded,old = prior_inputs(parent['root'], parent['pin'])
    check(value['binding'] == old, 'ATTEMPT_PRIOR_BINDING')
    binding = dict(parent=dict(root=str(root), pin=pin), prior_binding=old,
                   exhausted_counts=value['counts'], exhausted_comparisons=digest(value['comparisons']))
    return previous, report, metadata, tasks, messages, excluded, binding


def actions(parquet, addresses):
    offset = 0
    for batch in parquet.iter_batches(batch_size=128, columns=['instance_id', 'trajectory']):
        keys = sorted(k for k in addresses if offset <= k < offset+batch.num_rows)
        if keys:
            selected = batch.take(pa.array([k-offset for k in keys]))
            for i, address in enumerate(keys):
                task = selected.column('instance_id')[i].as_py()
                check(task == addresses[address], 'ATTEMPT_RAW_ADDRESS')
                turns = selected.column('trajectory')[i].values
                yield task, address, None, len(turns)
                for turn in range(POLICY['first_turn'], len(turns), POLICY['stride']):
                    value = turns[turn]
                    check(value['role'].as_py() == 'ai', 'ATTEMPT_SOURCE_ROLE')
                    text = value['text'].as_py()
                    check(isinstance(text, str) and bool(text), 'ATTEMPT_SOURCE_TEXT')
                    yield task, address, turn, text
        offset += batch.num_rows
        if addresses and offset > max(addresses):
            break


def discover(data):
    _, report, metadata, tasks, messages, excluded, _ = data
    counts, candidates = Counter(), []
    attempt_rows = {}
    attempt_tasks = set()
    canonical_addresses = {(r['shard'],r['row']) for r in metadata['selection']}
    begin = time.perf_counter()
    with ExitStack() as stack:
        streams = {s['path']:stack.enter_context(open_bound(Path(report['source_root']), s))
                   for s in metadata['scope']['shards']}
        hashed = time.perf_counter()
        for shard in metadata['scope']['shards']:
            parquet = pq.ParquetFile(streams[shard['path']])
            check(parquet.metadata.num_rows == shard['rows'], 'ATTEMPT_RAW_ROWS')
            addresses, offset = {}, 0
            for batch in parquet.iter_batches(batch_size=1024, columns=['instance_id']):
                for task in batch.column(0).to_pylist():
                    check(isinstance(task, str) and bool(task), 'ATTEMPT_IDENTIFIER')
                    if task not in tasks and (shard['path'],offset) not in canonical_addresses:
                        addresses[offset] = task
                    offset += 1
            check(offset == shard['rows'], 'ATTEMPT_IDENTIFIER_ROWS')
            attempt_rows[shard['path']] = len(addresses)
            attempt_tasks.update(addresses.values())
            for task, address, turn, value in actions(parquet, addresses):
                if turn is None:
                    counts['attempt_rows_read'] += 1
                    counts['recorded_turn_positions'] += value
                    counts['attempts_with_actions'] += value > POLICY['first_turn']
                    continue
                counts['action_positions'] += 1
                counts['action_utf8_bytes'] += len(value.encode())
                options = [*edit_bodies(value),
                    *[(dict(v, syntax_sha256=syntax(p)), p) for v,p in programs(value)]]
                counts['actions_without_eligible_program'] += not bool(options)
                for item, program in options:
                    row = dict(item, task=task, shard=shard['path'], row=address, turn=turn,
                        source_sha256=shard['sha256'], revision=metadata['scope']['revision'],
                        action_sha256=sha(value.encode()), action_bytes=len(value.encode()),
                        role='generated_proposal', historical_occurrence_established=False,
                        historical_file_state_established=False)
                    validate(row, value.encode(), program)
                    candidates.append((row, value.encode(), program))
    scanned = time.perf_counter()
    check(counts['attempt_rows_read'] == sum(attempt_rows.values()),
          'ATTEMPT_SOURCE_COVERAGE')
    counts['attempt_rows_by_shard'] = attempt_rows
    counts['distinct_tasks_read'] = len(attempt_tasks)
    selected, comparisons = choices(candidates, tasks, messages, excluded)
    chosen = time.perf_counter()
    return selected, comparisons, dict(counts), dict(raw_hash=hashed-begin,
        action_scan=scanned-hashed, comparison=chosen-scanned)


def build(parent_root, parent_pin, root):
    started = time.perf_counter()
    root = Path(root)
    check(not root.exists(), 'ATTEMPT_OUTPUT_EXISTS')
    data = inputs(parent_root, parent_pin)
    authenticated = time.perf_counter()
    selected, comparisons, counts, phases = discover(data)
    if selected is None:
        root.mkdir()
        publish(root/'exhaustion.json', dict(status='ATTEMPT_NO_ELIGIBLE_PROPOSAL',
            policy=POLICY, binding=data[-1], counts=counts, comparisons=comparisons,
            code=code_pins(), phases=phases, wall_seconds=time.perf_counter()-started))
        raise ValueError('ATTEMPT_NO_ELIGIBLE_PROPOSAL')
    row, action, program = selected
    root.mkdir()
    for name,value in (('policy.json', POLICY), ('comparisons.json', comparisons),
                       ('selected.json', row)):
        publish(root/name, value)
    write_bytes(root/'action.txt', action)
    write_bytes(root/'program.py', program)
    now = time.perf_counter()
    phases = dict(authentication=authenticated-started, **phases,
                  publication_overhead=now-started-(authenticated-started)-sum(phases.values()))
    report = dict(schema='remaining-attempt-source-v1', policy=digest(POLICY), scope=POLICY,
        binding=data[-1], code=code_pins(), selected=row, candidates=len(comparisons),
        eligible_candidates=sum(not r['reasons'] for r in comparisons), counts=counts,
        files={n:dict(bytes=(root/n).stat().st_size, sha256=sha((root/n).read_bytes()))
               for n in sorted(FILES)}, phases=phases, wall_seconds=now-started)
    pin = publish(root/'report.json', report)
    write_bytes(root/'COMMITTED', pin.encode())
    return RemainingAttemptSource(root, pin)


class RemainingAttemptSource(ProposedProgram):
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        check(not self.root.is_symlink(), 'ATTEMPT_SYMLINK')
        raw = (self.root/'report.json').read_bytes()
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text(), 'ATTEMPT_PIN')
        report = json.loads(raw)
        check(raw == canonical(report)+b'\n' and report['schema'] == 'remaining-attempt-source-v1'
              and report['scope'] == POLICY and report['policy'] == digest(POLICY)
              and report['code'] == code_pins() and set(report['files']) == FILES,
              'ATTEMPT_CONTRACT')
        payloads = {}
        for name,item in report['files'].items():
            path = self.root/name
            check(not path.is_symlink(), 'ATTEMPT_SYMLINK')
            raw = path.read_bytes()
            check(dict(bytes=len(raw), sha256=sha(raw)) == item, 'ATTEMPT_BYTES')
            payloads[name] = raw
        data = inputs(**dict(root=report['binding']['parent']['root'],
                             pin=report['binding']['parent']['pin']))
        check(report['binding'] == data[-1], 'ATTEMPT_BINDING')
        row = json.loads(payloads['selected.json'])
        comparisons = json.loads(payloads['comparisons.json'])
        check(len(comparisons) == report['candidates'] and comparisons
              and comparisons == sorted(comparisons, key=lambda r:rank(r['source']))
              and sum(not r['reasons'] for r in comparisons) == report['eligible_candidates'],
              'ATTEMPT_COMPARISONS')
        eligible = [r for r in comparisons if not r['reasons']]
        check(eligible and eligible[0]['source'] == row == report['selected']
              and row['turn'] >= POLICY['first_turn'] and row['turn'] % 2 == 0
              and json.loads(payloads['policy.json']) == POLICY, 'ATTEMPT_SELECTION')
        self.report, self.row = report, row
        self.action, self.program = payloads['action.txt'], payloads['program.py']
        validate(row, self.action, self.program)
        selected, _ = choices([(row, self.action, self.program)], *data[3:6])
        check(selected is not None, 'ATTEMPT_EXPOSED_SELECTION')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = build(a.parent, a.parent_pin, a.output)
    print(json.dumps(dict(pin=result.pin, selected=result.row, counts=result.report['counts'],
        candidates=result.report['candidates'], eligible=result.report['eligible_candidates'],
        phases=result.report['phases'], wall_seconds=result.report['wall_seconds'])))
