"""Retain changing source proposals without treating reported replies as outcomes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.proposed_program import validate
from claims.tool_temporal_source import scope
from senses.trajectory_population import open_bound, publish


SCHEMA = 'action-trajectory-census-v1'
POLICY = dict(selection='all_catalogued_attempts_with_distinct_program_bytes',
    ordering='shard_row_then_turn_and_span', outcome_used_for_selection=False,
    clock='recorded_turn_order_only', proposed_programs_executed=False,
    historical_occurrence_established=False, task_success_established=False,
    scientific_admission=False, serving=False, partition='development_inspection')


def pins():
    root = Path(__file__).resolve().parents[1]
    return {n: sha((root/n).read_bytes()) for n in (
        'claims/action_trajectory_census.py', 'claims/proposed_program.py',
        'claims/tool_temporal_source.py', 'senses/trajectory_population.py')}


def catalogue(root, pin):
    raw = (root/'report.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(), 'ACTION_CENSUS_PARENT_PIN')
    report = json.loads(raw)
    check(raw == canonical(report)+b'\n' and report['schema'] == 'remaining-attempt-source-v1',
          'ACTION_CENSUS_PARENT_SCHEMA')
    value = (root/'comparisons.json').read_bytes()
    check(dict(bytes=len(value), sha256=sha(value)) == report['files']['comparisons.json'],
          'ACTION_CENSUS_CATALOGUE_PIN')
    rows = json.loads(value)
    check(len(rows) == report['candidates'], 'ACTION_CENSUS_COUNT')
    groups = defaultdict(list)
    for item in rows:
        r = item['source']
        groups[r['shard'], r['row']].append(item)
    selected = {k: sorted(v, key=lambda x: (x['source']['turn'], x['source']['utf8_span']))
                for k, v in groups.items()
                if len({x['source']['program_sha256'] for x in v}) > 1}
    counts = dict(occurrences=len(rows), programs=len({x['source']['program_sha256'] for x in rows}),
        syntax=len({x['source']['syntax_sha256'] for x in rows}), attempts=len(groups),
        tasks=len({x['source']['task'] for x in rows}), changing_attempts=len(selected),
        selected_occurrences=sum(map(len, selected.values())))
    return report, selected, counts


def write(root, name, raw, files):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    files[name] = dict(bytes=len(raw), sha256=sha(raw))
    return name


def build(parent, parent_pin, root):
    start = time.perf_counter()
    check(not root.exists(), 'ACTION_CENSUS_OUTPUT_EXISTS')
    prior, selected, counts = catalogue(parent, parent_pin)
    raw_root = Path(prior['binding']['prior_binding']['source_root'])
    source_scope = scope()
    check(set(k[0] for k in selected) <= {s['path'] for s in source_scope['shards']},
          'ACTION_CENSUS_SHARD_SCOPE')
    root.mkdir()
    files, chains = {}, []
    hashing, reading, group_count = 0.0, 0.0, 0
    for shard in source_scope['shards']:
        addresses = sorted(k[1] for k in selected if k[0] == shard['path'])
        tick = time.perf_counter()
        with open_bound(raw_root, shard) as stream:
            hashing += time.perf_counter()-tick
            tick = time.perf_counter()
            parquet = pq.ParquetFile(stream)
            check(parquet.metadata.num_rows == shard['rows'], 'ACTION_CENSUS_RAW_ROWS')
            offset = 0
            for group in range(parquet.num_row_groups):
                length = parquet.metadata.row_group(group).num_rows
                wanted = [r for r in addresses if offset <= r < offset+length]
                if wanted:
                    table = parquet.read_row_group(group, columns=['instance_id', 'trajectory'])
                    group_count += 1
                    for address in wanted:
                        task = table.column('instance_id')[address-offset].as_py()
                        turns = table.column('trajectory')[address-offset].as_py()
                        entries = selected[shard['path'], address]
                        check(all(x['source']['task'] == task for x in entries), 'ACTION_CENSUS_TASK')
                        identity = dict(task=task, shard=shard['path'], row=address,
                            revision=source_scope['revision'], source_sha256=shard['sha256'])
                        name = digest(identity)
                        messages = []
                        for turn, message in enumerate(turns):
                            text = message['text']
                            check(text is None or isinstance(text, str), 'ACTION_CENSUS_MESSAGE')
                            path = (None if text is None else
                                    write(root, f'{name}/turn-{turn:04}.txt', text.encode(), files))
                            messages.append(dict(turn=turn, role=message['role'], file=path,
                                text_present=text is not None,
                                source_fields={k:v for k,v in message.items() if k != 'text'},
                                origin='recorded_source_report', observed_occurrence=False))
                        programs = []
                        for item in entries:
                            r = item['source']
                            check(r['source_sha256'] == shard['sha256'] and
                                  r['revision'] == source_scope['revision'], 'ACTION_CENSUS_REVISION')
                            message = turns[r['turn']]
                            check(message['role'] == 'ai', 'ACTION_CENSUS_ROLE')
                            action = message['text'].encode()
                            lo, hi = r['utf8_span']
                            program = action[lo:hi]
                            validate(r, action, program)
                            path = f'programs/{sha(program)}.py'
                            if path not in files:
                                write(root, path, program, files)
                            programs.append(dict(source=r, program=path, catalogue_rank=item['rank'],
                                prior_exclusions=item['reasons'], exposed_messages=item['exposed_messages']))
                        chains.append(dict(id=name, **identity, messages=messages, programs=programs))
                offset += length
            reading += time.perf_counter()-tick
    chains.sort(key=lambda x: (x['shard'], x['row']))
    check(len(chains) == counts['changing_attempts'], 'ACTION_CENSUS_COVERAGE')
    chain_pin = publish(root/'chains.json', chains)
    files['chains.json'] = dict(bytes=(root/'chains.json').stat().st_size, sha256=chain_pin)
    counts.update(selected_tasks=len({x['task'] for x in chains}),
        selected_programs=len({p['program'] for x in chains for p in x['programs']}),
        message_records=sum(len(x['messages']) for x in chains),
        message_files=sum(m['text_present'] for x in chains for m in x['messages']),
        missing_text_messages=sum(not m['text_present'] for x in chains for m in x['messages']),
        program_bytes=sum(v['bytes'] for k,v in files.items() if k.startswith('programs/')),
        message_bytes=sum(v['bytes'] for k,v in files.items() if '/turn-' in k),
        row_groups_read=group_count)
    wall = time.perf_counter()-start
    report = dict(schema=SCHEMA, policy=POLICY, code=pins(), counts=counts, files=files,
        source_scope=source_scope, source_root=str(raw_root),
        parent=dict(root=str(parent), pin=parent_pin),
        phases=dict(raw_hash=hashing, selected_row_read_and_write=reading,
                    other=wall-hashing-reading), wall_seconds=wall)
    pin = publish(root/'report.json', report)
    write(root, 'COMMITTED', pin.encode(), {})
    return dict(root=str(root), pin=pin, counts=counts, wall_seconds=wall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--parent-pin', required=True)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.parent_pin, args.root), sort_keys=True))
