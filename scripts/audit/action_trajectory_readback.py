"""Independent batch-source readback of the action-change census and its scope."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from senses.trajectory_population import open_bound, publish


def expected_chains(parent, pin):
    raw = (parent/'report.json').read_bytes()
    check(sha(raw) == pin == (parent/'COMMITTED').read_text(), 'ACTION_READ_PARENT')
    report = json.loads(raw)
    raw = (parent/'comparisons.json').read_bytes()
    check(dict(bytes=len(raw), sha256=sha(raw)) == report['files']['comparisons.json'],
          'ACTION_READ_CATALOGUE')
    groups = defaultdict(list)
    for item in json.loads(raw):
        row = item['source']
        groups[row['shard'], row['row']].append(item)
    return {k: sorted(v, key=lambda x: (x['source']['turn'], x['source']['utf8_span']))
            for k,v in groups.items() if len({r['source']['program_sha256'] for r in v}) > 1}


def check_chains(chains, expected, originals, payloads):
    check([(c['shard'], c['row']) for c in chains] == sorted(expected), 'ACTION_READ_MEMBERSHIP')
    for chain in chains:
        key = chain['shard'], chain['row']
        task, turns = originals[key]
        check(chain['task'] == task and all(r['source']['task'] == task for r in expected[key]),
              'ACTION_READ_TASK')
        identity = {k:chain[k] for k in ('task','shard','row','revision','source_sha256')}
        check(chain['id'] == digest(identity), 'ACTION_READ_ID')
        check(len(chain['messages']) == len(turns), 'ACTION_READ_MESSAGES')
        for i,(message, original) in enumerate(zip(chain['messages'], turns)):
            check(message['turn'] == i and message['role'] == original['role'] and
                  message['source_fields'] == {k:v for k,v in original.items() if k != 'text'},
                  'ACTION_READ_SOURCE_MESSAGE')
            if original['text'] is None:
                check(message['text_present'] is False and message['file'] is None,
                      'ACTION_READ_ABSENT_TEXT')
            else:
                check(message['text_present'] is True and
                      payloads[message['file']] == original['text'].encode(), 'ACTION_READ_TEXT')
            check(message['origin'] == 'recorded_source_report' and message['observed_occurrence'] is False,
                  'ACTION_READ_SCOPE')
        check([p['source'] for p in chain['programs']] == [x['source'] for x in expected[key]],
              'ACTION_READ_PROPOSALS')
        for p,item in zip(chain['programs'], expected[key]):
            row = p['source']
            check(all(row[k] == chain[k] for k in ('task','shard','row','revision','source_sha256')),
                  'ACTION_READ_SOURCE_BINDING')
            check(p['catalogue_rank'] == item['rank'] and p['prior_exclusions'] == item['reasons']
                  and p['exposed_messages'] == item['exposed_messages'], 'ACTION_READ_EXPOSURE')
            action = turns[row['turn']]['text'].encode()
            raw = payloads[p['program']]
            lo,hi = row['utf8_span']
            check(0 <= lo < hi <= len(action) and raw == action[lo:hi], 'ACTION_READ_SPAN')
            check(sha(action) == row['action_sha256'] and len(action) == row['action_bytes'] and
                  sha(raw) == row['program_sha256'] and len(raw) == row['program_bytes'],
                  'ACTION_READ_PROGRAM_BYTES')
            check(row['historical_occurrence_established'] is False and
                  row['historical_file_state_established'] is False, 'ACTION_READ_PROPOSAL_SCOPE')


def run(root, pin, output):
    start = time.perf_counter()
    raw = (root/'report.json').read_bytes()
    check(sha(raw) == pin == (root/'COMMITTED').read_text(), 'ACTION_READ_PIN')
    report = json.loads(raw)
    check(report['schema'] == 'action-trajectory-census-v1' and raw == canonical(report)+b'\n',
          'ACTION_READ_SCHEMA')
    check(all(report['policy'][k] is False for k in ('outcome_used_for_selection',
        'proposed_programs_executed', 'historical_occurrence_established',
        'task_success_established', 'scientific_admission', 'serving')), 'ACTION_READ_REPORT_SCOPE')
    payloads = {}
    for name,item in report['files'].items():
        path = root/name
        check(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),
              'ACTION_READ_PATH')
        data = path.read_bytes()
        check(dict(bytes=len(data), sha256=sha(data)) == item, 'ACTION_READ_ARTIFACT')
        payloads[name] = data
    expected = expected_chains(Path(report['parent']['root']), report['parent']['pin'])
    originals = {}
    for shard in report['source_scope']['shards']:
        with open_bound(Path(report['source_root']), shard) as stream:
            parquet = pq.ParquetFile(stream)
            check(parquet.metadata.num_rows == shard['rows'], 'ACTION_READ_ROWS')
            offset = 0
            for batch in parquet.iter_batches(batch_size=128, columns=['instance_id','trajectory']):
                for address in sorted(k[1] for k in expected if k[0] == shard['path']
                                      and offset <= k[1] < offset+batch.num_rows):
                    i = address-offset
                    originals[shard['path'],address] = (batch.column('instance_id')[i].as_py(),
                                                        batch.column('trajectory')[i].as_py())
                offset += batch.num_rows
    chains = json.loads(payloads['chains.json'])
    check_chains(chains, expected, originals, payloads)
    controls = {}
    for name, alter in (
        ('omitted_change', lambda x:x[0]['programs'].pop()),
        ('changed_task', lambda x:x[0].update(task='wrong-task')),
        ('changed_span', lambda x:x[0]['programs'][0]['source']['utf8_span'].__setitem__(0,0)),
        ('reported_as_observed', lambda x:x[0]['messages'][0].update(observed_occurrence=True))):
        changed = copy.deepcopy(chains)
        alter(changed)
        try:
            check_chains(changed, expected, originals, payloads)
        except ValueError as exc:
            controls[name] = str(exc)
        else:
            raise ValueError('ACTION_READ_CONTROL_ACCEPTED:'+name)
    scripts = Path(__file__).resolve().parents[1]
    for name,p in report['code'].items():
        check(sha((scripts/name).read_bytes()) == p, 'ACTION_READ_CODE')
    counts = dict(chains=len(chains), tasks=len({c['task'] for c in chains}),
        programs=len({p['program'] for c in chains for p in c['programs']}),
        occurrences=sum(len(c['programs']) for c in chains),
        messages=sum(len(c['messages']) for c in chains), files=len(payloads))
    for actual,key in [('chains','changing_attempts'),('tasks','selected_tasks'),
                       ('programs','selected_programs'),('occurrences','selected_occurrences'),
                       ('messages','message_records')]:
        check(counts[actual] == report['counts'][key], 'ACTION_READ_COUNTS')
    result = dict(schema='action-trajectory-readback-v1', root=str(root), pin=pin,
        counts=counts, controls=controls, source_rows_independently_equal=True,
        scope=report['policy'], code_sha256=sha(Path(__file__).read_bytes()),
        wall_seconds=time.perf_counter()-start, acceptance='author-checked only')
    result_pin = publish(output, result)
    return dict(path=str(output), pin=result_pin, **counts, controls=controls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.pin, args.output), sort_keys=True))
