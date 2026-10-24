"""A new proposal from a pinned catalogue, excluding the completed experience."""
from contextlib import ExitStack
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from claims.action_transfer_edit_source import parent, syntax
from claims.action_transfer_source import exposure, rank, write_bytes
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.proposed_program import ProposedProgram, validate
from senses.trajectory_population import open_bound, publish

POLICY = dict(schema='fresh-proposal-policy-v1', order='inherited_catalogue_rank',
    exclude='retained_tasks_program_substrings_syntax_and_previous_proposal_task',
    outcome_fields=[], historical_evaluation_untouched=False,
    dependency_closure_exposure_complete=False, executed=False, forecast_present=False,
    scientific_admission=False, serving=False)
FILES = {'policy.json','comparisons.json','selected.json','action.txt','program.py'}


def code_pins():
    return {'claims/fresh_proposal.py':sha(Path(__file__).read_bytes())}


def catalogue(root, pin):
    previous = ProposedProgram(root,pin)
    report, metadata = parent(**dict(root=previous.report['parent']['root'],
                                    pin=previous.report['parent']['pin']))
    tasks, messages, old = exposure(report['exposure']['root'],report['exposure']['pin'])
    rows = json.loads((previous.root/'candidates.json').read_bytes())
    check(len(rows) == previous.report['eligible_programs'] and rows == sorted(rows,key=rank),
          'FRESH_CATALOGUE_ORDER')
    tasks.add(previous.row['task'])
    messages.append(previous.action)
    excluded = set(json.loads((previous.root/'syntax-exclusions.json').read_bytes()))
    excluded.add(syntax(previous.program))
    return previous,report,metadata,rows,tasks,messages,excluded,old


def raw_candidates(source, by_group=False):
    previous, report, metadata, rows, *_ = source
    actions = {}
    with ExitStack() as stack:
        for shard in metadata['scope']['shards']:
            wanted = {r['row']:r['task'] for r in rows if r['shard'] == shard['path']}
            if not wanted:
                continue
            path = stack.enter_context(open_bound(Path(report['source_root']),shard))
            parquet = pq.ParquetFile(path)
            batches = (parquet.read_row_group(i,columns=['instance_id','trajectory'])
                       for i in range(parquet.num_row_groups)) if by_group else parquet.iter_batches(
                           batch_size=128,columns=['instance_id','trajectory'])
            offset = 0
            for batch in batches:
                addresses = sorted(k for k in wanted if offset <= k < offset+batch.num_rows)
                if addresses:
                    table = batch.take(pa.array([k-offset for k in addresses]))
                    for index,address in enumerate(addresses):
                        check(table.column('instance_id')[index].as_py() == wanted[address],
                              'FRESH_SOURCE_TASK')
                        turns = table.column('trajectory')[index].values
                        for row in rows:
                            if row['shard'] != shard['path'] or row['row'] != address:
                                continue
                            check(row['source_sha256'] == shard['sha256']
                                  and row['revision'] == metadata['scope']['revision'], 'FRESH_SOURCE_PIN')
                            turn = turns[row['turn']]
                            check(turn['role'].as_py() == 'ai', 'FRESH_SOURCE_ROLE')
                            action = turn['text'].as_py().encode()
                            left,right = row['utf8_span']
                            program = action[left:right]
                            validate(row,action,program)
                            actions[digest(row)] = action,program
                offset += batch.num_rows
                if offset > max(wanted):
                    break
    check(len(actions) == len(rows), 'FRESH_SOURCE_COVERAGE')
    return [(r,*actions[digest(r)]) for r in rows]


def choices(rows, tasks, messages, excluded):
    comparisons, selected = [],None
    for row,action,program in sorted(rows,key=lambda r:rank(r[0])):
        validate(row,action,program)
        reasons = []
        if row['task'] in tasks:
            reasons.append('EXPOSED_TASK')
        seen = [i for i,m in enumerate(messages) if program in m]
        if seen:
            reasons.append('EXPOSED_PROGRAM')
        if row['syntax_sha256'] in excluded:
            reasons.append('EXPOSED_SYNTAX')
        comparisons.append(dict(source=row,rank=rank(row)[0],reasons=reasons,exposed_messages=seen))
        if selected is None and not reasons:
            selected = row,action,program
    return selected,comparisons


def build(parent_root, parent_pin, root):
    started = time.perf_counter()
    root = Path(root)
    check(not root.exists(), 'FRESH_OUTPUT_EXISTS')
    data = catalogue(parent_root,parent_pin)
    authenticated = time.perf_counter()
    rows = raw_candidates(data)
    scanned = time.perf_counter()
    selected,comparisons = choices(rows,*data[4:7])
    check(selected is not None, 'FRESH_NO_UNEXPOSED_PROPOSAL')
    row,action,program = selected
    root.mkdir()
    for name,value in (('policy.json',POLICY),('comparisons.json',comparisons),('selected.json',row)):
        publish(root/name,value)
    write_bytes(root/'action.txt',action)
    write_bytes(root/'program.py',program)
    now = time.perf_counter()
    report = dict(schema='fresh-proposal-source-v1',parent=dict(root=str(parent_root),pin=parent_pin),
        code=code_pins(),policy=digest(POLICY),selected=row,candidates=len(rows),
        exclusions=dict(tasks=sorted(data[4]),syntax=sorted(data[6]),messages=len(data[5]),
                        original=data[7],previous_proposal=parent_pin),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},
        phases=dict(authentication=authenticated-started,raw_read=scanned-authenticated,
                    selection_publication=now-scanned),wall_seconds=now-started,scope=POLICY)
    pin = publish(root/'report.json',report)
    write_bytes(root/'COMMITTED',pin.encode())
    return FreshProposal(root,pin)


class FreshProposal(ProposedProgram):
    def __init__(self, root, pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text(), 'FRESH_PIN')
        report = json.loads(raw)
        check(raw == canonical(report)+b'\n' and report['schema'] == 'fresh-proposal-source-v1'
              and report['code'] == code_pins() and report['scope'] == POLICY
              and report['policy'] == digest(POLICY) and set(report['files']) == FILES,
              'FRESH_CONTRACT')
        payloads = {}
        for name,item in report['files'].items():
            path = self.root/name
            check(not path.is_symlink(), 'FRESH_SYMLINK')
            value = path.read_bytes()
            check(dict(bytes=len(value),sha256=sha(value)) == item, 'FRESH_BYTES')
            payloads[name] = value
        data = catalogue(report['parent']['root'],report['parent']['pin'])
        check(report['exclusions'] == dict(tasks=sorted(data[4]),syntax=sorted(data[6]),
            messages=len(data[5]),original=data[7],previous_proposal=data[0].pin), 'FRESH_EXCLUSIONS')
        row = json.loads(payloads['selected.json'])
        check(row == report['selected'] and row in data[3] and report['candidates'] == len(data[3]),
              'FRESH_SELECTION')
        self.report,self.row,self.action,self.program = report,row,payloads['action.txt'],payloads['program.py']
        selected,_ = choices([(row,self.action,self.program)],*data[4:7])
        check(selected is not None, 'FRESH_EXPOSED_SELECTION')
        check(json.loads(payloads['policy.json']) == POLICY, 'FRESH_POLICY')
