"""Select after both completed proposals; old source publications stay immutable."""
import json
from pathlib import Path
import time

from claims.action_transfer_edit_source import syntax
from claims.action_transfer_source import rank, write_bytes
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.proposed_program import ProposedProgram, validate
from claims.fresh_proposal import FreshProposal, catalogue as previous_catalogue, raw_candidates
from senses.trajectory_population import publish

POLICY = dict(schema='next-proposal-policy-v1', order='inherited_catalogue_rank',
    exclude='retained_tasks_program_substrings_syntax_and_both_completed_proposal_tasks',
    outcome_fields=[], historical_evaluation_untouched=False,
    dependency_closure_exposure_complete=False, executed=False, forecast_present=False,
    scientific_admission=False, serving=False)
FILES = {'policy.json','comparisons.json','selected.json','action.txt','program.py'}


def code_pins():
    return {n:sha((Path(__file__).parent/n).read_bytes()) for n in
            ('next_proposal.py','fresh_proposal.py')}


def catalogue(root, pin):
    previous = FreshProposal(root,pin)
    _,report,metadata,rows,tasks,messages,excluded,old = previous_catalogue(
        previous.report['parent']['root'],previous.report['parent']['pin'])
    tasks.add(previous.row['task'])
    messages.append(previous.action)
    excluded.add(syntax(previous.program))
    return previous,report,metadata,rows,tasks,messages,excluded,old


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
    check(not root.exists(), 'NEXT_OUTPUT_EXISTS')
    data = catalogue(parent_root,parent_pin)
    authenticated = time.perf_counter()
    rows = raw_candidates(data)
    scanned = time.perf_counter()
    selected,comparisons = choices(rows,*data[4:7])
    check(selected is not None, 'NEXT_NO_UNEXPOSED_PROPOSAL')
    row,action,program = selected
    root.mkdir()
    for name,value in (('policy.json',POLICY),('comparisons.json',comparisons),('selected.json',row)):
        publish(root/name,value)
    write_bytes(root/'action.txt',action)
    write_bytes(root/'program.py',program)
    now = time.perf_counter()
    report = dict(schema='next-proposal-source-v1',parent=dict(root=str(parent_root),pin=parent_pin),
        code=code_pins(),policy=digest(POLICY),selected=row,candidates=len(rows),
        exclusions=dict(tasks=sorted(data[4]),syntax=sorted(data[6]),messages=len(data[5]),
                        original=data[7],previous_proposal=parent_pin),
        files={n:dict(bytes=(root/n).stat().st_size,sha256=sha((root/n).read_bytes())) for n in FILES},
        phases=dict(authentication=authenticated-started,raw_read=scanned-authenticated,
                    selection_publication=now-scanned),wall_seconds=now-started,scope=POLICY)
    pin = publish(root/'report.json',report)
    write_bytes(root/'COMMITTED',pin.encode())
    return NextProposal(root,pin)


class NextProposal(ProposedProgram):
    def __init__(self, root, pin):
        self.root,self.pin = Path(root),pin
        raw = (self.root/'report.json').read_bytes()
        check(sha(raw) == pin == (self.root/'COMMITTED').read_text(), 'NEXT_PIN')
        report = json.loads(raw)
        check(raw == canonical(report)+b'\n' and report['schema'] == 'next-proposal-source-v1'
              and report['code'] == code_pins() and report['scope'] == POLICY
              and report['policy'] == digest(POLICY) and set(report['files']) == FILES,
              'NEXT_CONTRACT')
        payloads = {}
        for name,item in report['files'].items():
            path = self.root/name
            check(not path.is_symlink(), 'NEXT_SYMLINK')
            value = path.read_bytes()
            check(dict(bytes=len(value),sha256=sha(value)) == item, 'NEXT_BYTES')
            payloads[name] = value
        data = catalogue(report['parent']['root'],report['parent']['pin'])
        check(report['exclusions'] == dict(tasks=sorted(data[4]),syntax=sorted(data[6]),
            messages=len(data[5]),original=data[7],previous_proposal=data[0].pin), 'NEXT_EXCLUSIONS')
        row = json.loads(payloads['selected.json'])
        check(row == report['selected'] and row in data[3] and report['candidates'] == len(data[3]),
              'NEXT_SELECTION')
        self.report,self.row,self.action,self.program = report,row,payloads['action.txt'],payloads['program.py']
        selected,_ = choices([(row,self.action,self.program)],*data[4:7])
        check(selected is not None, 'NEXT_EXPOSED_SELECTION')
        check(json.loads(payloads['policy.json']) == POLICY, 'NEXT_POLICY')
