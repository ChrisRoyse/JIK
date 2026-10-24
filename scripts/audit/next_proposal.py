"""Read raw proposal addresses separately and exercise the selection boundaries."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.next_proposal import NextProposal, build, catalogue, choices, raw_candidates
from claims.proposed_program import validate
from senses.trajectory_population import publish


def read(root,pin):
    started = time.perf_counter()
    source = NextProposal(root,pin)
    data = catalogue(**dict(root=source.report['parent']['root'],pin=source.report['parent']['pin']))
    authenticated = time.perf_counter()
    rows = raw_candidates(data,by_group=True)
    scanned = time.perf_counter()
    selected,comparisons = choices(rows,*data[4:7])
    check(selected == (source.row,source.action,source.program)
          and comparisons == json.loads((root/'comparisons.json').read_bytes()), 'NEXT_RECOUNT')
    controls = {}
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as e:
            check(code in str(e), 'NEXT_CONTROL_CODE:'+name)
        else:
            raise AssertionError('NEXT_CONTROL_ACCEPTED:'+name)
        controls[name] = code
    refuse('changed_pin','NEXT_PIN',lambda:NextProposal(root,'0'*64))
    bad = copy.deepcopy(source.row)
    bad['utf8_span'][0] += 1
    refuse('changed_span','PROPOSAL_SPAN',lambda:validate(bad,source.action,source.program))
    refuse('changed_program','PROPOSAL_SPAN',lambda:validate(source.row,source.action,source.program+b' '))
    def excluded(tasks,messages,syntax,reason):
        chosen,comparison = choices([selected],tasks,messages,syntax)
        check(chosen is None and reason in comparison[0]['reasons'], 'NEXT_EXCLUSION_CONTROL')
        controls[reason] = 'rejected_before_selection'
    excluded(data[4]|{source.row['task']},data[5],data[6],'EXPOSED_TASK')
    excluded(data[4],data[5]+[source.program],data[6],'EXPOSED_PROGRAM')
    excluded(data[4],data[5],data[6]|{source.row['syntax_sha256']},'EXPOSED_SYNTAX')
    check(NextProposal(root,pin).report == source.report, 'NEXT_MUTATION')
    end = time.perf_counter()
    return dict(schema='next-proposal-readback-v1',acceptance='author-checked',number_status='measured',
        source=source.reference(),source_pin=pin,candidates=len(rows),
        distinct_programs=len({r[0]['program_sha256'] for r in rows}),controls=controls,
        selected_rank=next(i for i,c in enumerate(comparisons) if not c['reasons']),
        boundaries=['pinned_catalogue_to_raw_spans','exposure_to_selection','persisted_source_to_separate_row_group_readback'],
        phases=dict(authentication=authenticated-started,raw_read=scanned-authenticated,checks=end-scanned),
        wall_seconds=end-started,code_sha256=sha(Path(__file__).read_bytes()),
        executed=False,forecast_present=False,scientific_admission=False,serving=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('select','read'),required=True)
    for n in ('root','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True)
    a = p.parse_args()
    if a.stage == 'select':
        result = build(a.root,a.pin,a.output)
        print(json.dumps(dict(pin=result.pin,selected=result.row,phases=result.report['phases'])),flush=True)
    else:
        result = read(a.root,a.pin)
        pin = publish(a.output,result)
        print(json.dumps(dict(pin=pin,**result)),flush=True)
