"""Durable post-generation scoring. A complete sealed grid precedes private reads."""
import fcntl
import json
from pathlib import Path
import time

from claims.process_conditions import write_committed
from model.complete_answer_cohort import case_authority,read
from model.complete_answer_comparison import compare,grid
from model.complete_answer_evaluation import public_inputs
from model.complete_answer_summary import Summary
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def authority(root, pin, pack, scope):
    r = read(root/'report.json',pin); request = read(root.parent/'request.json',r['request'])
    need(r['complete'] and r['private_reference_received'] is False and r['optimizer_updates'] == 0
         and request['schema'] == 'public-pack-answer-cohort-v1'
         and r['pack'] == request['pack'] == dict(path=str(pack.root),pin=pack.pin), 'COHORT_SCORE_PUBLIC_AUTHORITY')
    b = read(root/'run/binding.json',r['binding'])
    ordinals = request['ordinals']; cases = pack.cases if ordinals is None else [pack.cases[i] for i in ordinals]
    need(b['cases'] == cases and b['authority'] == request['authority'] and scope in ('proof','all'), 'COHORT_SCORE_CASE_BINDING')
    need(scope != 'all' or cases == pack.cases, 'COHORT_SCORE_INCOMPLETE_FULL_SCOPE')
    actual = [(v['ordinal'],v['document']) for v in r['results']]
    need(actual == [(v['ordinal'],v['document']) for v in cases], 'COHORT_SCORE_COVERAGE')
    for n,v in b['checkpoints'].items():
        need(v['checkpoint'] == request['checkpoints'][n]['reference']['pin']
             and v['state'] == request['checkpoints'][n]['reference']['state'], 'COHORT_SCORE_CHECKPOINT')
    return r,b,cases


def sealed(root, result, row, binding, public, roster):
    need(result['path'] == f'case-{row["ordinal"]:06d}', 'COHORT_SCORE_CASE_PATH')
    path = root/'run'/result['path']; c = read(path/'case.json',result['pin'])
    _,pins = public_inputs(public,roster); a = case_authority(binding['authority'],row)
    expected = dict(case=row,authority=a,checkpoints=binding['checkpoints'],public=pins)
    need(c['complete'] and c['binding'] == expected and c['status'] == result['status'], 'COHORT_SCORE_CASE_SEAL')
    if c['status'] != 'generated':
        need(c['status'] == 'CONTROL_UNAVAILABLE_NATIVE_IDENTICAL' and c['generation'] is None
             and pins['correct']['prefix'] == pins['train_donor']['prefix'], 'COHORT_SCORE_GAP')
        return None,pins,a
    ref = c['generation']; need(Path(ref['path']).name == ref['path'], 'COHORT_SCORE_GENERATION_PATH')
    path = path/ref['path']; g = read(path/'report.json',ref['pin'])
    need(g['complete'] and g['reference_length_received'] is False and g['optimizer_updates'] == 0
         and g['checkpoints'] == binding['checkpoints'], 'COHORT_SCORE_GENERATION')
    traces = [dict(v,path=str(path/v['path'])) for v in g['traces']]
    proposals = dict(shared=['assembled'],own_slot=['assembled'],whole_proposals=list(roster))
    checked,models = grid(traces,expected=proposals,public=pins,roster=roster,authority=a)
    for n,v in models.items():need(v['model'] == binding['checkpoints'][n]['model'], 'COHORT_SCORE_MODEL')
    # Corrupt/unsealed trajectories must refuse before any private reference read.
    for v in checked:
        for event in replay(v['path'],v['pin']):del event
    return traces,pins,a


def run(root, pin, pack, reference, *, output, code, source, scope, after_case=None):
    root,output = Path(root),Path(output); output.mkdir(exist_ok=True)
    began = tick = time.perf_counter(); report,b,cases = authority(root,pin,pack,scope)
    identity = dict(schema='complete-answer-cohort-scores-v1',cohort=pin,pack=pack.pin,
        source=source,code=code,scope=scope,ordinals=[v['ordinal'] for v in cases])
    summary = Summary(); collections = {}; actions = dict(scored=0,reused=0,gap=0); refs=[]; gaps=[]
    phases = dict(authority=time.perf_counter()-tick,public_and_seals=0.,reference=0.,score=0.,persist=0.,summary=0.)
    with (output/'owner.lock').open('a+') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('COHORT_SCORE_OWNER_BUSY') from exc
        path = output/'binding.json'
        if path.exists():need(json.loads(path.read_bytes()) == identity, 'COHORT_SCORE_CHANGED_BINDING')
        else:write_committed(path,identity)
        binding_pin = file_sha(path)
        for row,result in zip(cases,report['results'],strict=True):
            tick = time.perf_counter(); ordinal = row['ordinal']; dest = output/f'case-{ordinal:06d}.json'
            if dest.exists():
                saved = json.loads(dest.read_bytes())
                need(saved['binding'] == binding_pin and saved['cohort_case'] == result['pin']
                     and saved['ordinal'] == ordinal and saved['document'] == row['document'], 'COHORT_SCORE_SAVED_CASE')
                need(Path(saved['payload']['path']).name == saved['payload']['path'], 'COHORT_SCORE_PAYLOAD_PATH')
                payload = read(output/saved['payload']['path'],saved['payload']['pin'])
                actions['reused'] += 1
                phases['persist'] += time.perf_counter()-tick
            else:
                public = pack.load(row['public']); traces,pins,a = sealed(root,result,row,b,public,pack.roster)
                phases['public_and_seals'] += time.perf_counter()-tick; tick = time.perf_counter()
                if traces is None:
                    payload = dict(status=result['status'],ordinal=ordinal,document=row['document'])
                    actions['gap'] += 1
                else:
                    private = reference(ordinal,public)
                    phases['reference'] += time.perf_counter()-tick; tick = time.perf_counter()
                    payload = compare(traces,expected=dict(shared=['assembled'],own_slot=['assembled'],whole_proposals=list(pack.roster)),
                        public=pins,authority=a,identity=private['identity'],future=private['future'],
                        roster=pack.roster,copy_frame=public['correct']['frames'][-1])
                    del private; actions['scored'] += 1
                phases['score'] += time.perf_counter()-tick; tick = time.perf_counter()
                # Incomplete attempts remain immutable; only the case commit makes a score current.
                attempt = 0
                while (output/f'score-{ordinal:06d}-{attempt:06d}.json').exists():attempt += 1
                target = output/f'score-{ordinal:06d}-{attempt:06d}.json'
                payload_pin = write_committed(target,payload)
                saved = dict(binding=binding_pin,cohort_case=result['pin'],ordinal=ordinal,document=row['document'],
                             payload=dict(path=target.name,pin=payload_pin))
                write_committed(dest,saved); phases['persist'] += time.perf_counter()-tick
                del public
                if after_case:after_case(ordinal)
            tick = time.perf_counter()
            refs.append(dict(path=dest.name,pin=file_sha(dest),ordinal=ordinal))
            if 'status' in payload:
                need(payload == dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',ordinal=ordinal,document=row['document'])
                     and result['status'] == payload['status'], 'COHORT_SCORE_SAVED_GAP')
                gaps.append(payload)
            else:
                meta=pack.rows[ordinal]; ident=payload['identity']
                need(ident['document'] == row['document'] and ident['source'] == fingerprint(source)
                     and ident['side'] == ('train','development','query')[int(meta['side'])]
                     and ident['component'] == int(meta['component'])
                     and ident['collection'] == source['collections'][int(meta['collection'])], 'COHORT_SCORE_PAYLOAD_DOCUMENT')
                summary.add(payload)
                collection = payload['identity']['collection']; collections.setdefault(collection,Summary()).add(payload)
            del payload; phases['summary'] += time.perf_counter()-tick
        tick = time.perf_counter()
        result = dict(complete=True,binding=binding_pin,cohort=pin,pack=pack.pin,scope=scope,
            full_population=cases == pack.cases,declared_cases=len(cases),scored_cases=len(summary.seen),
            control_unavailable=gaps,all_declared_cases_accounted=len(summary.seen)+len(gaps)==len(cases),
            cases=refs,actions=actions,summary=summary.finish() if summary.seen else None,
            by_collection={k:v.finish() for k,v in sorted(collections.items())},
            summary_metric_groups=len(summary.groups)+len(summary.pairs),
            no_metric_rows_retained=True,architecture_winner=False,semantic_validity_established=False)
        need(result['all_declared_cases_accounted'] and file_sha(output/'binding.json') == binding_pin,
             'COHORT_SCORE_FINAL_COVERAGE')
        phases['summary'] += time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'COHORT_SCORE_PHASES')
        result.update(phases_seconds=phases,seconds=seconds)
        return result
