"""Public-only, autonomous original-task transfer; references enter only scoring.

All QUERY tasks remain in the schedule, including those without a matched TRAIN
donor. Native windows stay typed. Reversal holds recipient request features fixed.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from claims.process_conditions import write_committed
from model.complete_answer_evaluation import ARCHITECTURES
from model.complete_answer_trace import collect, replay
from model.complete_answer_comparison import trace_score
from model.native_population_training import fingerprint
from text.constellation_corpus import canonical, digest, file_sha, need

POLICY = dict(query_status='previously_exposed_development', donor_split='train',
    donor_group='different', donor_selection='hash_order_public_structure_and_window_counts',
    reversal='reverse_frames_keep_recipient_request_features', references_in_generation=False,
    reference_length_in_generation=False, best_of_proposals=False, optimizer_updates=0)


def structure(pack, row):
    d = row['descriptor']
    fields = [{k:v for k,v in f.items() if k != 'clock'} for f in d['fields']]
    counts = [[int(pack.offsets[n][pack.index[f['input']]+1]-pack.offsets[n][pack.index[f['input']]])
               for n in pack.roster] for f in row['fields']]
    return dict(family=row['family'], operation=d['operation'], position_kind=d['position_kind'],
                fields=fields, windows=counts)


def schedule(pack):
    buckets = {}
    for row in pack.cases:
        if row['split'] == 'train':
            buckets.setdefault(digest(canonical(structure(pack,row))), []).append(row)
    rows = []
    for row in pack.cases:
        if row['split'] != 'query': continue
        signature = digest(canonical(structure(pack,row)))
        candidates = [r for r in buckets.get(signature,[]) if r['group'] != row['group'] and
                      [f['input'] for f in r['fields']] != [f['input'] for f in row['fields']]]
        donor = min(candidates,key=lambda r:digest(canonical(['task-transfer-donor-v1',row['task'],r['task']]))) if candidates else None
        rows.append(dict(ordinal=len(rows),task=row['task'],public=row['pin'],family=row['family'],
            group=row['group'],population=row['population'],structure=signature,
            donor=None if donor is None else dict(task=donor['task'],public=donor['pin']),
            donor_candidates=len(candidates),seed=int(digest(canonical(['task-transfer-noise-v1',row['task']]))[:15],16)))
    need(rows, 'TASK_TRANSFER_EMPTY')
    return dict(schema='task-answer-transfer-schedule-v1',pack=pack.pin,policy=POLICY,rows=rows,
        summary=dict(query=len(rows),with_donor=sum(r['donor'] is not None for r in rows),
                     families={f:sum(r['family']==f for r in rows) for f in sorted({r['family'] for r in rows})}))


def inputs(pack, row):
    public = pack.by_task.get(row['task'])
    need(public is not None and public['split']=='query' and public['pin']==row['public'], 'TASK_TRANSFER_QUERY')
    value = pack.load(row['task'],row['public']); frames = value['context']; request = value['request']
    modes = dict(correct=frames,reversed=frames[::-1]); unavailable = {}
    donor = row['donor']
    if donor is None:
        unavailable['train_donor'] = 'NO_MATCHED_TRAIN_SOURCE_GROUP'
    else:
        other = pack.by_task.get(donor['task'])
        need(other is not None and other['split']=='train' and other['group']!=public['group']
             and other['pin']==donor['public'] and structure(pack,other)==structure(pack,public),
             'TASK_TRANSFER_DONOR')
        alternate = pack.load(other['task'],other['pin'])['context']
        if fingerprint(alternate)==fingerprint(frames):
            unavailable['train_donor'] = 'MATCHED_TRAIN_NATIVE_IDENTICAL'
        else:modes['train_donor'] = alternate
    pins = {mode:dict(prefix=fingerprint(v),request=request.pin,descriptor=json.loads(request.descriptor))
            for mode,v in modes.items()}
    return modes,request,pins,unavailable


def checkpoint_set(loaded, target):
    need(set(loaded)==set(ARCHITECTURES), 'TASK_TRANSFER_ARCHITECTURES')
    expected = {a:list(loaded[a].model.proposals) for a in ARCHITECTURES}
    need(expected==dict(shared=['assembled'],own_slot=['assembled'],whole_proposals=list(target.roster)),
         'TASK_TRANSFER_PROPOSALS')
    first = loaded['shared']
    need(all(v.identity['step']==first.identity['step'] and v.identity['counts']==first.identity['counts']
         and v.identity['target']==first.identity['target'] and all(v.expected['binding'][k]==first.expected['binding'][k]
         for k in ('source','schedule')) for v in loaded.values()), 'TASK_TRANSFER_UNMATCHED_CHECKPOINTS')
    return expected


def check_case(root, pin, *, row, plan_pin, identities, expected, public, unavailable, authority, target_state, caches):
    root = Path(root)
    need(file_sha(root/'report.json')==pin and not (root/'failure.json').exists(), 'TASK_TRANSFER_CASE_PIN')
    r = json.loads((root/'report.json').read_bytes())
    need(r['complete'] and r['schema']=='task-answer-transfer-case-v1' and r['case']==row
         and r['plan']==plan_pin and r['checkpoints']==identities and r['expected']==expected
         and r['public']==public and r['unavailable']==unavailable and r['authority']==authority
         and r['target_state']==target_state and r['cache_bytes']==caches
         and r['reference_length_received'] is False and r['private_reference_received'] is False,
         'TASK_TRANSFER_CASE_BINDING')
    keys = {(a,p,m) for a,ps in expected.items() for p in ps for m in public}
    actual = [(v['architecture'],v['proposal'],v['mode']) for v in r['traces']]
    need(len(actual)==len(set(actual)) and set(actual)==keys, 'TASK_TRANSFER_GRID')
    for v in r['traces']:
        path = root/v['path']; need(path.resolve().is_relative_to(root.resolve()), 'TASK_TRANSFER_PATH')
        events = sum(1 for _ in replay(path,v['pin']))
        trace = json.loads((path/'report.json').read_bytes()); q = json.loads((path/'request.json').read_bytes())
        selected = public[v['mode']]
        need(q['authority']==authority and q['seed']==row['seed'] and q['context']==r['context']
             and q['model']==identities[v['architecture']]['model'] and q['proposal']==v['proposal']
             and q['target']==target_state and q['cache_bytes']==caches[v['architecture']]
             and q['roster']==r['roster'] and q['reference_visible'] is False
             and q['prefix']==selected['prefix'] and q['request']==selected['request']
             and q['request_descriptor']==selected['descriptor'] and events==trace['events'],
             'TASK_TRANSFER_TRACE_BINDING')
    return r


def run(loaded, pack, plan, *, target, output, authority, ordinals=None, context='complete-task-answer'):
    """No private source, target answer, reference length, or endpoint argument."""
    import fcntl
    output = Path(output); output.mkdir(exist_ok=True); began = time.perf_counter()
    need(plan==schedule(pack), 'TASK_TRANSFER_SCHEDULE')
    need(tuple(target.roster)==pack.roster and isinstance(context,str) and context, 'TASK_TRANSFER_INPUT_CONTRACT')
    plan_pin = digest(canonical(plan)); expected = checkpoint_set(loaded,target)
    identities = {a:v.identity for a,v in loaded.items()}; frozen = fingerprint(target.state_dict())
    selected = list(range(len(plan['rows']))) if ordinals is None else list(ordinals)
    need(selected and selected==sorted(set(selected)) and all(type(i) is int and 0<=i<len(plan['rows']) for i in selected),
         'TASK_TRANSFER_SELECTION')
    caches = {a:v.model.forecaster.predictor.cache_bytes for a,v in loaded.items()}
    binding = dict(plan=plan_pin,checkpoints=identities,expected=expected,authority=authority,ordinals=selected,
                   context=context,roster=list(target.roster),frozen=frozen,cache_bytes=caches)
    results = []; phases = dict(prepare=time.perf_counter()-began,inputs=0.,generation=0.,readback=0.,seal=0.)
    with (output/'owner.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        bp = output/'binding.json'
        if bp.exists():need(json.loads(bp.read_bytes())==binding, 'TASK_TRANSFER_RESUME_BINDING')
        else:write_committed(bp,binding)
        for i in selected:
            tick = time.perf_counter(); row = plan['rows'][i]
            modes,request,pins,unavailable = inputs(pack,row)
            case_authority = dict(authority,plan=plan_pin,task=row['task'],public=row['public'],ordinal=i)
            path = output/f'case-{i:06d}'; phases['inputs'] += time.perf_counter()-tick; tick = time.perf_counter()
            if (path/'report.json').exists():
                pin = file_sha(path/'report.json'); action = 'reused'
            else:
                need(not path.exists(), 'TASK_TRANSFER_INCOMPLETE_CASE_REQUIRES_READBACK'); path.mkdir()
                traces = []
                try:
                    for a in ARCHITECTURES:
                        checkpoint = loaded[a]; checkpoint.fresh()
                        for mode,frames in modes.items():
                            for j,proposal in enumerate(expected[a]):
                                dest = path/f'{a}-{mode}-{j}'
                                pin = collect(checkpoint.model,frames,request=request,target=target,proposal=proposal,
                                    context=context,seed=row['seed'],output=dest,authority=case_authority)
                                traces.append(dict(architecture=a,proposal=proposal,mode=mode,path=dest.name,pin=pin))
                        checkpoint.fresh()
                    pin = write_committed(path/'report.json',dict(schema='task-answer-transfer-case-v1',complete=True,
                        case=row,plan=plan_pin,checkpoints=identities,expected=expected,roster=list(target.roster),
                        authority=case_authority,public=pins,unavailable=unavailable,traces=traces,context=context,
                        target_state=frozen,cache_bytes=caches,
                        private_reference_received=False,reference_length_received=False,semantic_validity_established=False))
                    action = 'generated'
                except BaseException as exc:
                    write_committed(path/'failure.json',dict(error=type(exc).__name__,detail=str(exc))); raise
            phases['generation'] += time.perf_counter()-tick; tick = time.perf_counter()
            check_case(path,pin,row=row,plan_pin=plan_pin,identities=identities,expected=expected,public=pins,
                       unavailable=unavailable,authority=case_authority,target_state=frozen,caches=caches)
            results.append(dict(ordinal=i,task=row['task'],path=path.name,pin=pin,action=action))
            phases['readback'] += time.perf_counter()-tick
        tick = time.perf_counter()
        for v in loaded.values():v.fresh(physical=True)
        pack.fresh(physical=True); need(fingerprint(target.state_dict())==frozen, 'TASK_TRANSFER_TARGET_CHANGED')
        phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9<=sum(phases.values())/seconds<=1.001, 'TASK_TRANSFER_PHASES')
        r = dict(complete=True,binding=binding,results=results,declared_query=len(plan['rows']),evaluated=len(results),
            full_query=ordinals is None,model_loads=0,optimizer_updates=0,encoder_calls=0,phases=phases,seconds=seconds,
            semantic_validity_established=False,architecture_winner=False)
        dest = output/'report.json'
        if dest.exists():
            old = json.loads(dest.read_bytes())
            need(old['binding']==binding and [{k:v for k,v in x.items() if k!='action'} for x in old['results']]
                 ==[{k:v for k,v in x.items() if k!='action'} for x in results], 'TASK_TRANSFER_RESUME_CHANGED')
            dest = output/('reopen-'+digest(canonical(r))+'.json')
        return dest,write_committed(dest,r)


def score_case(root, pin, *, row, plan_pin, identities, expected, public, unavailable, authority, target_state, caches, future, roster):
    r = check_case(root,pin,row=row,plan_pin=plan_pin,identities=identities,expected=expected,
                   public=public,unavailable=unavailable,authority=authority,target_state=target_state,caches=caches)
    need(future and all(set(f)==set(roster) for f in future), 'TASK_TRANSFER_REFERENCE')
    arms = []
    for ref in r['traces']:
        arms.append(dict(architecture=ref['architecture'],proposal=ref['proposal'],mode=ref['mode'],
            trace=ref['pin'],metrics=trace_score(Path(root)/ref['path'],ref['pin'],future=future,roster=roster)))
    lookup = {(v['architecture'],v['proposal'],v['mode']):v['metrics'] for v in arms}; paired = []
    for a,ps in expected.items():
        for p in ps:
            for mode in ('reversed','train_donor'):
                v = lookup.get((a,p,mode)); correct = lookup[a,p,'correct']
                paired.append(dict(architecture=a,proposal=p,control=mode,available=v is not None,
                    reason=unavailable.get(mode),per_slot=None if v is None else
                    {n:correct['full_output_mean'][n]-v['full_output_mean'][n] for n in roster}))
    return dict(task=row['task'],family=row['family'],group=row['group'],population=row['population'],
        arms=arms,paired=paired,unavailable=unavailable,reference_atoms=len(future),reference_scored_after_generation=True,
        semantic_validity_established=False,proposal_selection='all_declared_no_best_of',query_status=POLICY['query_status'])
