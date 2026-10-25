"""One resident set of predictors, immutable public cases, durable case results."""
import fcntl
import json
from pathlib import Path
import time

from claims.process_conditions import write_committed
from model.complete_answer_comparison import grid
from model.complete_answer_evaluation import generate, public_inputs
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha, need


def read(path, pin):
    need(file_sha(path) == pin, 'ANSWER_COHORT_RECEIPT_PIN')
    return json.loads(Path(path).read_bytes())


def validate_cases(cases):
    need(isinstance(cases, list) and cases, 'ANSWER_COHORT_CASES')
    for row in cases:
        need(set(row) == {'ordinal', 'document', 'public', 'seed', 'context', 'reversal_distinct'},
             'ANSWER_COHORT_PUBLIC_FIELDS')
        need(type(row['ordinal']) is int and row['ordinal'] >= 0
             and type(row['document']) is int and row['document'] >= 0
             and type(row['seed']) is int and type(row['reversal_distinct']) is bool
             and isinstance(row['context'], str) and row['context'], 'ANSWER_COHORT_CASE_ID')
        need(set(row['public']) == {'path', 'pin', 'state'}, 'ANSWER_COHORT_ARTIFACT_FIELDS')
    ordinals = [r['ordinal'] for r in cases]; documents = [r['document'] for r in cases]
    need(ordinals == sorted(set(ordinals)) and len(set(documents)) == len(documents), 'ANSWER_COHORT_ORDER')


def case_authority(authority, row):
    return dict(authority, document=row['document'], ordinal=row['ordinal'], public=row['public']['pin'])


def generation_check(path, pin, public, loaded, target, authority):
    r = read(path/'report.json', pin)
    need(r['complete'] and r['authority'] == authority and r['private_reference_received'] is False
         and r['reference_length_received'] is False and r['optimizer_updates'] == 0,
         'ANSWER_COHORT_GENERATION_SCOPE')
    need(r['checkpoints'] == {n:v.identity for n,v in loaded.items()}, 'ANSWER_COHORT_GENERATION_CHECKPOINTS')
    _, pins = public_inputs(public, target.roster)
    expected = {n:list(v.model.proposals) for n,v in loaded.items()}
    rows = [dict(row, path=str(path/row['path'])) for row in r['traces']]
    grid(rows, expected=expected, public=pins, roster=target.roster, authority=authority)
    for row in rows:
        for event in replay(row['path'], row['pin']):
            del event
    return r


def one_case(loaded, public, target, row, output, authority):
    output.mkdir(exist_ok=True)
    _, pins = public_inputs(public, target.roster)
    binding = dict(case=row, authority=authority, checkpoints={n:v.identity for n,v in loaded.items()}, public=pins)
    binding = json.loads(json.dumps(binding, sort_keys=True))
    committed = output/'case.json'
    if committed.exists():
        result = json.loads(committed.read_bytes())
        need(result['binding'] == binding and result['complete'], 'ANSWER_COHORT_CASE_CHANGED')
        if result['status'] == 'generated':
            generation_check(output/result['generation']['path'], result['generation']['pin'], public, loaded, target, authority)
        else:
            need(result['status'] == 'CONTROL_UNAVAILABLE_NATIVE_IDENTICAL'
                 and pins['correct']['prefix'] == pins['train_donor']['prefix'], 'ANSWER_COHORT_FALSE_GAP')
        return result, 'reused'
    if pins['correct']['prefix'] == pins['train_donor']['prefix']:
        result = dict(complete=True, binding=binding, status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL', generation=None)
        write_committed(committed, result)
        return result, 'gap'
    attempts = sorted(output.glob('attempt-*'))
    for i, path in enumerate(attempts):
        need(path.is_dir() and path.name == f'attempt-{i:06d}', 'ANSWER_COHORT_ATTEMPT_ORDER')
    complete = [p for p in attempts if (p/'report.json').exists()]
    need(len(complete) <= 1, 'ANSWER_COHORT_AMBIGUOUS_GENERATION')
    if complete:
        path = complete[0]; pin = file_sha(path/'report.json'); kind = 'recovered'
    else:
        path = output/f'attempt-{len(attempts):06d}'
        pin = generate(loaded, public, target=target, seed=row['seed'], context=row['context'],
                       output=path, authority=authority)
        kind = 'generated'
    generation_check(path, pin, public, loaded, target, authority)
    result = dict(complete=True, binding=binding, status='generated',
        generation=dict(path=path.name, pin=pin), retained_incomplete_attempts=[p.name for p in attempts if p != path])
    write_committed(committed, result)
    return result, kind


def run(loaded, cases, load_public, *, target, output, authority):
    """Public artifacts are streamed one case at a time. No reference is accepted."""
    began = tick = time.perf_counter(); output = Path(output); output.mkdir(exist_ok=True)
    validate_cases(cases)
    binding = dict(schema='resident-answer-cohort-v1', authority=authority, cases=cases,
        checkpoints={n:v.identity for n,v in loaded.items()}, target=fingerprint(target.state_dict()),
        cache_bytes={n:v.model.forecaster.predictor.cache_bytes for n,v in loaded.items()})
    binding = json.loads(json.dumps(binding, sort_keys=True))
    counts = {n:0 for n in ('generated', 'reused', 'recovered', 'gap')}; results = []
    with (output/'owner.lock').open('a+') as owner:
        try:fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('ANSWER_COHORT_OWNER_BUSY') from exc
        path = output/'binding.json'
        if path.exists():need(json.loads(path.read_bytes()) == binding, 'ANSWER_COHORT_BINDING_CHANGED')
        else:write_committed(path, binding)
        root_pin = file_sha(path); phases = dict(open=time.perf_counter()-tick, public_read=0., cases=0., seal=0.)
        for row in cases:
            tick = time.perf_counter(); public = load_public(row['public'])
            phases['public_read'] += time.perf_counter()-tick; tick = time.perf_counter()
            dest = output/f"case-{row['ordinal']:06d}"
            result, kind = one_case(loaded, public, target, row, dest, case_authority(authority, row))
            counts[kind] += 1
            results.append(dict(ordinal=row['ordinal'], document=row['document'], status=result['status'],
                path=dest.name, pin=file_sha(dest/'case.json'), action=kind))
            del public, result
            phases['cases'] += time.perf_counter()-tick
        tick = time.perf_counter()
        need(file_sha(path) == root_pin and fingerprint(target.state_dict()) == binding['target'], 'ANSWER_COHORT_CHANGED')
        for checkpoint in loaded.values():checkpoint.fresh(physical=True)
        phases['seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'ANSWER_COHORT_PHASES')
    return dict(complete=True, binding=root_pin, results=results, actions=counts,
        checkpoint_loads=len(loaded), frozen_target_loads=1, private_reference_received=False,
        optimizer_updates=0, phases_seconds=phases, seconds=seconds,
        architecture_winner=False, capability_demonstrated=False)
