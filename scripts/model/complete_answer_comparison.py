"""Paired, post-generation evaluation of complete typed answer trajectories.

Every declared proposal and control stays in the denominator. Reference length
is used only here, after autonomous traces have been sealed, never for stopping.
"""
import json
import math
from pathlib import Path

from model.complete_answer_scores import native_similarity
from model.complete_answer_trace import replay
from text.constellation_corpus import file_sha, need

MODES = ('correct', 'reversed', 'train_donor')
INVARIANTS = ('model', 'target', 'roster', 'seed', 'context', 'cache_bytes',
              'serialized_event_bytes', 'observed_atoms', 'request', 'request_descriptor')


def grid(traces, *, expected, public, roster, authority):
    """Bind every arm to the same checkpoint and its independently named input."""
    need(set(public) == set(MODES) and expected and len(set(roster)) == len(roster),
         'ANSWER_COMPARISON_PROTOCOL')
    keys = {(a, p, m) for a, proposals in expected.items() for p in proposals for m in MODES}
    need(all(proposals and len(set(proposals)) == len(proposals) for proposals in expected.values()),
         'ANSWER_COMPARISON_PROPOSALS')
    actual = [(r['architecture'], r['proposal'], r['mode']) for r in traces]
    need(len(actual) == len(set(actual)) and set(actual) == keys, 'ANSWER_COMPARISON_GRID')
    models = {}; common = None; result = []
    for row in sorted(traces, key=lambda r:(r['architecture'], r['proposal'], MODES.index(r['mode']))):
        path = Path(row['path']); pin = row['pin']
        need(file_sha(path/'report.json') == pin, 'ANSWER_COMPARISON_REPORT_PIN')
        report = json.loads((path/'report.json').read_bytes())
        need(report['complete'] and report['reference_length_visible'] is False,
             'ANSWER_COMPARISON_REFERENCE_VISIBLE')
        need(file_sha(path/'request.json') == report['files']['request.json'], 'ANSWER_COMPARISON_REQUEST_PIN')
        request = json.loads((path/'request.json').read_bytes()); given = public[row['mode']]
        need(request['authority'] == authority and request['reference_visible'] is False
             and request['roster'] == list(roster) and request['proposal'] == row['proposal'],
             'ANSWER_COMPARISON_AUTHORITY')
        need(request['prefix'] == given['prefix'] and request['request'] == given['request']
             and request['request_descriptor'] == given['descriptor'], 'ANSWER_COMPARISON_PUBLIC_CONTEXT')
        invariant = {k:request[k] for k in INVARIANTS}
        architecture = row['architecture']
        need(models.setdefault(architecture, invariant) == invariant, 'ANSWER_COMPARISON_UNMATCHED_ARMS')
        shared = {k:v for k,v in invariant.items() if k != 'model'}
        if common is None: common = shared
        need(shared == common, 'ANSWER_COMPARISON_UNMATCHED_ARCHITECTURES')
        result.append(dict(row, manifest=request, manifest_pin=report['files']['request.json']))
    return result, models


def trace_score(path, pin, *, future, roster):
    expected = len(future); generated = 0; ended = False
    slots = {n:[-1.] * expected for n in roster}
    for event in replay(path, pin):
        if event['kind'] == 'end': ended = True; continue
        if generated < expected:
            for n in roster: slots[n][generated] = native_similarity(event['slots'][n], future[generated][n])
        generated += 1
    report = json.loads((Path(path)/'report.json').read_bytes())
    extra = max(0, generated-expected)
    return dict(expected_atoms=expected, generated_atoms=generated, ended=ended,
        ended_at_reference_boundary=ended and generated == expected, censored=not ended,
        matched_positions=min(generated, expected), missing=max(0, expected-generated), extra=extra,
        per_slot=slots, full_output_mean={n:(sum(v)-extra)/(expected+extra) for n,v in slots.items()},
        missing_similarity=-1., extra_similarity=-1., refusal=report['refusal'],
        reference_scored_after_generation=True, semantic_validity_established=False)


def compare(traces, *, expected, public, authority, identity, future, roster, copy_frame):
    # Admission of the entire grid precedes scoring; no outcome-dependent arm selection.
    rows, models = grid(traces, expected=expected, public=public, roster=roster, authority=authority)
    need(identity['side'] in ('development', 'query'), 'ANSWER_COMPARISON_NONHELDOUT_CASE')
    need(len(future) > 0 and set(copy_frame) == set(roster)
         and all(set(frame) == set(roster) for frame in future), 'ANSWER_COMPARISON_REFERENCE')
    baseline = {n:[native_similarity(copy_frame[n], frame[n]) for frame in future] for n in roster}
    arms = []; lookup = {}
    for row in rows:
        metrics = trace_score(row['path'], row['pin'], future=future, roster=roster)
        arm = {k:row[k] for k in ('architecture', 'proposal', 'mode', 'pin', 'manifest_pin')}
        arm['metrics'] = metrics; arms.append(arm)
        lookup[row['architecture'], row['proposal'], row['mode']] = metrics
    paired = []
    for architecture, proposals in expected.items():
        for proposal in proposals:
            correct = lookup[architecture, proposal, 'correct']
            for mode in MODES[1:]:
                control = lookup[architecture, proposal, mode]
                values = {}
                for n in roster:
                    difference = [x-y for x,y in zip(correct['per_slot'][n], control['per_slot'][n], strict=True)]
                    values[n] = dict(reference_position_delta=difference,
                        reference_mean_delta=math.fsum(difference)/len(future),
                        full_output_mean_delta=correct['full_output_mean'][n]-control['full_output_mean'][n])
                paired.append(dict(architecture=architecture, proposal=proposal, control=mode, per_slot=values,
                    both_ended=correct['ended'] and control['ended'],
                    both_exact_length=correct['ended_at_reference_boundary'] and control['ended_at_reference_boundary']))
    return dict(schema='paired-complete-answer-v1', identity=identity, models=models, public=public,
        expected={a:list(p) for a,p in expected.items()}, roster=list(roster), arms=arms, paired=paired,
        copy_current=dict(per_slot=baseline, oracle_reference_length=True, autonomous_stopping_baseline=False),
        proposal_selection='all_declared_proposals_separately', meaning_score_pooled=False,
        semantic_validity_established=False, architecture_winner=False, uncertainty_estimated=False)


def summarize(cases):
    """Each case is consumed once; retain scalar sufficient data, never native arrays."""
    seen = set(); groups = {}; pairs = {}; components = set(); roster = None; protocol = None; models = None
    for case in cases:
        ident = case['identity']; key = (ident['source'], ident['side'], ident['document'])
        need(key not in seen, 'ANSWER_COMPARISON_DUPLICATE_CASE'); seen.add(key)
        components.add((ident['source'], ident['side'], ident['component']))
        fixed = {a:{k:v for k,v in row.items() if k in
            ('model', 'target', 'roster', 'cache_bytes', 'serialized_event_bytes')} for a,row in case['models'].items()}
        if roster is None: roster, protocol, models = case['roster'], case['expected'], fixed
        need(case['roster'] == roster and case['expected'] == protocol, 'ANSWER_COMPARISON_COHORT_PROTOCOL')
        need(fixed == models, 'ANSWER_COMPARISON_COHORT_MODELS')
        declared = {(a,p,m) for a,ps in protocol.items() for p in ps for m in MODES}
        observed = [(r['architecture'],r['proposal'],r['mode']) for r in case['arms']]
        need(len(observed) == len(set(observed)) and set(observed) == declared, 'ANSWER_COMPARISON_COHORT_GRID')
        expected_pairs = {(a,p,m) for a,ps in protocol.items() for p in ps for m in MODES[1:]}
        actual_pairs = [(r['architecture'],r['proposal'],r['control']) for r in case['paired']]
        need(len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs,
             'ANSWER_COMPARISON_COHORT_PAIRS')
        for arm in case['arms']:
            key = (ident['source'], ident['side'], arm['architecture'], arm['proposal'], arm['mode'])
            metrics = arm['metrics']
            compact = {n:metrics[n] for n in ('expected_atoms', 'generated_atoms', 'matched_positions',
                'missing', 'extra', 'ended', 'ended_at_reference_boundary', 'censored', 'full_output_mean')}
            compact['slot_sum'] = {n:sum(metrics['per_slot'][n])-metrics['extra'] for n in roster}
            groups.setdefault(key, []).append(compact)
        for pair in case['paired']:
            key = (ident['source'], ident['side'], pair['architecture'], pair['proposal'], pair['control'])
            pairs.setdefault(key, []).append({n:pair['per_slot'][n]['full_output_mean_delta'] for n in roster})
    need(seen, 'ANSWER_COMPARISON_EMPTY_COHORT')
    result = []
    for key, rows in sorted(groups.items()):
        source, side, architecture, proposal, mode = key
        counts = {n:sum(int(row[n]) for row in rows) for n in
            ('expected_atoms', 'generated_atoms', 'matched_positions', 'missing', 'extra',
             'ended', 'ended_at_reference_boundary', 'censored')}
        denominator = counts['expected_atoms'] + counts['extra']
        result.append(dict(source=source, side=side, architecture=architecture, proposal=proposal, mode=mode,
            cases=len(rows), counts=counts, per_slot={n:dict(
                case_mean=math.fsum(row['full_output_mean'][n] for row in rows)/len(rows),
                atom_mean=math.fsum(row['slot_sum'][n] for row in rows)/denominator)
                for n in roster}))
    effects = []
    for key, rows in sorted(pairs.items()):
        source, side, architecture, proposal, control = key
        effects.append(dict(source=source, side=side, architecture=architecture, proposal=proposal,
            control=control, cases=len(rows), per_slot={n:dict(
                case_mean_delta=math.fsum(r[n] for r in rows)/len(rows),
                positive=sum(r[n] > 0 for r in rows), negative=sum(r[n] < 0 for r in rows),
                tie=sum(r[n] == 0 for r in rows)) for n in roster}))
    return dict(cases=len(seen), components=len(components), groups=result, paired=effects,
        uncertainty_estimated=False, architecture_winner=False, semantic_validity_established=False)
