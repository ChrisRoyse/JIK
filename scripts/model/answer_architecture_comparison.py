"""Finite-cohort paired diagnostics; native slots and proposals stay separate."""
from fractions import Fraction
import math

MODES = ('correct', 'reversed', 'train_donor')
POLICY = dict(schema='answer-architecture-comparison-v1',
    metric='existing_per_slot_full_output_mean', paired_unit='same_source_document',
    unknown_difference_bounds=[-2., 2.], aggregation='equal_case_exact_binary64_sum',
    proposal_selection='all_separately', best_of=False, uncertainty_estimated=False,
    semantic_validity_established=False, architecture_winner=False, capacity_matched=False)


def need(value, code):
    if not value:
        raise ValueError('MNLM_ANSWER_PAIR_' + code)


def units(x):
    need(type(x) is float and math.isfinite(x), 'NONFINITE')
    n, d = x.as_integer_ratio()
    return n << (1074 - (d.bit_length() - 1))


def average(total, count):
    return float(Fraction(total, (1 << 1074) * count))


def pairs(roster):
    names = [('shared', 'assembled'), ('own_slot', 'assembled')]
    names += [('whole_proposals', name) for name in roster]
    result = [('architecture', (*names[1], 'correct'), (*names[0], 'correct'))]
    for proposal in names[2:]:
        for base in names[:2]:
            result.append(('architecture', (*proposal, 'correct'), (*base, 'correct')))
    for name in names:
        for control in MODES[1:]:
            result.append(('context', (*name, 'correct'), (*name, control)))
    return [dict(kind=k, left=list(a), right=list(b)) for k, a, b in result]


def metric(m, roster, expected):
    e, g = m['expected_atoms'], m['generated_atoms']
    need(type(e) is int and e == expected > 0 and type(g) is int and g >= 0, 'COUNTS')
    need(all(type(m[n]) is int for n in ('matched_positions', 'missing', 'extra'))
         and m['matched_positions'] == min(e, g) and m['missing'] == max(0, e-g)
         and m['extra'] == max(0, g-e), 'COUNTS')
    need(all(type(m[n]) is bool for n in ('ended', 'censored', 'ended_at_reference_boundary'))
         and m['censored'] == (not m['ended'])
         and m['ended_at_reference_boundary'] == (m['ended'] and e == g), 'END')
    need((m['ended'] and m['refusal'] is None) or
         (not m['ended'] and isinstance(m['refusal'], str) and bool(m['refusal'])), 'REFUSAL')
    need(m['missing_similarity'] == m['extra_similarity'] == -1.
         and m['reference_scored_after_generation'] is True
         and m['semantic_validity_established'] is False, 'METRIC_POLICY')
    need(set(m['per_slot']) == set(m['full_output_mean']) == set(roster), 'SLOTS')
    for name in roster:
        values = m['per_slot'][name]
        need(len(values) == e and all(type(v) is float and math.isfinite(v) and -1 <= v <= 1
                                     for v in values), 'VALUES')
        need(values[min(e, g):] == [-1.] * max(0, e-g), 'MISSING_POSITIONS')
        v = m['full_output_mean'][name]
        need(type(v) is float and math.isfinite(v) and -1 <= v <= 1
             and v == (sum(values)-m['extra'])/(e+m['extra']), 'ARITHMETIC')


def validated_metrics(case, meta, roster):
    need(case['schema'] == 'paired-complete-answer-v1' and case['roster'] == list(roster), 'SCHEMA')
    ident = case['identity']
    need(all(ident[k] == meta[k] for k in ('document', 'component', 'side', 'collection')),
         'CASE_IDENTITY')
    need(ident['exposure'] == meta['exposure'], 'EXPOSURE')
    expected = dict(shared=['assembled'], own_slot=['assembled'], whole_proposals=list(roster))
    need(case['expected'] == expected and case['proposal_selection'] == 'all_declared_proposals_separately'
         and all(case[k] is False for k in ('meaning_score_pooled', 'architecture_winner',
                 'semantic_validity_established', 'uncertainty_estimated')), 'PROPOSALS')
    grid = {(a, p, m) for a, ps in expected.items() for p in ps for m in MODES}
    keys = [(r['architecture'], r['proposal'], r['mode']) for r in case['arms']]
    need(len(keys) == len(set(keys)) and set(keys) == grid, 'ARM_GRID')
    lookup = {}
    for key, arm in zip(keys, case['arms'], strict=True):
        metric(arm['metrics'], roster, ident['reference_atoms'])
        lookup[key] = arm['metrics']
    return lookup


def compare(case, meta, roster):
    lookup = validated_metrics(case, meta, roster)
    deltas, endings, dominance = [], [], []
    for pair in pairs(roster):
        left, right = lookup[tuple(pair['left'])], lookup[tuple(pair['right'])]
        delta = {n:left['full_output_mean'][n]-right['full_output_mean'][n] for n in roster}
        need(all(-2 <= v <= 2 for v in delta.values()), 'DELTA')
        deltas.append(delta)
        endings.append(dict(left_ended=left['ended'], right_ended=right['ended'],
            left_exact=left['ended_at_reference_boundary'], right_exact=right['ended_at_reference_boundary'],
            left_censored=left['censored'], right_censored=right['censored']))
        positive, negative = any(v > 0 for v in delta.values()), any(v < 0 for v in delta.values())
        dominance.append('mixed' if positive and negative else 'left' if positive else 'right' if negative else 'tie')
    return dict(meta=meta, source=case['identity']['source'], status='scored', deltas=deltas,
                endings=endings, dominance=dominance)


class Summary:
    def __init__(self, roster):
        self.roster = tuple(roster)
        self.comparisons = pairs(roster)
        self.groups = {}
        self.seen = set()
        self.component_sides = {}
        self.source = None

    def add(self, case):
        m = case['meta']; d = m['document']
        if self.source is None: self.source = case['source']
        need(case['source'] == self.source, 'SOURCE')
        need(d not in self.seen, 'DUPLICATE_CASE'); self.seen.add(d)
        need(m['side'] in ('development', 'query'), 'SIDE')
        need(self.component_sides.setdefault(m['component'], m['side']) == m['side'], 'COMPONENT_SPLIT')
        need(case['status'] in ('scored', 'CONTROL_UNAVAILABLE_NATIVE_IDENTICAL'), 'STATUS')
        if case['status'] == 'scored':
            need(len(case['deltas']) == len(case['endings']) == len(case['dominance'])
                 == len(self.comparisons), 'PAIR_GRID')
        for collection in ('ALL', m['collection']):
            for index in range(len(self.comparisons)):
                key = (collection, m['side'], index)
                s = self.groups.setdefault(key, dict(cases=0, unknown=0, components=set(),
                    distinct_address_order_cases=0,
                    dominance={n:0 for n in ('left', 'right', 'mixed', 'tie')}, endings={},
                    slots={n:dict(total=0, positive=0, negative=0, tie=0) for n in self.roster}))
                s['cases'] += 1; s['components'].add(m['component'])
                s['distinct_address_order_cases'] += int(m['reversal_distinct'])
                if case['status'] != 'scored':
                    s['unknown'] += 1; continue
                delta = case['deltas'][index]
                need(set(delta) == set(self.roster), 'PAIR_SLOTS')
                need(all(type(v) is float and math.isfinite(v) and -2 <= v <= 2
                         for v in delta.values()), 'DELTA')
                positive, negative = any(v > 0 for v in delta.values()), any(v < 0 for v in delta.values())
                expected = 'mixed' if positive and negative else 'left' if positive else 'right' if negative else 'tie'
                need(case['dominance'][index] == expected, 'DOMINANCE')
                end = case['endings'][index]
                need(set(end) == {a+'_'+b for a in ('left', 'right') for b in ('ended', 'exact', 'censored')}
                     and all(type(v) is bool for v in end.values())
                     and all(end[a+'_ended'] != end[a+'_censored']
                             and (not end[a+'_exact'] or end[a+'_ended']) for a in ('left', 'right')), 'PAIR_END')
                s['dominance'][case['dominance'][index]] += 1
                for name, value in case['endings'][index].items():
                    s['endings'][name] = s['endings'].get(name, 0) + int(value)
                for name, value in delta.items():
                    t = s['slots'][name]; t['total'] += units(value)
                    t['positive'] += value > 0; t['negative'] += value < 0; t['tie'] += value == 0

    def finish(self):
        need(self.seen, 'EMPTY')
        groups = []
        for (collection, side, index), s in sorted(self.groups.items()):
            known = s['cases']-s['unknown']; slots = {}
            for name, t in s['slots'].items():
                # These bound unknown finite-cohort scores; they are not sampling intervals.
                slots[name] = dict(paired_mean=average(t['total'], known) if known else None,
                    declared_mean_bounds=[average(t['total'] + units(sign*2.*s['unknown']), s['cases'])
                                          for sign in (-1, 1)],
                    positive=t['positive'], negative=t['negative'], tie=t['tie'])
            groups.append(dict(collection=collection, side=side, comparison=index,
                declared_cases=s['cases'], scored_cases=known, unavailable=s['unknown'],
                distinct_address_order_cases=s['distinct_address_order_cases'],
                components=len(s['components']), slots=slots, dominance=s['dominance'], endings=s['endings']))
        return dict(cases=len(self.seen), source=self.source, comparisons=self.comparisons, groups=groups, policy=POLICY,
                    intervals='bounds_for_unobserved_declared_cases_not_confidence_intervals')
