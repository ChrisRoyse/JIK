"""Compare every whole-answer proposal with two fixed exact-TRAIN controls."""
from model.answer_architecture_comparison import Summary, metric, need, validated_metrics

ARMS = ('modal_control', 'clarify_conflicts')
STRATA = ('no_exact_prefix', 'unique_answer', 'conflicting_answers')
BASELINE_POLICY = dict(schema='legal-exact-answer-native-scores-v1',
    metric='complete_answer_scores.native_similarity_fp64', missing_similarity=-1.,
    extra_similarity=-1., arms=list(ARMS), all_cases_in_denominator=True,
    reference_length_used_for_selection=False, donor_end='TRAIN_document_END',
    semantic_validity_established=False, admitted_senses=0)
POLICY = dict(schema='answer-baseline-comparison-v1',
    metric='existing_per_slot_full_output_mean', paired_unit='same_source_document',
    unknown_difference_bounds=[-2., 2.], aggregation='equal_case_exact_binary64_sum',
    proposal_selection='all_separately', best_of=False, uncertainty_estimated=False,
    semantic_validity_established=False, architecture_winner=False, capacity_matched=False,
    baseline_selection='frozen_exact_TRAIN_prefix', strongest_baseline_established=False,
    no_match_is_observed_refusal=True, availability_strata=list(STRATA))


def pairs(roster):
    names = [('shared', 'assembled'), ('own_slot', 'assembled')]
    names += [('whole_proposals', n) for n in roster]
    return [dict(kind='baseline', left=[*n, 'correct'], right=['exact_train', a, 'correct'])
            for n in names for a in ARMS]


def validate_baseline(v, meta, roster):
    need(v['identity'] == {k:meta[k] for k in ('document', 'component', 'side', 'collection')},
         'BASELINE_IDENTITY')
    status = v['selection_status']; donor = v['donor']
    need(type(status) is int and status in (0, 1, 2) and type(donor) is int
         and (donor == -1 if status == 0 else donor >= 0 and donor != meta['document']),
         'BASELINE_SELECTION')
    need(set(v['arms']) == set(ARMS), 'BASELINE_ARMS')
    a, b = (v['arms'][n] for n in ARMS)
    expected = a['expected_atoms']
    for m in (a, b): metric(m, roster, expected)
    if status == 0:
        need(a == b and a['generated_atoms'] == 0 and a['refusal'] == 'NO_EXACT_TRAIN_PREFIX',
             'BASELINE_NO_MATCH')
    else:
        need(a['ended'] and a['generated_atoms'] == v['native']['donor_atoms'] > 0,
             'BASELINE_DONOR_END')
        if status == 1: need(a == b, 'BASELINE_UNIQUE')
        else:
            need(b['generated_atoms'] == 0 and b['refusal'] == 'CONFLICTING_TRAIN_ANSWERS',
                 'BASELINE_CONFLICT')
    return STRATA[status]


def compare(payload, baseline, meta, roster, source):
    stratum = validate_baseline(baseline, meta, roster)
    row = dict(meta=meta, source=source, baseline_status=stratum,
               baseline_donor=baseline['donor'], selection_row=baseline['selection_row'])
    if payload.get('status') == 'CONTROL_UNAVAILABLE_NATIVE_IDENTICAL':
        need(payload == dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',
                             ordinal=meta['ordinal'], document=meta['document']), 'GAP_IDENTITY')
        return dict(row, status=payload['status'])
    need(payload['identity']['reference_atoms'] == baseline['arms'][ARMS[0]]['expected_atoms'],
         'BASELINE_REFERENCE_LENGTH')
    lookup = validated_metrics(payload, meta, roster)
    deltas, endings, dominance = [], [], []
    for pair in pairs(roster):
        left = lookup[tuple(pair['left'])]; right = baseline['arms'][pair['right'][1]]
        d = {n:left['full_output_mean'][n]-right['full_output_mean'][n] for n in roster}
        deltas.append(d)
        endings.append(dict(left_ended=left['ended'], right_ended=right['ended'],
            left_exact=left['ended_at_reference_boundary'], right_exact=right['ended_at_reference_boundary'],
            left_censored=left['censored'], right_censored=right['censored']))
        pos, neg = any(x > 0 for x in d.values()), any(x < 0 for x in d.values())
        dominance.append('mixed' if pos and neg else 'left' if pos else 'right' if neg else 'tie')
    return dict(row, status='scored', deltas=deltas, endings=endings, dominance=dominance)


class BaselineSummary:
    def __init__(self, roster):
        self.roster = tuple(roster); self.strata = {}

    def add(self, row):
        need(row['baseline_status'] in STRATA, 'BASELINE_STRATUM')
        for key in ('ALL', row['baseline_status']):
            if key not in self.strata:
                s = self.strata[key] = Summary(self.roster); s.comparisons = pairs(self.roster)
            self.strata[key].add(row)

    def finish(self):
        need('ALL' in self.strata, 'EMPTY')
        result = self.strata['ALL'].finish(); result['policy'] = POLICY
        result['groups'] = []
        for name, s in sorted(self.strata.items()):
            result['groups'].extend(dict(g, baseline_status=name) for g in s.finish()['groups'])
        result['stratum_cases'] = {n:len(self.strata[n].seen) if n in self.strata else 0 for n in STRATA}
        return result
