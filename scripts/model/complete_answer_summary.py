"""Streaming equivalent of the existing complete-answer cohort summary.

Each finite binary64 addend is accumulated exactly as an integer multiple of
2**-1074. Round once at the existing fsum boundary, then divide as before.
Only case/component identities grow with the cohort, never metric dictionaries.
"""
from fractions import Fraction
import math

from model.complete_answer_comparison import MODES
from text.constellation_corpus import need

COUNTS = ('expected_atoms','generated_atoms','matched_positions','missing','extra',
          'ended','ended_at_reference_boundary','censored')


def units(value):
    need(type(value) is float and math.isfinite(value), 'ANSWER_SUMMARY_NONFINITE')
    n,d = value.as_integer_ratio()
    return n << (1074-(d.bit_length()-1))


def rounded(value):
    return float(Fraction(value, 1 << 1074))


class Summary:
    def __init__(self):
        self.seen = set(); self.components = set(); self.groups = {}; self.pairs = {}
        self.roster = self.protocol = self.models = None

    def add(self, case):
        ident = case['identity']; key = (ident['source'],ident['side'],ident['document'])
        need(key not in self.seen, 'ANSWER_COMPARISON_DUPLICATE_CASE'); self.seen.add(key)
        self.components.add((ident['source'],ident['side'],ident['component']))
        fixed = {a:{k:v for k,v in row.items() if k in
            ('model','target','roster','cache_bytes','serialized_event_bytes')} for a,row in case['models'].items()}
        if self.roster is None:self.roster,self.protocol,self.models = case['roster'],case['expected'],fixed
        need(case['roster'] == self.roster and case['expected'] == self.protocol, 'ANSWER_COMPARISON_COHORT_PROTOCOL')
        need(fixed == self.models, 'ANSWER_COMPARISON_COHORT_MODELS')
        expected = {(a,p,m) for a,ps in self.protocol.items() for p in ps for m in MODES}
        actual = [(r['architecture'],r['proposal'],r['mode']) for r in case['arms']]
        need(len(actual) == len(set(actual)) and set(actual) == expected, 'ANSWER_COMPARISON_COHORT_GRID')
        expected_pairs = {(a,p,m) for a,ps in self.protocol.items() for p in ps for m in MODES[1:]}
        actual_pairs = [(r['architecture'],r['proposal'],r['control']) for r in case['paired']]
        need(len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs,
             'ANSWER_COMPARISON_COHORT_PAIRS')
        for arm in case['arms']:
            key = (ident['source'],ident['side'],arm['architecture'],arm['proposal'],arm['mode'])
            state = self.groups.setdefault(key,dict(cases=0,counts={n:0 for n in COUNTS},
                means={n:0 for n in self.roster},sums={n:0 for n in self.roster}))
            state['cases'] += 1; m = arm['metrics']
            for n in COUNTS:state['counts'][n] += int(m[n])
            for n in self.roster:
                state['means'][n] += units(m['full_output_mean'][n])
                # Preserve the prior per-case sum rounding before the cohort fsum.
                state['sums'][n] += units(sum(m['per_slot'][n])-m['extra'])
        for pair in case['paired']:
            key = (ident['source'],ident['side'],pair['architecture'],pair['proposal'],pair['control'])
            state = self.pairs.setdefault(key,dict(cases=0,slots={n:dict(total=0,positive=0,negative=0,tie=0)
                                                                 for n in self.roster}))
            state['cases'] += 1
            for n in self.roster:
                v = pair['per_slot'][n]['full_output_mean_delta']; slot = state['slots'][n]
                slot['total'] += units(v); slot['positive'] += v > 0; slot['negative'] += v < 0; slot['tie'] += v == 0

    def finish(self):
        need(self.seen, 'ANSWER_COMPARISON_EMPTY_COHORT'); groups = []; paired = []
        for key,s in sorted(self.groups.items()):
            source,side,architecture,proposal,mode = key; d = s['counts']['expected_atoms']+s['counts']['extra']
            groups.append(dict(source=source,side=side,architecture=architecture,proposal=proposal,mode=mode,
                cases=s['cases'],counts=dict(s['counts']),per_slot={n:dict(
                    case_mean=rounded(s['means'][n])/s['cases'],atom_mean=rounded(s['sums'][n])/d) for n in self.roster}))
        for key,s in sorted(self.pairs.items()):
            source,side,architecture,proposal,control = key
            paired.append(dict(source=source,side=side,architecture=architecture,proposal=proposal,control=control,
                cases=s['cases'],per_slot={n:dict(case_mean_delta=rounded(v['total'])/s['cases'],
                    positive=v['positive'],negative=v['negative'],tie=v['tie']) for n,v in s['slots'].items()}))
        return dict(cases=len(self.seen),components=len(self.components),groups=groups,paired=paired,
            uncertainty_estimated=False,architecture_winner=False,semantic_validity_established=False)
