"""Recompute successor bounds and corrections from persisted arrays and scores."""
import copy
from pathlib import Path

import mpmath as mp
import numpy as np

from audit.proposal_recognition import read, evidence
from audit.region_selection_readback import native_error
from audit.typed_correction_readback_math import retained as retained_rows
from audit.brier_region_readback_math import first_solve as solve_rows
from claims.codata_source import check


def reader_view(learning):
    # The reusable readers use these historical field names; no persisted binding changes.
    value = dict(learning,binding=copy.deepcopy(learning['binding']))
    value['binding']['fresh_feedback_learning'] = value['binding']['successor_feedback_learning']
    value['binding']['typed_correction']['rows'] = value['binding']['successor_targets']['rows']
    value['binding']['brier_region'] = value['binding']['successor_targets']
    return value


def retained(learning):
    return retained_rows(reader_view(learning))


def first_solve(root,learning,start):
    return solve_rows(root,reader_view(learning),start)


def floors(learning,baseline,sealed,native):
    binding = learning['binding']
    description = binding['successor_targets']
    pairs = binding['successor_feedback_learning']['pairs']
    rows,bounds = description['rows'],description['bounds']
    check(baseline['scores'] == learning['before'],'SUCCESSOR_READER_BASELINE')
    prior = Path(binding['region_correction']['diagnostic']['root'])
    audit = read(prior/'report.json')
    check(evidence(prior/'arrays.npz') == audit['arrays'],'SUCCESSOR_READER_OLD_REFERENCES')
    bank = binding['successor_feedback_learning']['regions']['bank']['rows']
    new_order = sorted(pairs,key=lambda k:pairs[k]['region'])
    with np.load(prior/'arrays.npz',allow_pickle=False) as z, mp.workdps(80):
        for key,lanes in bounds.items():
            pair = pairs[key]
            before = baseline['scores'][key]['termination']['score']
            original = sealed[pair['condition']]['models']['corrected']
            check(before['outcome'] == original['outcome'] == bank[pair['region']]['outcome'],
                  'SUCCESSOR_READER_OUTCOME')
            scores = dict(common=before['common'],**before['instruments'])
            original = dict(common=original['common'],**original['instruments'])
            check(set(lanes) == set(scores),'SUCCESSOR_READER_BOUND_ROSTER')
            for lane,b in lanes.items():
                ceiling = min(scores[lane]['brier'],original[lane]['brier'])
                check(0 < ceiling <= 2 and b['ceiling'] == ceiling and
                      b['expanded_brier'] == scores[lane]['brier'] and
                      b['sealed_brier'] == original[lane]['brier'],'SUCCESSOR_READER_CEILING')
                residual = mp.sqrt(mp.mpf(ceiling)/2)
                q = 1-residual
                margin = mp.mpf(0) if ceiling == 2 else mp.log((len(bank)-1)*q/residual)
                check(abs(b['probability_floor']-float(q)) < 1e-15 and
                      abs(b['margin']-float(margin)) < 1e-12 and b['vacuous'] == (ceiling == 2),
                      'SUCCESSOR_READER_BOUND')
            for lane,all_rows in rows.items():
                rr = [r for r in all_rows if r['key'] == key]
                check(len(rr) == len(bank)-1 and {r['target'] for r in rr} == {pair['region']} and
                      {r['rival'] for r in rr} == set(range(len(bank)))-{pair['region']},
                      'SUCCESSOR_READER_CONSTRAINT_ROSTER')
                j = audit['instruments'].index(lane)
                references = [z[f'start_0_{j}_reference_{r}'] for r in range(17)]
                references += [native[lane][pairs[k]['observation']] for k in new_order]
                for r in rr:
                    cost = float(np.float32(native_error(references[r['rival']],native[lane][pair['observation']])))
                    check(abs(cost-r['native_cost']) < 1e-7,'SUCCESSOR_READER_NATIVE_DISTANCE')
                    check(r['kind'] == 'native' and r['lane_margin'] == lanes[lane]['margin'] and
                          r['common_share'] == lanes['common']['margin']/len(rows) and
                          r['floor'] == max(r['native_cost'],r['lane_margin'],r['common_share']),
                          'SUCCESSOR_READER_FLOOR')
