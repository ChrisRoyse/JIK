"""Explicit TRAIN-count probability baselines for a frozen three-level region code."""
import numpy as np

POLICY = dict(version='region-count-probabilities-v1', half_count=0.5,
              copy_weight='TRAIN_copy_hit_fraction_per_level', held_out_fitting=False,
              baselines=['uniform', 'marginal', 'joint_prefix', 'root_transition', 'persistence'],
              score='conditional_CE_and_conditional_Brier; sum_CE_is_joint_log_score')


def need(ok, code):
    if not ok: raise ValueError('REGION_COUNT_'+code)


def validate(counts, sizes):
    need(len(sizes) == 3 and all(type(n) is int and n > 1 for n in sizes), 'SIZES')
    need(counts['fit_uses_held_out'] is False and counts['scientific_admission'] is False
         and counts['support'] == 'all_TRAIN_within_document_successor_occurrences', 'FIT_SCOPE')
    total = counts['rows']; need(type(total) is int and total > 0, 'ROWS')
    joint = np.asarray(counts['joint']); transition = np.asarray(counts['root_transition'])
    copy = np.asarray(counts['copy_level'])
    need(joint.shape == tuple(sizes) and transition.shape == (sizes[0], sizes[0]) and copy.shape == (3,), 'LAYOUT')
    for value in (joint, transition, copy):
        need(value.dtype.kind in 'iu' and bool((value >= 0).all()) and bool((value <= total).all()), 'COUNTS')
    need(int(joint.sum()) == int(transition.sum()) == total, 'TOTAL')
    for j in range(3):
        expected = joint.sum(axis=tuple(k for k in range(3) if k != j))
        need(expected.tolist() == counts['level_histograms'][j], 'MARGINALS')
    need(transition.sum(0).tolist() == counts['level_histograms'][0], 'TRANSITION_TARGET')
    return joint.astype(np.float64), transition.astype(np.float64), copy.astype(np.float64)/total


def address(target, current, sizes):
    need(len(target) == len(current) == 3 and all(type(v) is int and 0 <= v < sizes[j]
         for values in (target, current) for j,v in enumerate(values)), 'ADDRESS')


def probabilities_valid(probabilities, sizes):
    need(len(probabilities) == 3, 'PROBABILITY_LEVELS')
    for p, n in zip(probabilities, sizes):
        need(p.shape == (n,) and bool(np.isfinite(p).all()) and bool((p > 0).all())
             and bool((p <= 1).all()) and abs(float(p.sum())-1) <= 1e-12, 'PROBABILITY')


class Tables:
    def __init__(self, counts, sizes, policy=POLICY):
        need(policy == POLICY, 'POLICY'); self.sizes = tuple(sizes)
        joint, transition, self.copy = validate(counts, sizes)
        self.joint = joint + POLICY['half_count']; self.root = self.joint.sum((1, 2))
        self.middle = self.joint.sum(2)
        self.root /= self.root.sum()
        self.middle /= self.middle.sum(1, keepdims=True)
        self.joint /= self.joint.sum(2, keepdims=True)
        self.transition = transition + POLICY['half_count']
        self.transition /= self.transition.sum(1, keepdims=True)
        self.marginal = []
        for j in range(3):
            p = joint.sum(axis=tuple(k for k in range(3) if k != j)) + POLICY['half_count']
            self.marginal.append(p/p.sum())
        self.uniform = [np.full(n, 1/n, dtype=np.float64) for n in sizes]

    def distributions(self, target, current):
        address(target, current, self.sizes)
        a,b,_ = target
        joint = [self.root, self.middle[a], self.joint[a,b]]
        persistence = []
        for j, p in enumerate(joint):
            value = (1-self.copy[j])*p; value[current[j]] += self.copy[j]; persistence.append(value)
        out = dict(uniform=self.uniform, marginal=self.marginal, joint_prefix=joint,
                   root_transition=[self.transition[current[0]], *joint[1:]], persistence=persistence)
        for ps in out.values(): probabilities_valid(ps, self.sizes)
        return out

    def scores(self, target, current):
        out = {}
        for name, probabilities in self.distributions(target, current).items():
            out[name] = dict(ce=[float(-np.log(p[y])) for p,y in zip(probabilities, target)],
                brier=[float(np.dot(p,p)-2*p[y]+1) for p,y in zip(probabilities, target)])
        return out
