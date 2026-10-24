"""Refine represented residuals across the full supported row space at fixed limits."""
import mpmath as mp
import numpy as np
import torch

from claims.codata_source import check,sha
from model import coordinate_region_solve

RULE = 'full_rowspace_represented_residual_refinement_v1'


def residual_readback(g,rhs,delta):
    """Keep short products in extended precision and reduce blocks at 80 digits."""
    parts = [[] for _ in rhs]
    with mp.workdps(80):
        for begin in range(0,g.shape[1],256):
            values = np.sum(g[:,begin:begin+256]*delta[begin:begin+256].astype(np.longdouble),
                            axis=1,dtype=np.longdouble)
            for i,value in enumerate(values):
                parts[i].append(mp.mpf(np.format_float_scientific(value,precision=25,unique=False)))
        return np.array([mp.nstr(mp.fsum(values)-mp.mpf(float(b)),75)
                         for values,b in zip(parts,rhs)],dtype=np.longdouble)


def refine(gradients,deficit,report):
    active = np.array(report['active'])
    norms = np.array(report['norms'])
    g = gradients[active]/norms[active,None]
    rhs = deficit[active]/norms[active]
    check(np.array_equal(rhs,report['rhs']),'ROWSPACE_NORMALIZATION')
    extended = g.astype(np.longdouble)
    weights = np.array(report['weights'],dtype=np.longdouble)
    ideal = extended.T@weights
    computed = ideal.astype(np.float64)
    support = weights > 0
    check(bool(support.any()),'ROWSPACE_EMPTY_FACE')
    remaining = report['iteration_limit']-report['iterations']
    check(remaining > 1,'ROWSPACE_BUDGET')
    u,s,vh = np.linalg.svd(g[support],full_matrices=False)
    threshold = max(g[support].shape)*np.finfo(np.float64).eps*s[0]
    rank = int(np.sum(s > threshold))
    full_rank = rank == int(support.sum())
    tolerance = report['tolerance']
    trace,seen = [],set()
    passed = False
    for _ in range(remaining-1):
        residual = residual_readback(extended,rhs,computed)
        stationarity = computed.astype(np.longdouble)-ideal
        actual = dict(minimum_multiplier=float(weights.min()),minimum_residual=float(residual.min()),
            complementarity=float(np.max(np.abs(weights*residual))),
            stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
        passed = bool(np.isfinite(computed).all() and weights.min() >= 0 and residual.min() >= -tolerance
            and np.max(np.abs(weights*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
        trace.append(actual)
        pin = sha(computed.tobytes())
        if passed or not full_rank or pin in seen or len(trace) == remaining-1:
            break
        seen.add(pin)
        correction = vh.T@((u.T@(-residual[support].astype(np.float64)))/s)
        computed += correction
    result = dict(report,method=RULE,rowspace_predecessor=report['method'],
        rowspace_original_actual=report['actual'],rowspace_refinement=dict(support=np.flatnonzero(support).tolist(),
            rank=rank,singular_values=s.tolist(),rank_threshold=float(threshold),
            condition=float(s[0]/s[-1]),trace=trace),
        iterations=report['iterations']+1+len(trace),actual=actual,actual_certified=passed,
        products=(residual+rhs).astype(np.float64).tolist(),norm=float(np.linalg.norm(computed)),
        status='certified_minimum_norm' if passed else 'rowspace_parameter_certificate_failed')
    check(result['iterations'] <= report['iteration_limit'],'ROWSPACE_BUDGET')
    return computed,result


def solve(gradients,deficit):
    delta,report = coordinate_region_solve.solve(gradients,deficit)
    if delta is not None or report['status'] != 'coordinate_parameter_certificate_failed':
        return delta,report
    if report['iteration_limit']-report['iterations'] <= 1:
        return None,dict(report,rowspace_status='iteration_limit')
    computed,result = refine(gradients.detach().cpu().numpy(),deficit.detach().cpu().numpy(),report)
    return (torch.from_numpy(computed).to(gradients.device) if result['actual_certified'] else None),result
