"""Bound coordinate rounding error before rank-revealing primal refinement."""
import numpy as np
from scipy.linalg import qr
import torch

from claims.codata_source import check,sha
from model import coordinate_region_solve
from model.rowspace_region_solve import residual_readback

RULE = 'rounding_bounded_coordinate_refinement_v1'


def refine(gradients,deficit,report):
    active = np.array(report['active'])
    norms = np.array(report['norms'])
    g = gradients[active]/norms[active,None]
    rhs = deficit[active]/norms[active]
    check(np.array_equal(rhs,report['rhs']),'QUANTUM_NORMALIZATION')
    extended = g.astype(np.longdouble)
    weights = np.array(report['weights'],dtype=np.longdouble)
    ideal = extended.T@weights
    computed = ideal.astype(np.float64)
    support = weights > 0
    size = int(support.sum())
    check(size > 0,'QUANTUM_EMPTY_FACE')
    remaining = report['iteration_limit']-report['iterations']
    check(remaining > 1,'QUANTUM_BUDGET')
    tolerance = report['tolerance']
    spacing = np.abs(np.spacing(computed)).astype(np.longdouble)
    quantum = np.max(np.abs(extended*weights[:,None]),axis=0)*spacing/2
    allowance = np.longdouble(tolerance)/(2*size)
    eligible = np.flatnonzero(quantum <= allowance)
    selected = np.array([],dtype=int)
    singular = np.array([],dtype=float)
    rank = 0
    threshold = 0.
    if len(eligible) >= size:
        _,_,pivots = qr(g[support][:,eligible],mode='economic',pivoting=True)
        selected = eligible[pivots[:size]]
        singular = np.linalg.svd(g[support][:,selected],compute_uv=False)
        threshold = size*np.finfo(np.float64).eps*singular[0]
        rank = int(np.sum(singular > threshold))
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
        if passed or rank != size or pin in seen or len(trace) == remaining-1:
            break
        seen.add(pin)
        try:
            correction = np.linalg.solve(g[support][:,selected],-residual[support].astype(np.float64))
        except np.linalg.LinAlgError:
            break
        computed[selected] += correction
    result = dict(report,method=RULE,quantum_predecessor=report['method'],
        quantum_original_actual=report['actual'],quantum_refinement=dict(support=np.flatnonzero(support).tolist(),
            eligible=len(eligible),coordinates=selected.tolist(),quantum=quantum[selected].astype(float).tolist(),
            allowance=float(allowance),rank=rank,singular_values=singular.tolist(),rank_threshold=float(threshold),
            condition=float(singular[0]/singular[-1]) if rank == size else None,trace=trace),
        iterations=report['iterations']+1+len(trace),actual=actual,actual_certified=passed,
        products=(residual+rhs).astype(np.float64).tolist(),norm=float(np.linalg.norm(computed)),
        status='certified_minimum_norm' if passed else 'quantum_parameter_certificate_failed')
    check(result['iterations'] <= report['iteration_limit'],'QUANTUM_BUDGET')
    return computed,result


def solve(gradients,deficit):
    delta,report = coordinate_region_solve.solve(gradients,deficit)
    if delta is not None or report['status'] != 'coordinate_parameter_certificate_failed':
        return delta,report
    if report['iteration_limit']-report['iterations'] <= 1:
        return None,dict(report,quantum_status='iteration_limit')
    computed,result = refine(gradients.detach().cpu().numpy(),deficit.detach().cpu().numpy(),report)
    return (torch.from_numpy(computed).to(gradients.device) if result['actual_certified'] else None),result
