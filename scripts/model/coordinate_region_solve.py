"""Representable primal refinement without relaxing the original KKT tolerance."""
import numpy as np
from scipy.linalg import qr
import torch

from claims.codata_source import check
from model import precise_region_solve

RULE = 'coordinate_primal_certificate_refinement_v1'


def solve(gradients, deficit):
    delta, report = precise_region_solve.solve(gradients,deficit)
    if delta is not None or report['status'] != 'precise_parameter_certificate_failed':
        return delta,report
    if report['iterations'] >= report['iteration_limit']:
        return None,dict(report,coordinate_status='iteration_limit')
    active = np.array(report['active'])
    norms = np.array(report['norms'])
    g = gradients.detach().cpu().numpy()[active]/norms[active,None]
    rhs = deficit.detach().cpu().numpy()[active]/norms[active]
    check(np.array_equal(rhs,report['rhs']), 'COORDINATE_SOLVE_NORMALIZATION')
    extended, weights = g.astype(np.longdouble), np.array(report['weights'],dtype=np.longdouble)
    computed = (extended.T@weights).astype(np.float64)
    residual = np.sum(extended*computed.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs
    support = weights > 0
    check(bool(support.any()), 'COORDINATE_SOLVE_EMPTY_FACE')
    _,_,pivots = qr(g[support],mode='economic',pivoting=True)
    selected = pivots[:int(support.sum())]
    try:
        correction = np.linalg.solve(g[support][:,selected],-residual[support].astype(np.float64))
    except np.linalg.LinAlgError:
        return None,dict(report,coordinate_status='singular_face',iterations=report['iterations']+1)
    original = computed.copy()
    computed[selected] += correction
    residual = np.sum(extended*computed.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs
    stationarity = computed.astype(np.longdouble)-extended.T@weights
    tolerance = report['tolerance']
    actual = dict(minimum_multiplier=float(weights.min()),minimum_residual=float(residual.min()),
        complementarity=float(np.max(np.abs(weights*residual))),
        stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
    passed = bool(np.isfinite(computed).all() and weights.min() >= 0 and residual.min() >= -tolerance
        and np.max(np.abs(weights*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
    result = dict(report,method=RULE,coordinate_original_actual=report['actual'],
        coordinate_refinement=dict(coordinates=selected.tolist(),rows=int(support.sum()),
            correction_norm=float(np.linalg.norm(correction)),
            represented_norm=float(np.linalg.norm(computed-original))),
        iterations=report['iterations']+1,actual=actual,actual_certified=passed,
        products=(residual+rhs).astype(np.float64).tolist(),norm=float(np.linalg.norm(computed)),
        status='certified_minimum_norm' if passed else 'coordinate_parameter_certificate_failed')
    return (torch.from_numpy(computed).to(gradients.device) if passed else None),result
