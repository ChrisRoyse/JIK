"""Choose correction coordinates by representable spacing at the same KKT limits."""
import numpy as np
from scipy.linalg import qr
import torch

from claims.codata_source import check
from model import coordinate_region_solve

RULE = 'spacing_aware_coordinate_certificate_v1'


def refine(gradients,deficit,report):
    active = np.array(report['active'])
    norms = np.array(report['norms'])
    g = gradients[active]/norms[active,None]
    rhs = deficit[active]/norms[active]
    check(np.array_equal(rhs,report['rhs']),'REPRESENTABLE_NORMALIZATION')
    extended = g.astype(np.longdouble)
    weights = np.array(report['weights'],dtype=np.longdouble)
    ideal = extended.T@weights
    computed = ideal.astype(np.float64)
    support = weights > 0
    check(bool(support.any()),'REPRESENTABLE_EMPTY_FACE')
    remaining = report['iteration_limit']-report['iterations']
    check(remaining > 1,'REPRESENTABLE_BUDGET')
    spacing = np.abs(np.spacing(computed))
    spacing = np.maximum(spacing,np.finfo(np.float64).tiny)
    weighted = g[support]/spacing
    weighted /= np.max(np.abs(weighted))
    _,_,pivots = qr(weighted,mode='economic',pivoting=True)
    selected = pivots[:int(support.sum())]
    tolerance = report['tolerance']
    trace = []
    seen = set()
    passed = False
    for _ in range(remaining-1):
        residual = np.sum(extended*computed.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs
        stationarity = computed.astype(np.longdouble)-ideal
        actual = dict(minimum_multiplier=float(weights.min()),minimum_residual=float(residual.min()),
            complementarity=float(np.max(np.abs(weights*residual))),
            stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
        passed = bool(np.isfinite(computed).all() and weights.min() >= 0 and residual.min() >= -tolerance
            and np.max(np.abs(weights*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
        trace.append(actual)
        if passed or computed.tobytes() in seen:
            break
        seen.add(computed.tobytes())
        # Do not take an unmeasured move when the inherited iteration budget ends.
        if len(trace) == remaining-1:
            break
        try:
            correction = np.linalg.solve(g[support][:,selected],-residual[support].astype(np.float64))
        except np.linalg.LinAlgError:
            break
        computed[selected] += correction
    result = dict(report,method=RULE,spacing_predecessor=report['method'],
        spacing_original_actual=report['actual'],spacing_refinement=dict(coordinates=selected.tolist(),
            spacings=spacing[selected].tolist(),trace=trace),
        iterations=report['iterations']+1+len(trace),actual=actual,actual_certified=passed,
        products=(residual+rhs).astype(np.float64).tolist(),norm=float(np.linalg.norm(computed)),
        status='certified_minimum_norm' if passed else 'spacing_parameter_certificate_failed')
    check(result['iterations'] <= report['iteration_limit'],'REPRESENTABLE_BUDGET')
    return computed,result


def solve(gradients,deficit):
    delta,report = coordinate_region_solve.solve(gradients,deficit)
    if delta is not None or report['status'] != 'coordinate_parameter_certificate_failed':
        return delta,report
    if report['iteration_limit']-report['iterations'] <= 1:
        return None,dict(report,spacing_status='iteration_limit')
    computed,result = refine(gradients.detach().cpu().numpy(),deficit.detach().cpu().numpy(),report)
    return (torch.from_numpy(computed).to(gradients.device) if result['actual_certified'] else None),result
