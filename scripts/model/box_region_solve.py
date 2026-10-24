"""Refine represented constraints inside the unchanged stationarity box."""
import numpy as np
from scipy.linalg import qr
import torch

from claims.codata_source import check,sha
from model import coordinate_region_solve
from model.rowspace_region_solve import residual_readback

RULE = 'stationarity_box_rounding_bounded_refinement_v1'


def refine(gradients,deficit,report):
    active = np.array(report['active'])
    norms = np.array(report['norms'])
    g = gradients[active]/norms[active,None]
    rhs = deficit[active]/norms[active]
    check(np.array_equal(rhs,report['rhs']),'BOX_NORMALIZATION')
    extended = g.astype(np.longdouble)
    weights = np.array(report['weights'],dtype=np.longdouble)
    ideal = extended.T@weights
    computed = ideal.astype(np.float64)
    support = weights > 0
    size = int(support.sum())
    check(size > 0,'BOX_EMPTY_FACE')
    remaining = report['iteration_limit']-report['iterations']
    check(remaining > 1,'BOX_BUDGET')
    tolerance = report['tolerance']
    exact_lower,exact_upper = ideal-tolerance,ideal+tolerance
    lower,upper = exact_lower.astype(np.float64),exact_upper.astype(np.float64)
    lower = np.where(lower.astype(np.longdouble) < exact_lower,np.nextafter(lower,np.inf),lower)
    upper = np.where(upper.astype(np.longdouble) > exact_upper,np.nextafter(upper,-np.inf),upper)
    check(bool(((lower <= computed)&(computed <= upper)).all()),'BOX_INITIAL_STATIONARITY')
    spacing = np.maximum(np.abs(np.spacing(lower)),np.abs(np.spacing(upper))).astype(np.longdouble)
    quantum = np.max(np.abs(extended*weights[:,None]),axis=0)*spacing/2
    allowance = np.longdouble(tolerance)/(2*size)
    available = (quantum <= allowance)&(lower < upper)
    initial_count = int(available.sum())
    trace,selections,seen = [],[],set()
    passed,spent = False,0
    while spent < remaining:
        residual = residual_readback(extended,rhs,computed)
        stationarity = computed.astype(np.longdouble)-ideal
        actual = dict(minimum_multiplier=float(weights.min()),minimum_residual=float(residual.min()),
            complementarity=float(np.max(np.abs(weights*residual))),
            stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
        passed = bool(np.isfinite(computed).all() and weights.min() >= 0 and residual.min() >= -tolerance
            and np.max(np.abs(weights*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
        trace.append(actual)
        spent += 1
        pin = sha(computed.tobytes()+available.tobytes())
        if passed or pin in seen or remaining-spent < 2 or int(available.sum()) < size:
            break
        seen.add(pin)
        eligible = np.flatnonzero(available)
        _,_,pivots = qr(g[support][:,eligible],mode='economic',pivoting=True)
        selected = eligible[pivots[:size]]
        matrix = g[support][:,selected]
        singular = np.linalg.svd(matrix,compute_uv=False)
        threshold = size*np.finfo(np.float64).eps*singular[0]
        rank = int(np.sum(singular > threshold))
        spent += 1
        item = dict(eligible=len(eligible),coordinates=selected.tolist(),rank=rank,
            condition=float(singular[0]/singular[-1]) if rank == size else None,
            quantum=quantum[selected].astype(float).tolist())
        selections.append(item)
        if rank != size:
            break
        try:
            correction = np.linalg.solve(matrix,-residual[support].astype(np.float64)).astype(np.longdouble)
        except np.linalg.LinAlgError:
            break
        original = computed[selected].copy()
        current = original.astype(np.longdouble)
        ratios = np.full(size,np.inf,dtype=np.longdouble)
        positive,negative = correction > 0,correction < 0
        ratios[positive] = (upper[selected][positive].astype(np.longdouble)-current[positive])/correction[positive]
        ratios[negative] = (lower[selected][negative].astype(np.longdouble)-current[negative])/correction[negative]
        first = int(np.argmin(ratios))
        scale = min(np.longdouble(1),ratios[first])
        check(scale >= 0,'BOX_NEGATIVE_SCALE')
        computed[selected] = np.clip((current+scale*correction).astype(np.float64),lower[selected],upper[selected])
        if scale < 1:
            computed[selected[first]] = upper[selected[first]] if correction[first] > 0 else lower[selected[first]]
        saturated = (computed[selected] == lower[selected])|(computed[selected] == upper[selected])
        available[selected[saturated]] = False
        if np.array_equal(computed[selected],original):
            available[selected] = False
        item.update(scale=float(scale),saturated=int(saturated.sum()),remaining=int(available.sum()))
    result = dict(report,method=RULE,box_predecessor=report['method'],box_original_actual=report['actual'],
        box_refinement=dict(support=np.flatnonzero(support).tolist(),eligible=initial_count,
            allowance=float(allowance),selections=selections,trace=trace,remaining=int(available.sum())),
        iterations=report['iterations']+spent,actual=actual,actual_certified=passed,
        products=(residual+rhs).astype(np.float64).tolist(),norm=float(np.linalg.norm(computed)),
        status='certified_minimum_norm' if passed else 'box_parameter_certificate_failed')
    check(result['iterations'] <= report['iteration_limit'],'BOX_BUDGET')
    return computed,result


def solve(gradients,deficit):
    delta,report = coordinate_region_solve.solve(gradients,deficit)
    if delta is not None or report['status'] != 'coordinate_parameter_certificate_failed':
        return delta,report
    if report['iteration_limit']-report['iterations'] <= 1:
        return None,dict(report,box_status='iteration_limit')
    computed,result = refine(gradients.detach().cpu().numpy(),deficit.detach().cpu().numpy(),report)
    return (torch.from_numpy(computed).to(gradients.device) if result['actual_certified'] else None),result
