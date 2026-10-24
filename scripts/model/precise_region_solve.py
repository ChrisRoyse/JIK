"""Repair numerical optimality certificates without relaxing their original tolerance."""
import mpmath as mp
import numpy as np
import torch

from claims.codata_source import check
from model import region_correction_solve

RULE = 'pairwise_extended_gram_high_precision_face_v1'


def decimal(value):
    return np.format_float_scientific(value,precision=25,unique=False)


def face_solve(gram, rhs, initial, tolerance, budget):
    """Refine the actual active face, charging every iteration to the inherited budget."""
    with mp.workdps(80):
        matrix = [[mp.mpf(v) for v in row] for row in gram]
        b = [mp.mpf(float(v)) for v in rhs]
        weights = [mp.mpf(float(v)) if v > tolerance else mp.mpf(0) for v in initial]
        active = [i for i,w in enumerate(weights) if w > 0]
        trace = []
        for _ in range(budget):
            if not active:
                active.append(int(np.argmax(rhs)))
            try:
                solution = mp.lu_solve(mp.matrix([[matrix[i][j] for j in active] for i in active]),
                                      mp.matrix([b[i] for i in active]))
            except (ValueError,ZeroDivisionError):
                return None,trace,'singular_face'
            candidate = [mp.mpf(0) for _ in rhs]
            for i,value in zip(active,solution):
                candidate[i] = value
            negative = [i for i in active if candidate[i] < 0]
            trace.append(dict(active=active.copy(),negative=negative))
            if negative:
                ratios = [(weights[i]/(weights[i]-candidate[i]),i) for i in negative]
                ratio,dropped = min(ratios)
                weights = [w+ratio*(v-w) for w,v in zip(weights,candidate)]
                weights[dropped] = mp.mpf(0)
                active.remove(dropped)
                if min(weights) < 0:
                    return None,trace,'negative_multiplier'
                continue
            weights = candidate
            residual = [mp.fsum(v*w for v,w in zip(row,weights))-value for row,value in zip(matrix,b)]
            if min(residual) >= -tolerance and max(abs(w*r) for w,r in zip(weights,residual)) <= tolerance:
                return [mp.nstr(w,75) for w in weights],trace,'certified_face'
            entered = min(range(len(rhs)),key=lambda i:residual[i])
            if entered in active:
                return None,trace,'face_residual'
            active.append(entered)
        return None,trace,'iteration_limit'


def solve(gradients, deficit):
    delta, old = region_correction_solve.solve(gradients,deficit)
    if delta is not None or 'weights' not in old:
        return delta,dict(old,method='original_fp64')
    check(np.finfo(np.longdouble).nmant >= 63,'PRECISE_SOLVE_RUNTIME_PRECISION')
    active = np.array(old['active'])
    norms = np.array(old['norms'])
    g = gradients.detach().cpu().numpy()[active]/norms[active,None]
    rhs = deficit.detach().cpu().numpy()[active]/norms[active]
    check(np.array_equal(rhs,old['rhs']),'PRECISE_SOLVE_NORMALIZATION')
    extended = g.astype(np.longdouble)
    gram = np.stack([np.sum(extended*row,axis=1,dtype=np.longdouble) for row in extended])
    precise = [[decimal(v) for v in row] for row in gram]
    remaining = old['iteration_limit']-old['iterations']
    weights,trace,status = face_solve(precise,rhs,old['weights'],old['tolerance'],remaining)
    report = dict(old,method=RULE,original_status=old['status'],original_weights=old['weights'],
        precise_gram=precise,precise_refinement=trace,precise_refinement_status=status,
        iterations=old['iterations']+len(trace),weights=weights,precision_digits=80,
        extended_mantissa=np.finfo(np.longdouble).nmant)
    if weights is None:
        return None,dict(report,status='precise_face_failed')
    w = np.array(weights,dtype=np.longdouble)
    computed = (extended.T@w).astype(np.float64)
    residual = np.sum(extended*computed.astype(np.longdouble),axis=1,dtype=np.longdouble)-rhs.astype(np.longdouble)
    stationarity = computed.astype(np.longdouble)-extended.T@w
    tolerance = old['tolerance']
    actual = dict(minimum_multiplier=float(w.min()),minimum_residual=float(residual.min()),
        complementarity=float(np.max(np.abs(w*residual))),
        stationarity_max=float(np.max(np.abs(stationarity))),tolerance=tolerance)
    passed = bool(np.isfinite(computed).all() and w.min() >= 0 and residual.min() >= -tolerance
        and np.max(np.abs(w*residual)) <= tolerance and np.max(np.abs(stationarity)) <= tolerance)
    report.update(actual=actual,products=(residual+rhs).astype(np.float64).tolist(),
                  norm=float(np.linalg.norm(computed)),actual_certified=passed)
    if not passed:
        return None,dict(report,status='precise_parameter_certificate_failed')
    return torch.from_numpy(computed).to(gradients.device),dict(report,status='certified_minimum_norm')
