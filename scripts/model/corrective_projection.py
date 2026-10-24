"""Seek a strict common descent direction without changing any metric ceiling."""
import numpy as np
import torch
from scipy.optimize import minimize

from claims.codata_source import check


def refine(gram, weights, tolerance, budget):
    """Refine the active simplex face; every affine solve spends one iteration."""
    weights = weights.copy()
    weights[weights <= tolerance] = 0.
    weights /= weights.sum()
    active = list(np.flatnonzero(weights > 0))
    trace = []
    for _ in range(budget):
        block = gram[np.ix_(active,active)]
        size = len(active)
        system = np.block([[block,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
        rhs = np.zeros(size+1)
        rhs[-1] = 1.
        affine, _, rank, _ = np.linalg.lstsq(system,rhs,rcond=None)
        candidate = np.zeros(len(weights))
        candidate[active] = affine[:-1]
        negative = np.flatnonzero(candidate < 0)
        trace.append(dict(active=len(active),rank=int(rank),negative=len(negative)))
        if len(negative):
            ratios = weights[negative]/(weights[negative]-candidate[negative])
            dropped = int(negative[np.argmin(ratios)])
            fraction = float(ratios.min())
            weights += fraction*(candidate-weights)
            weights[dropped] = 0.
            active.remove(dropped)
            if not np.all(weights >= 0):
                return weights,trace,'negative_weight'
            continue
        weights = candidate
        products = gram@weights
        energy = float(weights@products)
        if np.min(products-energy) >= -tolerance and np.max(np.abs(weights*(products-energy))) <= tolerance:
            return weights,trace,'certified_face'
        entered = int(np.argmin(products))
        if entered in active:
            return weights,trace,'face_residual'
        active.append(entered)
    return weights,trace,'iteration_limit'


def project(gradients, displacement):
    check(gradients.ndim == 2 and displacement.ndim == 1 and gradients.shape[1] == len(displacement)
          and bool(torch.isfinite(gradients).all()) and bool(torch.isfinite(displacement).all()),
          'CORRECTION_PROJECTION_INPUT')
    norms = torch.linalg.vector_norm(gradients,dim=1)
    active = norms > 0
    g = gradients[active]/norms[active,None]
    gram = (g@g.T).cpu().numpy()
    count = len(g)
    report = dict(rule='minimum_norm_common_descent_v1',rows=len(gradients),active_rows=count,
                  gram=gram.tolist(),proposal_norm=float(torch.linalg.vector_norm(displacement)))
    if not count:
        return None, dict(report,status='no_active_gradients')
    # A minimum-norm convex combination v satisfies g_i.v >= ||v||².
    # Thus -v is strictly inward for every active row when ||v|| > 0.
    tolerance = 64*np.finfo(np.float64).eps*max(1.,float(np.linalg.norm(gram,np.inf)))
    solved = minimize(lambda w:.5*float(w@gram@w),np.full(count,1./count),
        jac=lambda w:gram@w,method='SLSQP',bounds=[(0.,1.)]*count,
        constraints=[dict(type='eq',fun=lambda w:float(w.sum()-1.),jac=lambda w:np.ones(count))],
        options=dict(maxiter=3*count,ftol=tolerance))
    weights, refinement, refinement_status = refine(gram,solved.x,tolerance,max(0,3*count-int(solved.nit)))
    products = gram@weights
    energy = float(weights@products)
    report.update(weights=weights.tolist(),products=products.tolist(),energy=energy,tolerance=tolerance,
                  iterations=int(solved.nit)+len(refinement),solver_iterations=int(solved.nit),
                  refinement=refinement,refinement_status=refinement_status,
                  solver_success=bool(solved.success),solver_message=str(solved.message))
    if not solved.success or refinement_status != 'certified_face':
        return None, dict(report,status='solver_failed')
    if not (np.isfinite(weights).all() and np.all(weights >= 0) and
            abs(weights.sum()-1.) <= tolerance and np.min(products-energy) >= -tolerance and
            np.max(np.abs(weights*(products-energy))) <= tolerance):
        return None, dict(report,status='simplex_kkt_failed')
    if energy <= 0 or np.min(products) <= 0:
        return None, dict(report,status='no_certified_common_descent')
    delta = -g.T@torch.from_numpy(weights).to(displacement.device)
    check(bool(torch.isfinite(delta).all()),'CORRECTION_PROJECTION_FINITE')
    length = float(torch.linalg.vector_norm(delta))
    scale = report['proposal_norm']/length
    delta = delta*scale
    actual = (g@delta).tolist()
    report.update(projected_norm=length,scale=scale,projected_dot=actual)
    if not all(v < 0 for v in actual):
        return None, dict(report,status='reconstructed_direction_not_inward')
    return delta, dict(report,status='common_descent')
