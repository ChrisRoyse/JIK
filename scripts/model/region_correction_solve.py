"""Minimum-norm correction of native-cost region margins, with a dual certificate."""
import numpy as np
from scipy.optimize import minimize
import torch

from claims.codata_source import check
from model.slack_update import refine


def solve(gradients, deficit):
    check(gradients.ndim == 2 and deficit.shape == (len(gradients),)
          and bool(torch.isfinite(gradients).all()) and bool(torch.isfinite(deficit).all()),
          'REGION_SOLVE_INPUT')
    norms = torch.linalg.vector_norm(gradients,dim=1)
    active = norms > 0
    report = dict(norms=norms.tolist(),active=active.tolist(),deficit=deficit.tolist())
    if bool(((~active) & (deficit > 0)).any()):
        return None,dict(report,status='unreachable_margin')
    if not bool((deficit > 0).any()):
        return None,dict(report,status='margins_satisfied')
    g = gradients[active]/norms[active,None]
    rhs = (deficit[active]/norms[active]).cpu().numpy()
    gram = (g@g.T).cpu().numpy()
    count = len(g)
    tolerance = 64*np.finfo(np.float64).eps*max(1.,float(np.linalg.norm(gram,np.inf)),float(np.max(np.abs(rhs))))
    fitted = minimize(lambda w:.5*float(w@gram@w)-float(rhs@w),np.zeros(count),
        jac=lambda w:gram@w-rhs,method='SLSQP',bounds=[(0.,None)]*count,
        options=dict(maxiter=3*count,ftol=tolerance))
    weights,trace,status = refine(gram,rhs,fitted.x,tolerance,max(0,3*count-int(fitted.nit)))
    residual = gram@weights-rhs
    report.update(gram=gram.tolist(),rhs=rhs.tolist(),weights=weights.tolist(),residual=residual.tolist(),
        tolerance=tolerance,solver_success=bool(fitted.success),solver_message=str(fitted.message),
        iterations=int(fitted.nit)+len(trace),iteration_limit=3*count,refinement=trace,refinement_status=status)
    if not fitted.success or status != 'certified_face':
        return None,dict(report,status='solver_failed')
    if not (np.isfinite(weights).all() and np.min(weights) >= 0 and np.min(residual) >= -tolerance
            and np.max(np.abs(weights*residual)) <= tolerance):
        return None,dict(report,status='dual_certificate_failed')
    delta = g.T@torch.from_numpy(weights).to(gradients.device)
    products = g@delta
    report.update(products=products.tolist(),norm=float(torch.linalg.vector_norm(delta)))
    if not bool(torch.isfinite(delta).all()) or bool((products < torch.from_numpy(rhs).to(products.device)-tolerance).any()):
        return None,dict(report,status='primal_certificate_failed')
    return delta,dict(report,status='certified_minimum_norm')
