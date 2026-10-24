"""Outcome-blind reconstruction closure for a CONDITIONAL assay geometry.

Fit h(Z) from disjoint design readings only. The assay must retain all of Z and
append h(Z): sigma(Z,h(Z))=sigma(Z). Dropping Z or substituting predictions is
forbidden. This is a finite-sample geometry experiment, not a CMI certificate.
Affine and single-coordinate periodic residuals cover a limited model family.
"""
import numpy as np
from scipy.optimize import minimize_scalar

RIDGE = 1e-6


def fit(panel, candidate):
    """No labels, ground truth, evaluation readings or admission score accepted."""
    z, x = np.asarray(panel, float), np.asarray(candidate, float)
    if z.ndim != 2 or x.ndim != 2 or len(z) != len(x) or len(z) < 50:
        raise ValueError("MNLM_CONDITIONAL_GEOMETRY_DESIGN")
    if not np.isfinite(z).all() or not np.isfinite(x).all():
        raise ValueError("MNLM_CONDITIONAL_GEOMETRY_FINITE")
    mean = z.mean(0); scale = z.std(0); scale[scale == 0] = 1
    normalized = (z-mean)/scale
    a = np.column_stack((np.ones(len(z)), normalized))
    q, singular, _ = np.linalg.svd(a, full_matrices=False)
    q = q[:, singular > singular[0]*1e-8]
    residual = x-q@(q.T@x)
    best = np.zeros(x.shape[1]); choices = [None]*x.shape[1]
    # Uniform angular-frequency grid derived only from design count and span.
    # 4*n bins; up to one cycle per mean design spacing (irregular samples).
    # No control-frequency constants are used; aliases remain a model limitation.
    for column in range(z.shape[1]):
        coordinate = z[:, column]; span = np.ptp(coordinate)
        if span == 0:
            continue
        step = np.pi/(2*span)
        frequencies = step*np.arange(1, 4*len(z)+1)
        phase = coordinate[:, None]*frequencies[None, :]
        sine, cosine = np.sin(phase), np.cos(phase)
        sine -= q@(q.T@sine); cosine -= q@(q.T@cosine)
        ss = np.sum(sine*sine, axis=0); cc = np.sum(cosine*cosine, axis=0)
        sc = np.sum(sine*cosine, axis=0); determinant = ss*cc-sc*sc
        usable = determinant > 1e-12
        sx, cx = sine.T@residual, cosine.T@residual
        score = np.where(usable[:, None],
            (cc[:, None]*sx*sx+ss[:, None]*cx*cx-2*sc[:, None]*sx*cx)
            /np.where(usable, determinant, 1)[:, None], -np.inf)
        winners = np.argmax(score, axis=0)
        for target, index in enumerate(winners):
            if score[index, target] > best[target]:
                best[target] = score[index, target]
                choices[target] = (column, float(frequencies[index]), float(step))
    models = []
    for target, choice in enumerate(choices):
        value = x[:, target]
        def solve(matrix):
            gram = matrix.T@matrix
            coef = np.linalg.solve(gram+RIDGE*np.eye(len(gram)), matrix.T@value)
            return coef, float(np.sum((value-matrix@coef)**2))
        coef, error = solve(a)
        model = dict(kind="affine", coefficients=coef.tolist(), design_sse=error)
        if choice is not None:
            column, frequency, step = choice
            coordinate = z[:, column]
            def design(f):
                return np.column_stack((a, np.sin(f*coordinate), np.cos(f*coordinate)))
            def objective(f):
                return solve(design(f))[1]
            refined = minimize_scalar(objective, bounds=(max(step/100, frequency-step), frequency+step),
                                      method="bounded", options={"xatol": 1e-10})
            if not refined.success:
                raise ValueError("MNLM_CONDITIONAL_GEOMETRY_OPTIMIZER")
            spectral_coef, spectral_error = solve(design(refined.x))
            if spectral_error < error:
                model = dict(kind="periodic", coefficients=spectral_coef.tolist(),
                             column=column, frequency=float(refined.x), design_sse=spectral_error)
        models.append(model)
    return dict(schema="conditional-reconstruction-geometry-v1", mean=mean.tolist(),
                scale=scale.tolist(), panel_dimensions=z.shape[1], candidate_dimensions=x.shape[1],
                design_rows=len(z), ridge=RIDGE, models=models, labels_used=False,
                preserves_panel="append_prediction_to_complete_panel")


def predict(model, panel):
    z = np.asarray(panel, float)
    if z.ndim != 2 or z.shape[1] != model["panel_dimensions"] or not np.isfinite(z).all():
        raise ValueError("MNLM_CONDITIONAL_GEOMETRY_COMPLETE_PANEL")
    a = np.column_stack((np.ones(len(z)), (z-np.array(model["mean"]))/np.array(model["scale"])))
    columns = []
    for item in model["models"]:
        matrix = a
        if item["kind"] == "periodic":
            phase = item["frequency"]*z[:, item["column"]]
            matrix = np.column_stack((a, np.sin(phase), np.cos(phase)))
        elif item["kind"] != "affine":
            raise ValueError("MNLM_CONDITIONAL_GEOMETRY_MODEL")
        columns.append(matrix@np.asarray(item["coefficients"]))
    return np.column_stack(columns)
