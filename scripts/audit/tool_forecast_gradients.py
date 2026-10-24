"""Independent classification gradients over typed native windows; no updates."""
import base64

import numpy as np
import torch
import torch.nn.functional as F

from claims.codata_source import check
from audit.tool_native_recognition import scores


def inspect(inputs, record, core, extra, hidden, label):
    """Rebuild the score expression in float64, checking the NumPy readback."""
    refs = record["region_context"]["definition"]["history"]
    request = record["request"]
    expected = [inputs.population.lookup[request["task"], t]
                for t in range(1, request["turn"])]
    check(refs == expected, "FORECAST_DIAG_HISTORY")
    units = record["region_context"]["definition"]["units"]
    native_hidden = np.frombuffer(base64.b64decode(hidden, validate=True), dtype="<f4")
    native_hidden = native_hidden.reshape(len(inputs.specs), -1).astype(np.float64)
    expected_scores, _ = scores(inputs, record, core, extra, hidden)
    tensors, parameters, errors = [], {}, {}
    as_tensor = lambda a: torch.tensor(a, dtype=torch.float64)
    for lane, spec in enumerate(inputs.specs):
        name = spec.name
        array, offsets = inputs.arrays[name]
        pw, pb = (as_tensor(core[f"projectors.{name}.{part}"])
                  for part in ("weight", "bias"))
        qw, qb = (as_tensor(extra[f"query.{name}.{part}"]).requires_grad_()
                  for part in ("weight", "bias"))
        parameters[f"query.{name}.weight"] = qw
        parameters[f"query.{name}.bias"] = qb

        def project(unit):
            windows = as_tensor(array[offsets[unit]:offsets[unit+1]])
            return F.normalize(windows@pw.T+pb, dim=-1)

        state = torch.zeros(native_hidden.shape[1], dtype=torch.float64)
        for ref in refs:
            for window in project(ref["unit"]):
                state = torch.tanh((state+window)@qw.T+qb)
        context = F.normalize(as_tensor(native_hidden[lane])+state, dim=-1)
        query = F.normalize((project(request["reference"]["unit"])+context)@qw.T+qb, dim=-1)
        values = torch.stack([(query@project(unit).T*10).max(dim=1).values.mean()
                              for unit in units])
        error = float(np.abs(values.detach().numpy()-expected_scores[name]).max())
        check(error < 1e-10, "FORECAST_DIAG_SCORE")
        errors[name] = error
        tensors.append(values)
    # Lane log-normalizers cancel under the common-region cross-entropy.
    logits = torch.stack(tensors).sum(dim=0)
    loss = F.cross_entropy(logits[None], torch.tensor([label]))/len(inputs.specs)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    return dict(loss=float(loss.detach()), errors=errors,
                gradients={k: g.numpy() for k, g in zip(parameters, gradients)})


def geometry(gradients, names, extra, final):
    """Only parameter-gradient inner products; native slots are never flattened."""
    def dot(a, b):
        return float(sum(np.sum(a[k]*b[k]) for k in a))
    norms = {name: dot(g, g)**.5 for name, g in gradients.items()}
    pairs = []
    for i, left in enumerate(names):
        for right in names[i+1:]:
            denominator = norms[left]*norms[right]
            check(denominator > 0, "FORECAST_DIAG_ZERO_GRADIENT")
            pairs.append(dict(left=left, right=right,
                              cosine=dot(gradients[left], gradients[right])/denominator))
    update = {k: final[k].astype(np.float64)-extra[k].astype(np.float64)
              for k in next(iter(gradients.values()))}
    return dict(norms=norms, pairs=pairs,
                observed_parameter_displacement_derivative={
                    name: dot(g, update) for name, g in gradients.items()})
