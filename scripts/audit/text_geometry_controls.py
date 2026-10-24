"""Reachability and refusal controls on real, provenance-bound training vectors."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check
from model.text_region_geometry import spherical_place
from text.constellation_corpus import file_sha


def controls(model, data):
    device = model.core.initial.device
    bases, targets, bounds, indices, lanes, regions = [], [], [], [], [], []
    for lane, name in enumerate(data.lanes):
        for index in data.fit:
            region = data.book["codes"][index]
            bases.append(data.native[name][data.refs[region]][0])
            targets.append(data.native[name][index][0])
            bounds.append(data.geometry["angles"][lane][region])
            indices.append(index)
            lanes.append(lane)
            regions.append(region)
    base = F.normalize(torch.from_numpy(np.stack(bases)).to(device), dim=-1)
    target = F.normalize(torch.from_numpy(np.stack(targets)).to(device), dim=-1)
    radius = torch.tensor(bounds, device=device)[:, None]
    dot = (target*base).sum(-1, keepdim=True).clamp(-1, 1)
    tangent = target-dot*base
    length = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    angle = torch.atan2(length, dot)
    fraction = (angle/radius.clamp_min(1e-7)).clamp(0., 1.-1e-4)
    raw = tangent/length.clamp_min(1e-7)*math.sqrt(base.shape[-1])*torch.atanh(fraction)
    with torch.no_grad():
        predicted = spherical_place(base, raw, radius)
        stationary = spherical_place(base, torch.zeros_like(raw), radius)
        error = 1-(predicted*target).sum(-1)
        check(float(error.max()) < 2e-6 and torch.allclose(stationary, base, atol=2e-6, rtol=0),
              "GEOMETRY_ANALYTIC_REACHABILITY")
    probe = torch.zeros_like(raw, requires_grad=True)
    value = (spherical_place(base, probe, radius)*target).sum()
    gradient, = torch.autograd.grad(value, probe)
    check(torch.isfinite(gradient).all() and float(gradient.abs().max()) > 0, "GEOMETRY_ZERO_GRADIENT")
    denied = []
    for name, bad in (("negative", -torch.ones_like(radius)), ("nonfinite", torch.full_like(radius, float("nan")))):
        try:
            spherical_place(base, raw, bad)
        except ValueError as exc:
            check(str(exc) == "MNLM_CODATA_GEOMETRY_BOUNDS", "GEOMETRY_CONTROL_CODE")
            denied.append(name)
        else:
            raise ValueError("MNLM_GEOMETRY_INVALID_BOUNDS_ACCEPTED")
    saved = model.placement_angles.detach().clone()
    try:
        with torch.no_grad():
            model.placement_angles[0, 0] += .01
        try:
            model.check_reference()
        except ValueError as exc:
            check(str(exc) == "MNLM_CODATA_GEOMETRY_CHANGED", "GEOMETRY_CONTROL_CODE")
            denied.append("mutated_geometry")
        else:
            raise ValueError("MNLM_GEOMETRY_MUTATION_ACCEPTED")
    finally:
        with torch.no_grad():
            model.placement_angles.copy_(saved)
    model.check_reference()
    path = data.control_output/"geometry-control.npz"
    with path.open("xb") as handle:
        np.savez(handle, base=base.cpu().numpy(), target=target.cpu().numpy(), predicted=predicted.cpu().numpy(),
                 radius=radius.cpu().numpy(), indices=np.array(indices), lanes=np.array(lanes), regions=np.array(regions))
    return dict(samples=len(indices), analytic_native_error_max=float(error.max()),
        zero_displacement_gradient_max=float(gradient.abs().max()), refused=denied,
        artifact=path.name, sha256=file_sha(path), learned_capability=False,
        interpretation="analytic inverse placement on training vectors; not network predictions")
