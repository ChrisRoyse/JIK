"""Per-instrument SIGReg, independently implemented from the ECF formula.

Balestriero and LeCun, LeJEPA (arXiv:2511.08544). Reference algorithm:
galilai-group/lejepa, commit c293d291ca87cd4fddee9d3fffe4e914c7272052.
No reference source is vendored. Native instrument geometry remains frozen;
only each learned typed latent is regularized. This is not a sense or D2 audit.
"""
from pathlib import Path

import torch

from model.text_atom_associations import AtomAssociationData
from senses.contract import digest
from text.constellation_corpus import file_sha, need


RECIPE = dict(slices=256, knots=17, upper_frequency=3., seed=20261021,
    seed_rule="base_plus_persisted_optimizer_step", slot_policy="independent_per_instrument",
    target="standard_normal_characteristic_function", lambda_choices=[0., .02],
    source="https://arxiv.org/abs/2511.08544")


def directions(values, step):
    need(values.ndim == 3 and values.is_cuda and values.dtype == torch.float32
         and type(step) is int and 0 <= step <= 16, "SIGREG_INPUT_CONTRACT")
    generator = torch.Generator(device=values.device).manual_seed(RECIPE["seed"]+step)
    matrix = torch.randn(values.shape[0], values.shape[2], RECIPE["slices"],
                         device=values.device, generator=generator)
    return matrix / torch.linalg.vector_norm(matrix, dim=1, keepdim=True)


def statistic(values, projection):
    """N times the Gaussian-weighted integral of squared ECF error, per slot."""
    need(values.ndim == 3 and projection.shape == (values.shape[0], values.shape[2], 256)
         and torch.isfinite(values).all() and torch.isfinite(projection).all(),
         "SIGREG_NONFINITE_OR_SHAPE")
    grid = torch.linspace(0., 3., 17, device=values.device, dtype=values.dtype)
    expected = torch.exp(-.5*grid*grid)
    phase = torch.einsum("snd,sdk->snk", values, projection)[..., None]*grid
    real = torch.cos(phase).mean(dim=1)
    imaginary = torch.sin(phase).mean(dim=1)
    discrepancy = ((real-expected)**2 + imaginary**2)*expected
    return 2*values.shape[1]*torch.trapezoid(discrepancy, grid, dim=-1).mean(dim=-1)


def projected_states(model, data, hidden):
    return torch.stack([model.query[name](hidden[:, lane]) for lane, name in enumerate(data.lanes)])


class SigregData(AtomAssociationData):
    def __init__(self, corpus, bank, config, strength):
        super().__init__(corpus, bank, config)
        need(strength in RECIPE["lambda_choices"], "SIGREG_UNDECLARED_STRENGTH")
        self.strength = strength
        self.bind = {**self.bind, "regularizer": dict(recipe=RECIPE, strength=strength,
            code=file_sha(Path(__file__)),
            driver=file_sha(Path(__file__).parents[1]/"audit/text_sigreg_learning.py"))}
        self.pin = digest(self.bind)

    def objective(self, model, optimizer, hidden, succession, relation):
        steps = {int(s["step"].item()) for s in optimizer.state.values()}
        need(len(steps) <= 1, "SIGREG_OPTIMIZER_CURSOR")
        step = next(iter(steps)) if steps else 0
        values = projected_states(model, self, hidden)
        penalties = statistic(values, directions(values, step))
        regularizer = penalties.mean()
        loss = succession + (1-self.strength)*relation + self.strength*regularizer
        return loss, dict(sigreg=float(regularizer.detach()), sigreg_step=step,
            per_slot_sigreg=penalties.detach().cpu().tolist(),
            per_slot_coordinate_variance=values.detach().var(dim=1, unbiased=False).mean(-1).cpu().tolist())
