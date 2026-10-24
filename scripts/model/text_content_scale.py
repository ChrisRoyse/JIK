"""Explicit content-scale variant; same native bytes, parameters and positions."""
from pathlib import Path

import torch
from torch.nn import functional as F

from model.native_prefix_transformer import NativePrefixTransformer
from model.text_associations import AssociationData
from model.text_semantic_data import build_semantic
from senses.contract import digest
from text.constellation_corpus import file_sha, need


class ContentTransformer(NativePrefixTransformer):
    def encode(self, frames, projectors):
        def balanced(layer):
            return lambda x: F.layer_norm(layer(x), (self.config.width,))
        return super().encode(frames, {n: balanced(projectors[n]) for n in projectors})


class ContentData(AssociationData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config)
        self.bind = {**self.bind, "content_input": dict(
            method="per_typed_projected_window_parameter_free_layer_norm",
            code=file_sha(Path(__file__)),
            driver=file_sha(Path(__file__).parents[1]/"audit/text_content_learning.py"))}
        self.pin = digest(self.bind)

    def representation_controls(self, model):
        return {**super().representation_controls(model), "input_scale": self.input_scale,
                "same_parameter_tensors_before_and_after_normalization": True}


def build_content(data, args):
    model = build_semantic(data, args)
    old = model.core.sequence_transformer
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}
    # Restore identical tensors and RNG; only the forward procedure differs.
    with torch.random.fork_rng(devices=[]):
        new = ContentTransformer(old.config, old.specs).to(old.query.device)
    new.load_state_dict(old.state_dict(), strict=True)
    new.train(old.training)
    model.core.sequence_transformer = new
    need(set(before) == set(model.state_dict()) and all(torch.equal(v, model.state_dict()[n])
        for n, v in before.items()), "CONTENT_SCALE_PARAMETER_CHANGE")
    data.input_scale = {}
    with torch.no_grad():
        for lane, spec in enumerate(model.specs):
            raw, normalized, position = [], [], []
            for group in data.groups:
                for atom, index in enumerate(group["indices"][:-1]):
                    values = data.cached[index][spec.name]
                    z = model.core.projectors[spec.name](values)
                    local = old.position(torch.arange(len(values), device=z.device))
                    added = old.kind[lane]+old.position(torch.tensor([atom], device=z.device))+local
                    raw.append(z.square().mean())
                    normalized.append(F.layer_norm(z, (data.config.width,)).square().mean())
                    position.append(added.square().mean())
            data.input_scale[spec.name] = dict(projected_rms=float(torch.stack(raw).mean().sqrt()),
                normalized_rms=float(torch.stack(normalized).mean().sqrt()),
                position_and_type_rms=float(torch.stack(position).mean().sqrt()))
    return model
