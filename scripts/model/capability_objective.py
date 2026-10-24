"""Vector-derived supervision and full-surface VJP through predicted constellations."""
from dataclasses import replace

import torch
from torch.nn import functional as F

from claims.capability_population import need
from model.constellation import Constellation
from model.stream_surface import StreamSurfaceDecoder, whole_surface_loss


class PredictedSurfaceDecoder(StreamSurfaceDecoder):
    def validate_inputs(self, native):
        # Validate the same frozen identity/shape/mask contract while allowing the
        # predictor's derivatives. memories() continues to use original tensors.
        checked = {n: replace(x, values=x.values.detach()) for n, x in native.items()}
        return super().validate_inputs(checked)

    def render_loss(self, raw, predicted, *, backward=False, meaning_loss=None):
        need(isinstance(predicted, Constellation) and predicted.origin == 'predicted',
             'CAPABILITY_DECODER_PREDICTION_REQUIRED')
        predicted.validate(self.specs, self.config)
        leaves = {n: replace(x, values=x.values.detach().requires_grad_(backward))
                  for n, x in predicted.slots.items()}
        result = whole_surface_loss(self, raw, leaves, backward=backward)
        gradients = {}
        if backward:
            values, derivatives = [], []
            for name, leaf in leaves.items():
                gradient = leaf.values.grad
                need(gradient is not None and bool(torch.isfinite(gradient).all()),
                     'CAPABILITY_DECODER_GRADIENT')
                values.append(predicted.slots[name].values); derivatives.append(gradient)
                gradients[name] = gradient.detach().clone()
            # One predictor graph, one VJP; byte-window activations are released
            # each batch. This is the chain rule, not a detached training path.
            if meaning_loss is not None:
                need(meaning_loss.ndim == 0 and bool(torch.isfinite(meaning_loss)),
                     'CAPABILITY_DECODER_MEANING_LOSS')
                values.append(meaning_loss); derivatives.append(torch.ones_like(meaning_loss))
            torch.autograd.backward(values, derivatives)
        return result, gradients


def native_similarity(a, b):
    """Whole ordered arrays retained; this loss measures bidirectional coverage.

It is explicitly not a proof of sequence/order equivalence. Window order remains
in the predictor, renderer and persisted source traces.
"""
    matrix = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
    return (matrix.max(0).values.mean() + matrix.max(1).values.mean()) / 2


def mathematical_labels(view, target):
    """Nearest frozen training region, derived only from target vectors for loss."""
    scores = []
    with torch.no_grad():
        for i, spec in enumerate(view.specs):
            truth = target[spec.name]
            truth.check(spec, view.config)
            t = truth.values[0, ~truth.padding[0]]
            values, padding = getattr(view, f'world_values_{i}'), getattr(view, f'world_padding_{i}')
            scores.append(torch.stack([native_similarity(t, x[~p]) for x, p in zip(values, padding)]))
        joint = torch.stack(scores).mean(0)
        label = joint.argmax().reshape(1)
    return label, {s.name: x for s, x in zip(view.specs, scores)}


def next_constellation_loss(view, output, target):
    label, assignments = mathematical_labels(view, target)
    region, placement = [], []
    for spec in view.specs:
        name = spec.name
        region.append(F.cross_entropy(output['regions'][name], label))
        guess = output['predicted_constellation'].slots[name]
        actual = target[name]
        placement.append(1 - native_similarity(guess.values[0, ~guess.padding[0]],
                                                actual.values[0, ~actual.padding[0]]))
    loss = torch.stack(region).mean() + torch.stack(placement).mean()
    need(bool(torch.isfinite(loss)), 'CAPABILITY_MEANING_LOSS')
    return loss, dict(label=int(label), region_loss=float(torch.stack(region).mean().detach()),
        placement_loss=float(torch.stack(placement).mean().detach()),
        assignment_scores={n: x.detach().cpu().tolist() for n, x in assignments.items()},
        labels='automatically_derived_from_typed_vectors', external_semantic_labels=False)
