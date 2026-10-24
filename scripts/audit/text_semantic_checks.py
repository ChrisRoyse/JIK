"""Real-source boundary controls for the native semantic Transformer connection."""
from pathlib import Path

import torch

from claims.codata_source import check
from model.text_semantic_data import PrefixBatch
from senses.contract import digest
from text.constellation_corpus import file_sha


def controls(model, data):
    group = data.groups[0]
    left = group["indices"][3]
    native = data.batch(left)
    controls = []
    def deny(name, indices):
        try:
            PrefixBatch(data, left, indices=indices).validate(model.specs, model.config)
        except Exception as exc:
            check(getattr(exc, "code", str(exc)) == "MNLM_CODATA_SEMANTIC_PREFIX_CUTOFF", "SEMANTIC_CONTROL_CODE")
            controls.append(name)
        else:
            raise ValueError("MNLM_SEMANTIC_BAD_PREFIX_ACCEPTED:"+name)
    prefix = list(native.indices)
    deny("future_atom", prefix+[group["indices"][4]])
    deny("omitted_atom", prefix[1:])
    deny("reordered_atoms", list(reversed(prefix)))
    deny("foreign_document", [data.groups[1]["indices"][0]]+prefix[1:])
    deny("bool_aliased_index", [bool(prefix[0])]+prefix[1:])
    check(all(data.region_indices(i) == data.refs for i in data.successor), "SEMANTIC_VOCABULARY_CHANGED")
    result = dict(code_sha256=file_sha(Path(__file__)), prefix_controls=controls,
        fixed_region_vocabulary=True, book=data.book_pin, shared_initial=data.shared_initial,
        prefix_shapes=data.shapes, train_region_occupancy=data.book["train_occupancy"])
    if data.transformer:
        sequence = model.core.sequence_transformer
        frames = native.frames
        replacement = next(f for f in frames[:-1]
            if all(f[n].shape == frames[-1][n].shape for n in f))
        with torch.no_grad():
            base = sequence.encode(frames, model.core.projectors)
            changed = sequence.encode(frames[:-1]+(replacement,), model.core.projectors)
            reordered = sequence.encode(tuple(reversed(frames[:-1]))+frames[-1:], model.core.projectors)
        check(torch.equal(base[:-1], changed[:-1]), "SEMANTIC_FUTURE_LEAK")
        check(not torch.equal(base[-1], changed[-1]) and not torch.equal(base[-1], reordered[-1]),
              "SEMANTIC_IGNORED_READING_OR_ORDER")
        prior = model.start(group["context"])
        for index in prefix[:-1]:
            with torch.no_grad():
                prior = data.forward(model, prior, index, digest(dict(control="semantic", index=index)))["states"][0]
        out = data.forward(model, prior, left, digest(dict(control="semantic", index=left)))
        gradient, = torch.autograd.grad(out["states"][0].identity_hidden[0, 0], sequence.query)
        check(torch.isfinite(gradient).all() and gradient.abs().max() > 0, "SEMANTIC_IDENTITY_DISCONNECTED")
        result.update(causal_prior_exact=True, changed_current_max=float((base[-1]-changed[-1]).abs().max()),
            changed_past_order_max=float((base[-1]-reordered[-1]).abs().max()),
            transformer_to_identity_gradient_max=float(gradient.abs().max()))
    return result
