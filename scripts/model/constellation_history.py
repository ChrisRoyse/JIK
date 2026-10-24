"""A trainable interpretation of an attested, ordered constellation prefix."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from claims.codata_source import check
from model.constellation import Constellation


@dataclass(frozen=True)
class CausalHistory:
    context: str
    rows: tuple[str, ...]
    frames: tuple[Constellation, ...]

    def validate(self, model, states):
        check(len(states) == 1 and self.context == states[0].context, "CONSTELLATION_HISTORY_CONTEXT")
        check(self.rows and len(self.rows) == len(self.frames), "CONSTELLATION_HISTORY_ROSTER")
        elements, tokens = 0, {s.name: 0 for s in model.specs}
        for row, frame in zip(self.rows, self.frames):
            check(frame.origin == "observed", "CONSTELLATION_HISTORY_ORIGIN")
            frame.validate(model.specs, model.config)
            for spec in model.specs:
                x = frame.slots[spec.name]
                check(x.rows == (row,) and not bool(x.absent.any()), "CONSTELLATION_HISTORY_FRAME")
                elements += x.values.numel()
                tokens[spec.name] += x.values.shape[1]
        check(elements <= model.config.max_native_elements
              and max(tokens.values()) <= model.config.max_native_tokens, "CONSTELLATION_HISTORY_BUDGET")


def interpret(model, hidden, states, history):
    if history is None:
        return hidden
    check(isinstance(history, CausalHistory), "CONSTELLATION_HISTORY_ROSTER")
    history.validate(model, states)
    contexts = []
    for lane, spec in enumerate(model.specs):
        current = torch.zeros_like(hidden[:, lane])
        for frame in history.frames:
            slot = frame.slots[spec.name]
            projected = F.normalize(model.core.projectors[spec.name](slot.values), dim=-1)
            for window in range(projected.shape[1]):
                update = torch.tanh(model.query[spec.name](current+projected[:, window]))
                current = torch.where(slot.padding[:, window, None], current, update)
        contexts.append(hidden[:, lane]+current)
    return torch.stack(contexts, dim=1)
