"""Intrinsic identity dynamics with a separately versioned native world-region bank."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from model.attention import Linear
from model.identity_loop import ContextState, reference_hash
from model.native import NativeInput
from senses.contract import require


@dataclass(frozen=True)
class WorldState(ContextState):
    world_id: str = ""


class WorldLoop(nn.Module):
    def __init__(self, core, references, *, world_source_id):
        super().__init__()
        self.core, self.config, self.specs = core, core.config, core.specs
        self.standard_id, self.source_id = core.standard_id, core.source_id
        self.world_source_id = world_source_id
        self.reference_rows = {}
        require(set(references) == {s.name for s in self.specs}, "MNLM_WORLD_ROSTER", "typed bank required")
        sizes, rows = [], None
        for spec in self.specs:
            x = references[spec.name]
            x.check(spec, self.config)
            require(not bool(x.absent.any()) and (rows is None or rows == x.rows),
                    "MNLM_WORLD_ALIGNMENT", "world references must share attested atom identities")
            rows = x.rows
            self.reference_rows[spec.name] = x.rows
            sizes.append(x.values.numel())
        require(0 < len(rows) <= 709 and sum(sizes) <= self.config.max_native_elements,
                "MNLM_WORLD_BANK_BUDGET", "bounded real reference bank required")
        h = self.config.width
        expected = sum(p.numel() for p in core.parameters()) + sum(
            h*h+h + h*s.dims+s.dims for s in self.specs)
        require(expected <= self.config.max_parameters, "MNLM_WORLD_PARAMETER_BUDGET", "architecture exceeds contract")
        self.world_id = reference_hash(self.specs, references, world_source_id)
        self.query, self.place = nn.ModuleDict(), nn.ModuleDict()
        for i, spec in enumerate(self.specs):
            x = references[spec.name]
            self.register_buffer(f"world_values_{i}", x.values.detach().clone())
            self.register_buffer(f"world_padding_{i}", x.padding.detach().clone())
            self.register_buffer(f"world_absent_{i}", x.absent.detach().clone())
            owners = torch.arange(len(x.rows), device=x.values.device)[:, None].expand_as(x.padding)[~x.padding]
            self.register_buffer(f"world_owner_{i}", owners)
            self.query[spec.name], self.place[spec.name] = Linear(h, h), Linear(h, spec.dims)
        require(sum(p.numel() for p in self.parameters()) == expected,
                "MNLM_WORLD_PARAMETER_COUNT", "parameter accounting differs")

    def references(self):
        return {s.name: NativeInput(s.identity, self.reference_rows[s.name], getattr(self, f"world_values_{i}"),
                                    getattr(self, f"world_padding_{i}"), getattr(self, f"world_absent_{i}"))
                for i, s in enumerate(self.specs)}

    def check_reference(self):
        self.core.check_reference()
        require(reference_hash(self.specs, self.references(), self.world_source_id) == self.world_id,
                "MNLM_WORLD_REFERENCE_DRIFT", "restore the pinned world bank")
        for i, spec in enumerate(self.specs):
            padding = getattr(self, f"world_padding_{i}")
            owners = torch.arange(len(self.reference_rows[spec.name]), device=padding.device)[:, None]
            require(torch.equal(owners.expand_as(padding)[~padding], getattr(self, f"world_owner_{i}")),
                    "MNLM_WORLD_OWNER_DRIFT", "restore the source-to-region mapping")

    def start(self, context):
        return self.bind_state(self.core.start(context))

    def bind_state(self, state):
        return WorldState(state.context, state.standard_id, state.sequence, state.hidden,
                          state.last_event, self.world_id)

    def forward(self, native, states, *, allowed, phase, event_ids, evidence_origin):
        require(all(isinstance(s, WorldState) and s.world_id == self.world_id for s in states),
                "MNLM_WORLD_CONTEXT_BANK", "context belongs to another world-bank version")
        rows = next(iter(self.reference_rows.values()))
        require(allowed.dtype == torch.bool and allowed.shape == (len(rows),)
                and allowed.device == self.core.initial.device and bool(allowed.any()),
                "MNLM_WORLD_ALLOWED", "current checked source regions required")
        output = self.core(native, states, phase=phase, event_ids=event_ids, evidence_origin=evidence_origin)
        hidden = torch.stack([s.hidden for s in output["states"]])
        regions, slots, owner = {}, {}, {}
        for i, spec in enumerate(self.specs):
            name = spec.name
            book = getattr(self, f"world_values_{i}")[~getattr(self, f"world_padding_{i}")]
            owners = getattr(self, f"world_owner_{i}")
            require(len(states)*len(book) <= self.config.max_attention_elements,
                    "MNLM_WORLD_SCORE_BUDGET", "reduce active contexts, never truncate source windows")
            q = F.normalize(self.query[name](hidden[:, i]), dim=-1)
            keys = F.normalize(self.core.projectors[name](book), dim=-1)
            scores = (q @ keys.T * 10.).masked_fill(~allowed[owners][None], -torch.inf)
            selected = scores.argmax(-1)
            regions[name] = scores
            owner[name] = owners[selected]
            slots[name] = F.normalize(book[selected]+.1*self.place[name](hidden[:, i]), dim=-1)
        # Identity and world region spaces remain distinct and inspectable.
        return {**output, "states": tuple(self.bind_state(s) for s in output["states"]),
                "identity_regions": output["regions"], "identity_slots": output["slots"],
                "regions": regions, "slots": slots, "world_owner": owner, "world_bank": self.world_id}
