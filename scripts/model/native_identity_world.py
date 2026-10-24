"""Separate sensory and identity instruments coupled inside the recurrent model.

This is an engineering consumer of candidate instruments, not admission. Native
world slots remain separately typed in ConstellationWorld. The fixed textual
standard has its original instruments; no visual/audio standard is fabricated.
"""
from dataclasses import dataclass

import torch
from torch import nn

from model.attention import Attention, Linear
from model.constellation_world import ConstellationWorld
from model.contracts import validate_bank
from model.identity_loop import ContextState
from model.native import NativeInput
from model.stream_attention import blocks, observe
from model.event_temporal_input import FEATURES, check_catalogue
from model.episode_input import EpisodeBatch, validate_observation
from model.world_loop import WorldState
from senses.contract import canonical, digest, require
import json


@dataclass(frozen=True)
class SensoryState(WorldState):
    identity_hidden: torch.Tensor | None = None


class SensoryIdentityCore(nn.Module):
    def __init__(self, identity, specs, temporal_catalogue):
        super().__init__()
        validate_bank(specs)
        self.identity, self.specs, self.config = identity, tuple(specs), identity.config
        self.standard_id, self.source_id = identity.standard_id, identity.source_id
        self._temporal = canonical(temporal_catalogue)
        self.temporal_id = digest(temporal_catalogue)
        h, k = self.config.width, len(self.specs)
        expected = sum(p.numel() for p in identity.parameters()) + k*h + 12*h*h + 14*h
        expected += sum(s.dims*h+h + 10*h*h+10*h for s in specs)
        expected += k*(FEATURES*h+h)
        require(expected <= self.config.max_parameters, "MNLM_SENSORY_PARAMETER_BUDGET",
                "declare a bounded architecture before allocation")
        self.initial = nn.Parameter(torch.zeros(k, h))
        self.projectors = nn.ModuleDict({s.name: Linear(s.dims, h) for s in specs})
        self.temporal_projectors = nn.ModuleDict({s.name: Linear(FEATURES, h) for s in specs})
        self.observe = nn.ModuleDict({s.name: Attention(self.config) for s in specs})
        self.update = nn.ModuleDict({s.name: nn.GRUCell(h, h) for s in specs})
        self.mix, self.to_identity, self.from_identity = (Attention(self.config) for _ in range(3))
        self.norm = nn.LayerNorm(h)
        require(sum(p.numel() for p in self.parameters()) == expected,
                "MNLM_SENSORY_PARAMETER_COUNT", "parameter accounting changed")

    def check_reference(self):
        self.identity.check_reference()
        require(digest(json.loads(self._temporal)) == self.temporal_id,
                "MNLM_TEMPORAL_DRIFT", "restore the fixed temporal source catalogue")

    def start(self, context):
        state = self.identity.start(context)
        return SensoryState(context, self.standard_id, 0, self.initial.clone(),
                            identity_hidden=state.hidden)

    def observe_native(self, native, prior, phase_value):
        updates = []
        for i, spec in enumerate(self.specs):
            x = native[spec.name]
            stream = x.blocks(self.config) if isinstance(native, EpisodeBatch) else blocks(
                x.values, x.padding, self.config, positions=x.positions())
            counts = (x.count,) if isinstance(native, EpisodeBatch) else tuple((~x.padding).sum(1).tolist())
            value = observe(self.observe[spec.name], self.projectors[spec.name],
                (prior[:, i]+phase_value)[:, None], stream, counts=counts, config=self.config,
                positioner=self.temporal_projectors[spec.name])[:, 0]
            updates.append(self.update[spec.name](value+phase_value, prior[:, i]))
        return torch.stack(updates, 1)

    def forward(self, native, states, *, phase, event_ids, evidence_origin, advance_phase=True):
        validate_observation(native, self.specs, self.config)
        require(digest(json.loads(self._temporal)) == self.temporal_id,
                "MNLM_TEMPORAL_DRIFT", "restore the fixed temporal source catalogue")
        if isinstance(native, EpisodeBatch):
            native.check_catalogue(json.loads(self._temporal))
        else:
            check_catalogue(native, json.loads(self._temporal))
        b, k, h = len(states), len(self.specs), self.config.width
        device = self.initial.device
        require(b > 0 and all(isinstance(s, SensoryState) and s.hidden.shape == (k, h)
                and s.hidden.dtype == torch.float32 and s.hidden.device == device
                and bool(torch.isfinite(s.hidden).all()) and s.identity_hidden is not None for s in states),
                "MNLM_SENSORY_STATE", "restore both typed context states")
        require(type(phase) is int and 0 <= phase < 5, "MNLM_LOOP_PHASE", "named phase required")
        require((b == 1 if isinstance(native, EpisodeBatch) else
                 all(len(x.rows) == b and x.values.device == device and not bool(x.absent.any())
                     for x in native.values())), "MNLM_SENSORY_OBSERVATION",
                "complete source-bound sensory observations required")
        prior = torch.stack([s.hidden for s in states])
        identity_states = tuple(ContextState(s.context, s.standard_id, s.sequence,
                                             s.identity_hidden, s.last_event) for s in states)
        prior_identity = torch.stack([s.hidden for s in identity_states])
        require(prior_identity.shape == (b, len(self.identity.specs), h)
                and prior_identity.device == device and prior_identity.dtype == torch.float32
                and bool(torch.isfinite(prior_identity).all()),
                "MNLM_SENSORY_IDENTITY_STATE", "restore typed identity context")
        phase_value = self.identity.phase_embedding.weight[phase]
        typed = self.observe_native(native, prior, phase_value)
        typed = self.norm(typed+self.mix(typed, typed))
        coupling = self.to_identity(prior_identity+phase_value, typed)
        # Explicitly absent: these source events have no readings under the
        # original identity runtime. Their separately typed sensory path is above.
        rows = next(iter(native.values())).rows
        absent = {s.name: NativeInput(s.identity, rows, torch.zeros(b, 1, s.dims, device=device),
                    torch.ones(b, 1, dtype=torch.bool, device=device),
                    torch.ones(b, dtype=torch.bool, device=device)) for s in self.identity.specs}
        result = self.identity(absent, identity_states, phase=phase, event_ids=event_ids,
                               evidence_origin=evidence_origin, external_observation=coupling,
                               advance_phase=advance_phase)
        identity_hidden = torch.stack([s.hidden for s in result["states"]])
        hidden = self.norm(typed+self.from_identity(typed, identity_hidden))
        require(bool(torch.isfinite(hidden).all()), "MNLM_SENSORY_NONFINITE",
                "reject state and restore preceding committed version")
        return {**result, "states": tuple(SensoryState(s.context, s.standard_id, s.sequence,
            hidden[j], s.last_event, identity_hidden=s.hidden) for j, s in enumerate(result["states"]))}


class NativeIdentityWorld(ConstellationWorld):
    def bind_state(self, state):
        require(isinstance(state, SensoryState), "MNLM_SENSORY_STATE",
                "world and identity context must travel together")
        return SensoryState(state.context, state.standard_id, state.sequence, state.hidden,
                            state.last_event, self.world_id, state.identity_hidden)
