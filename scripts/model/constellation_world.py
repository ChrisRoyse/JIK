"""One region, complete typed native slots, and bounded placement per native window."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from claims.codata_source import check
from model.constellation import Constellation
from model.constellation_history import interpret
from model.native import NativeInput
from model.world_loop import WorldLoop, WorldState
from model.window_maxsim import maxsim
from model.episode_input import EpisodeBatch, validate_observation


class ConstellationWorld(WorldLoop):
    @classmethod
    def from_world(cls, world):
        # Preserve all retained weights and RNG state. No fitting or new parameters.
        with torch.random.fork_rng(devices=[]):
            result = cls(world.core, world.references(), world_source_id=world.world_source_id)
        result.to(world.core.initial.device)
        result.load_state_dict(world.state_dict(), strict=True)
        result.train(world.training)
        result.check_reference()
        return result

    def forward(self, native, states, *, allowed, phase, event_ids, evidence_origin, history=None,
                advance_phase=True):
        validate_observation(native, self.specs, self.config)
        check(all(isinstance(s, WorldState) and s.world_id == self.world_id for s in states),
              "CONSTELLATION_CONTEXT")
        rows = next(iter(self.reference_rows.values()))
        check(allowed.dtype == torch.bool and allowed.shape == (len(rows),)
              and allowed.device == self.core.initial.device and bool(allowed.any()),
              "CONSTELLATION_ALLOWED")
        mode = {} if advance_phase is True else dict(advance_phase=advance_phase)
        output = self.core(native, states, phase=phase, event_ids=event_ids,
                           evidence_origin=evidence_origin, **mode)
        hidden = torch.stack([s.hidden for s in output["states"]])
        placement_hidden = interpret(self, hidden, states, history)
        visible = allowed.nonzero().flatten()
        scores, region_support = {}, None
        # Every current native window conditions recognition in its own typed lane.
        # Only scalar match evidence is reduced; native slots/windows stay intact.
        for i, spec in enumerate(self.specs):
            observation = native[spec.name]
            if not isinstance(native, EpisodeBatch):
                check(not bool(observation.absent.any()), "CONSTELLATION_RECOGNITION_ABSENT")
            values, padding = getattr(self, f"world_values_{i}"), getattr(self, f"world_padding_{i}")
            book, mask = values[visible], padding[visible]
            context = F.normalize(placement_hidden[:, i], dim=-1)[:, None, :]
            keys = F.normalize(self.core.projectors[spec.name](book), dim=-1)
            if isinstance(native, EpisodeBatch):
                per_region = self._stream_regions(spec, observation, context, keys, mask)
            else:
                observed = F.normalize(self.core.projectors[spec.name](observation.values), dim=-1)
                q = F.normalize(self.query[spec.name](context+observed), dim=-1)
                matches = maxsim(q, keys, mask, self.config)
                matches = matches.masked_fill(observation.padding[:, :, None], 0.)
                per_region = matches.sum(1)/(~observation.padding).sum(1)[:, None]
            check(bool(torch.isfinite(per_region).all()), "CONSTELLATION_REGION_NONFINITE")
            full = torch.full((len(states), len(rows)), -torch.inf, device=context.device)
            full[:, visible] = per_region
            scores[spec.name] = full
            support = F.log_softmax(per_region, dim=-1)
            region_support = support if region_support is None else region_support+support
        selected = visible[region_support.argmax(-1)]
        common = torch.full_like(next(iter(scores.values())), -torch.inf)
        common[:, visible] = region_support
        constellation = self._placement(placement_hidden, selected, event_ids)
        return {**output, "states": tuple(self.bind_state(s) for s in output["states"]),
                "identity_regions": output["regions"], "identity_slots": output["slots"],
                "regions": scores, "constellation_regions": common,
                "slots": {n: s.values for n, s in constellation.slots.items()},
                "predicted_echo": {n: s.values for n, s in constellation.slots.items()},
                "predicted_constellation": constellation,
                "recognition": "typed-native-causal-prefix-v1" if history is not None else "typed-native-window-context-v1",
                "placement_hidden": placement_hidden,
                "world_owner": {s.name: selected for s in self.specs}, "world_bank": self.world_id}

    def _stream_regions(self, spec, observation, context, keys, mask):
        def score(values, state, reference):
            observed = F.normalize(self.core.projectors[spec.name](values.to(state.device)), dim=-1)
            q = F.normalize(self.query[spec.name](state+observed), dim=-1)
            return maxsim(q, reference, mask, self.config).sum(1)

        total, count = None, 0
        for block in observation.blocks(self.config):
            check(block.offset == count and not bool(block.padding.any()), "EPISODE_REGION_COVERAGE")
            part = (checkpoint(score, block.values, context, keys, use_reentrant=False,
                               preserve_rng_state=False) if torch.is_grad_enabled()
                    else score(block.values, context, keys))
            total = part if total is None else total+part
            count += block.values.shape[1]
        check(count == observation.count and total is not None, "EPISODE_REGION_COVERAGE")
        return total/count

    def supervised_placement(self, output, selected, *, allowed, event_ids):
        """Train placement within the checked label; forward never accepts a label."""
        rows = next(iter(self.reference_rows.values()))
        check(torch.is_grad_enabled(), "CONSTELLATION_SUPERVISION_TRAINING_ONLY")
        check(selected.dtype == torch.long and selected.shape == (len(output["states"]),)
              and selected.device == self.core.initial.device
              and bool(((selected >= 0) & (selected < len(rows))).all()),
              "CONSTELLATION_SUPERVISION_REGION")
        check(allowed.dtype == torch.bool and allowed.shape == (len(rows),)
              and allowed.device == selected.device and bool(allowed[selected].all()),
              "CONSTELLATION_SUPERVISION_CUTOFF")
        check(output["world_bank"] == self.world_id
              and all(s.world_id == self.world_id for s in output["states"]),
              "CONSTELLATION_SUPERVISION_WORLD")
        return self._placement(output["placement_hidden"], selected, event_ids)

    def _placement(self, hidden, selected, event_ids):
        rows = next(iter(self.reference_rows.values()))
        region_ids = tuple(rows[j] for j in selected.tolist())
        prediction_rows = tuple("prediction:"+e for e in event_ids)
        slots = {}
        for i, spec in enumerate(self.specs):
            book = getattr(self, f"world_values_{i}")[selected]
            padding = getattr(self, f"world_padding_{i}")[selected]
            width = int((~padding).sum(1).max())
            book, padding = book[:, :width], padding[:, :width]
            base = F.normalize(book, dim=-1)
            window_state = hidden[:, i, None, :] + self.core.projectors[spec.name](book)
            residual = torch.tanh(self.place[spec.name](window_state))
            residual = .1*residual / torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1.)
            predicted = F.normalize(base+residual, dim=-1).masked_fill(padding[:, :, None], 0.)
            # A placement radius is architectural, not a calibrated acceptance gate.
            check(bool(torch.isfinite(predicted).all()), "CONSTELLATION_NONFINITE")
            slots[spec.name] = NativeInput(spec.identity, prediction_rows, predicted, padding,
                                          torch.zeros(len(selected), dtype=torch.bool, device=predicted.device))
        return Constellation(slots, "predicted", region_ids).validate(self.specs, self.config)
