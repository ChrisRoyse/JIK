"""Journaled growth from checked experience; transport changes references, never weights."""
import copy
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.constellation_world import ConstellationWorld
from model.loop_training_checkpoint import tensor_fingerprint
from model.tool_kernel import ToolKernel, build
from model.tool_loop import ToolLoop
from model import tool_context_regions
from model import tool_observed_inputs
from model import tool_bank_recovery
from model.world_checkpoint import load, save


def parameters(model):
    return digest(tensor_fingerprint(dict(model.named_parameters())))


def transport(model, inputs, states):
    with torch.random.fork_rng(devices=[]):
        enlarged = ConstellationWorld(model.core, inputs.batch(inputs.world_units, model.config,
                                      model.core.initial.device), world_source_id=inputs.source_id)
    enlarged.to(model.core.initial.device).eval()
    enlarged.query.load_state_dict(model.query.state_dict())
    enlarged.place.load_state_dict(model.place.state_dict())
    check(parameters(enlarged) == parameters(model), "TOOL_GROWTH_PARAMETER_DRIFT")
    check(enlarged.standard_id == model.standard_id, "TOOL_GROWTH_IDENTITY_DRIFT")
    for spec in model.specs:
        old, new = model.references()[spec.name], enlarged.references()[spec.name]
        check(new.rows[:len(old.rows)] == old.rows, "TOOL_GROWTH_REGION_ORDER")
        for i in range(len(old.rows)):
            check(torch.equal(old.values[i, ~old.padding[i]], new.values[i, ~new.padding[i]]),
                  "TOOL_GROWTH_WINDOW_DRIFT")
    return enlarged, {c: enlarged.bind_state(s) for c, s in states.items()}


def state_pin(states):
    return digest({c: dict(sequence=s.sequence, last_event=s.last_event,
                           hidden=tensor_fingerprint(s.hidden)) for c, s in states.items()})


class GrowingToolLoop(ToolLoop):
    def __init__(self, *args, growth_root, **kwargs):
        self.growth_root = Path(growth_root).resolve()
        self.growths, self.repairs, self.routing_versions = [], [], {}
        self.context_views = {}
        routing = kwargs.get("routing")
        check(routing is not None, "TOOL_GROWTH_KERNEL_REQUIRED")
        self.routing_versions[routing.pin] = routing
        extension = dict(schema="tool-checked-private-growth-v1", code=sha(Path(__file__).read_bytes()),
                         private_code=sha(Path(tool_context_regions.__file__).read_bytes()),
                         observed_code=sha(Path(tool_observed_inputs.__file__).read_bytes()),
                         recovery_code=sha(Path(tool_bank_recovery.__file__).read_bytes()))
        super().__init__(*args, extension=extension, **kwargs)

    def routing_for(self, pin):
        check(pin in self.routing_versions, "TOOL_ROUTING_VERSION")
        return self.routing_versions[pin]

    def _prediction(self, request, native):
        if request["phase"] == 2:
            return tool_context_regions.predict(self, request, native)
        return super()._prediction(request, native)

    def _growth_plan(self, keys):
        self._fresh()
        self._bank_checked()
        check(not self.pending and all(s.sequence % 5 == 0 for c, s in self.states.items()
                                      if c not in self.dirty),
              "TOOL_GROWTH_PENDING")
        check(isinstance(keys, list) and keys and len(keys) == len(set(keys)), "TOOL_GROWTH_KEYS")
        pair_map = copy.deepcopy(self.routing.pairs)
        additions, supports = [], {}
        view = copy.copy(self.inputs)
        for key in keys:
            item = self.eligible(key)
            req = item["record"]["request"]
            check(self.source.by_task[req["task"]]["partition"] == "train", "TOOL_GROWTH_SPLIT")
            action = req["reference"]
            action_ep = item["record"]["support"][str(req["turn"])]
            self._checked(action_ep)
            if "execution" in item:
                entry = item["execution"]
                self.execution_evidence(key)
                view, response = tool_observed_inputs.append(view, self.execution_inputs,
                                                             entry["observation"])
                unit, target_ep = response["unit"], entry["episode"]
                pair_map[str(unit)] = dict(action=action, response=response)
                additions.append(unit)
                supports[str(unit)] = dict(action=action_ep, response=target_ep, forecast=key)
                continue
            response = self.inputs.population.lookup[req["task"], req["turn"]+1]
            unit = response["unit"]
            check(unit not in self.inputs.world_units and str(unit) not in pair_map,
                  "TOOL_GROWTH_ALREADY_PRESENT")
            target_ep = item["target_episode"]
            self._checked(target_ep)
            check(item["target_unit"] == unit and self.memory.episodes[target_ep]["source"] ==
                  self.source.reference(req["task"], req["turn"]+1), "TOOL_GROWTH_SOURCE")
            pair_map[str(unit)] = dict(action=action, response=response)
            additions.append(unit)
            supports[str(unit)] = dict(action=action_ep, response=target_ep, forecast=key)
        view.world_units = list(self.inputs.world_units)+additions
        return view, pair_map, supports

    def grow(self, name, keys, *, expected=None):
        check(isinstance(name, str) and name not in ("", ".", "..") and
              "/" not in name and "\\" not in name, "TOOL_GROWTH_PATH")
        check(all(g["name"] != name for g in (*self.growths, *self.repairs)), "TOOL_GROWTH_DUPLICATE")
        inputs, pair_map, supports = self._growth_plan(keys)
        old_parameters, old_state = parameters(self.model), state_pin(self.states)
        model, states = transport(self.model, inputs, self.states)
        root = self.growth_root/name
        check(root.resolve().parent == self.growth_root, "TOOL_GROWTH_PATH")
        if expected is None:
            check(not root.exists(), "TOOL_GROWTH_ARTIFACT_EXISTS")
            root.mkdir(parents=True)
            kernel_pin = build(inputs, root/"kernel", pair_map=pair_map)
            checkpoint_pin = save(root/"model", model, tuple(states.values()))
        else:
            kernel_pin, checkpoint_pin = expected["kernel"], expected["checkpoint"]
        # Read actual persisted checkpoint and graph before the journal can select either.
        restored, restored_states, _ = load(root/"model", expected=checkpoint_pin,
            standard=model.standard_id, world=model.world_id, device=self.device)
        restored = ConstellationWorld.from_world(restored).eval()
        restored_states = {s.context: s for s in restored_states}
        check(tensor_fingerprint(restored.state_dict()) == tensor_fingerprint(model.state_dict())
              and state_pin(restored_states) == old_state
              and all(s.world_id == restored.world_id for s in restored_states.values()),
              "TOOL_GROWTH_CHECKPOINT_READBACK")
        routing = ToolKernel(inputs, root/"kernel", kernel_pin, expected_pairs=pair_map)
        body = dict(kind="grow", name=name, keys=keys, prior_head=self.head,
            old_world=self.model.world_id, world=restored.world_id,
            old_checkpoint=self.active_checkpoint, checkpoint=checkpoint_pin,
            old_kernel=self.routing.pin, kernel=kernel_pin, units=inputs.world_units,
            supports=supports, parameters=old_parameters, context_state=old_state,
            standard=self.model.standard_id)
        if expected is None:
            self.journal.append(body)
        else:
            check(body == expected, "TOOL_GROWTH_REPLAY")
        self.inputs, self.model, self.states, self.routing = inputs, restored, restored_states, routing
        self.context_views.clear()
        self.active_checkpoint = checkpoint_pin
        self.routing_versions[kernel_pin] = routing
        self.allowed = torch.ones(len(inputs.world_units), dtype=torch.bool, device=self.device)
        for unit, support in supports.items():
            self.bank[int(unit)] = support["response"]
            self.route_memory[digest(pair_map[unit]["action"])] = support["action"]
        self.growths.append(body)
        return body

    def _replay(self, body):
        if body.get("kind") == "grow":
            self.grow(body["name"], body["keys"], expected=body)
        elif body.get("kind") == "repair_bank":
            self.repair_bank(body["name"], expected=body)
        elif body.get("kind") == "start_context":
            self.start_context(body["context"], body["task"], expected=body)
        else:
            super()._replay(body)

    def repair_bank(self, name, *, expected=None):
        return tool_bank_recovery.repair(self, name, expected=expected)

    def start_context(self, context, task, *, expected=None):
        return tool_bank_recovery.start_context(self, context, task, expected=expected)

    def snapshot(self):
        return dict(**super().snapshot(), growths=self.growths,
                    repairs=self.repairs, region_units=list(self.inputs.world_units), kernel=self.routing.pin)
