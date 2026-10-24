"""Retire invalidated region dependencies without laundering recurrent state."""
import copy

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_world import ConstellationWorld
from model.loop_training_checkpoint import tensor_fingerprint
from model.tool_kernel import PLAN, ToolKernel, build
from model.world_checkpoint import load, save


def plan(session):
    session._fresh()
    keep, removed, routes = {}, {}, {}
    check(set(session.bank) == set(session.inputs.world_units), "TOOL_RECOVERY_BANK_JOIN")
    for unit in session.inputs.world_units:
        pair = session.routing.pairs[str(unit)]
        action = digest(pair["action"])
        support = dict(action=session.route_memory[action], response=session.bank[unit])
        invalid = []
        for ep in support.values():
            state = session.memory.state.get(ep)
            check(state is not None, "TOOL_RECOVERY_MEMORY_JOIN")
            if not state["active"] or state["grounded_scope"] is None:
                invalid.append(ep)
            else:
                # A damaged receipt/file is not authority to retire a region.
                session._support_checked(ep)
        if invalid:
            removed[str(unit)] = dict(support=support, invalid=sorted(set(invalid)))
        else:
            keep[str(unit)] = copy.deepcopy(pair)
            routes[action] = support["action"]
    check(removed, "TOOL_RECOVERY_NOT_NEEDED")
    check(len(keep) >= PLAN["top_k"], "TOOL_RECOVERY_INSUFFICIENT_BANK")
    inputs = copy.copy(session.inputs)
    inputs.world_units = [u for u in session.inputs.world_units if str(u) in keep]
    inputs.source_ancestry = (*getattr(inputs, "source_ancestry", ()), inputs.source_id)
    inputs.source_id = digest(dict(parent=inputs.source_id, retired=removed, units=inputs.world_units))
    return inputs, keep, removed, routes


def repair(session, name, *, expected=None):
    from model.tool_growth import parameters, state_pin
    check(isinstance(name, str) and name not in ("", ".", "..")
          and "/" not in name and "\\" not in name, "TOOL_GROWTH_PATH")
    check(all(g["name"] != name for g in (*session.growths, *session.repairs)),
          "TOOL_GROWTH_DUPLICATE")
    inputs, pairs, removed, routes = plan(session)
    with torch.random.fork_rng(devices=[]):
        model = ConstellationWorld(session.model.core, inputs.batch(inputs.world_units,
            session.model.config, session.model.core.initial.device), world_source_id=inputs.source_id)
    model.to(session.device).eval()
    model.query.load_state_dict(session.model.query.state_dict())
    model.place.load_state_dict(session.model.place.state_dict())
    check(parameters(model) == parameters(session.model)
          and model.standard_id == session.model.standard_id, "TOOL_RECOVERY_PARAMETER_DRIFT")
    positions = {u:i for i,u in enumerate(session.inputs.world_units)}
    indices = [positions[u] for u in inputs.world_units]
    for spec in model.specs:
        old, new = session.model.references()[spec.name], model.references()[spec.name]
        check(new.rows == tuple(old.rows[i] for i in indices), "TOOL_RECOVERY_REGION_ORDER")
        for j, i in enumerate(indices):
            check(torch.equal(old.values[i, ~old.padding[i]], new.values[j, ~new.padding[j]]),
                  "TOOL_RECOVERY_WINDOW_DRIFT")
    states = {c: model.bind_state(s) for c, s in session.states.items()}
    before_state = state_pin(session.states)
    root = session.growth_root/name
    check(root.resolve().parent == session.growth_root, "TOOL_GROWTH_PATH")
    if expected is None:
        check(not root.exists(), "TOOL_GROWTH_ARTIFACT_EXISTS")
        root.mkdir(parents=True)
        kernel_pin = build(inputs, root/"kernel", pair_map=pairs)
        checkpoint_pin = save(root/"model", model, tuple(states.values()))
    else:
        kernel_pin, checkpoint_pin = expected["kernel"], expected["checkpoint"]
    restored, restored_states, _ = load(root/"model", expected=checkpoint_pin,
        standard=model.standard_id, world=model.world_id, device=session.device)
    restored = ConstellationWorld.from_world(restored).eval()
    restored_states = {s.context: s for s in restored_states}
    check(tensor_fingerprint(restored.state_dict()) == tensor_fingerprint(model.state_dict())
          and state_pin(restored_states) == before_state, "TOOL_RECOVERY_CHECKPOINT_READBACK")
    routing = ToolKernel(inputs, root/"kernel", kernel_pin, expected_pairs=pairs)
    canceled = {c:k for c,k in session.pending.items() if session.forecasts[k]["invalidated"]}
    body = dict(kind="repair_bank", name=name, prior_head=session.head, removed=removed,
        units=inputs.world_units, old_world=session.model.world_id, world=restored.world_id,
        old_checkpoint=session.active_checkpoint, checkpoint=checkpoint_pin,
        old_kernel=session.routing.pin, kernel=kernel_pin, parameters=parameters(restored),
        context_state=before_state, standard=restored.standard_id, canceled=canceled,
        dirty=sorted(session.dirty), source=inputs.source_id)
    if expected is None:
        session.journal.append(body)
    else:
        check(body == expected, "TOOL_RECOVERY_REPLAY")
    session.inputs, session.model, session.states, session.routing = inputs, restored, restored_states, routing
    session.bank = {u:session.bank[u] for u in inputs.world_units}
    session.route_memory = routes
    session.context_views.clear()
    session.active_checkpoint = checkpoint_pin
    session.routing_versions[kernel_pin] = routing
    session.allowed = torch.ones(len(inputs.world_units), dtype=torch.bool, device=session.device)
    for c, k in canceled.items():
        session.pending.pop(c)
        session.forecasts[k]["status"] = "canceled"
    session.repairs.append(body)
    return body


def start_context(session, context, task, *, expected=None):
    from model.tool_growth import state_pin
    session._fresh()
    session._bank_checked()
    check(isinstance(context, str) and 0 < len(context) <= 128 and context not in session.contexts,
          "TOOL_CONTEXT_EXISTS")
    check(task in session.source.by_task and session.source.by_task[task]["status"] == "ready"
          and task not in session.contexts.values(), "TOOL_CONTEXT_TASK")
    state = session.model.start(context)
    body = dict(kind="start_context", prior_head=session.head, context=context, task=task,
                world=session.model.world_id, standard=session.model.standard_id,
                state=state_pin({context:state}))
    if expected is None:
        session.journal.append(body)
    else:
        check(body == expected, "TOOL_CONTEXT_REPLAY")
    session.contexts[context], session.states[context] = task, state
    session.cursor[context], session.support[context] = 0, {}
    return body
