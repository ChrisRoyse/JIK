"""Private, occurrence-checked native region views; shared publication stays separate."""
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_world import ConstellationWorld
from model.constellation_history import CausalHistory
from model.tool_execution import checked_echo
from model import process_anchor


def definition(session, request, support, routing):
    context, task, cutoff = request["context"], request["task"], request["turn"]
    check(session.contexts.get(context) == task and request["phase"] == 2
          and request["reference"] == session.inputs.population.lookup.get((task, cutoff))
          and request["reference"]["role"] == "generated_proposal", "TOOL_PRIVATE_CONTEXT")
    check(set(support) == {str(t) for t in range(1, cutoff+1)}, "TOOL_PRIVATE_PREFIX")
    router = session.routing_for(routing["kernel"])
    check(routing["query"] == request["reference"], "TOOL_PRIVATE_ROUTING")
    shared_world = session.model.world_id if router is session.routing else None
    for g in (*session.growths, *session.repairs):
        if g["old_kernel"] == router.pin:
            shared_world = g["old_world"]
        if g["kernel"] == router.pin:
            shared_world = g["world"]
    check(shared_world is not None, "TOOL_PRIVATE_WORLD")
    private = []
    for turn in range(1, cutoff+1):
        ep = support[str(turn)]
        session._checked(ep)
        check(session.memory.episodes[ep]["source"] == session.source.reference(task, turn),
              "TOOL_PRIVATE_SOURCE")
        if turn >= 3 and turn % 2 == 1:
            ref = session.inputs.population.lookup[task, turn]
            check(turn < cutoff and ref["role"] == "recorded_response", "TOOL_PRIVATE_CUTOFF")
            private.append(dict(reference=ref, episode=ep))
    shared = [router.inputs.world_units[i] for i in routing["allowed"]]
    units = list(dict.fromkeys(shared+[p["reference"]["unit"] for p in private]))
    result = dict(schema="tool-private-regions-v2", source=router.inputs.source_id,
                context=context, task=task, cutoff=cutoff, kernel=router.pin,
                shared_world=shared_world, shared=shared, private=private, units=units,
                history=[session.inputs.population.lookup[task, turn] for turn in range(1, cutoff)])
    echoes = {}
    for key, item in session.forecasts.items():
        req = item["record"]["request"]
        if "execution" in item and req["context"] == context and req["turn"]+1 < cutoff:
            binding = checked_echo(session, key, item["execution"])
            echoes[str(req["turn"]+1)] = dict(binding=binding, episode=item["execution"]["episode"])
    if echoes:
        result["observed_echoes"] = echoes
    return result


def attach(view, model):
    # Views own only immutable native reference buffers. Each call uses the actual
    # candidate's parameter objects, so gradients and optimizer steps reach it.
    check(view.standard_id == model.standard_id and view.specs == model.specs,
          "TOOL_PRIVATE_MODEL")
    view.core, view.query, view.place = model.core, model.query, model.place
    return view


def make_view(inputs, model, desc, execution_inputs=None):
    references = inputs.batch(desc["units"], model.config, model.core.initial.device)
    with torch.random.fork_rng(devices=[]):
        view = ConstellationWorld(model.core, references, world_source_id=digest(desc))
    view.to(model.core.initial.device).eval()
    attach(view, model)
    # The source-checked roster preserves occurrences, including repeated units.
    echoes = desc.get("observed_echoes", {})
    providers = []
    for r in desc["history"]:
        echo = echoes.get(str(r["turn"]))
        if echo is None:
            providers.append((inputs, r["unit"], r))
        else:
            check(execution_inputs is not None and
                  execution_inputs.checked(echo["binding"]["observation"]) == echo["binding"],
                  "TOOL_HISTORY_EXECUTION")
            provider = execution_inputs.for_observation(echo["binding"]["observation"])
            providers.append((provider, echo["binding"]["unit"], echo["binding"]))
    lengths = {s.name: sum(int(provider.arrays[s.name][1][unit+1]-provider.arrays[s.name][1][unit])
                           for provider,unit,_ in providers) for s in model.specs}
    check(sum(lengths[s.name]*s.dims for s in model.specs) <= model.config.max_native_elements
          and max(lengths.values()) <= model.config.max_native_tokens, "CONSTELLATION_HISTORY_BUDGET")
    frames = tuple(provider.observed(r if provider is inputs else r["observation"],
                                    model.config, model.core.initial.device) for provider,_,r in providers)
    view.context_history = CausalHistory(desc["context"],
        tuple(next(iter(f.slots.values())).rows[0] for f in frames), frames)
    view.execution_anchor_bank = process_anchor.bank(inputs, desc["units"])
    view.check_reference()
    return view


def predict(session, request, native):
    route = session.routing.route(request["reference"])
    support = {str(t): ep for t, ep in session.support[request["context"]].items()}
    desc = definition(session, request, support, route)
    key = digest(desc)
    saved = session.context_views.get(request["context"])
    if saved is None or saved[0] != key:
        saved = (key, make_view(session.inputs, session.model, desc, session.execution_inputs))
        session.context_views[request["context"]] = saved
    view = attach(saved[1], session.model)
    allowed = torch.ones(len(desc["units"]), dtype=torch.bool, device=session.device)
    with torch.no_grad():
        output = view(native.slots, (view.bind_state(session.states[request["context"]]),),
                      allowed=allowed, phase=2, event_ids=[digest(request)],
                      evidence_origin=[request["origin"]], history=view.context_history)
    # The persistent recurrent state remains bound to the shared model checkpoint.
    output["states"] = tuple(session.model.bind_state(s) for s in output["states"])
    output["execution_anchor"] = process_anchor.predict(output, view.execution_anchor_bank, allowed)
    return output, route, dict(definition=desc, world=view.world_id)


def historical(session, record):
    saved = record["region_context"]
    router = session.routing_for(record["routing"]["kernel"])
    check(record["routing"] == router.route(record["request"]["reference"]),
          "TOOL_PRIVATE_ROUTING")
    desc = definition(session, record["request"], record["support"], record["routing"])
    check(saved["definition"] == desc and record["world"] == desc["shared_world"],
          "TOOL_PRIVATE_HISTORY")
    view = make_view(session.inputs, session.model, desc, session.execution_inputs)
    check(saved["world"] == view.world_id, "TOOL_PRIVATE_VIEW_PIN")
    return view, desc["units"]
