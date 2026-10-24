"""Bind native runtime conditions before Claim and replay that binding during learning."""
from claims.codata_source import check
from claims.episodic_memory import digest
from claims.tool_temporal_source import CUTOFFS


def checked(session, record):
    provider = session.condition_inputs
    check(provider is not None, "TOOL_CONDITION_NATIVE_MISSING")
    binding = record["native"]
    check(binding == provider.checked(binding["condition"]), "TOOL_CONDITION_NATIVE_BINDING")
    state = session.memory.state.get(record["episode"])
    ep = provider.conditions.episode(binding["condition"])
    check(ep["id"] == record["episode"] and state is not None and state["active"]
          and state["grounded_scope"] == "precommitted_execution_conditions"
          and state["receipt"] == provider.conditions.assess(ep), "TOOL_CONDITION_UNAVAILABLE")
    return record


def bind(session, context, key, *, expected=None):
    session._fresh()
    check(session.condition_inputs is not None, "TOOL_CONDITION_NATIVE_MISSING")
    check(context in session.contexts and not session.needs_review(context), "TOOL_CONTEXT_REVIEW")
    state = session.states[context]
    action = 2+2*(state.sequence//5)
    check(state.sequence % 5 == 1 and action in CUTOFFS and context not in session.pending,
          "TOOL_CONDITION_PHASE")
    address = (context, action)
    check(address not in session.condition_bindings, "TOOL_CONDITION_DUPLICATE")
    provider = session.condition_inputs.for_condition(key)
    c = provider.conditions.commitment
    ref = c["program_source"]
    check(c["source"] == session.source.pin and ref["task"] == session.contexts[context]
          and ref == session.source.reference(ref["task"], ref["turn"])
          and ref["turn"] <= session.cursor[context], "TOOL_CONDITION_SOURCE")
    native = provider.checked(key)
    ep = provider.conditions.episode(key)
    records = session._intake(ep)
    record = dict(native=native, episode=ep["id"])
    body = dict(kind="execution_conditions", context=context, action=action, sequence=state.sequence,
                conditions=record, memory=records, prior_head=session.head)
    if expected is None:
        session.journal.append(body)
    else:
        check(body == expected, "TOOL_CONDITION_REPLAY")
    session.condition_bindings[address] = record
    return body


def request_binding(session, context, action, phase):
    if phase not in (1, 2):
        return None
    record = session.condition_bindings.get((context, action))
    return None if record is None else checked(session, record)


def native_action(session, request):
    record = request.get("execution_conditions")
    if record is None:
        return session.inputs.observed(request["reference"], session.model.config, session.device)
    checked(session, record)
    check(record == session.condition_bindings.get((request["context"], request["turn"]))
          and request["phase"] in (1,2), "TOOL_CONDITION_REQUEST")
    return session.condition_inputs.merge(request["reference"], record["native"],
                                           session.model.config, session.device)


def match_observation(session, record, observation):
    conditions = record["request"].get("execution_conditions")
    if conditions is not None:
        checked(session, conditions)
        binding = session.observations.execution_conditions(observation)
        check(binding["condition"] == conditions["native"]["condition"]
              and binding["commitment"] == conditions["native"]["commitment"],
              "TOOL_EXECUTION_CONDITION_MISMATCH")
