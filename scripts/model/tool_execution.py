"""Attach independently observed process facts to a committed tool forecast."""
from claims.codata_source import check
from claims.process_observation import SCOPE
from model import tool_conditions


def condition_attachment(session, observation):
    """Keep preconditions as separate evidence and invalidate their dependents."""
    registry = session.observations.for_observation(observation)
    if "condition_commitment" not in registry.manifest:
        return None, []
    inputs = registry.condition_inputs()
    key = session.observations.entries[observation]["condition"]
    ep = inputs.episode(key)
    if ep["id"] in session.memory.episodes:
        state = session.memory.state[ep["id"]]
        check(state["active"] and state["grounded_scope"] == "precommitted_execution_conditions"
              and state["receipt"] == inputs.assess(ep), "TOOL_PRECONDITION_UNAVAILABLE")
        return ep["id"], []
    records = [session.memory.apply(dict(op="intake", episode=ep))]
    records.append(session.memory.apply(dict(op="check_source", episode=ep["id"])))
    records.append(session.memory.apply(dict(op="promote", episode=ep["id"],
                                            receipt=records[-1]["event"]["result"]["id"])))
    return ep["id"], records


def echo_binding(session, context, observation):
    check(session.execution_inputs is not None, "TOOL_EXECUTION_NATIVE_MISSING")
    check(session.states[context].sequence % 5 == 3 and context in session.pending,
          "TOOL_EXECUTION_ECHO_PHASE")
    key = session.pending[context]
    check(key not in session.executions, "TOOL_DUPLICATE_EXECUTION")
    item = session.forecasts[key]
    request = item["record"]["request"]
    check(observation in session.observations.entries
          and session.observations.entries[observation]["target"] ==
              session.source.reference(request["task"], request["turn"]+1),
          "TOOL_EXECUTION_FORECAST")
    tool_conditions.match_observation(session, item["record"], observation)
    return session.execution_inputs.checked(observation)


def stage_echo(session, key, binding, target_episode):
    item = session.forecasts[key]
    supports = sorted(set((*session.record_support(item["record"]), target_episode)))
    observation = binding["observation"]
    condition, records = condition_attachment(session, observation)
    if condition is not None:
        supports = sorted(set((*supports, condition)))
    ep = session.observations.episode(observation, supports)
    records.append(session.memory.apply(dict(op="intake", episode=ep)))
    records.append(session.memory.apply(dict(op="check_source", episode=ep["id"])))
    records.append(session.memory.apply(dict(op="promote", episode=ep["id"],
                                            receipt=records[-1]["event"]["result"]["id"])))
    return dict(observation=observation, episode=ep["id"],
                comparison=session.observations.compare_report(observation), native=binding), records


def checked_echo(session, key, entry):
    check(session.executions.get(key) == entry and session.execution_inputs is not None,
          "TOOL_EXECUTION_NATIVE_MISSING")
    state = session.memory.state[entry["episode"]]
    ep = session.memory.episodes[entry["episode"]]
    check(state["active"] and state["grounded_scope"] == SCOPE
          and state["receipt"] == session.memory.checker.assess(ep), "TOOL_EXECUTION_UNAVAILABLE")
    check(entry["native"] == session.execution_inputs.checked(entry["observation"]), "TOOL_EXECUTION_NATIVE_BINDING")
    return entry["native"]


def observe(session, key, observation, *, expected=None):
    session._fresh()
    check(session.observations is not None, "TOOL_NO_EXECUTION_CHECKER")
    check(key not in session.executions, "TOOL_DUPLICATE_EXECUTION")
    item = session.eligible(key)
    registry = session.observations
    check(observation in registry.entries, "TOOL_EXECUTION_UNKNOWN")
    target = session.source.reference(item["record"]["request"]["task"],
                                      item["record"]["request"]["turn"]+1)
    check(registry.entries[observation]["target"] == target, "TOOL_EXECUTION_FORECAST")
    tool_conditions.match_observation(session, item["record"], observation)
    # A source report and a new occurrence are separate episodes, even when the
    # process result agrees. The link is explicit reexecution, never time travel.
    supports = sorted(set((*session.record_support(item["record"]), item["target_episode"])))
    condition, records = condition_attachment(session, observation)
    if condition is not None:
        supports = sorted(set((*supports, condition)))
    ep = registry.episode(observation, supports)
    comparison = registry.compare_report(observation)
    records.extend([session.memory.apply(dict(op="intake", episode=ep)),
                    session.memory.apply(dict(op="check_source", episode=ep["id"]))])
    receipt = records[-1]["event"]["result"]
    records.append(session.memory.apply(dict(op="promote", episode=ep["id"], receipt=receipt["id"])))
    body = dict(kind="execution_observation", forecast=key, observation=observation,
                episode=ep["id"], comparison=comparison, memory=records,
                observation_set=registry.for_observation(observation).pin, prior_head=session.head)
    if expected is None:
        session.journal.append(body)
    else:
        check(body == expected, "TOOL_EXECUTION_REPLAY")
    session.executions[key] = dict(observation=observation, episode=ep["id"], comparison=comparison)
    return body


def checked(session, key):
    session.eligible(key)
    check(key in session.executions, "TOOL_EXECUTION_MISSING")
    entry = session.executions[key]
    state = session.memory.state[entry["episode"]]
    ep = session.memory.episodes[entry["episode"]]
    check(state["active"] and state["grounded_scope"] == SCOPE and
          state["receipt"] == session.memory.checker.assess(ep), "TOOL_EXECUTION_UNAVAILABLE")
    native = checked_echo(session, key, entry) if "native" in entry else None
    return dict(receipt=state["receipt"], comparison=entry["comparison"],
                native_feedback_available=native is not None, learned_outcome_understanding=False)
