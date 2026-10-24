"""Enroll a captured native outcome only after its pending forecast was sealed."""
from pathlib import Path

from claims.codata_source import check
from claims import forecast_commitment
from claims.process_observation import ProcessObservations
from model.process_input import ProcessInputs
from model import process_batches


def enroll(session, spec, *, expected=None):
    session._fresh()
    check(not any(e["spec"]["forecast_pin"] == spec["forecast_pin"]
                  for e in session.execution_enrollments), "EXECUTION_ALREADY_ENROLLED")
    check(session.observations is None or session.execution_enrollments,
          "EXECUTION_ENROLLMENT_LEGACY_REGISTRY")
    seal = forecast_commitment.read(Path(spec["forecast_root"]), spec["forecast_pin"])
    check(seal["head"] == session.head and seal["contract"] == session.journal.pin
          and seal["conditions"] == session.condition_inputs.identity, "EXECUTION_ENROLLMENT_HEAD")
    for key, record in seal["forecasts"].items():
        check(key not in session.executions and
              all(key not in e["links"] for e in session.execution_enrollments), "EXECUTION_ALREADY_ENROLLED")
        item = session.forecasts.get(key)
        check(item is not None and item["record"] == record and not item["invalidated"]
              and session.pending.get(record["request"]["context"]) == key, "EXECUTION_ENROLLMENT_PENDING")
    observations = ProcessObservations(Path(spec["observations"]), spec["observation_pin"], session.source)
    check(observations.manifest.get("forecast", {}).get("pin") == spec["forecast_pin"],
          "EXECUTION_ENROLLMENT_SEAL")
    observed_conditions = observations.condition_inputs()
    check(all(session.condition_inputs.conditions.for_condition(entry["condition"]).pin ==
              observed_conditions.pin for entry in observations.entries.values()),
          "EXECUTION_ENROLLMENT_CONDITIONS")
    links = {}
    for observation, entry in observations.entries.items():
        matches = [(k,r) for k,r in seal["forecasts"].items()
                   if r["request"]["task"] == entry["target"]["task"]
                   and r["request"]["turn"]+1 == entry["target"]["turn"]]
        check(len(matches) == 1, "EXECUTION_ENROLLMENT_TARGET")
        key, record = matches[0]
        forecast_commitment.match(seal, record, entry, observations.read(observation))
        check(key not in links, "EXECUTION_ENROLLMENT_DUPLICATE")
        links[key] = observation
    check(set(links) == set(seal["forecasts"]), "EXECUTION_ENROLLMENT_ROSTER")
    base = session.inputs if session.execution_inputs is None else session.execution_inputs.base
    check(base.specs == session.inputs.specs and base.source_id in (
              session.inputs.source_id, *getattr(session.inputs, "source_ancestry", ())),
          "EXECUTION_ENROLLMENT_SOURCE_ANCESTRY")
    native = ProcessInputs(observations, base, Path(spec["population"]), Path(spec["native"]),
        population_pin=spec["population_pin"], native_pin=spec["native_pin"])
    combined = process_batches.append(session.execution_inputs, native)
    body = dict(kind="execution_enrollment", prior_head=session.head, spec=spec,
                links=links, native=native.identity)
    if expected is None:
        session.journal.append(body)
    else:
        check(body == expected, "EXECUTION_ENROLLMENT_REPLAY")
    session.observations, session.execution_inputs = combined.observations, combined
    session.memory.checker.observations = combined.observations
    session.execution_enrollment = body
    session.execution_enrollments.append(body)
    return body
