"""Assess continued learners against their authenticated prior withheld checks."""
import math

from claims.codata_source import check


def metric(value):
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          "TOOL_RETENTION_METRIC")
    return value


def compare(previous, current, keys, instruments):
    check(keys and len(set(keys)) == len(keys)
          and set(keys) <= previous.keys() and set(keys) <= current.keys(),
          "TOOL_RETENTION_ROSTER")
    comparisons = {}
    for key in keys:
        before = previous[key]["assessment"]["instruments"]
        after = current[key]["assessment"]["instruments"]
        check(set(before) == set(after) == set(instruments), "TOOL_RETENTION_ROSTER")
        comparisons[key] = {}
        for name, identity in instruments.items():
            check(before[name]["instrument"] == after[name]["instrument"] == identity,
                  "TOOL_RETENTION_INSTRUMENT")
            old, new = metric(before[name]["prediction_loss"]), metric(after[name]["prediction_loss"])
            comparisons[key][name] = dict(previous=old, candidate=new, retained=new <= old)
    return comparisons


def combine(result, predecessor, comparisons):
    """Keep strict gain unchanged; retention can only add a rejection."""
    retained = all(v["retained"] for group in comparisons.values() for v in group.values())
    check((predecessor is None) == (not comparisons), "TOOL_RETENTION_PREDECESSOR")
    ceilings = {}
    gain = True
    for key, group in result["comparisons"].items():
        for name, values in group.items():
            new, parent, persistence = (metric(values[k]) for k in ("candidate", "parent", "persistence"))
            check(values["better_than_parent"] == (new < parent)
                  and values["better_than_persistence"] == (new < persistence),
                  "TOOL_RETENTION_COMPARISON")
            gain &= new < parent and new < persistence
            zero = [kind for kind, value in (("parent", parent), ("persistence", persistence)) if value == 0.]
            if zero:
                ceilings.setdefault(key, {})[name] = zero
    check(result["metric_gate_passed"] == gain, "TOOL_RETENTION_GAIN")
    reasons = list(result["reasons"])
    if not retained:
        reasons.append("RETAINED_PREDICTION_REGRESSION")
    if ceilings:
        reasons.append("STRICT_GAIN_BASELINE_AT_ZERO")
    return dict(result, gain_gate_passed=bool(gain), metric_gate_passed=bool(gain and retained),
        retention=dict(schema="per_instrument_predecessor_retention_v1", required=predecessor is not None,
                       predecessor=predecessor, comparisons=comparisons, passed=retained),
        strict_gain_ceilings=ceilings, reasons=reasons)


def assess(data, result, candidate):
    predecessor, comparisons = None, {}
    if data.continuation_name is not None:
        session = data.session
        prior = session.decisions[-1]
        expected = data.binding["continuation"]
        check(prior["candidate"] == expected["candidate"] == data.continuation_name
              and prior["pin"] == expected["pin"], "TOOL_RETENTION_PREDECESSOR")
        saved = session.journal.payload(prior["payload"])
        check(saved["result"] == prior["result"]
              and set(prior["evaluate"]) <= set(data.plan["evaluate"]), "TOOL_RETENTION_ROSTER")
        predecessor = dict(candidate=prior["candidate"], pin=prior["pin"], payload=prior["payload"],
                           scope="quarantined_learner_checks")
        comparisons = compare(saved["candidate"], candidate, prior["evaluate"],
                              {s.name: s.identity for s in data.model.specs})
    return combine(result, predecessor, comparisons)
