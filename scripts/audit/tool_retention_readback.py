"""Reconstruct retention and strict-gain ceilings from already scored SQL payloads."""
import math

from claims.codata_source import check


def inspect(body, payload, history, payloads, instruments):
    result = body["result"]
    predecessor, comparisons = None, {}
    if "continuation" in body:
        check(history and history[-1]["candidate"] == body["continuation"], "RETENTION_READ_PREDECESSOR")
        old = history[-1]
        predecessor = dict(candidate=old["candidate"], pin=old["pin"], payload=old["payload"],
                           scope="quarantined_learner_checks")
        previous = payloads[old["payload"]]["candidate"]
        check(set(old["evaluate"]) <= set(body["evaluate"]), "RETENTION_READ_ROSTER")
        for key in old["evaluate"]:
            comparisons[key] = {}
            for name, identity in instruments.items():
                before = previous[key]["assessment"]["instruments"][name]
                after = payload["candidate"][key]["assessment"]["instruments"][name]
                check(before["instrument"] == after["instrument"] == identity, "RETENTION_READ_INSTRUMENT")
                a, b = before["prediction_loss"], after["prediction_loss"]
                check(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0, "RETENTION_READ_METRIC")
                comparisons[key][name] = dict(previous=a, candidate=b, retained=b <= a)
    retained = all(v["retained"] for group in comparisons.values() for v in group.values())
    expected = dict(schema="per_instrument_predecessor_retention_v1", required=predecessor is not None,
                    predecessor=predecessor, comparisons=comparisons, passed=retained)
    check(result.get("retention") == expected, "RETENTION_READ_COMPARISON")
    ceilings, gain = {}, True
    for key in body["evaluate"]:
        for name in instruments:
            parent = payload["baseline"][key]["assessment"]["instruments"][name]["prediction_loss"]
            new = payload["candidate"][key]["assessment"]["instruments"][name]
            gain &= new["prediction_loss"] < parent and new["prediction_loss"] < new["persistence_loss"]
            zero = [kind for kind, value in (("parent", parent), ("persistence", new["persistence_loss"]))
                    if value == 0.]
            if zero:
                ceilings.setdefault(key, {})[name] = zero
    check(result.get("strict_gain_ceilings") == ceilings, "RETENTION_READ_CEILING")
    check(result.get("gain_gate_passed") == bool(gain)
          and result["metric_gate_passed"] == bool(gain and retained), "RETENTION_READ_GATE")
    reasons = ([] if gain else ["WITHHELD_PER_INSTRUMENT_GATE_FAILED"])
    reasons += ["DEVELOPMENT_REPLAY_NOT_ADMITTED_CAPABILITY_EVIDENCE"]
    if not retained:
        reasons.append("RETAINED_PREDICTION_REGRESSION")
    if ceilings:
        reasons.append("STRICT_GAIN_BASELINE_AT_ZERO")
    check(result["reasons"] == reasons, "RETENTION_READ_REASONS")
    return dict(comparisons=sum(len(v) for v in comparisons.values()),
                regressions=sum(not v["retained"] for group in comparisons.values() for v in group.values()),
                zero_baselines=sum(len(v) for v in ceilings.values()))
