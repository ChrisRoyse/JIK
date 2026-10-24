"""Fault real saved metric records without changing any training data."""
import copy

from claims.codata_source import check
from model.tool_retention import compare


def controls(previous, current, keys, instruments, refuse, prefix):
    equality = compare(previous, previous, keys, instruments)
    check(all(v["retained"] for group in equality.values() for v in group.values()),
          "RETENTION_EQUALITY")
    key, name = keys[0], next(iter(instruments))

    def reject(label, code, mutate):
        altered = copy.deepcopy(current)
        mutate(altered)
        refuse(prefix+"_"+label, code, lambda: compare(previous, altered, keys, instruments))

    reject("missing_key", "TOOL_RETENTION_ROSTER", lambda p: p.pop(key))
    reject("missing_lane", "TOOL_RETENTION_ROSTER",
           lambda p: p[key]["assessment"]["instruments"].pop(name))
    reject("wrong_instrument", "TOOL_RETENTION_INSTRUMENT",
           lambda p: p[key]["assessment"]["instruments"][name].update(instrument="wrong"))
    for label, value in (("nonfinite", float("nan")), ("negative", -1.)):
        reject(label, "TOOL_RETENTION_METRIC",
               lambda p: p[key]["assessment"]["instruments"][name].update(prediction_loss=value))
    regressed = copy.deepcopy(previous)
    regressed[key]["assessment"]["instruments"][name]["prediction_loss"] += .001

    def require_retention():
        result = compare(previous, regressed, keys, instruments)
        check(all(v["retained"] for group in result.values() for v in group.values()),
              "TOOL_RETENTION_REGRESSION")

    refuse(prefix+"_single_slot_regression", "TOOL_RETENTION_REGRESSION", require_retention)
