"""UCR-v1: typed evidence, explicit absence and immutable observation identities (#15).

IDs and source locators are lineage, never numeric features. A present channel is
an observation, not evidence of independence or permission to assert a meaning.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime

VERSION = "ucr-v1"
CHANNELS = ("surface", "fields", "relations", "history", "provenance",
            "co_reference", "structure", "behaviour", "perception")
TYPES = {"number", "integer", "boolean", "category", "text", "reference", "clock"}


class SenseError(RuntimeError):
    def __init__(self, code, message, remediation="Correct the input and retry; no state was accepted."):
        self.code, self.message, self.remediation = code, message, remediation
        super().__init__(f"{code}: {message}")

    def envelope(self):
        return {"code": self.code, "message": self.message, "remediation": self.remediation}


def require(ok, code, message):
    if not ok:
        raise SenseError(code, message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SenseError("MNLM_UCR_VALUE", str(exc)) from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def absent(reason):
    return {"state": "absent", "reason": reason, "items": []}


def present(items):
    return {"state": "present", "reason": None, "items": items} if items else absent("not_observed")


def field(name, value, kind, unit=None, clock_role=None):
    return {"name": name, "value": value, "type": kind, "unit": unit, "clock_role": clock_role}


def time_value(value):
    require(isinstance(value, str), "MNLM_UCR_CLOCK", "clock must be an ISO timestamp with timezone")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(dt.tzinfo is not None, "MNLM_UCR_CLOCK", "timezone is required")
        return dt.timestamp()
    except ValueError as exc:
        raise SenseError("MNLM_UCR_CLOCK", value) from exc


def validate_field(f):
    require(isinstance(f, dict) and set(f) == {"name", "value", "type", "unit", "clock_role"}, "MNLM_UCR_FIELD", "typed field keys")
    k, v = f["type"], f["value"]
    require(isinstance(f["name"], str) and bool(f["name"]) and isinstance(k, str) and k in TYPES, "MNLM_UCR_FIELD", "field name/type")
    if k in ("integer", "number"):
        require(type(v) in ((int,) if k == "integer" else (int, float)) and (type(v) is int or math.isfinite(v)),
                "MNLM_UCR_FIELD", "numeric fields must retain finite numeric values")
    elif k == "boolean":
        require(type(v) is bool, "MNLM_UCR_FIELD", "boolean required")
    else:
        require(isinstance(v, str), "MNLM_UCR_FIELD", "string required")
    if k == "clock":
        time_value(v)
        require(f["clock_role"] in ("event", "observed", "valid_from", "valid_to"), "MNLM_UCR_CLOCK", "declare clock role")
    else:
        require(f["clock_role"] is None, "MNLM_UCR_CLOCK", "clock role on non-clock field")
    require(f["unit"] is None or isinstance(f["unit"], str), "MNLM_UCR_FIELD", "unit must be named or absent")


def validate_record(r):
    require(isinstance(r, dict) and set(r) == {"schema", "atom_id", "locator", "unit", "referents", "channels"}, "MNLM_UCR_SCHEMA", "unknown or missing record keys")
    require(r["schema"] == VERSION and isinstance(r["channels"], dict) and set(r["channels"]) == set(CHANNELS), "MNLM_UCR_SCHEMA", "all channels must be explicit")
    loc = r["locator"]
    require(isinstance(loc, dict) and set(loc) == {"source", "shard", "row", "span", "record_digest", "manifest_digest"}, "MNLM_UCR_LOCATOR", "locator keys")
    require(all(isinstance(loc[k], str) and loc[k] for k in ("source", "shard", "record_digest", "manifest_digest")), "MNLM_UCR_LOCATOR", "source and digests required")
    require(all(is_sha256(loc[k]) for k in ("record_digest", "manifest_digest")), "MNLM_UCR_LOCATOR", "digests must be 64 lowercase hex characters")
    require(type(loc["row"]) is int and loc["row"] >= 0 and isinstance(loc["span"], list) and len(loc["span"]) == 2
            and all(type(v) is int and v >= 0 for v in loc["span"]) and loc["span"][0] < loc["span"][1], "MNLM_UCR_LOCATOR", "nonempty source span required; audio declares sample units in structure")
    require(isinstance(r["unit"], str) and r["unit"], "MNLM_UCR_UNIT", "sibling/label unit required")
    require(isinstance(r["referents"], list), "MNLM_UCR_REFERENT", "referents must be explicit")
    for ref in r["referents"]:
        require(isinstance(ref, dict) and set(ref) == {"key", "method", "confidence", "evidence"} and isinstance(ref["key"], str) and bool(ref["key"])
                and ref["method"] in ("source_identifier", "attested_mapping") and ref["confidence"] == 1
                and bool(ref["evidence"]), "MNLM_UCR_REFERENT", "only evidence-backed exact identities may join")
    for name, ch in r["channels"].items():
        require(isinstance(ch, dict) and set(ch) == {"state", "reason", "items"} and isinstance(ch["items"], list), "MNLM_UCR_CHANNEL", name)
        require((ch["state"] == "absent" and not ch["items"] and isinstance(ch["reason"], str) and bool(ch["reason"]))
                or (ch["state"] == "present" and ch["reason"] is None and bool(ch["items"])), "MNLM_UCR_CHANNEL", name)
        for item in ch["items"]:
            validate_item(name, item)
    canonical(r)  # rejects non-finite values even in nested evidence
    body = {k: v for k, v in r.items() if k != "atom_id"}
    require(r["atom_id"] == digest(body), "MNLM_UCR_ID", "record does not match its content address")
    return r


def validate_item(channel, item):
    require(isinstance(item, dict), "MNLM_UCR_ITEM", channel)
    if channel == "surface":
        require(set(item) == {"text"} and isinstance(item["text"], str), "MNLM_UCR_ITEM", "surface text")
    elif channel in ("fields", "provenance", "structure"):
        validate_field(item)
    elif channel == "relations":
        require(set(item) == {"rel_type", "direction", "polarity", "target_referent", "qualifiers", "time", "source"}, "MNLM_UCR_RELATION", "relation keys")
        require(item["direction"] in ("out", "in", "symmetric") and item["polarity"] in ("positive", "negative", "unknown")
                and all(isinstance(item[k], str) and item[k] for k in ("rel_type", "target_referent", "source")), "MNLM_UCR_RELATION", "typed direction/polarity and source required")
        require(isinstance(item["qualifiers"], list), "MNLM_UCR_RELATION", "qualifiers must be typed fields")
        for f in item["qualifiers"]:
            validate_field(f)
        if item["time"] is not None:
            time_value(item["time"])
    elif channel in ("history", "co_reference"):
        require(set(item) == {"atom_id", "unit", "source", "observed_at"}, "MNLM_UCR_CONTEXT", "context must identify a retained observation")
        require(all(isinstance(item[k], str) and item[k] for k in ("atom_id", "unit", "source")), "MNLM_UCR_CONTEXT", "context identities required")
        if item["observed_at"] is not None:
            time_value(item["observed_at"])
    elif channel == "behaviour":
        require(set(item) == {"event", "mode", "evidence", "measurements"} and item["mode"] in ("observed", "static")
                and isinstance(item["event"], str) and bool(item["event"]) and bool(item["evidence"])
                and isinstance(item["measurements"], list), "MNLM_UCR_BEHAVIOUR", "attested observation/static trace required")
        for f in item["measurements"]:
            validate_field(f)
    elif channel == "perception":
        require(set(item) == {"modality", "sha256", "media_type", "measurements"} and item["modality"] in ("image", "audio", "video", "sensor")
                and is_sha256(item["sha256"]) and isinstance(item["media_type"], str) and bool(item["media_type"])
                and isinstance(item["measurements"], list), "MNLM_UCR_PERCEPTION", "content-bound percept required")
        for f in item["measurements"]:
            validate_field(f)


def seal_record(r):
    r["atom_id"] = digest({k: v for k, v in r.items() if k != "atom_id"})
    return validate_record(r)


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
