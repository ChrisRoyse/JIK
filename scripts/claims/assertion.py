"""Typed assertions and verdicts for executable claim checking (operator review recommendation 4). The acceptance unit is
"this source supports this typed assertion under these conditions", not "this source exists". An assertion names a
subject, a relation, an object, a polarity, qualifiers, an optional time interval, the evidence it cites and its
epistemic status. A verdict is one of four outcomes that never collapse into each other:

  supported            a cited evidence item (or, for a hypothesis with no evidence, a row the checker found) holds the triple
  contradicted         the source holds the triple of a negated assertion, or holds a different object where the property
                       is single-valued for that subject in the source
  unsupported_missing  nothing in the source for or against; explicitly NOT supported ("no counter-evidence" lands here)
  evidence_mismatch    a cited reference does not resolve, or resolves to a record that does not hold the triple

Validation fails closed: a missing or unknown field, a wrong type or an unknown enum value raises ClaimError with a
typed code. Only `qualifiers` ({}), `time` (None) and `evidence` ([]) are optional; `epistemic_status: claimed`
requires at least one evidence item (a claim with nothing cited is a hypothesis, which asks the source itself)."""
from __future__ import annotations

import json
import re
from datetime import date
from dataclasses import asdict, dataclass, field

POLARITIES = ("asserted", "negated")
EPISTEMIC = ("claimed", "hypothesis")
OUTCOMES = ("supported", "contradicted", "unsupported_missing", "evidence_mismatch")
ENTITY = re.compile(r"Q[1-9][0-9]*")
PROPERTY = re.compile(r"P[1-9][0-9]*")
ASSERTION_FIELDS = {"subject", "relation", "object", "polarity", "qualifiers", "time", "evidence", "epistemic_status"}
REQUIRED = {"subject", "relation", "object", "polarity", "epistemic_status"}
EVIDENCE_FIELDS = {"source", "shard", "row"}
TIME_FIELDS = {"start", "end"}
VERDICT_FIELDS = {"outcome", "assertion", "reason", "cited", "source_rows", "source_objects", "conditions"}


class ClaimError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class EvidenceRef:
    source: str  # a corpus source id, e.g. wikidata5m
    shard: str   # a canonical shard of that source, e.g. wikidata5m_transductive_train
    row: int     # the canonical row; the triple id is shard#row

    @property
    def triple_id(self) -> str:
        return f"{self.shard}#{self.row}"


@dataclass(frozen=True)
class Interval:
    start: str | None  # ISO-8601 date or None (open)
    end: str | None


@dataclass(frozen=True)
class Assertion:
    subject: str
    relation: str
    object: str  # an entity id (Q...) or a literal string
    polarity: str
    epistemic_status: str
    qualifiers: dict = field(default_factory=dict)
    time: Interval | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    @property
    def object_is_entity(self) -> bool:
        return bool(ENTITY.fullmatch(self.object))

    def to_json(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


@dataclass(frozen=True)
class Verdict:
    outcome: str
    assertion: dict
    reason: str  # a typed reason code, e.g. CITED_ROW_HOLDS_TRIPLE
    cited: list = field(default_factory=list)  # per cited ref: {triple_id, resolves, holds}
    source_rows: list = field(default_factory=list)  # triple ids of source rows holding (s, p, o)
    source_objects: list = field(default_factory=list)  # objects the source holds for (s, p)
    conditions: dict = field(default_factory=dict)  # what the verdict assumed (functional basis, world assumption)

    def __post_init__(self):
        if not isinstance(self.outcome, str) or self.outcome not in OUTCOMES:
            raise ClaimError("MNLM_CLAIM_BAD_OUTCOME", f"{self.outcome!r} is not one of {OUTCOMES}")

    def to_json(self) -> dict:
        return asdict(self)


def _need(d, allowed: set, required: set, where: str) -> None:
    if not isinstance(d, dict):
        raise ClaimError("MNLM_CLAIM_NOT_OBJECT", f"{where} must be a JSON object, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise ClaimError("MNLM_CLAIM_UNKNOWN_FIELD", f"{where} has unknown field(s) {sorted(unknown)}")
    missing = required - set(d)
    if missing:
        raise ClaimError("MNLM_CLAIM_MISSING_FIELD", f"{where} is missing field(s) {sorted(missing)}")


def _str(v, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise ClaimError("MNLM_CLAIM_BAD_TYPE", f"{where} must be a non-empty string, got {v!r}")
    return v


def parse_evidence(d, i: int) -> EvidenceRef:
    _need(d, EVIDENCE_FIELDS, EVIDENCE_FIELDS, f"evidence[{i}]")
    row = d["row"]
    if not isinstance(row, int) or isinstance(row, bool) or row < 0:
        raise ClaimError("MNLM_CLAIM_BAD_TYPE", f"evidence[{i}].row must be an integer, got {row!r}")
    return EvidenceRef(_str(d["source"], f"evidence[{i}].source"), _str(d["shard"], f"evidence[{i}].shard"), row)


def parse_assertion(d) -> Assertion:
    """Validate one assertion object (fail closed) and return the typed Assertion."""
    _need(d, ASSERTION_FIELDS, REQUIRED, "assertion")
    subject = _str(d["subject"], "subject")
    if not ENTITY.fullmatch(subject):
        raise ClaimError("MNLM_CLAIM_BAD_SUBJECT", f"subject {subject!r} is not an entity id Q<n>")
    relation = _str(d["relation"], "relation")
    if not PROPERTY.fullmatch(relation):
        raise ClaimError("MNLM_CLAIM_BAD_RELATION", f"relation {relation!r} is not a property id P<n>")
    obj = _str(d["object"], "object")
    if _str(d["polarity"], "polarity") not in POLARITIES:
        raise ClaimError("MNLM_CLAIM_BAD_POLARITY", f"polarity {d['polarity']!r} is not one of {POLARITIES}")
    if _str(d["epistemic_status"], "epistemic_status") not in EPISTEMIC:
        raise ClaimError("MNLM_CLAIM_BAD_EPISTEMIC", f"epistemic_status {d['epistemic_status']!r} is not one of {EPISTEMIC}")
    qualifiers = d.get("qualifiers", {})
    if not isinstance(qualifiers, dict) or any(not isinstance(k, str) or not PROPERTY.fullmatch(k) for k in qualifiers):
        raise ClaimError("MNLM_CLAIM_BAD_QUALIFIERS", f"qualifiers must be an object keyed by property ids, got {qualifiers!r}")
    time = d.get("time")
    if time is not None:
        _need(time, TIME_FIELDS, TIME_FIELDS, "time")
        for k in TIME_FIELDS:
            if time[k] is not None and not isinstance(time[k], str):
                raise ClaimError("MNLM_CLAIM_BAD_TYPE", f"time.{k} must be an ISO date string or null")
            if time[k] is not None:
                try:
                    parsed = date.fromisoformat(time[k])
                    if parsed.isoformat() != time[k]:
                        raise ValueError("not a canonical date")
                except ValueError as exc:
                    raise ClaimError("MNLM_CLAIM_BAD_TIME", f"time.{k} must be YYYY-MM-DD") from exc
        if time["start"] is not None and time["end"] is not None and time["start"] > time["end"]:
            raise ClaimError("MNLM_CLAIM_BAD_TIME", "time.start is after time.end")
        time = Interval(time["start"], time["end"])
    ev = d.get("evidence", [])
    if not isinstance(ev, list):
        raise ClaimError("MNLM_CLAIM_BAD_TYPE", "evidence must be a list")
    evidence = tuple(parse_evidence(e, i) for i, e in enumerate(ev))
    if d["epistemic_status"] == "claimed" and not evidence:
        raise ClaimError("MNLM_CLAIM_NO_EVIDENCE", "a claimed assertion must cite at least one evidence item; state it as a hypothesis to ask the source")
    return Assertion(subject, relation, obj, d["polarity"], d["epistemic_status"], dict(qualifiers), time, evidence)


def parse_assertion_json(text: str) -> Assertion:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaimError("MNLM_CLAIM_BAD_JSON", str(e)) from e
    return parse_assertion(d)


def parse_verdict(d) -> Verdict:
    """Validate a persisted verdict (fail closed) so a reader never trusts an outcome string it did not check."""
    _need(d, VERDICT_FIELDS, VERDICT_FIELDS, "verdict")
    parse_assertion(d["assertion"])
    _str(d["reason"], "reason")
    if not isinstance(d["conditions"], dict) or any(not isinstance(d[k], list) for k in ("cited", "source_rows", "source_objects")):
        raise ClaimError("MNLM_CLAIM_BAD_TYPE", "verdict collections have wrong types")
    return Verdict(**d)
