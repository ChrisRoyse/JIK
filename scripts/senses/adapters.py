"""Canonical source formats -> UCR. Nothing in this module is a lens.

O(record bytes + explicit fields + atoms). Prepare once per canonical record;
atom views reuse the prepared metadata. Oracle labels/tests stay out of channels.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
import json

from .contract import CHANNELS, VERSION, absent, digest, field, present, require, seal_record

PLANS = json.loads((Path(__file__).resolve().parents[2] / "config" / "field_plans.json").read_text())["sources"]
ORACLE_FIELDS = {"anchors", "label", "y", "y_name", "unit_tests", "tests_execution_status",
                 "average_test_score", "llm_judgement"}
UNITS = {"ActionGeo_Lat": "degree_latitude", "ActionGeo_Long": "degree_longitude",
         "NumMentions": "count", "NumSources": "count", "NumArticles": "count"}


def exact_ref(key, evidence, method="source_identifier"):
    return {"key": key, "method": method, "confidence": 1, "evidence": evidence}


def relation(kind, target, source, qualifiers=None):
    return {"rel_type": kind, "direction": "out", "polarity": "positive", "target_referent": target,
            "qualifiers": qualifiers or [], "time": None, "source": source}


def clock(value, role):
    raw = str(value)
    fmt = "%Y%m%d%H%M%S" if len(raw) == 14 and raw.isdigit() else "%Y%m%d" if len(raw) == 8 and raw.isdigit() else "%Y-%m-%d"
    try:
        return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).isoformat(), role
    except ValueError:
        from .contract import time_value
        time_value(raw)
        return raw, role


def typed_fields(values, plan, roles=None):
    out = []
    for name, kind in sorted(plan.items()):
        v = values.get(name)
        if v is None or v == "" or name in ORACLE_FIELDS:
            continue
        if kind in ("scalar", "ordinal", "cyclic", "number", "integer"):
            require(not isinstance(v, bool), "MNLM_UCR_FIELD", f"{name}: boolean is not a scalar")
            try:
                value = int(v) if type(v) is int or (isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v)) else float(v)
            except (ValueError, TypeError):
                require(False, "MNLM_UCR_FIELD", f"{name}: expected number")
            require(kind != "integer" or type(value) is int, "MNLM_UCR_FIELD", f"{name}: exact integer required")
            out.append(field(name, value, "integer" if type(value) is int else "number", UNITS.get(name)))
        elif kind == "boolean":
            require(type(v) is bool, "MNLM_UCR_FIELD", f"{name}: boolean required")
            out.append(field(name, v, "boolean"))
        elif kind == "clock":
            role = (roles or {}).get(name, "event")
            value, role = clock(v, role)
            out.append(field(name, value, "clock", "UTC", role))
        elif kind == "set":
            require(isinstance(v, list), "MNLM_UCR_FIELD", f"{name}: expected list")
            out.extend(field(name, str(x), "category") for x in v)
        else:
            typ = {"reference": "reference", "text-label": "text", "free-text": "text",
                   "categorical": "category", "category": "category", "text": "text"}.get(kind)
            require(typ is not None, "MNLM_UCR_FIELD", f"{name}: unknown field kind {kind}")
            out.append(field(name, str(v), typ))
    return out


def prepare(envelope, mapping=None):
    """Check the retained record once before constructing any of its atom views."""
    import blake3
    source, shard, rec = envelope["source"], envelope["shard"], envelope["record"]
    text, f = rec["text"], rec["f"]
    require(isinstance(text, str) and isinstance(f, dict), "MNLM_UCR_CANON", "text and explicit fields required")
    require(rec["b3"] == blake3.blake3(text.encode()).hexdigest()[:32] and rec["n"] == len(text.encode()),
            "MNLM_UCR_CANON_DIGEST", "retained canonical text changed")
    manifest = envelope["source_manifest"]
    require(manifest.get("source") == source and manifest.get("shard") == shard and not manifest.get("error")
            and bool(manifest.get("out_blake3")), "MNLM_UCR_MANIFEST", "source/shard manifest mismatch")
    ch = {c: absent("not_supplied_by_source") for c in CHANNELS}
    refs, rels = [], []
    if mapping is not None:
        require(mapping.get("contract") == "ucr-adapter-v1" and mapping.get("source") == source,
                "MNLM_UCR_ADAPTER", "adapter contract/source mismatch")
        plan = mapping["fields"]
        for name, namespace in mapping.get("referent_fields", {}).items():
            if f.get(name):
                refs.append(exact_ref(namespace + ":" + str(f[name]), name))
    elif source in PLANS:
        plan = PLANS[source]["explicit"]
    elif "edition" in f and "ref" in f:
        plan = {"book": "reference", "chapter": "ordinal", "verse": "reference", "lang": "categorical", "role": "categorical"}
    else:
        require(False, "MNLM_UCR_ADAPTER_REQUIRED", f"{source}: supply a declarative source adapter")
    ch["fields"] = present(typed_fields(f, plan, {"DATEADDED": "observed", "Day": "event", "date_filed": "event"}))
    prov = [field("source", source, "reference"), field("manifest", digest(manifest), "reference")]
    for key in ("edition", "licence", "lang", "generation_algorithm", "domain", "court_id"):
        if f.get(key) is not None:
            prov.append(field(key, str(f[key]), "category"))
    ch["provenance"] = present(prov)
    if source == "wikidata5m":
        subject = f.get("entity_id", f.get("s"))
        if subject:
            refs.append(exact_ref("wikidata:" + subject, "entity_id/s"))
        if all(k in f for k in ("s", "p", "o")):
            require(text.split("\t") == [f["s"], f["p"], f["o"]], "MNLM_UCR_CANON", "triple text and fields disagree")
            rels.append(relation(f["p"], "wikidata:" + f["o"], source))
    elif source == "wikipedia-en-20231101":
        refs.append(exact_ref("wikipedia:en:" + str(f["id"]), "id"))
        # A page title is not a QID join. Only an attested mapping may add one.
    elif source == "gdelt-events-90d":
        refs.append(exact_ref("gdelt-event:" + f["GlobalEventID"], "GlobalEventID"))
        if f.get("Actor1Code") and f.get("Actor2Code"):
            refs.append(exact_ref("gdelt-dyad:" + digest([f["Actor1Code"], f["Actor2Code"]]), "Actor1Code/Actor2Code"))
            rels.append(relation("gdelt:" + f["EventCode"], "gdelt-actor:" + f["Actor2Code"], source,
                                 [field("actor1", f["Actor1Code"], "reference")]))
        # The TSV contains outcome fields. It is structured data, not a second text sense.
        ch["surface"] = absent("structured_row_not_independent_surface")
    elif source == "cuyahoga-eighth-district":
        if f.get("opinion_id"):
            refs.append(exact_ref("courtlistener-opinion:" + str(f["opinion_id"]), "opinion_id"))
        if f.get("docket_id"):
            rels.append(relation("in_docket", "courtlistener-docket:" + str(f["docket_id"]), source))
    elif source == "opencodeinstruct":
        refs.append(exact_ref("oci-task:" + str(f["id"]), "id"))
        # Labels and execution outcomes are oracles, never a behaviour sense.
        ch["behaviour"] = absent("no_independent_execution_observation")
    if "edition" in f and f.get("role") == "verse":
        # Coordinate equivalence does not establish semantic equivalence. Different
        # versifications require an attested map; no automatic cross-edition join.
        refs.append(exact_ref("verse:" + f["edition"] + ":" + f["ref"], "edition/ref"))
    ch["relations"] = present(rels)
    from .code_syntax import SyntaxSource
    return {"source": source, "shard": shard, "rec": rec, "manifest": manifest,
            "channels": ch, "refs": refs, "record_digest": digest(rec),
            "syntax": SyntaxSource(text) if rec["kind"] == "prose" else None}


def atom(prepared, start=0, end=None, role=None):
    p, rec = prepared, prepared["rec"]
    end = len(rec["text"]) if end is None else end
    require(0 <= start < end <= len(rec["text"]), "MNLM_UCR_SPAN", "atom span outside canonical text")
    loc = {"source": p["source"], "shard": p["shard"], "row": rec["row"], "span": [start, end],
           "record_digest": p["record_digest"], "manifest_digest": digest(p["manifest"])}
    ch = copy.deepcopy(p["channels"])
    # Identifier triples are retained as typed relations; do not embed raw QIDs.
    if rec["kind"] == "prose":
        ch["surface"] = present([{"text": rec["text"][start:end]}])
    ch["structure"] = present([field("kind", rec["kind"], "category"),
                               field("role", str(role if role is not None else rec["f"].get("role", "record")), "category"),
                               field("char_start", start, "integer", "character"), field("char_end", end, "integer", "character")])
    if p["syntax"] is not None:
        ch["structure"]["items"].extend(p["syntax"].for_span(start, end))
    return seal_record({"schema": VERSION, "locator": loc, "unit": digest([p["source"], p["shard"], rec["row"]]),
                        "referents": p["refs"], "channels": ch})


def native(record):
    """A new source may supply UCR directly; it never registers another lens."""
    from .contract import validate_record
    return validate_record(copy.deepcopy(record))
