"""Question-specific visibility over universal observations, never new instruments.

Allowlisting channels is mandatory. Context shares the target's mask and excludes
all declared label units. A policy is evidence to review, not a leakage theorem.
"""
from __future__ import annotations

import copy
import re

from .contract import CHANNELS, absent, canonical, digest, present, require, seal_record
from .scope import check_source
from .store import validate_history_clock
from training_scope import POLICY_SHA256


def validate_policy(p, question):
    keys = {"schema", "question", "allow_channels", "deny_fields", "deny_relations", "blocked_units", "label_aliases", "rationale"}
    require(isinstance(p, dict), "MNLM_UCR_POLICY", "explicit visibility policy required")
    version = p.get("schema")
    require(set(p) == keys | ({"history_clock"} if version == "ucr-view-v2" else set()),
            "MNLM_UCR_POLICY", "explicit channel and label-unit masking policy required")
    require(version in ("ucr-view-v1", "ucr-view-v2") and p["question"] == question and p["rationale"], "MNLM_UCR_POLICY", question)
    require(isinstance(p["allow_channels"], list) and all(isinstance(c, str) for c in p["allow_channels"]), "MNLM_UCR_POLICY", "channel list required")
    require(set(p["allow_channels"]) <= set(CHANNELS) and len(set(p["allow_channels"])) == len(p["allow_channels"]), "MNLM_UCR_POLICY", "unknown or duplicate channel")
    for key in ("deny_fields", "deny_relations", "blocked_units", "label_aliases"):
        require(isinstance(p[key], list) and all(isinstance(x, str) and x for x in p[key]), "MNLM_UCR_POLICY", key)
    if version == "ucr-view-v2":
        clock = validate_history_clock(p["history_clock"])
        require(clock["field"] not in p["deny_fields"], "MNLM_UCR_ORDER_POLICY",
                "a withheld field cannot select historical evidence")
    return p


def masked(r, policy):
    out = copy.deepcopy(r)
    allow, deny = set(policy["allow_channels"]), set(policy["deny_fields"])
    aliases = sorted(set(policy["label_aliases"]), key=lambda s: (-len(s), s))
    regex = re.compile("|".join(re.escape(a) for a in aliases), re.IGNORECASE) if aliases else None
    for c in CHANNELS:
        if c not in allow:
            out["channels"][c] = absent("question_policy_excluded")
            continue
        # Missing time, unavailable modality and withheld evidence are different
        # observations. A visibility view must not erase their recorded reasons.
        if out["channels"][c]["state"] == "absent":
            continue
        items = out["channels"][c]["items"]
        if c in ("fields", "structure", "provenance"):
            items = [x for x in items if x["name"] not in deny]
        elif c == "relations":
            items = [x for x in items if x["rel_type"] not in policy["deny_relations"]]
            for x in items:
                x["qualifiers"] = [q for q in x["qualifiers"] if q["name"] not in deny]
        elif c in ("behaviour", "perception"):
            for x in items:
                x["measurements"] = [q for q in x["measurements"] if q["name"] not in deny]
        # Mask literal aliases in every string-bearing measurement, not identifiers
        # in lineage. Otherwise a sibling field can trivially return the label.
        def scrub(value):
            if isinstance(value, str):
                return regex.sub("[MASKED]", value) if regex else value
            if isinstance(value, list):
                return [scrub(x) for x in value]
            if isinstance(value, dict):
                return {k: scrub(v) if k in ("value", "text", "event") or isinstance(v, (dict, list)) else v for k, v in value.items()}
            return value
        out["channels"][c] = present([scrub(x) for x in items]) if items else absent("question_policy_redacted")
    return seal_record(out)


def view(store, record, policy, generation, surface=None):
    validate_policy(policy, policy.get("question"))
    # A calibration file is not an alternative source of sensory evidence.
    # Accept the retained text or the policy-masked version, never an arbitrary
    # replacement (including prose invented for a channel that was absent).
    retained = store.get(record["atom_id"], generation)
    require(canonical(retained) == canonical(record), "MNLM_UCR_VIEW_BINDING", "view input differs from retained observation")
    # Historical generations remain auditable. Consumption through any view,
    # including direct callers, must obey today's operator training scope.
    check_source(retained["locator"]["source"])
    target = masked(record, policy)
    if surface is not None:
        raw = "\n".join(x["text"] for x in record["channels"]["surface"]["items"])
        visible = "\n".join(x["text"] for x in target["channels"]["surface"]["items"])
        require(surface in (raw, visible), "MNLM_UCR_SURFACE_BINDING",
                "calibration text is not the retained atom or its declared masked view; export the correct source span")
    contexts = {}
    for channel in ("history", "co_reference"):
        if channel not in policy["allow_channels"]:
            contexts[channel] = []
            continue
        raw = store.context(record, channel, generation, policy["blocked_units"],
                            history_clock=policy.get("history_clock") if channel == "history" else None)
        for dependency in raw:
            check_source(dependency["locator"]["source"])
        contexts[channel] = [masked(r, policy) for r in raw]
    return target, contexts, {"generation": generation, "policy_digest": digest(policy),
                              "training_scope_sha256": POLICY_SHA256,
                              "atom": record["atom_id"], "context_atoms": {c: [x["atom_id"] for x in rows] for c, rows in contexts.items()}}
