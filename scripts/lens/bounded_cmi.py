"""Assumption-bound binary-witness CMI lower bound; commissioning only (#12).

For C=h(Z), enclose P(U|z), P(V|z), P(UV|z) by cell confidence intervals
widened by declared population-uniform conditional variation allowances. A
covariance interval separated from zero bounds full-Z MI by Pinsker. Sum
disjoint cells using lower mass bounds, then apply conditional DPI. Z is kept
complete; h(Z) is an arithmetic partition, never substituted for the estimand.

Validity requires IID units, fixed witnesses/partition, and TRUE variation
allowances holding almost surely for the entire population in each cell.
These assumptions cannot be established by fitting this evaluation sample.
In particular, a caller's hash/justification is NOT proof of its assumptions.
No return value from this experimental module authorizes production admission.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math

import numpy as np
import slot_layout
from lenscommon import LensError

from finite_cmi import ALPHA, BIT_FLOOR, MIN_UNITS, SEEDS, canonical, digest, interval, require

SCHEMA = "bounded-variation-cmi-commissioning-v1"


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {"schema", "alpha", "seeds", "comparisons"}
            and plan["schema"] == SCHEMA and plan["alpha"] == ALPHA and plan["seeds"] == list(SEEDS),
            "MNLM_CMI_PLAN", "freeze the comparison/seed roster and unchanged error budget")
    require(isinstance(plan["comparisons"], dict) and bool(plan["comparisons"]),
            "MNLM_CMI_PLAN", "predeclared comparisons required")
    for name, c in plan["comparisons"].items():
        require(isinstance(name, str) and bool(name) and isinstance(c, dict)
                and set(c) == {"candidate_event", "outcome_event", "panel_ids", "evaluation_units",
                               "partition", "variation", "assumptions"}
                and all(isinstance(c[k], str) and bool(c[k]) for k in ("candidate_event", "outcome_event"))
                and type(c["evaluation_units"]) is int and c["evaluation_units"] >= MIN_UNITS
                and isinstance(c["panel_ids"], list) and bool(c["panel_ids"])
                and all(isinstance(x, str) and x for x in c["panel_ids"])
                and len(c["panel_ids"]) == len(set(c["panel_ids"])),
                "MNLM_CMI_PLAN", "fixed witnesses, panel identities and evaluation extent required")
        p = c["partition"]
        require(isinstance(p, dict), "MNLM_CMI_PARTITION", "explicit fixed partition required")
        if p.get("kind") == "one-cell":
            require(set(p) == {"kind"}, "MNLM_CMI_PARTITION", "unknown one-cell parameters")
            cells = {"all"}
        else:
            require(set(p) == {"kind", "lens", "coordinate", "cut"}
                    and p["kind"] == "dense-coordinate-cut" and p["lens"] in c["panel_ids"]
                    and type(p["coordinate"]) is int and p["coordinate"] >= 0
                    and type(p["cut"]) in (int, float) and math.isfinite(p["cut"]),
                    "MNLM_CMI_PARTITION", "fixed native dense coordinate and finite cut required")
            cells = {"below", "at_or_above"}
        a = c["assumptions"]
        require(isinstance(a, dict) and set(a) == {"sampling", "sampling_basis", "regularity_basis",
                "basis_sha256", "scope", "evaluation_used"} and a["sampling"] == "iid-independent-units"
                and isinstance(a["sampling_basis"], str) and bool(a["sampling_basis"]),
                "MNLM_CMI_ASSUMPTIONS", "substantiate independent units; unique IDs alone are insufficient")
        require(isinstance(a["regularity_basis"], str) and bool(a["regularity_basis"])
                and a["scope"] == "full-native-Z-population-almost-sure" and a["evaluation_used"] is False
                and isinstance(a["basis_sha256"], str) and len(a["basis_sha256"]) == 64
                and all(x in "0123456789abcdef" for x in a["basis_sha256"]),
                "MNLM_CMI_REGULARITY_UNESTABLISHED", "supply an external full-population variation argument; never fit it to evaluation")
        v = c["variation"]
        require(isinstance(v, dict) and set(v) == cells,
                "MNLM_CMI_REGULARITY_UNESTABLISHED", "every partition cell requires explicit variation allowances")
        require(all(isinstance(e, dict) and set(e) == {"u", "v", "both"}
                    and all(probability(x) for x in e.values()) for e in v.values()),
                "MNLM_CMI_REGULARITY_UNESTABLISHED", "allowances must be known finite probabilities in [0,1]")


def native_parts(state, comparison):
    """Validate ALL typed conditioning values, including unpartitioned members."""
    try:
        slots = json.loads(state)
    except (ValueError, TypeError) as exc:
        require(False, "MNLM_CMI_NATIVE_STATE", f"invalid full native state: {type(exc).__name__}")
    require(isinstance(slots, list) and all(isinstance(s, dict) for s in slots)
            and [s.get("id") for s in slots] == comparison["panel_ids"],
            "MNLM_CMI_PANEL_INCOMPLETE", "complete ordered conditioning roster required")
    decoded = {}
    for s in slots:
        try:
            kind, members = s["kind"], s["members"]
            expected = {"dense": {"X"}, "multi": {"tokens"},
                        "sparse": {"data", "indices", "vocabulary_extent"}}[kind] | {"absent"}
            require(set(s) == {"id", "kind", "members"} and
                    (set(members) == expected or set(members) == expected | {"windows"}),
                    "MNLM_CMI_NATIVE_STATE", "missing or extra native member")
            arrays = {}
            for key, m in members.items():
                require(set(m) == {"shape", "dtype", "bytes"} and m["dtype"] in ("|b1", "<f4", "<i8", "<i4")
                        and isinstance(m["shape"], list) and all(type(n) is int and n >= 0 for n in m["shape"]),
                        "MNLM_CMI_NATIVE_STATE", "typed numeric native shape/dtype required")
                raw = base64.b64decode(m["bytes"], validate=True)
                dtype = np.dtype(m["dtype"])
                require(len(raw) == math.prod(m["shape"])*dtype.itemsize,
                        "MNLM_CMI_NATIVE_STATE", "native byte extent differs from shape")
                arrays[key] = np.frombuffer(raw, dtype=dtype).reshape(m["shape"])
            if kind == "dense":
                require(arrays["X"].ndim == 1, "MNLM_CMI_NATIVE_STATE", "one native dense observation required")
                arrays["X"] = arrays["X"][None, :]
            else:
                k = "tokens" if kind == "multi" else "data"
                arrays["offsets" if kind == "multi" else "indptr"] = np.asarray([0, len(arrays[k])], np.int64)
                if kind == "sparse":
                    extent = arrays.pop("vocabulary_extent")
                    require(extent.shape == (1,) and extent.dtype.str == "<i8", "MNLM_CMI_NATIVE_STATE", "sparse extent required")
                    arrays["shape"] = np.asarray([1, int(extent[0])], np.int64)
            info = slot_layout.validate(arrays, kind)
            require(info["n"] == 1, "MNLM_CMI_NATIVE_STATE", "native state must contain exactly one observation")
            decoded[s["id"]] = (kind, arrays)
        except (KeyError, ValueError, TypeError, LensError) as exc:
            require(False, "MNLM_CMI_NATIVE_STATE", f"malformed native state: {type(exc).__name__}")
    return decoded


def cell_of(state, comparison):
    """Partition complete state; arithmetic grouping never replaces full native Z."""
    decoded = native_parts(state, comparison)
    p = comparison["partition"]
    if p["kind"] == "one-cell":
        return "all"
    kind, arrays = decoded[p["lens"]]
    require(kind == "dense" and not arrays["absent"][0] and p["coordinate"] < arrays["X"].shape[1],
            "MNLM_CMI_PARTITION_NATIVE", "partition requires a present finite native dense coordinate")
    return "below" if float(arrays["X"][0, p["coordinate"]]) < p["cut"] else "at_or_above"


def cell_bound(n, total, u, v, both, alpha, allowance):
    require(all(type(x) is int for x in (n, total, u, v, both)) and 0 <= n <= total
            and 0 <= u <= n and 0 <= v <= n and max(0, u+v-n) <= both <= min(u, v),
            "MNLM_CMI_COUNT", "inconsistent binary contingency table")
    require(set(allowance) == {"u", "v", "both"} and all(probability(x) for x in allowance.values()),
            "MNLM_CMI_REGULARITY_UNESTABLISHED", "explicit finite variation required")
    mass = interval(n, total, alpha)
    cell = {k: interval(count, n, alpha) for k, count in (("u",u),("v",v),("both",both))}
    widened = {k: (max(0., b[0]-allowance[k]), min(1., b[1]+allowance[k])) for k,b in cell.items()}
    a, b, c = widened["u"], widened["v"], widened["both"]
    covariance = (c[0]-a[1]*b[1], c[1]-a[0]*b[0])
    d = max(0., covariance[0], -covariance[1])
    supported = n >= MIN_UNITS and min(v, n-v) >= MIN_UNITS
    bits = 8*d*d/math.log(2) if supported else 0.
    return dict(n=n, u=u, v=v, both=both, mass_interval=mass, cell_intervals=cell,
                full_z_intervals=widened, covariance_interval=covariance, allowance=allowance,
                supported=supported, bits_lcb_at_every_z=bits, weighted_bits_lcb=mass[0]*bits)


def estimate(plan, comparison, seed, rows):
    """Rows=(independent unit ID, COMPLETE native state bytes, bool U, bool V)."""
    validate_plan(plan)
    require(comparison in plan["comparisons"] and type(seed) is int and seed in SEEDS,
            "MNLM_CMI_COMPARISON", "comparison/seed absent from frozen roster")
    c = plan["comparisons"][comparison]
    require(len(rows) == c["evaluation_units"], "MNLM_CMI_SAMPLE_PLAN", "evaluation extent differs; no optional stopping")
    require(all(isinstance(r, (list, tuple)) and len(r) == 4 and isinstance(r[0], str) and r[0]
                and isinstance(r[1], bytes) and type(r[2]) is bool and type(r[3]) is bool for r in rows),
            "MNLM_CMI_ROW", "unique IDs, complete native bytes and boolean witnesses required")
    require(len({r[0] for r in rows}) == len(rows), "MNLM_CMI_UNIT_REUSE", "one observation per independent unit")
    counts = {k: [0,0,0,0] for k in c["variation"]}
    input_hash = hashlib.sha256()
    for unit, state, u, v in rows:
        cell = cell_of(state, c)
        values = counts[cell]
        values[0] += 1; values[1] += u; values[2] += v; values[3] += u and v
        # Length-prefixed full bytes retain every coordinate even if h ignores it.
        for part in (unit.encode(), state, bytes([u, v])):
            input_hash.update(len(part).to_bytes(8, "big")); input_hash.update(part)
    tests = sum(len(x["variation"]) for x in plan["comparisons"].values())
    alpha = ALPHA/(len(SEEDS)*tests*4)
    results = {k: cell_bound(*count[:1], len(rows), *count[1:], alpha, c["variation"][k])
               for k,count in counts.items()}
    lower = math.fsum(x["weighted_bits_lcb"] for x in results.values())
    return dict(schema=SCHEMA, plan_sha256=digest(plan), comparison=comparison, seed=seed,
                input_sha256=input_hash.hexdigest(), evaluation_units=len(rows), cells=results,
                interval_alpha=alpha, family_alpha=ALPHA, bits_lcb=lower,
                bound_exceeds_floor=lower >= BIT_FLOOR, conditional_on_declared_assumptions=True,
                assumptions_proven_by_this_estimator=False, production_admission=False,
                scope="commissioning only; full native Z with explicit population variation allowances")
