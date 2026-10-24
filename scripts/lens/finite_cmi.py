"""Conservative exact-context Shannon-CMI lower bound; commissioning ONLY (#12).

For fixed binary U=f(X), V=g(Y), conditional DPI gives I(X;Y|Z)>=I(U;V|Z).
Keep Z exact. Choose atomic Z states on an independent outcome-blind design set.
Simultaneous binomial intervals bound d=P(UV|z)-P(U|z)P(V|z). Binary TV=2|d|;
Pinsker gives I(U;V|z)>=8*d^2/ln(2) bits. Weight by lower bounds on P(Z=z).
Unselected contexts contribute zero to this bound, NOT zero true information.

Validity assumes IID units, fixed witnesses/roster independent of evaluation,
and complete native conditioning. Unique unit IDs cannot establish IID sampling.
No generic continuous-CMI claim, adaptive budget reuse, or production admission.
Cost O(input bytes+n+contexts) and O(design-context bytes) counting state.
Clopper/Pearson (1934); conditional DPI/Pinsker; derivation in #12 preflight.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math

from scipy.stats import beta

SCHEMA = "exact-context-cmi-commissioning-v1"
SEEDS = (42, 43, 44, 45, 46)
DESIGN_UNITS = 128
MIN_UNITS = 50
BIT_FLOOR = 0.05
ALPHA = 0.05


class BoundError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def require(ok, code, message):
    if not ok:
        raise BoundError(code, message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {"schema", "alpha", "seeds", "design_units", "comparisons", "assumptions"},
            "MNLM_CMI_PLAN", "freeze the complete comparison roster and sampling assumptions")
    require(plan["schema"] == SCHEMA and plan["alpha"] == ALPHA and
            plan["seeds"] == list(SEEDS) and plan["design_units"] == DESIGN_UNITS,
            "MNLM_CMI_PLAN", "commissioning constants differ from the frozen contract")
    a = plan["assumptions"]
    require(isinstance(a, dict) and set(a) == {"sampling", "justification", "witnesses", "conditioning"} and
            a["sampling"] == "iid-independent-units" and
            isinstance(a["justification"], str) and bool(a["justification"].strip()) and
            a["witnesses"] == "fixed-before-evaluation" and a["conditioning"] == "complete-native-state",
            "MNLM_CMI_ASSUMPTIONS", "declare and substantiate IID units, fixed witnesses and full native Z")
    c = plan["comparisons"]
    require(isinstance(c, dict) and bool(c) and all(isinstance(k, str) and k for k in c),
            "MNLM_CMI_PLAN", "empty or invalid predeclared comparison roster")
    for spec in c.values():
        require(isinstance(spec, dict) and set(spec) == {"candidate_event", "outcome_event", "panel_ids", "evaluation_units"} and
                all(isinstance(spec[k], str) and bool(spec[k].strip()) for k in ("candidate_event", "outcome_event")) and
                isinstance(spec["panel_ids"], list) and all(isinstance(i, str) and i for i in spec["panel_ids"]) and
                len(set(spec["panel_ids"])) == len(spec["panel_ids"]) and
                type(spec["evaluation_units"]) is int and spec["evaluation_units"] >= MIN_UNITS,
                "MNLM_CMI_PLAN", "name both frozen witness functions for each comparison")


@lru_cache(maxsize=16384)
def interval(k, n, alpha):
    """Two-sided Clopper-Pearson; slight outward rounding never raises the LCB."""
    require(type(k) is int and type(n) is int and 0 <= k <= n and math.isfinite(alpha) and 0 < alpha < 1,
            "MNLM_CMI_COUNT", "need integer 0<=k<=n and a finite error budget in (0,1)")
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.isf(alpha / 2, k + 1, n - k))
    require(math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= 1,
            "MNLM_CMI_NUMERICS", "binomial interval failed; inspect runtime and exact counts")
    # Numerical allowance, not a relaxed statistical threshold. Tested separately
    # against inversion of binomial tails; these are floating, not formal intervals.
    return (max(0.0, lo - 1e-12), min(1.0, hi + 1e-12))


def context_bound(n, total, u, v, both, alpha):
    require(all(type(x) is int for x in (n, total, u, v, both)) and 0 <= n <= total and
            0 <= u <= n and 0 <= v <= n and max(0, u + v - n) <= both <= min(u, v),
            "MNLM_CMI_COUNT", "invalid binary contingency table")
    mass = interval(n, total, alpha)
    a, b, c = interval(u, n, alpha), interval(v, n, alpha), interval(both, n, alpha)
    cov = (c[0] - a[1] * b[1], c[1] - a[0] * b[0])
    d = max(0.0, cov[0], -cov[1])
    supported = n >= MIN_UNITS and min(v, n - v) >= MIN_UNITS
    info = 8 * d * d / math.log(2) if supported else 0.0
    return {"n": n, "u": u, "v": v, "both": both, "mass_interval": mass,
            "u_interval": a, "v_interval": b, "both_interval": c,
            "covariance_interval": cov, "bits_lcb_given_context": info,
            "weighted_bits_lcb": mass[0] * info,
            "supported": supported}


def estimate(plan, comparison, seed, design, evaluation):
    """Rows: design=(unit, exact bytes); eval=(unit, exact bytes, bool U, bool V).

    Full byte equality is used in grouping, never a digest as the equality key.
    Design rows contain no witnesses/outcomes; all are selected, without ranking.
    Repeat calls on the same frozen data reproduce a result, not new evidence.
    """
    validate_plan(plan)
    require(comparison in plan["comparisons"] and seed in SEEDS and type(seed) is int,
            "MNLM_CMI_COMPARISON", "comparison/seed absent from frozen multiplicity roster")
    require(len(design) == DESIGN_UNITS and len(evaluation) >= MIN_UNITS,
            "MNLM_CMI_SAMPLE", "need 128 separate design units and at least 50 evaluation units")
    require(len(evaluation) == plan["comparisons"][comparison]["evaluation_units"],
            "MNLM_CMI_SAMPLE_PLAN", "evaluation extent differs from frozen plan; do not peek or stop adaptively")
    for rows, size in ((design, 2), (evaluation, 4)):
        require(all(isinstance(r, (tuple, list)) and len(r) == size and
                    isinstance(r[0], str) and bool(r[0]) and isinstance(r[1], bytes) for r in rows),
                "MNLM_CMI_ROW", "rows require a nonempty independent-unit identity and exact state bytes")
    du, eu = [r[0] for r in design], [r[0] for r in evaluation]
    require(len(set(du + eu)) == len(du) + len(eu), "MNLM_CMI_UNIT_REUSE",
            "repeated unit or overlapping design/evaluation: select one observation per independent unit")
    require(all(type(r[2]) is bool and type(r[3]) is bool for r in evaluation),
            "MNLM_CMI_WITNESS", "frozen events must yield booleans; do not silently coerce labels")
    contexts = {state: [0, 0, 0, 0] for _, state in design}
    alpha = ALPHA / (len(SEEDS) * len(plan["comparisons"]) * 4 * len(contexts))
    for _, state, u, v in evaluation:
        if state in contexts:
            count = contexts[state]
            count[0] += 1
            count[1] += int(u)
            count[2] += int(v)
            count[3] += int(u and v)
    contributions = []
    for state, (n, u, v, both) in sorted(contexts.items()):
        contributions.append({"state_sha256": hashlib.sha256(state).hexdigest(),
                              "state_bytes": len(state),
                              **context_bound(n, len(evaluation), u, v, both, alpha)})
    lower = math.fsum(c["weighted_bits_lcb"] for c in contributions)
    supported = sum(c["n"] for c in contributions if c["supported"])
    status = ("MNLM_CMI_BOUND_EXCEEDS_FLOOR" if lower >= BIT_FLOOR else
              "MNLM_CMI_EXACT_CONTEXT_SUPPORT" if supported == 0 else "MNLM_CMI_INSUFFICIENT")
    return {"schema": SCHEMA, "comparison": comparison, "seed": seed, "plan_sha256": digest(plan),
            "evaluation_units": len(evaluation), "design_units": len(design),
            "selected_contexts": len(contexts), "supported_evaluation_units": supported,
            "unselected_evaluation_units": len(evaluation) - sum(c["n"] for c in contributions),
            "interval_alpha": alpha, "family_alpha": ALPHA, "bits_lcb": lower,
            "code": status, "contexts": contributions, "production_admission": False,
            "scope": "commissioning; conditional on declared assumptions; not independently accepted"}
