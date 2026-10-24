"""Finite, declared source-document acquisition/retention; never lens admission."""
import math

from claims.codata_source import check


POLICY = dict(schema="text-candidate-policy-v1", scope="exposed_source_documents_only",
    acquisition="each_native_error_strictly_below_parent_and_persistence; region_logloss_below_parent",
    retention="each_native_error_and_region_logloss_no_increase",
    stages={"cycle-1": dict(parent="initial-training", acquire=["general", "legal"], retain=[]),
            "cycle-2": dict(parent="cycle-1", acquire=["science"], retain=["general", "legal"])},
    scientific_admission=False, serving=False, conditional_bits_measured=False)


def value(number):
    check(type(number) in (int, float) and math.isfinite(number) and number >= 0,
          "TEXT_SELECTION_METRIC")
    return number


def assess(name, parent, candidate, groups, instruments, *, retained, established):
    check(name in POLICY["stages"], "TEXT_SELECTION_STAGE")
    stage = POLICY["stages"][name]
    declared = {g["domain"]: g for g in groups if g["split"] == "development"}
    check(len(declared) == 3 and set(declared) == {"general", "legal", "science"}
          and set(parent) == set(candidate) == set(declared), "TEXT_SELECTION_DOCUMENT_ROSTER")
    checks, acquired, preserved = {}, True, True
    for domain in stage["acquire"]+stage["retain"]:
        old, new = parent[domain], candidate[domain]
        check(old["count"] == new["count"] == len(declared[domain]["indices"])-1
              and set(old["instruments"]) == set(new["instruments"]) == set(instruments),
              "TEXT_SELECTION_UNIT_COVERAGE")
        old_nll, new_nll = value(old["region_nll_bits"]), value(new["region_nll_bits"])
        acquisition = domain in stage["acquire"]
        lanes = {}
        passed = new_nll < old_nll if acquisition else new_nll <= old_nll
        for lane in instruments:
            a, b = old["instruments"][lane], new["instruments"][lane]
            before, after, persistent = value(a["prediction"]), value(b["prediction"]), value(b["persistence"])
            check(persistent == value(a["persistence"]), "TEXT_SELECTION_BASELINE_CHANGED")
            ok = after < min(before, persistent) if acquisition else after <= before
            lanes[lane] = dict(parent=before, candidate=after, persistence=persistent, passed=ok)
            passed &= ok
        checks[domain] = dict(document=declared[domain]["context"],
            role="acquisition" if acquisition else "retention", count=new["count"],
            region_logloss=dict(parent=old_nll, candidate=new_nll), instruments=lanes, passed=bool(passed))
        if acquisition:
            acquired &= passed
        else:
            preserved &= passed
    reasons = []
    if retained != stage["parent"]:
        reasons.append("UNRETAINED_PARENT")
    if set(stage["retain"]) != set(established):
        reasons.append("ESTABLISHED_DOMAIN_MISMATCH")
    if not acquired:
        reasons.append("ACQUISITION_GATE_FAILED")
    if not preserved:
        reasons.append("RETENTION_GATE_FAILED")
    return dict(eligible_diagnostic_lineage=not reasons, reasons=reasons, checks=checks,
        acquisition_passed=bool(acquired), retention_passed=bool(preserved),
        serving=False, meaning_asserted=False, conditional_bits_measured=False)
