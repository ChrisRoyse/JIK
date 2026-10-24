"""Stage 3 freeze and pin (issue #12, spec §19.3 step 8): the admitted lenses of every anchor family, with their frozen
contracts and lens_ids, become a named panel version with a machine-checkable invariant list (lens count, the BLAKE3 of
the sorted lens_ids, every weights digest, dtype f32, no CPU fallback, the runtime they were probed on). The panel file
is content-addressed, ledgered (`lenses.panel`) and copied into the repository as config/panel.<version>.json so Stage 4
on any host measures with the same instruments. A panel that already exists refuses MNLM_PANEL_EXISTS unless --supersede
names it as superseded in the new panel. The P0 gates of #6 row 3 are evaluated here and written, pass or fail.
    python -I scripts/lens/panel.py --root R --out O --version p0-v1 --panel-version panel-p0-v1 [--repo C:/code/MNLM]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import blake3
import math

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogue import by_name  # noqa: E402
from admission_state import load_reports, validate_ledger  # noqa: E402
from lenscommon import FAMILIES, HOST, RETIRED_QUESTIONS, LensError, sealed_registry, Paths, ledger_append, lens_root, now_iso, spec_family, write_json  # noqa: E402

MIN_CANDIDATES = 40
MIN_LENSES = 5
MIN_FAMILIES = 3
MAX_PAIRWISE_CORR = 0.6


def max_corr(reports: dict[str, dict], questions: list[str], a: str, b: str) -> float | None:
    """|Spearman| between two lenses, the maximum over the family's questions on which both were read; None if no question
    could read it (fail closed: an unreadable correlation is not a correlation below 0.6)."""
    worst = None
    for q in questions:
        corr = reports[q]["correlation"]
        if a in corr["order"] and b in corr["order"]:
            v = corr["matrix"][corr["order"].index(a)][corr["order"].index(b)]
            if v is None or not math.isfinite(float(v)):
                return None
            worst = max(worst if worst is not None else -1.0, abs(float(v)))
    return worst


def union_panel(reports: dict[str, dict]) -> dict:
    """spec §19.1: lenses are admitted per anchor family and the panel is the union over the family's questions; Def. 13.3 D1
    binds the union too, so a lens admitted on a later question that correlates above 0.6 with a lens already in the union
    (on any question of the family) is a duplicate of the panel and is excluded, in question order then admission order."""
    out: dict[str, dict] = {}
    for fam in reports:
        out.setdefault(spec_family(fam), {"questions": [], "admitted_by_question": {}, "admitted_union": [], "union_excluded": [], "served": False})["questions"].append(fam)
    for a, u in out.items():
        for fam in u["questions"]:
            u["admitted_by_question"][fam] = list(reports[fam]["admission"]["panel"])
            for nm in reports[fam]["admission"]["panel"]:
                if nm in u["admitted_union"]:
                    continue
                pairs = {p: max_corr(reports, u["questions"], nm, p) for p in u["admitted_union"]}
                undefined = [p for p, v in pairs.items() if v is None]
                if undefined:
                    u["union_excluded"].append({"lens": nm, "question": fam, "reason": "correlation_undefined", "against": undefined})
                    continue
                dup = max(pairs, key=pairs.get) if pairs else None
                if dup is not None and pairs[dup] > MAX_PAIRWISE_CORR:
                    u["union_excluded"].append({"lens": nm, "question": fam, "reason": "correlation", "duplicate_of": dup, "spearman": pairs[dup], "gate": MAX_PAIRWISE_CORR})
                    continue
                u["admitted_union"].append(nm)
        u["perspectives"] = {nm: by_name(nm)["perspective"] for nm in u["admitted_union"]}
        u["perspectives_admitted"] = sorted(set(u["perspectives"].values()))
        # D4 counts admitted lenses (each passed D1/D2); perspectives are reported, not counted (operator 2026-10-18).
        u["candidate_count_gate"] = len(u["admitted_union"]) >= MIN_LENSES
        # A union across unrelated questions is not five senses jointly present
        # at an asserted meaning. Current usable-information assays cannot certify D2.
        u["served"] = False
        u["status"] = "MNLM_SHANNON_CONDITIONAL_UNMEASURED"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="p0-v1")
    ap.add_argument("--panel-version", default="panel-p0-v1")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--supersede", action="store_true")
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    root = lens_root(paths)
    out = root / "panel.json"
    superseded = None
    if out.exists():
        if not args.supersede:
            raise LensError("MNLM_PANEL_EXISTS", f"{out} is frozen; pass --supersede to freeze a successor")
        superseded = json.loads(out.read_text(encoding="utf-8")).get("panel_blake3")
    registry = sealed_registry(root)
    admission = json.loads((root / "assay" / "admission.json").read_text(encoding="utf-8"))
    validate_ledger(paths.ledger, registry, admission)
    validated = load_reports(root, registry, admission)
    families = {}
    admitted: dict[str, list[str]] = {}
    bits: dict[str, dict] = {}
    reports = {}
    retired_q: dict[str, str] = {}
    for fam in FAMILIES:
        if fam not in admission["digests"]:
            continue  # a declared question without an assay is not part of this freeze; the admission ledger names what was assayed
        if fam in RETIRED_QUESTIONS:
            retired_q[fam] = RETIRED_QUESTIONS[fam]
            continue  # a retired question stays on disk with its assay; it never counts toward a family (decisions D-3/D-4, #12)
        rep = reports[fam] = validated[fam]
        stale = [nm for nm in rep["admission"]["panel"] if registry["lenses"].get(nm, {}).get("error") == "MNLM_LENS_RETIRED"]
        if stale:
            raise LensError("MNLM_PANEL_RETIRED_LENS_ADMITTED", f"{fam}: its assay record admits retired lenses {stale}; re-assay the question before freezing")
        panel = rep["admission"]["panel"]
        families[fam] = {"admitted": panel, "served": rep["served"], "n_eff": rep["n_eff"], "sufficiency": rep["sufficiency"], "H_Y_bits": rep["H_Y_bits"], "n": rep["n"],
                         "assay_blake3": admission["digests"][fam], "eligible_standalone": rep["admission"]["eligible"]}
    anchor_families = union_panel(reports)
    for fam, rep in reports.items():
        for nm in rep["admission"]["panel"]:
            if nm not in anchor_families[spec_family(fam)]["admitted_union"]:
                continue  # excluded from the union by D1: not a panel lens
            admitted.setdefault(nm, []).append(fam)
            bits.setdefault(nm, {})[fam] = {k: rep["lenses"][nm][k] for k in ("bits_mean", "lcb_min", "ucb_max", "n_used", "coverage", "fingerprint_suspect")}
            bits[nm][fam]["admitted_by"] = (rep["admission"].get("admitted_by") or {}).get(nm, {"rule": "conditional_gain"})
    lenses = []
    for nm in sorted(admitted):
        e = registry["lenses"][nm]
        lenses.append({"name": nm, "lens_id": e["lens_id"], "contract": e["contract"], "probe": e["probe"], "admitted_for": admitted[nm], "bits": bits[nm]})
    ok_lenses = {n: e for n, e in registry["lenses"].items() if not e.get("error")}
    served = [a for a, v in sorted(anchor_families.items()) if v["served"]]
    gates = {"candidates_with_contracts_and_probes": len(ok_lenses), "candidates_ge_40": len(ok_lenses) >= MIN_CANDIDATES,
             "perspectives_per_family": {a: v["perspectives_admitted"] for a, v in anchor_families.items()}, "rule": "usable candidates are diagnostic; Shannon conditional evidence and per-meaning coverage are not commissioned",
             "shannon_conditional_certified": False,
             "every_registered_lens_probe_byte_identical": all(e["probe"]["byte_identical"] for e in ok_lenses.values()),
             "families_served_ge5_lenses": served, "families_served_ge3": len(served) >= MIN_FAMILIES, "questions_served_ge5_lenses": [f for f, v in families.items() if v["served"]],
             "sufficiency_reported_per_family": all("sufficiency" in v for v in families.values()),
             "families_sufficient": [f for f, v in families.items() if v["sufficiency"]["sufficient"]],
             "registration_failures": {n: e["error"] for n, e in registry["lenses"].items() if e.get("error")}, "retired_questions": retired_q}
    ids = sorted(l["lens_id"] for l in lenses)
    panel = {"panel_version": args.panel_version, "contract": "panel-v1", "population_version": args.version, "lenses": lenses, "families": families, "anchor_families": anchor_families, "gates": gates,
             "invariants": {"lens_count": len(lenses), "lens_ids_blake3": blake3.blake3("".join(ids).encode()).hexdigest(), "weights_sha256": {l["name"]: l["contract"]["weights_sha256"] for l in lenses},
                            "dtype": "f32", "cpu_fallback": False, "runtime": registry["runtime"], "registry_blake3": registry.get("registry_blake3_before_ledger"),
                            "admission_ledger_seq": admission.get("ledger_seq"), "calibration_ledger_seq": None},
             "superseded_panel_blake3": superseded, "frozen_at": now_iso(), "host": HOST}
    from universal import coverage_report
    panel["universal"] = coverage_report(registry, reports)
    if any(e.get("spec", {}).get("role") == "universal_instrument" for e in registry["lenses"].values()):
        universal_names = panel["universal"]["candidate_roster"]
        panel["legacy_diagnostic_lenses"] = panel["lenses"]
        panel["candidate_lenses"] = [{"name": n, **{k: registry["lenses"][n][k] for k in ("lens_id", "contract", "probe")}}
                                     for n in universal_names]
        panel["lenses"] = []
        panel["contract"] = "universal-panel-v1"
        panel["invariants"]["lens_count"] = 0
        panel["invariants"]["lens_ids_blake3"] = blake3.blake3(b"").hexdigest()
        panel["invariants"]["weights_sha256"] = {}
        gates["legacy_candidates_with_contracts_and_probes"] = gates["candidates_with_contracts_and_probes"]
        gates["candidates_with_contracts_and_probes"] = len(universal_names)
        gates["candidates_ge_40"] = len(universal_names) >= MIN_CANDIDATES
        gates["families_served_ge5_lenses"] = []
        gates["families_served_ge3"] = False
        gates["questions_served_ge5_lenses"] = []
        gates["families_sufficient"] = []
        served = []
        lenses, ids = [], []
    cal = root / "cal" / "cal.summary.json"
    if cal.exists():
        panel["invariants"]["calibration_ledger_seq"] = json.loads(cal.read_text(encoding="utf-8")).get("ledger_seq")
    # Publish once, with the ledger binding already attached. A ledger failure
    # leaves the previous accepted panel untouched, never an unsealed live file.
    body = (json.dumps(panel, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    digest = blake3.blake3(body).hexdigest()
    e = ledger_append(paths, f"panel@{HOST}", "lenses.panel", f"stage3/{args.panel_version}", {"panel_blake3": digest, "lens_count": len(lenses), "lens_ids_blake3": panel["invariants"]["lens_ids_blake3"],
                                                                                                "families_served": served, "gates": {k: v for k, v in gates.items() if isinstance(v, bool)}})
    panel["panel_blake3"], panel["ledger_seq"], panel["ledger_head"] = digest, e["seq"], e["hash"]
    write_json(out, panel)
    repo_copy = Path(args.repo) / "config" / f"{args.panel_version}.json"
    write_json(repo_copy, panel)
    print(json.dumps({"panel_blake3": digest, "lenses": len(lenses), "served": served, "gates": {k: v for k, v in gates.items() if isinstance(v, bool)}, "ledger_seq": e["seq"], "repo_copy": str(repo_copy)}), flush=True)
    return 0


if __name__ == "__main__":
    from lenscommon import run_cli
    sys.exit(run_cli(main))
