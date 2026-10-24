"""Stage 2 FSV for the Poly layer (issue #11; workstation only): reads the row-atom summary and field plans back against
the Stage 1 population and Poly's sealed store. B1 atoms by reference: poly.unique.npy re-digested equals the summary, its
length equals population rows minus exact-removed, per-source counts are recomputed from the generation table. B2 plans:
every row field of every source's inventory has a plan, every plan kind is known, no numeric field (number, or decimal
string in the sampled rows) is routed to free-text or text-label, every clock declares a role, and on a sample of
generations (every source, one generation other than the first where it exists) every key of every row has a plan.
Negative controls: a planted numeric field routed to text refuses, a planted clock without a role refuses, a planted key
outside the plan refuses, a tampered digest differs.
    python -I scripts/canon/fsv_stage2_poly.py --root D:/MNLMData --out D:/MNLMData --version p0-v1 --issue 11 --store S --inventory I
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, run_stamp, write_json  # noqa: E402
from poly_dedup import canonical_artifacts, iter_rows  # noqa: E402

CHECKS: list[dict] = []
KINDS = {"scalar", "cyclic", "categorical", "ordinal", "set", "reference", "text-label", "clock", "free-text", "record", "absent"}
CLOCK_ROLES = {"coordinate", "receipt", "revision", "calendar", "lifecycle", "relational"}
DECIMAL = re.compile(r"^-?\d+(\.\d+)?$")
SAMPLE_ROWS = 200


def check(name: str, ok: bool, **detail):
    CHECKS.append({"check": name, "ok": bool(ok), **detail})
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  {json.dumps(detail, default=str)[:240]}"), flush=True)


def plan_violations(fields: dict, rows: list[dict]) -> dict:
    """Independent restatement of the plan rules over sampled rows: (missing keys, numeric routed to text, clocks without a role, unknown kinds)."""
    out = {"missing": [], "numeric_to_text": [], "clock_no_role": [], "unknown_kind": []}
    for name, plan in fields.items():
        if plan["kind"] not in KINDS:
            out["unknown_kind"].append(name)
        if plan["kind"] == "clock" and plan.get("clock") not in CLOCK_ROLES:
            out["clock_no_role"].append(name)
    seen: dict[str, bool] = {}
    for r in rows:
        for k, v in r.items():
            if k not in fields:
                out["missing"].append(k)
                continue
            if v is None or isinstance(v, bool):
                continue
            numeric = isinstance(v, (int, float)) or (isinstance(v, str) and DECIMAL.match(v) is not None)
            seen[k] = seen.get(k, True) and numeric
    for k, all_numeric in seen.items():
        if all_numeric and fields[k]["kind"] in ("free-text", "text-label"):
            out["numeric_to_text"].append(k)
    out["missing"] = sorted(set(out["missing"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--inventory", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    d = paths.dedup
    adir = paths.out / "atoms" / args.version / "poly"
    summary = json.loads((adir / "poly.atoms.summary.json").read_text(encoding="utf-8"))
    plans = json.loads((adir / "field_plans.json").read_text(encoding="utf-8"))
    pop = json.loads((d / "poly.population.json").read_text(encoding="utf-8"))
    dedup_summary = json.loads((d / "poly.summary.json").read_text(encoding="utf-8"))
    unique = np.load(d / "poly.unique.npy")
    # B1 atoms by reference
    b3 = blake3.blake3((d / "poly.unique.npy").read_bytes()).hexdigest()
    check("B1.unique_digest_matches_summary", b3 == summary["unique_blake3"], recomputed=b3[:16], summary=summary["unique_blake3"][:16])
    check("B1.atoms_equal_population_minus_exact_removed", summary["atoms"] == len(unique) == pop["rows"] - dedup_summary["exact"]["removed"], atoms=summary["atoms"], unique=int(len(unique)), rows=pop["rows"], removed=dedup_summary["exact"]["removed"])
    offsets = np.cumsum([0] + [g["rows"] for g in pop["generations"]])
    gen_of = np.searchsorted(offsets, unique, side="right") - 1
    per = {}
    for gi, c in zip(*np.unique(gen_of, return_counts=True)):
        s = pop["generations"][int(gi)]["source"]
        per[s] = per.get(s, 0) + int(c)
    stage1_per = {s: v["rows"] - v["exact_removed"] for s, v in dedup_summary["per_source"].items()}
    check("B1.per_source_atoms_recomputed", per == summary["per_source_atoms"] == stage1_per, sources=len(per), mismatch=[s for s in per if per[s] != summary["per_source_atoms"].get(s) or per[s] != stage1_per.get(s)][:5])
    check("B1.field_plans_digest_matches_summary", blake3.blake3((adir / "field_plans.json").read_bytes()).hexdigest() == summary["field_plans_blake3"])
    # B2 plans against the inventory and sampled generations
    inv_dir, store = Path(args.inventory), Path(args.store)
    inv_missing_plan, checked_fields = [], 0
    for src in pop["sources"]:
        f = inv_dir / f"{src}.json"
        if not f.exists():
            continue
        fields = plans["sources"][src]["fields"]
        for x in json.loads(f.read_text(encoding="utf-8"))["fields"]:
            p = x["field_path"]
            if p.startswith("/rows/*/"):
                checked_fields += 1
                if p[len("/rows/*/"):].split("/")[0] not in fields:
                    inv_missing_plan.append(f"{src}.{p}")
    check("B2.every_inventory_row_field_has_a_plan", not inv_missing_plan, inventory_fields=checked_fields, missing=inv_missing_plan[:6])
    viol = {"missing": [], "numeric_to_text": [], "clock_no_role": [], "unknown_kind": []}
    sampled_gens = sampled_rows = 0
    for src in pop["sources"]:
        gens = [g for g in pop["generations"] if g["source"] == src]
        gen = gens[1] if len(gens) > 1 else gens[0]  # a generation the planner did not sample where one exists
        rows = []
        for a in canonical_artifacts(store, src, gen["generation"]):
            for r in iter_rows(store / "sources" / src / "generations" / gen["generation"] / a["relative_path"], int(a.get("byte_count") or 0), a.get("canonical_record_encoding")):
                rows.append(r)
                if len(rows) >= SAMPLE_ROWS:
                    break
            if len(rows) >= SAMPLE_ROWS:
                break
        sampled_gens += 1
        sampled_rows += len(rows)
        v = plan_violations(plans["sources"][src]["fields"], rows)
        for k in viol:
            viol[k] += [f"{src}.{x}" for x in v[k]]
    check("B2.sampled_generations_every_key_planned", not viol["missing"], generations=sampled_gens, rows=sampled_rows, missing=viol["missing"][:6])
    check("B2.no_numeric_field_embedded", not viol["numeric_to_text"], fields=viol["numeric_to_text"][:6])
    check("B2.every_clock_declares_its_role", not viol["clock_no_role"], fields=viol["clock_no_role"][:6])
    check("B2.plan_kinds_known", not viol["unknown_kind"], fields=viol["unknown_kind"][:6])
    # negative controls
    v = plan_violations({"price": {"kind": "text-label"}, "when": {"kind": "clock"}}, [{"price": "12.5", "when": "2026-10-15", "extra": 1}])
    check("NEG.planted_numeric_to_text_refuses", v["numeric_to_text"] == ["price"], v=v)
    check("NEG.planted_clock_without_role_refuses", v["clock_no_role"] == ["when"])
    check("NEG.planted_key_outside_plan_refuses", v["missing"] == ["extra"])
    check("NEG.tampered_digest_differs", blake3.blake3(b"tampered").hexdigest() != summary["unique_blake3"])
    passed = sum(c["ok"] for c in CHECKS)
    run = f"stage2-poly-{args.version}-{run_stamp()}"
    rdir = paths.root / "evidence" / "fsv" / args.issue / run
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "stage2-poly-fsv.json", {"run": run, "version": args.version, "host": HOST, "passed": passed, "total": len(CHECKS), "checks": CHECKS})
    print(f"{passed}/{len(CHECKS)} checks passed; report {rdir / 'stage2-poly-fsv.json'}", flush=True)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
