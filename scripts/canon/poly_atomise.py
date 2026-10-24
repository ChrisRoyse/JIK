"""Stage 2 for the Poly sealed structured layer (issue #11; workstation only; Poly's tree is read, never written).
Row atoms by reference: every unique row of the Stage 1 population (poly.unique.npy) is one atom, identified by its
population index -> (generation, row) through poly.population.json; no bytes are copied. Field plans per source (spec §18
"each field receives a plan"): the field list comes from Poly's canonical field inventory (paths under /rows/*/) or,
for a source without an inventory, from the first generation's rows (recorded as plan_source=rows); the kind is decided
by a fixed rule table over the inventory value kinds, the field name and the values of the first generation's first
SAMPLE_ROWS rows (a string field whose sampled values all parse as decimals is a scalar; ISO-8601 or epoch-millisecond
fields are clocks with a declared role; identifiers are references; labels and descriptions are text-labels; everything
else explicit is categorical; a null-only field is a typed absence). The rule that fired is recorded per field.
Time-series windows (spec §18, W=8) are not built here: Poly's native weave is the declared contract for windows and is
referenced, not re-derived (stated in the summary).
Output: <out>/atoms/<version>/poly/field_plans.json, poly.atoms.summary.json, one ledger entry `atoms.build`.
Cost: O(sources x fields) + one generation's rows per source; never a pass over the 85 M rows.
    python -I scripts/canon/poly_atomise.py --root D:/MNLMData --out D:/MNLMData --version p0-v1 \
        --store C:/code/poly/var/poly/native-reset-v1/state/production-source-store --inventory C:/code/poly/var/poly/native-reset-v1/state/canonical-field-inventory
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HOST, Paths, digest_file, ledger_append, now_iso, write_json  # noqa: E402

CONTRACT_ID = "poly-atoms-v1"
SAMPLE_ROWS = 500
DECIMAL = re.compile(r"^-?\d+(\.\d+)?$")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
REF_NAME = re.compile(r"(^|_)(id|ids|ticker|sha256|url|uri|alias|series|symbol|code|cik|cusip|isin|slug|hash|key|address)($|_)")
TEXT_NAME = re.compile(r"(^|_)(title|name|description|headline|summary|text|question|notes?|body|label|rules|comment|reason|message)($|_)")
REVISION_NAME = re.compile(r"(^|_)(revision_id|vintage_id|vintage|revision)($|_)")
CLOCK_NAME = re.compile(r"(^|_)(time|at|date|timestamp|period|datetime|ts|day|month|year|quarter|unix_ms|epoch)($|_)|_unix_ms$|_time$|_date$")


class PolyAtomError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def inventory_fields(inv_dir: Path, src: str) -> dict[str, list[str]] | None:
    f = inv_dir / f"{src}.json"
    if not f.exists():
        return None
    out: dict[str, list[str]] = {}
    for x in json.loads(f.read_text(encoding="utf-8"))["fields"]:
        p = x["field_path"]
        if not p.startswith("/rows/*/"):
            continue  # generation metadata (/schema_version, /source_of_truth), not a row field
        name = p[len("/rows/*/"):].split("/")[0]
        out.setdefault(name, [])
        out[name] = sorted(set(out[name]) | set(x["value_kinds"]) | ({"nested"} if "/" in p[len("/rows/*/"):] else set()))
    return out


def first_rows(store: Path, pop: dict, src: str) -> list[dict]:
    gen = next((g for g in pop["generations"] if g["source"] == src), None)
    if gen is None:
        raise PolyAtomError("MNLM_POLY_SOURCE_WITHOUT_GENERATION", src)
    m = json.loads((store / "sources" / src / "generations" / gen["generation"] / "manifest.json").read_text(encoding="utf-8"))
    from poly_dedup import iter_rows  # the Stage 1 reader: one record per declared canonical encoding, nothing guessed

    rows: list[dict] = []
    for a in m.get("artifacts", []):
        if a.get("role") == "canonical":
            for r in iter_rows(store / "sources" / src / "generations" / gen["generation"] / a["relative_path"], int(a.get("byte_count") or 0), a.get("canonical_record_encoding")):
                rows.append(r)
                if len(rows) >= SAMPLE_ROWS:
                    return rows
    return rows


def decide(name: str, kinds: list[str], values: list, receipt_clocks: set[str]) -> dict:
    """Plan for one field: kind, clock role when a clock, the rule that fired and the sampled value kinds."""
    vals = [v for v in values if v is not None]
    sample_kinds = sorted({type(v).__name__ for v in values})
    if "nested" in kinds or any(isinstance(v, (dict, list)) and v for v in vals):
        return {"kind": "record", "rule": "nested object or array kept as an explicit record"}
    if name in receipt_clocks:
        return {"kind": "clock", "clock": "receipt", "rule": "Poly receipt clock"}
    if REVISION_NAME.search(name):
        return {"kind": "clock", "clock": "revision", "rule": "revision or vintage identifier"}
    if not vals and set(kinds) <= {"null", "empty_array", "empty_object"}:
        return {"kind": "absent" if set(kinds) <= {"null"} else "set", "rule": "null-only or empty container in every observation"}
    if all(isinstance(v, bool) for v in vals) and vals:
        return {"kind": "categorical", "rule": "boolean"}
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) and vals
    numeric_str = all(isinstance(v, str) and DECIMAL.match(v) for v in vals) and vals
    if CLOCK_NAME.search(name) and (numeric or numeric_str or (vals and all(isinstance(v, str) and ISO.match(v) for v in vals))):
        return {"kind": "clock", "clock": "coordinate", "rule": "clock-named field with epoch or ISO-8601 values"}
    if numeric or numeric_str:
        return {"kind": "scalar", "rule": "numeric values" if numeric else "decimal strings (exact string kept beside the scalar)"}
    if vals and all(isinstance(v, str) and ISO.match(v) for v in vals):
        return {"kind": "clock", "clock": "coordinate", "rule": "ISO-8601 values"}
    if REF_NAME.search(name):
        return {"kind": "reference", "rule": "identifier-named field"}
    if TEXT_NAME.search(name):
        return {"kind": "text-label", "rule": "label or description field: text-embedding lens on the label beside the exact string"}
    return {"kind": "categorical", "rule": "explicit string, encoded never embedded", "sample_kinds": sample_kinds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--inventory", required=True)
    args = ap.parse_args()
    paths = Paths(args.root, args.out, args.version)
    d = paths.dedup
    pop = json.loads((d / "poly.population.json").read_text(encoding="utf-8"))
    unique = np.load(d / "poly.unique.npy")
    store, inv_dir = Path(args.store), Path(args.inventory)
    receipt = set(pop["receipt_clocks"])
    plans: dict[str, dict] = {}
    kinds_count = collections.Counter()
    beyond: dict[str, int] = {}
    for src in pop["sources"]:
        inv = inventory_fields(inv_dir, src)
        rows = first_rows(store, pop, src)
        names = dict(inv) if inv is not None else {}
        unlisted = sorted({k for r in rows for k in r} - set(names))  # the rows are the truth; an inventory names what its readback observed
        for k in unlisted:
            names[k] = []
        fields = {}
        for name in sorted(names):
            values = [r.get(name) for r in rows if name in r]
            fields[name] = decide(name, names[name], values, receipt)
            if name in unlisted and inv is not None:
                fields[name]["not_in_inventory"] = True
            kinds_count[fields[name]["kind"]] += 1
        beyond[src] = len(unlisted) if inv is not None else 0
        plans[src] = {"plan_source": "inventory" if inv is not None else "rows", "fields": fields, "sampled_rows": len(rows), "fields_beyond_inventory": unlisted if inv is not None else []}
    adir = paths.out / "atoms" / args.version / "poly"
    adir.mkdir(parents=True, exist_ok=True)
    write_json(adir / "field_plans.json", {"contract": CONTRACT_ID, "receipt_clocks": sorted(receipt), "sources": plans})
    # per-source unique counts from the generation table (cumulative row offsets) and the unique index
    offsets = np.cumsum([0] + [g["rows"] for g in pop["generations"]])
    gen_of_unique = np.searchsorted(offsets, unique, side="right") - 1
    src_names = sorted(pop["sources"])
    src_index = {s: i for i, s in enumerate(src_names)}
    src_of_gen = np.array([src_index[g["source"]] for g in pop["generations"]])
    per_source = {s: int(c) for s, c in zip(src_names, np.bincount(src_of_gen[gen_of_unique], minlength=len(src_names)))}
    summary = {"version": args.version, "contract": CONTRACT_ID, "atoms": int(len(unique)), "atom_identity": "population index -> (generation, row) via poly.population.json",
               "unique_blake3": digest_file(d / "poly.unique.npy")[0], "population_rows": pop["rows"], "sources": len(plans), "fields": sum(len(p["fields"]) for p in plans.values()),
               "plan_kinds": dict(kinds_count), "plan_source": dict(collections.Counter(p["plan_source"] for p in plans.values())), "per_source_atoms": per_source,
               "fields_beyond_inventory": {k: v for k, v in beyond.items() if v},
               "windows": "not built: Poly's native weave windows (W=8, declared clocks, no resampling) are the contract and are referenced at Stage 3/4, not re-derived",
               "field_plans_blake3": digest_file(adir / "field_plans.json")[0], "host": HOST, "ts": now_iso()}
    write_json(adir / "poly.atoms.summary.json", summary)
    e = ledger_append(paths, f"poly_atomise@{HOST}", "atoms.build", f"stage2/{args.version}/poly", summary)
    summary["ledger_seq"], summary["ledger_head"] = e["seq"], e["hash"]
    write_json(adir / "poly.atoms.summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_source_atoms"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
