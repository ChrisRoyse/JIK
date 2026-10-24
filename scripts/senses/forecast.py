"""Frozen, group-held-out later-observation tasks over ordered UCR sources.

O(export bytes + rows), one episode plus one selected pair per group in memory.
These are diagnostic recorded-index outcomes, not causal/physical certification.
No source-name branch, fitted threshold, label balancing, or encoder label access.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import CHANNELS, SenseError, canonical, digest, require, validate_record
from senses.pipeline import file_hash, read_bound
from senses.views import validate_policy
from training_scope import POLICY_SHA256
from senses.scope import check_source

SPLITS = ("train", "validation", "test")
PLAN_KEYS = {"schema", "purpose", "source", "export_sha256", "rows", "horizon", "fields", "ordinal_field",
             "ordinal_unit", "seed", "split_tenths", "selection", "series", "label", "allow_channels",
             "assumptions", "admission"}


def validate_plan(p):
    require(isinstance(p, dict) and set(p) == PLAN_KEYS and p["schema"] == "ucr-forecast-plan-v1"
            and p["purpose"] == "recorded_temporal_diagnostic" and p["admission"] is False,
            "MNLM_FORECAST_PLAN", "explicit diagnostic plan required")
    require(type(p["horizon"]) is int and 1 <= p["horizon"] <= 1024 and type(p["rows"]) is int and p["rows"] > 0
            and type(p["seed"]) is int and 0 <= p["seed"] < 2**32,
            "MNLM_FORECAST_PLAN", "bounded positive horizon/count and integer seed required")
    require(isinstance(p["fields"], list) and 1 <= len(p["fields"]) <= 32
            and all(isinstance(n, str) and n for n in p["fields"]) and len(set(p["fields"])) == len(p["fields"]),
            "MNLM_FORECAST_PLAN", "unique explicit numeric target fields required")
    require(p["split_tenths"] == [6, 2, 2]
            and p["selection"] == "one_pair_per_unit; minimum_identity_hash; no_outcome_filter"
            and p["series"] == "locator_source_and_shard; contiguous_zero_based_rows"
            and p["label"] == "future_minus_current; exact_sign; no_physical_units_claim"
            and p["allow_channels"] == ["surface", "fields", "structure", "provenance", "perception"]
            and isinstance(p["ordinal_field"], str) and bool(p["ordinal_field"])
            and isinstance(p["ordinal_unit"], str) and bool(p["ordinal_unit"]) and bool(p["assumptions"]),
            "MNLM_FORECAST_PLAN", "unsupported selection, label, split or visibility contract")
    check_source(p["source"])
    require(len(p["export_sha256"]) == 64 and all(c in "0123456789abcdef" for c in p["export_sha256"]),
            "MNLM_FORECAST_PLAN", "pin the complete source export")
    return p


def split_for(unit, seed):
    bucket = int(digest(["forecast-split-v1", seed, unit]), 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def rank(record, seed):
    loc = record["locator"]
    # No value, outcome, atom/payload hash, language or availability in selection.
    return digest(["forecast-start-v1", seed, record["unit"], loc["source"], loc["shard"], loc["row"]])


def values(record, names):
    fields = record["channels"]["fields"]["items"]
    found = {n: [f for f in fields if f["name"] == n] for n in names}
    require(all(len(v) == 1 and v[0]["type"] in ("number", "integer") for v in found.values()),
            "MNLM_FORECAST_TARGET", "target missing, duplicated or nonnumeric; no silent row selection")
    return [found[n][0]["value"] for n in names]


def select(records, plan):
    """Exhaust input (including trailing digest check) before returning candidates."""
    chosen, inventory, seen_series, seen_atoms = {}, [], set(), set()
    episode, series, count = [], None, 0

    def finish():
        if not episode:
            return
        candidates = len(episode) - plan["horizon"]
        inventory.append({"source": series[0], "shard": series[1], "unit": episode[0]["unit"],
                          "rows": len(episode), "eligible_starts": max(0, candidates),
                          "status": "eligible" if candidates > 0 else "shorter_than_horizon"})
        for i in range(max(0, candidates)):
            current, future = episode[i], episode[i + plan["horizon"]]
            key, score = current["unit"], rank(current, plan["seed"])
            if key not in chosen or score < chosen[key][0]:
                chosen[key] = (score, current, future)

    for r in records:
        validate_record(r)
        loc = r["locator"]
        require(loc["source"] == plan["source"], "MNLM_FORECAST_SOURCE", "source outside pinned scope")
        require(r["atom_id"] not in seen_atoms, "MNLM_FORECAST_DUPLICATE", "duplicate source observation")
        seen_atoms.add(r["atom_id"])
        key = (loc["source"], loc["shard"])
        if key != series:
            finish()
            require(key not in seen_series, "MNLM_FORECAST_ORDER", "series reappeared after closure")
            seen_series.add(key)
            series, episode = key, []
        clocks = [f for f in r["channels"]["structure"]["items"] if f["name"] == plan["ordinal_field"]]
        require(loc["row"] == len(episode) and loc["span"] == [len(episode), len(episode) + 1]
                and len(clocks) == 1 and clocks[0]["type"] == "integer"
                and clocks[0]["unit"] == plan["ordinal_unit"] and clocks[0]["value"] == len(episode)
                and (not episode or episode[0]["unit"] == r["unit"]),
                "MNLM_FORECAST_ORDER", "contiguous ordinal series and stable group required")
        # Validate even unselected rows: missing values cannot silently alter scope.
        values(r, plan["fields"])
        episode.append(r)
        count += 1
    finish()
    require(count == plan["rows"], "MNLM_FORECAST_COUNT", "complete source row count changed")
    return chosen, inventory


def example(current, future, plan):
    a, b = current["locator"], future["locator"]
    require(current["unit"] == future["unit"] and (a["source"], a["shard"]) == (b["source"], b["shard"])
            and b["row"] - a["row"] == plan["horizon"],
            "MNLM_FORECAST_TARGET_ORDER", "future must belong to same series at declared later index")
    before, after = values(current, plan["fields"]), values(future, plan["fields"])
    def signature(r):
        by_name = {f["name"]: f for f in r["channels"]["fields"]["items"]}
        return [[by_name[n][k] for k in ("type", "unit", "clock_role")] for n in plan["fields"]]
    require(signature(current) == signature(future), "MNLM_FORECAST_TARGET_UNITS", "target types/units changed across time")
    delta = [y - x for x, y in zip(before, after)]
    require(all(math.isfinite(x) for x in delta), "MNLM_FORECAST_TARGET", "nonfinite displacement")
    return {"atom": current["atom_id"], "future_atom": future["atom_id"], "unit": current["unit"],
            "split": split_for(current["unit"], plan["seed"]), "current_locator": a, "future_locator": b,
            "before": before, "after": after, "delta": delta,
            "sign": [(x > 0) - (x < 0) for x in delta]}


def policy_for(units):
    return {"schema": "ucr-view-v1", "question": "recorded-future-state", "allow_channels":
            ["surface", "fields", "structure", "provenance", "perception"], "deny_fields": [],
            "deny_relations": [], "blocked_units": sorted(units), "label_aliases": [],
            "rationale": "Current observation only; all context disabled. Future/label records live outside the predictor store. "
                         "Behavior withheld because command/state alignment is uncommissioned. Diagnostic, not admission."}


def put(path, value):
    with path.open("xb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())


def build(export, recipe, out):
    started = time.perf_counter()
    plan_bytes = recipe.read_bytes()
    plan = validate_plan(json.loads(plan_bytes))
    manifest_path = export.with_suffix(".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest["sha256"] == plan["export_sha256"] and manifest["count"] == plan["rows"],
            "MNLM_FORECAST_SOURCE", "recipe and source manifest disagree")
    out.mkdir(parents=True, exist_ok=False)
    selected, inventory = select(read_bound(export, plan["export_sha256"]), plan)
    selection_done = time.perf_counter()
    rows, inputs, futures = [], [], []
    for unit in sorted(selected):
        _, current, future = selected[unit]
        rows.append(example(current, future, plan))
        inputs.append(current)
        futures.append(future)
    require(rows and {x["split"] for x in rows} == set(SPLITS), "MNLM_FORECAST_SPLIT", "empty split or population")
    require(not {r["atom_id"] for r in inputs} & {r["atom_id"] for r in futures},
            "MNLM_FORECAST_LEAK", "future record entered current input population")
    policy = policy_for(selected)
    validate_policy(policy, policy["question"])
    payloads = {"plan.json": canonical(plan), "inventory.json": canonical(inventory),
                "policies.json": canonical({policy["question"]: policy}),
                "inputs.jsonl": b"".join(canonical(r) + b"\n" for r in inputs),
                "future-evidence.jsonl": b"".join(canonical(r) + b"\n" for r in futures),
                "labels.jsonl": b"".join(canonical(r) + b"\n" for r in rows)}
    files = {}
    for name, raw in payloads.items():
        put(out / name, raw)
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    report = {"schema": "ucr-forecast-bundle-v1", "acceptance": "author-checked only", "source_sha256": plan["export_sha256"],
              "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "recipe_sha256": hashlib.sha256(plan_bytes).hexdigest(), "training_scope_sha256": POLICY_SHA256,
              "files": files, "rows": len(rows), "source_rows": plan["rows"], "series": len(inventory),
              "groups": len(selected), "short_series": sum(x["eligible_starts"] == 0 for x in inventory),
              "splits": dict(Counter(x["split"] for x in rows)), "anchor_families": ["recorded_temporal_realisation"],
              "labels_per_row": len(plan["fields"]), "iid_established": False, "stage3_admitted": False,
              "stage4_accepted": False, "retention_owner": "ChrisRoyse/MNLM#12",
              "cost_seconds": {"source_select": selection_done - started, "publish": time.perf_counter() - selection_done}}
    report["wall_seconds"] = time.perf_counter() - started
    report["code_sha256"] = {n: file_hash(Path(__file__).parent / n) for n in ("forecast.py", "contract.py", "views.py")}
    put(out / "manifest.json", canonical(report))  # acceptance marker last, after full digest exhaustion
    return report


def load_bundle(root):
    """Consumption requires sealed bytes and exact predictor/label alignment."""
    m = json.loads((root / "manifest.json").read_bytes())
    require(m.get("schema") == "ucr-forecast-bundle-v1" and m.get("stage3_admitted") is False
            and m.get("stage4_accepted") is False and m.get("iid_established") is False
            and m.get("training_scope_sha256") == POLICY_SHA256, "MNLM_FORECAST_MANIFEST", "unsupported/stale bundle")
    expected = {"plan.json", "inventory.json", "policies.json", "inputs.jsonl", "future-evidence.jsonl", "labels.jsonl"}
    require(set(m["files"]) == expected, "MNLM_FORECAST_MANIFEST", "exact artifact inventory required")
    data = {}
    for name, entry in m["files"].items():
        raw = (root / name).read_bytes()
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                "MNLM_FORECAST_DIGEST", name)
        data[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
    plan = validate_plan(data["plan.json"])
    require(m["source_sha256"] == plan["export_sha256"] and m["source_rows"] == plan["rows"],
            "MNLM_FORECAST_SOURCE", "bundle changed the frozen source scope")
    inputs, futures, labels = (data[n] for n in ("inputs.jsonl", "future-evidence.jsonl", "labels.jsonl"))
    require(len(inputs) == len(futures) == len(labels) == m["rows"] == m["groups"],
            "MNLM_FORECAST_COUNT", "bundle row alignment changed")
    units = [r["unit"] for r in inputs]
    require(units == sorted(set(units)), "MNLM_FORECAST_SPLIT", "one ordered example per group required")
    for a, b, label in zip(inputs, futures, labels):
        validate_record(a)
        validate_record(b)
        require(a["locator"]["source"] == b["locator"]["source"] == plan["source"],
                "MNLM_FORECAST_SOURCE", "observations outside declared source")
        require(canonical(example(a, b, plan)) == canonical(label), "MNLM_FORECAST_LABEL", "label/order/split drift")
    require(dict(Counter(x["split"] for x in labels)) == m["splits"]
            and set(m["splits"]) == set(SPLITS), "MNLM_FORECAST_SPLIT", "split count drift")
    require(data["policies.json"] == {"recorded-future-state": policy_for(units)},
            "MNLM_FORECAST_POLICY", "input policy changed; future/context exclusion required")
    require(not {r["atom_id"] for r in inputs} & {r["atom_id"] for r in futures},
            "MNLM_FORECAST_LEAK", "future predictor contamination")
    return m, data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("export", "recipe", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    args = ap.parse_args()
    try:
        r = build(args.export, args.recipe, args.out)
        print(json.dumps({k: r[k] for k in ("rows", "groups", "series", "splits", "short_series", "wall_seconds")}))
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        raise SystemExit(3)
