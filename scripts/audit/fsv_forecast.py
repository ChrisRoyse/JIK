"""Cold source/order/selection replay and actual input-isolation boundaries (#12)."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import SenseError, canonical, digest, require, seal_record
from senses.forecast import build, example, load_bundle, select, validate_plan
from senses.pipeline import file_hash, read_bound
from senses.store import Store
from senses.instruments import measure, slot_digest
from universal import ChannelReader


def run(export, bundle, out):
    started, checks = time.perf_counter(), []
    out.mkdir(parents=True, exist_ok=False)

    def check(name, passed, **detail):
        row = {"name": name, "pass": bool(passed), **detail}
        checks.append(row)
        print(json.dumps(row), flush=True)
        (out / "checks.json").write_bytes(canonical(checks))
        require(passed, "MNLM_FORECAST_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False)

    m, data = load_bundle(bundle)
    plan = data["plan.json"]
    inputs, futures, labels = [data[k] for k in ("inputs.jsonl", "future-evidence.jsonl", "labels.jsonl")]
    before = {n: file_hash(bundle / n) for n in ("manifest.json", *m["files"])}
    # Independent replay: exhaustive source index, no select/rank/split helper.
    series, retained, n = {}, {}, 0
    wanted = {r["atom_id"] for r in inputs + futures}
    for r in read_bound(export, m["source_sha256"]):
        loc = r["locator"]
        series.setdefault((loc["source"], loc["shard"]), []).append((loc["row"], r["unit"], r["atom_id"]))
        if r["atom_id"] in wanted:
            retained[r["atom_id"]] = r
        n += 1
    expected = {}
    for key, positions in series.items():
        check_order = [x[0] for x in positions] == list(range(len(positions))) and len({x[1] for x in positions}) == 1
        require(check_order, "MNLM_FORECAST_FSV_SOURCE", "source series not contiguous")
        for i in range(max(0, len(positions) - plan["horizon"])):
            step, unit, atom = positions[i]
            message = ["forecast-start-v1", plan["seed"], unit, key[0], key[1], step]
            score = hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                              separators=(",", ":")).encode()).hexdigest()
            candidate = (score, atom, positions[i + plan["horizon"]][2])
            if unit not in expected or candidate[0] < expected[unit][0]:
                expected[unit] = candidate
    check("complete_source_selection_replay", n == m["source_rows"] and len(expected) == len(inputs)
          and all((r["atom"], r["future_atom"]) == expected[r["unit"]][1:] for r in labels),
          source_rows=n, series=len(series), groups=len(expected))
    check("exact_current_and_future_bytes", all(retained[r["atom_id"]] == r for r in inputs + futures))
    for row in labels:
        a, b = retained[row["atom"]], retained[row["future_atom"]]
        av = {x["name"]: x["value"] for x in a["channels"]["fields"]["items"]}
        bv = {x["name"]: x["value"] for x in b["channels"]["fields"]["items"]}
        delta = [bv[k] - av[k] for k in plan["fields"]]
        require(delta == row["delta"] and [0 if x == 0 else 1 if x > 0 else -1 for x in delta] == row["sign"],
                "MNLM_FORECAST_FSV_LABEL", "cold label differs from retained measurements")
        h = hashlib.sha256(json.dumps(["forecast-split-v1", plan["seed"], row["unit"]],
                                     separators=(",", ":")).encode()).hexdigest()
        bucket = int(h, 16) % 10
        require(row["split"] == ("train" if bucket <= 5 else "validation" if bucket <= 7 else "test"),
                "MNLM_FORECAST_FSV_SPLIT", "group split drift")
    check("labels_and_group_splits_rederived", True, splits=m["splits"], iid_claim=False)
    split_units = {s: {r["unit"] for r in labels if r["split"] == s} for s in m["splits"]}
    check("no_group_crosses_splits", sum(len(s) for s in split_units.values()) == len(set.union(*split_units.values())))

    # Small REAL source series for ordering controls; these are faults, not training data.
    first_key = next(iter(series))
    controls_source = []
    for r in read_bound(export, m["source_sha256"]):
        if (r["locator"]["source"], r["locator"]["shard"]) == first_key:
            controls_source.append(r)
    small = {**plan, "rows": len(controls_source)}
    refused("nonpositive_horizon", lambda: validate_plan({**plan, "horizon": 0}), "MNLM_FORECAST_PLAN")
    refused("undeclared_behavior_input", lambda: validate_plan({**plan, "allow_channels": plan["allow_channels"] + ["behaviour"]}),
            "MNLM_FORECAST_PLAN")
    refused("gap_in_sequence", lambda: select(iter(controls_source[1:]), {**small, "rows": small["rows"] - 1}), "MNLM_FORECAST_ORDER")
    changed = copy.deepcopy(controls_source)
    changed[2]["unit"] = "wrong-group"
    seal_record(changed[2])
    refused("group_changes_inside_series", lambda: select(iter(changed), small), "MNLM_FORECAST_ORDER")
    changed = copy.deepcopy(controls_source)
    changed[2]["channels"]["fields"]["items"].pop()
    seal_record(changed[2])
    refused("missing_target_is_not_silently_selected_away", lambda: select(iter(changed), small), "MNLM_FORECAST_TARGET")
    refused("target_is_not_future", lambda: example(futures[0], inputs[0], plan), "MNLM_FORECAST_TARGET_ORDER")
    unit_drift = copy.deepcopy(futures[0])
    unit_drift["channels"]["fields"]["items"][0]["unit"] = "different-unit"
    seal_record(unit_drift)
    refused("target_units_change", lambda: example(inputs[0], unit_drift, plan), "MNLM_FORECAST_TARGET_UNITS")
    # Mutate values only (including outcomes): selection identity and group split must not move.
    mutated = copy.deepcopy(controls_source)
    for r in mutated:
        for f in r["channels"]["fields"]["items"]:
            f["value"] = -123.0
        seal_record(r)
    a, _ = select(iter(controls_source), small)
    b, _ = select(iter(mutated), small)
    check("selection_does_not_read_outcomes", {k: (v[0], v[1]["locator"]["row"]) for k, v in a.items()}
          == {k: (v[0], v[1]["locator"]["row"]) for k, v in b.items()})

    def changed_bundle(name, filename, change, code):
        target = out / name
        target.mkdir()
        for f in ("manifest.json", *m["files"]):
            shutil.copyfile(bundle / f, target / f)
        obj = copy.deepcopy(data[filename])
        change(obj)
        raw = b"".join(canonical(r) + b"\n" for r in obj) if filename.endswith(".jsonl") else canonical(obj)
        (target / filename).write_bytes(raw)
        mm = copy.deepcopy(m)
        mm["files"][filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        (target / "manifest.json").write_bytes(canonical(mm))
        refused(name, lambda: load_bundle(target), code)

    changed_bundle("resealed_wrong_label", "labels.jsonl", lambda r: r[0]["sign"].__setitem__(0, 7), "MNLM_FORECAST_LABEL")
    changed_bundle("resealed_split_tamper", "labels.jsonl", lambda r: r[0].__setitem__("split", "training-cheat"), "MNLM_FORECAST_LABEL")
    changed_bundle("resealed_context_injection", "policies.json",
                   lambda p: p["recorded-future-state"]["allow_channels"].append("history"), "MNLM_FORECAST_POLICY")
    # Actual consumer: ONLY current records are ingested. Labels/futures never enter this store.
    store_path = out / "predictors.sqlite"
    store = Store(store_path, create=True, records_only=True)
    try:
        store.ingest(iter(inputs), {"input_sha256": before["inputs.jsonl"], "count": len(inputs)}, None)
        generation = store.head
        check("predictor_store_cold_readback", all(store.get(r["atom_id"], generation) == r for r in inputs))
        refused("future_record_cannot_be_opened", lambda: store.get(futures[0]["atom_id"], generation), "MNLM_UCR_NOT_FOUND")
    finally:
        store.close()
    reader = ChannelReader(store_path, bundle / "policies.json", generation)
    signatures, mask_count, first_provenance = {}, 0, None
    try:
        policy = reader.policy("recorded-future-state")
        for r in inputs:
            loc = r["locator"]
            row = {**loc, "atom": r["atom_id"], "text": "\n".join(x["text"] for x in r["channels"]["surface"]["items"])}
            visible, contexts, binding = reader.read(row, policy)
            require(not any(contexts.values()) and not visible["channels"]["behaviour"]["items"],
                    "MNLM_FORECAST_FSV_INPUT", "future/sibling or uncommissioned action input")
            require(all(not binding["context_atoms"][c] for c in ("history", "co_reference")),
                    "MNLM_FORECAST_FSV_INPUT", "unexpected dependency")
            for channel in ("surface", "fields", "structure", "provenance"):
                slot = measure(visible, channel)
                if channel == "provenance":
                    if first_provenance is None:
                        first_provenance = slot["tokens"].copy()
                    require(np.array_equal(first_provenance, slot["tokens"]),
                            "MNLM_FORECAST_FSV_PROVENANCE", "constant finding must agree in exact native arrays")
                signatures.setdefault(channel, []).append(slot_digest(slot))
            mask_count += 1
        check("actual_views_exclude_context_and_commands", mask_count == len(inputs), rows=mask_count)
        check("native_constant_provenance_reported", len(set(signatures["provenance"])) == 1,
              native_states={c: len(set(v)) for c, v in signatures.items()})
    finally:
        reader.close()
    check("bundle_unchanged", before == {n: file_hash(bundle / n) for n in before})
    report = {"acceptance": "author-checked only", "checks": checks, "passed": len(checks), "failed": 0,
              "bundle_manifest_sha256": before["manifest.json"], "stage3_admitted": False, "stage4_accepted": False,
              "independent_acceptance": False, "wall_seconds": time.perf_counter() - started}
    (out / "report.json").write_bytes(canonical(report))
    (out / "SHA256.json").write_bytes(canonical({str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
        for p in out.rglob("*") if p.is_file() and p.name != "SHA256.json"}))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("export", "bundle", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.export, a.bundle, a.out)), flush=True)
