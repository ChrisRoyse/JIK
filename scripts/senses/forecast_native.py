"""Forecast predictors through existing native publication, in fixed eight-row shards.

No evaluator labels/futures enter the store or row files. Instrument identities
are reusable; the forecasting task supplies only visibility and a held-out scope.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical, require
from senses.forecast import load_bundle, put, SPLITS
from senses.pipeline import file_hash
from senses.store import Store
from senses.pixels import contract as pixel_contract
from universal import specs as universal_specs
from image_ucr import SPEC
from production.input_plan import code_hashes, environment, validate
from production.stream import run
from training_scope import POLICY_SHA256


def publish(bundle, blobs, out, *, vision=False):
    start = time.perf_counter()
    manifest, data = load_bundle(bundle)
    require(manifest["rows"] == 416, "MNLM_FORECAST_NATIVE_SCOPE", "commission a new cost plan for a changed population")
    out.mkdir(parents=True, exist_ok=False)
    rows_root = out / "rows"
    rows_root.mkdir()
    records = data["inputs.jsonl"]
    store_path = out / "predictors.sqlite"
    db = Store(store_path, create=True, records_only=True)
    try:
        db.ingest(iter(records), {"input_sha256": manifest["files"]["inputs.jsonl"]["sha256"], "count": len(records)}, None)
        generation = db.head
        require(all(db.get(r["atom_id"], generation) == r for r in records),
                "MNLM_FORECAST_STORE", "current predictor readback mismatch")
    finally:
        db.close()
    instruments = {s["name"]: {"spec": s, "weights_sha256": s["weights_sha256"], "tier": "deterministic",
                               "slot_budget_bytes": 256 << 10}
                   for s in universal_specs() if s["channel"] in ("surface", "fields", "structure", "provenance")}
    image_spec = SPEC
    weight_sha = pixel_contract()["weights_sha256"]
    if vision:
        from vision_encoder import SPEC as image_spec
        from vision_assets import FILES
        weight_sha = FILES["model.safetensors"][1]
    instruments[image_spec["name"]] = {"spec": image_spec, "weights_sha256": weight_sha,
          "tier": "multi" if vision else "deterministic", "slot_budget_bytes": 8 << 20,
          **({"revision": image_spec["revision"]} if vision else {})}
    shards, groups = {}, {}
    for split in SPLITS:
        selected = [r for r, label in zip(records, data["labels.jsonl"]) if label["split"] == split]
        groups[split] = []
        for offset in range(0, len(selected), 8):
            key = f"{split}-{offset // 8:03d}"
            rows = [{"i": i, "atom": r["atom_id"], "source": r["locator"]["source"],
                     "shard": r["locator"]["shard"], "row": r["locator"]["row"], "span": r["locator"]["span"],
                     "text": "\n".join(x["text"] for x in r["channels"]["surface"]["items"])}
                    for i, r in enumerate(selected[offset:offset + 8])]
            path = rows_root / (key + ".jsonl")
            put(path, b"".join(canonical(r) + b"\n" for r in rows))
            shards[key] = {"n": len(rows), "rows_sha256": file_hash(path), "rows_bytes": path.stat().st_size,
                           "family": "recorded-future-state", "tier": "deterministic",
                           "instruments": sorted(n for n, i in instruments.items() if i["tier"] == "deterministic")}
            group = [key]
            if vision:
                visual_key = key + "-vision"
                put(rows_root / (visual_key + ".jsonl"), path.read_bytes())
                shards[visual_key] = {**shards[key], "tier": "multi", "instruments": [image_spec["name"]]}
                group.append(visual_key)
            groups[split].append(group)
    policies = bundle / "policies.json"
    plan = {"schema": "tier-input-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "generation": generation, "policies_sha256": file_hash(policies),
            "blobs_manifest_sha256": file_hash(blobs), "code_sha256": code_hashes(), "environment": environment(),
            "batch_rows": 8, "slot_budget_bytes": 8 << 20, "metadata_budget_bytes": 64 << 10,
            "blob_budget_bytes": 512 << 20, "work_budget_bytes": 1152 << 20, "instruments": instruments, "shards": shards}
    validate(plan)
    put(out / "plan.json", canonical(plan))
    prepare_done = time.perf_counter()
    result = run(plan, store_path, policies, rows_root, out / "native", blobs_path=blobs)
    put(out / "stream.json", canonical(result))
    consumer = {"schema": "forecast-native-v1", "purpose": "recorded_temporal_diagnostic",
                "bundle_manifest_sha256": file_hash(bundle / "manifest.json"), "groups": groups,
                "plan_sha256": file_hash(out / "plan.json"), "instruments": sorted(instruments),
                "source_generation": generation, "rows": len(records), "meaning_asserted": False,
                "visual_features_only": vision, "new_outcome_evaluation": False,
                "stage3_admitted": False, "stage4_accepted": False,
                "cost_seconds": {"prepare": prepare_done - start, "native": time.perf_counter() - prepare_done}}
    consumer["wall_seconds"] = time.perf_counter() - start
    put(out / "consumer.json", canonical(consumer))
    print(json.dumps({"rows": len(records), "shards": len(shards), "instruments": len(instruments),
                      "wall_seconds": consumer["wall_seconds"], "stage4_accepted": False}), flush=True)
    return consumer


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("bundle", "blobs", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    ap.add_argument("--vision", action="store_true", help="replace raw RGB with frozen DINOv2; no new outcome evaluation")
    a = ap.parse_args()
    publish(a.bundle, a.blobs, a.out, vision=a.vision)
