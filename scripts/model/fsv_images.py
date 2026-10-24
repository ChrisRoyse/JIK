"""Real camera bytes -> native publication/cache -> six-encoder consumer.

Four retained observations are a wiring proof, not an efficacy or admission test.
CPU is the declared execution device for this bounded consumer experiment.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
import torch

from senses.contract import CHANNELS, SenseError, canonical, digest, require
from senses.pipeline import file_hash, read_bound
from senses.store import Store
from senses.pixels import contract as pixel_contract
from universal import specs as universal_specs, ChannelReader
from image_ucr import SPEC, measure_image_ucr, visible_image
from model.contracts import ModelConfig
from model.network import SensoryLanguageModel
from model.publication_input import read_publication
from model.checkpoint import save, load, state_hash
from production.input_plan import code_hashes, environment
from production.stream import run
from production.publication import Publication
from training_scope import POLICY_SHA256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vision", action="store_true", help="also exercise frozen DINOv2; still one perception channel")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started, checks = time.monotonic(), []
    def check(name, passed, **data):
        checks.append({"name": name, "pass": bool(passed), **data})
        print(json.dumps(checks[-1]), flush=True)
        (out / "checks.json").write_bytes(canonical(checks))
        require(passed, "MNLM_IMAGE_FSV", name)
    def refuses(name, call, code):
        try:
            call()
        except SenseError as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False)

    manifest = json.loads((args.export / "observations.ucr.manifest.json").read_bytes())
    selected, n = [], 0
    for r in read_bound(args.export / "observations.ucr.jsonl", manifest["sha256"]):
        if len(selected) < 4:
            selected.append(r)
        n += 1
    check("complete_export_digest_and_count", n == manifest["count"] == 23702, full_count=n, proof_count=len(selected))
    (out / "selected.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in selected))
    store_path = out / "source.sqlite"
    store = Store(store_path, create=True, records_only=True)
    try:
        store.ingest(iter(selected), {"input_sha256": file_hash(out / "selected.jsonl"), "count": len(selected)}, None)
        generation = store.head
        check("source_store_cold_readback", all(store.get(r["atom_id"], generation) == r for r in selected))
    finally:
        store.close()
    policy = {"schema": "ucr-view-v1", "question": "image-wiring", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "bounded wiring experiment; no labels, no scientific admission"}
    policies = out / "policies.json"
    policies.write_bytes(canonical({"image-wiring": policy}))
    rows = [{"i": i, "atom": r["atom_id"], "source": r["locator"]["source"], "shard": r["locator"]["shard"],
             "row": r["locator"]["row"], "span": r["locator"]["span"], "text": "\n".join(x["text"] for x in r["channels"]["surface"]["items"])}
            for i, r in enumerate(selected)]
    rr = out / "rows"
    rr.mkdir()
    (rr / "images.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
    blobs = args.export / "blobs.json"
    image = {"spec": SPEC, "weights_sha256": pixel_contract()["weights_sha256"], "tier": "deterministic", "slot_budget_bytes": 4 << 20}
    instruments = {s["name"]: {"spec": s, "weights_sha256": s["weights_sha256"], "tier": "deterministic", "slot_budget_bytes": 4 << 20}
                   for s in universal_specs() if s["channel"] in ("surface", "fields", "behaviour", "structure", "provenance")}
    instruments[SPEC["name"]] = image
    if args.vision:
        from vision_assets import FILES
        from vision_encoder import SPEC as VISION
        weight_sha = FILES["model.safetensors"][1]
        instruments[VISION["name"]] = {"spec": VISION, "weights_sha256": weight_sha, "revision": VISION["revision"],
                                       "tier": "deterministic", "slot_budget_bytes": 4 << 20}
    plan = {"schema": "tier-input-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "generation": generation, "policies_sha256": file_hash(policies),
            "blobs_manifest_sha256": file_hash(blobs), "code_sha256": code_hashes(), "environment": environment(),
            "batch_rows": 2, "slot_budget_bytes": 4 << 20, "metadata_budget_bytes": 1 << 20,
            "blob_budget_bytes": 64 << 20, "work_budget_bytes": 192 << 20,
            "cache": {"schema": "atom-cache-v1", "max_db_bytes": 8 << 20}, "instruments": instruments,
            "shards": {"images": {"n": len(rows), "rows_sha256": file_hash(rr / "images.jsonl"),
                                  "rows_bytes": (rr / "images.jsonl").stat().st_size, "family": "image-wiring",
                                  "tier": "deterministic", "instruments": sorted(instruments)}}}
    (out / "plan.json").write_bytes(canonical(plan))
    reader = ChannelReader(store_path, policies, generation)
    try:
        refuses("image_label_mask_cannot_reopen_pixels", lambda: visible_image(reader, {**policy, "label_aliases": ["robot"]}, rows[0]), "MNLM_IMAGE_MASK_UNSUPPORTED")
        refuses("image_field_mask_cannot_reopen_pixels", lambda: visible_image(reader, {**policy, "deny_fields": ["width"]}, rows[0]), "MNLM_IMAGE_MASK_UNSUPPORTED")
        excluded, _ = visible_image(reader, {**policy, "allow_channels": ["surface"]}, rows[0])
        check("explicit_image_exclusion", not excluded["channels"]["perception"]["items"])
    finally:
        reader.close()
    (out / "cal").mkdir()
    (out / "cal/image-wiring.jsonl").write_bytes((rr / "images.jsonl").read_bytes())
    direct = measure_image_ucr(SPEC, {"image-wiring": rows}, out / "direct", {}, store_path, policies, blobs, file_hash(blobs))
    check("assay_native_image_path", direct["probe"]["present"] == 4 and "error" not in direct)
    if args.vision:
        from model.vision_checks import boundaries
        boundaries(selected[0], blobs, out, check, refuses)
        from vision_ucr import measure_vision_ucr
        from lenscommon import take_lease, release_lease
        from measure import nvidia_smi
        (out / "assay-resident-before.json").write_bytes(canonical(nvidia_smi()))
        take_lease("15", "four-image learned visual assay")
        try:
            visual = measure_vision_ucr(VISION, {"image-wiring": rows}, out / "direct-vision", {},
                                        store_path, policies, blobs, file_hash(blobs))
            check("assay_native_vision_path", visual["probe"]["present"] == 4 and "error" not in visual)
        finally:
            release_lease()
    first = run(plan, store_path, policies, rr, out / "publication", blobs_path=blobs, cache_root=out / "cache")
    (out / "stream-first.json").write_bytes(canonical(first))
    count = len(instruments)
    check("native_instruments_published", first["checkpoints_computed"] == count
          and (first["gpu_peak_allocated_bytes"] > 0 if args.vision else first["gpu_peak_allocated_bytes"] == 0))
    cached = run(plan, store_path, policies, rr, out / "cached-publication", blobs_path=blobs, cache_root=out / "cache")
    (out / "stream-cached.json").write_bytes(canonical(cached))
    check("all_native_cache_hits", cached["cache"]["atom_hits"] == 4 * count and not cached["loaded_instruments"])
    config = ModelConfig()
    specs, native, model_rows, receipt = read_publication(out / "publication/publication", [["images"]], config)
    torch.set_num_threads(1)
    torch.manual_seed(1375)
    torch.use_deterministic_algorithms(True)
    model = SensoryLanguageModel(config, specs).eval()
    prefix = torch.full((4, 4), 256, dtype=torch.long)
    with torch.inference_mode():
        result = model(prefix, native)
        memories, counts = model.memories(native)
        check("distinct_encoders_count_channels_once", len(memories) == count and result["instrument_count"].tolist() == [count] * 4
              and result["channel_count"].tolist() == [6] * 4, dimensions={s.name: s.dims for s in specs})
        x = native[SPEC["name"]]
        changed = x.values.clone()
        changed[:, :, :192] = 0  # negative control only, never published as source
        different = model(prefix, {**native, SPEC["name"]: replace(x, values=changed)})["logits"]
        check("actual_pixel_values_affect_consumer", not torch.equal(result["logits"], different))
        if args.vision:
            v = native[VISION["name"]]
            check("learned_vision_keeps_native_tokens", tuple(v.values.shape) == (4, 362, 384))
            different = model(prefix, {**native, VISION["name"]: replace(v, values=torch.zeros_like(v.values))})["logits"]
            check("learned_visual_values_affect_consumer", not torch.equal(result["logits"], different))
        check("deterministic_native_replay", torch.equal(result["logits"], model(prefix, native)["logits"]))
    refuses("unadmitted_senses_cannot_assert_meaning", lambda: model.assert_meaning(native), "MNLM_MODEL_UNCOMMISSIONED")
    save(model, out / "checkpoint", step=0, input_sha256=digest(receipt))
    cold, _ = load(out / "checkpoint", "cpu")
    with torch.inference_mode():
        check("cold_model_byte_replay", state_hash(model) == state_hash(cold) and torch.equal(result["logits"], cold(prefix, native)["logits"]))
    # The source image pixels, UCR bytes and declarations remain immutable.
    check("source_export_unchanged", file_hash(args.export / "observations.ucr.jsonl") == manifest["sha256"])
    report = {"acceptance": "author-checked only", "checks": checks, "passed": len(checks), "failed": 0,
              "rows": 4, "instruments": count, "channels": 6, "model_parameters": sum(p.numel() for p in model.parameters()),
              "semantic_understanding_demonstrated": False, "stage3_admitted": False, "stage4_accepted": False,
              "wall_seconds": time.monotonic() - started}
    (out / "report.json").write_bytes(canonical(report))
    # Evidence SQLite files are closed and checkpointed before sealing.
    import sqlite3
    for path in (store_path, out / "cache/index.sqlite", out / "publication/publication/index.sqlite",
                 out / "cached-publication/publication/index.sqlite"):
        if path.exists():
            db = sqlite3.connect(path)
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("PRAGMA journal_mode=DELETE")
            db.close()
    (out / "SHA256.json").write_bytes(canonical({str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
                                                 for p in out.rglob("*") if p.is_file() and p.name != "SHA256.json"}))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
