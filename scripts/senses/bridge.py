"""Versioned Bridge RLDS -> provisional UCR, with exact primary JPEG blobs.

Cost O(selected bytes + primary pixels + output); one episode resident. Raw CRC
and file digest checks finish before the export manifest is published. Fields
kept in raw storage are enumerated, never silently repurposed as predictors.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.bridge_inspect import META, STEP, SCHEMA, episode
from tfrecord import decode, frames, RecordError
from senses.contract import CHANNELS, VERSION, SenseError, absent, canonical, digest, field, present, require, seal_record
from senses.pipeline import file_hash, write_records

POLICY_PATH = Path(__file__).resolve().parents[2] / "config/bridge-adapter.json"
SELECTION = "6a6c61cdb101f95e2215c9d733df44fa1c34cf429394fb586712dad145afe43d"
ACQUISITION = "8dc7a87239a4eac6453ad4e37990a977ee45b9a2dcf77beaf4b9e99d66d6d2fc"
ADAPTER = "bridge-observations-v2"


def policy():
    raw = POLICY_PATH.read_bytes()
    p = json.loads(raw)
    require(p["schema"] == "bridge-observation-policy-v1" and p["selection_sha256"] == SELECTION
            and p["camera"] == "image_0" and p["admitted"] is False and p["image_shape"] == [256, 256, 3]
            and p["commands"] == "raw_recorded; never_relabel_from_next_state"
            and p["availability"] == "retained_nonconstant_rgb_jpeg; preserve_provider_flag_conflict"
            and p["physical_units"] == "uncommissioned"
            and p["timing"] == "record_order_only; action_state_alignment_uncommissioned"
            and p["split_group"] == "provider_collection_file; independence_unproven"
            and p["withheld"] == ["reward", "discount", "is_first", "is_last", "is_terminal", "language_embedding", "image_1", "image_2", "image_3"],
            "MNLM_BRIDGE_POLICY", "unsupported observation policy")
    return p, hashlib.sha256(raw).hexdigest()


def primary_image(raw):
    """No reconstruction, resizing or invented camera coordinates."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            require(im.format == "JPEG" and im.mode == "RGB" and im.size == (256, 256),
                    "MNLM_BRIDGE_IMAGE", "expected source RGB 256x256 JPEG")
            im.load()
            require(not all(a == b for a, b in im.getextrema()), "MNLM_BRIDGE_PLACEHOLDER",
                    "spatially constant primary frame; retain as absent")
    except (OSError, ValueError) as exc:
        raise SenseError("MNLM_BRIDGE_IMAGE", "primary JPEG cannot be decoded") from exc
    return hashlib.sha256(raw).hexdigest()


def durable(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def step_record(data, path, step, shard, offset, payload_sha, binding, image_sha, image_error=None):
    collection = digest(["bridge-provider-collection", path])
    episode_ref = digest([collection, data[META + "episode_id"][0]])
    # Every episode in one collection stays together; no IID assertion.
    r = {"schema": VERSION, "locator": {"source": "bridge-v2-rlds", "shard": shard + f"#offset={offset}",
         "row": step, "span": [step, step + 1], "record_digest": payload_sha, "manifest_digest": digest(binding)},
         "unit": collection, "referents": [], "channels": {c: absent("not_supplied_by_source") for c in CHANNELS}}
    ch = r["channels"]
    instruction = data[STEP + "language_instruction"][step].decode("utf-8")
    has_language = bool(data[META + "has_language"][0])
    require(bool(instruction) == has_language, "MNLM_BRIDGE_LANGUAGE", "availability contradicts instruction bytes")
    ch["surface"] = present([{"text": instruction}]) if has_language else absent("source_language_absent")
    ch["fields"] = present([field(f"state_component_{i}", v, "number") for i, v in
                            enumerate(data[STEP + "observation/state"][step * 7:(step + 1) * 7])])
    ch["behaviour"] = present([{"event": "recorded_command", "mode": "observed",
        "evidence": {"raw_sha256": binding["raw_sha256"], "episode_payload_sha256": payload_sha,
                     "offset": offset, "step": step, "alignment": "uncommissioned; no next-state relabeling"},
        "measurements": [field(f"command_component_{i}", v, "number") for i, v in
                         enumerate(data[STEP + "action"][step * 7:(step + 1) * 7])]}])
    ch["structure"] = present([field("kind", "observation", "category"),
                                field("span_unit", "step", "category"),
                                field("observation_type", "recorded_robot_step", "category"),
                                field("step_index", step, "integer", "sequence_index")])
    ch["provenance"] = present([field("source", "bridge-v2-rlds", "reference"),
        field("episode", episode_ref, "reference"), field("collection", collection, "reference"),
        field("raw_sha256", binding["raw_sha256"], "reference"), field("policy_sha256", binding["policy_sha256"], "reference"),
        field("license", "CC-BY-4.0", "category"), field("physical_units", "uncommissioned", "category"),
        field("image_0_provider_present", bool(data[META + "has_image_0"][0]), "boolean"),
        field("image_0_availability_conflict", image_sha is not None and not data[META + "has_image_0"][0], "boolean")])
    if image_sha is not None:
        ch["perception"] = present([{"modality": "image", "sha256": image_sha, "media_type": "image/jpeg",
            "measurements": [field("width", 256, "integer", "pixel"), field("height", 256, "integer", "pixel"),
                             field("channels", 3, "integer"), field("layout", "RGB", "category")]}])
    else:
        ch["perception"] = absent(image_error or "source_image_absent")
    ch["history"] = absent("ordered_steps_retained; physical_timestamps_not_supplied")
    return seal_record(r)


def input_plan(root):
    cdir = root / "census/bridge-observations-v1"
    selected = (cdir / "selection.json").read_bytes()
    require(hashlib.sha256(selected).hexdigest() == SELECTION, "MNLM_BRIDGE_SELECTION", "selection changed")
    wanted = json.loads(selected)["sources"]["bridge-v2-rlds"]["paths"]
    acquisition_raw = (root / "raw" / f"bounded-{SELECTION}-{ACQUISITION}.json").read_bytes()
    require(hashlib.sha256(acquisition_raw).hexdigest() == ACQUISITION, "MNLM_BRIDGE_ACQUISITION",
            "sealed acquisition report changed; receipts alone cannot redefine source identity")
    acquired = json.loads(acquisition_raw)
    accepted = {f["path"]: f for f in acquired["files"]}
    require(len(accepted) == len(acquired["files"]) == 12 and set(accepted) == set(wanted),
            "MNLM_BRIDGE_ACQUISITION", "acquisition differs from selected scope")
    receipts = {}
    for line in (root / "raw/bridge-v2-rlds/receipts.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        require(r["path"] not in receipts or receipts[r["path"]] == r, "MNLM_BRIDGE_RECEIPT", "conflicting receipts")
        receipts[r["path"]] = r
    require(len(wanted) == 12 and len(set(wanted)) == 12 and set(wanted) <= receipts.keys(),
            "MNLM_BRIDGE_SELECTION", "need the complete selected scope")
    result = []
    for url in wanted:
        r = receipts[url]
        require(r["sha256"] == accepted[url]["sha256"] and r["bytes"] == accepted[url]["bytes"],
                "MNLM_BRIDGE_RECEIPT", "receipt drift from sealed acquisition")
        path = root / "raw/bridge-v2-rlds" / url.rsplit("/", 1)[-1]
        require(r["size_match"] and r["verified"] is not False and path.stat().st_size == r["bytes"],
                "MNLM_BRIDGE_RECEIPT", "raw file is not completely acquired")
        result.append({"path": path, "sha256": r["sha256"], "bytes": r["bytes"]})
    return result


def export(root, out):
    p, policy_sha = policy()
    plan = input_plan(root)
    out.mkdir(parents=True, exist_ok=False)
    (out / "blobs").mkdir()
    binding = {"adapter": ADAPTER, "policy": p, "policy_sha256": policy_sha,
               "selection_sha256": SELECTION, "source": "bridge-v2-rlds", "scientific_admission": False,
               "raw": [{"name": f["path"].name, "sha256": f["sha256"], "bytes": f["bytes"]} for f in plan]}
    blobs, counts, episodes_seen = {}, Counter(), set()
    # field disposition includes every source key; annotations stay in raw bytes.
    disposition = {k: "retained_raw_only" for k in SCHEMA}
    for k in ("action", "observation/state", "language_instruction", "observation/image_0"):
        disposition[STEP + k] = "provisional_observation"
    for k in SCHEMA:
        if k.startswith(META):
            disposition[k] = "source_identity_or_availability; raw_preserved"
    durable(out / "field-disposition.json", canonical(disposition) + b"\n")

    def records():
        with (out / "episodes.jsonl").open("xb") as inventory:
            for source in plan:
                per_source = {"adapter": binding["adapter"], "policy_sha256": policy_sha, "raw_sha256": source["sha256"]}
                for offset, payload in frames(source["path"], source["sha256"]):
                    n, data, source_path, language = episode(decode(payload))
                    key = (source_path, data[META + "episode_id"][0])
                    require(key not in episodes_seen, "MNLM_BRIDGE_DUPLICATE_EPISODE", "duplicate source episode")
                    episodes_seen.add(key)
                    payload_sha = hashlib.sha256(payload).hexdigest()
                    counts["episodes"] += 1
                    for step in range(n):
                        raw = data[STEP + "observation/image_0"][step]
                        sha, error = None, None
                        try:
                            sha = primary_image(raw)
                        except SenseError as exc:
                            error = exc.code
                            counts[error] += 1
                        if sha and sha not in blobs:
                            target = (out / "blobs" / (sha + ".jpg")).resolve()
                            durable(target, raw)
                            blobs[sha] = {"sha256": sha, "path": str(target), "bytes": len(raw), "media_type": "image/jpeg"}
                        r = step_record(data, source_path, step, source["path"].name, offset, payload_sha, per_source, sha, error)
                        counts["steps"] += 1
                        counts["primary_images"] += sha is not None
                        yield r
                    inventory.write(canonical({"shard": source["path"].name, "offset": offset, "steps": n,
                        "source_path": source_path, "episode_id": key[1], "payload_sha256": payload_sha}) + b"\n")
            inventory.flush()
            os.fsync(inventory.fileno())
        durable(out / "blobs.json", canonical({"schema": "ucr-blobs-v1", "blobs": [blobs[k] for k in sorted(blobs)]}) + b"\n")
        binding.update(blobs_manifest_sha256=file_hash(out / "blobs.json"), episodes_sha256=file_hash(out / "episodes.jsonl"),
                       field_disposition_sha256=file_hash(out / "field-disposition.json"), counts=dict(counts))
    result = write_records(out / "observations.ucr.jsonl", records(), binding)
    report = {"acceptance": "author-checked only", "counts": dict(counts), "unique_images": len(blobs),
              "retained_image_bytes": sum(b["bytes"] for b in blobs.values()), "export_sha256": result["sha256"],
              "stage3_admitted": False, "stage4_accepted": False}
    durable(out / "report.json", canonical(report) + b"\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(export(args.root, args.out)), flush=True)
    except (SenseError, RecordError) as exc:
        print(json.dumps({"code": exc.code, "message": str(exc), "remediation": "Restore pinned source bytes or review the declared adapter policy; do not reuse partial output."}), file=sys.stderr)
        raise SystemExit(3)
