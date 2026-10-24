"""Physical visual boundaries. Derived corruptions are controls, never training data."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image

from senses.blobs import Blobs
from senses.contract import canonical, seal_record, absent
from senses.pipeline import file_hash
from senses.pixels import decode_rgb
from lenscommon import take_lease, release_lease
from measure import nvidia_smi
from vision_encoder import VisionEncoder, image_array
from vision_assets import check as check_asset


def boundaries(record, blobs_path, out, check, refuses):
    (out / "vision-resident-before.json").write_bytes(canonical(nvidia_smi()))
    blobs = Blobs(blobs_path, file_hash(blobs_path))
    original = copy.deepcopy(record["channels"]["perception"]["items"][0])
    raw = blobs.read(original["sha256"], original["media_type"])
    rgb = decode_rgb(raw, original["media_type"])
    before = file_hash(blobs_path)
    root = out / "derived-negative-controls"
    root.mkdir()
    variants = {"crop": rgb[:255, :253], "changed": 255 - rgb, "long": np.tile(rgb[:1], (1, 3, 1))}
    entries, items = [], {}
    for name, pixels in variants.items():
        path = root / (name + ".png")
        Image.fromarray(pixels).save(path)
        item = copy.deepcopy(original)
        item.update(sha256=file_hash(path), media_type="image/png")
        for f in item["measurements"]:
            if f["name"] == "width":
                f["value"] = pixels.shape[1]
            if f["name"] == "height":
                f["value"] = pixels.shape[0]
        items[name] = item
        entries.append({"sha256": item["sha256"], "path": str(path.resolve()), "bytes": path.stat().st_size,
                        "media_type": "image/png"})
    manifest = root / "blobs.json"
    manifest.write_bytes(canonical({"schema": "ucr-blobs-v1", "blobs": entries}))
    controls = Blobs(manifest, file_hash(manifest))
    padded, region = image_array(items["crop"], controls)
    check("pad_preserves_all_original_pixels", padded.shape == (266, 266, 3)
          and np.array_equal(padded[:255, :253], variants["crop"])
          and not padded[255:].any() and not padded[:, 253:].any(), geometry=region)
    refuses("elongated_image_budget", lambda: image_array(items["long"], controls), "MNLM_VISION_BUDGET")
    bad = copy.deepcopy(original)
    next(f for f in bad["measurements"] if f["name"] == "width")["value"] += 1
    refuses("retained_geometry_mismatch", lambda: image_array(bad, blobs), "MNLM_PIXEL_GEOMETRY")
    corrupt = root / "changed.png"
    retained = corrupt.read_bytes()
    corrupt.write_bytes(retained[:-1] + bytes([retained[-1] ^ 1]))
    refuses("mutated_blob_cannot_be_read", lambda: image_array(items["changed"], controls), "MNLM_UCR_BLOB_DIGEST")
    corrupt.write_bytes(retained)
    try:
        check_asset(corrupt, len(retained), "0" * 64)
    except ValueError as exc:
        check("changed_model_asset_pin_refused", "MNLM_VISION_ASSET_DRIFT" in str(exc))
    else:
        check("changed_model_asset_pin_refused", False)

    take_lease("15", "real-image visual instrument boundaries")
    encoder = None
    try:
        encoder = VisionEncoder()
        import torch
        check("effective_runtime_attested", encoder.identity["runtime"]["deterministic"] is True
              and encoder.identity["runtime"]["matmul_tf32"] is False
              and encoder.identity["runtime"]["cudnn_tf32"] is False
              and encoder.identity["runtime"]["cudnn_benchmark"] is False)
        torch.use_deterministic_algorithms(False)
        try:
            refuses("runtime_drift_refused", lambda: encoder.measure(record, blobs), "MNLM_VISION_RUNTIME_DRIFT")
        finally:
            torch.use_deterministic_algorithms(True)
        from production.session import Session
        # Reuse the runtime state audit without creating another model resident.
        shell = object.__new__(Session)
        shell.lens = encoder
        state = shell.state_digest()
        from vision_encoder import bind_checkpoint
        intact = encoder.model.state_dict()
        bad_state = dict(intact)
        bad_state.pop(next(iter(bad_state)))
        refuses("partial_checkpoint_refused_before_mutation", lambda: bind_checkpoint(encoder.model, bad_state),
                "MNLM_VISION_CHECKPOINT_LAYOUT")
        bad_state = dict(intact)
        key = next(iter(bad_state))
        bad_state[key] = bad_state[key].flatten()[:1]
        refuses("wrong_shape_refused_before_mutation", lambda: bind_checkpoint(encoder.model, bad_state),
                "MNLM_VISION_CHECKPOINT_LAYOUT")
        check("failed_load_did_not_mutate_weights", state == shell.state_digest())
        a = encoder.measure(record, blobs)
        b = encoder.measure(record, blobs)
        check("native_visual_determinism", a["tokens"].tobytes() == b["tokens"].tobytes()
              and a["regions"] == b["regions"], shape=a["shape"], digest=hashlib.sha256(a["tokens"].tobytes()).hexdigest())
        np.save(out / "visual-witness.npy", a["tokens"], allow_pickle=False)
        (out / "visual-identity.json").write_bytes(canonical(encoder.identity))
        modified = copy.deepcopy(record)
        modified["channels"]["perception"]["items"] = [items["changed"]]
        c = encoder.measure(seal_record(modified), controls)
        check("real_changed_pixels_change_visual_features", not np.array_equal(a["tokens"], c["tokens"]))
        modified["channels"]["perception"]["items"] = [items["crop"]]
        c = encoder.measure(seal_record(modified), controls)
        check("non_aligned_image_has_all_spatial_tokens", c["shape"] == [362, 384] and c["regions"][0]["width"] == 253)
        modified["channels"]["perception"]["items"] = [original] * 5
        refuses("too_many_images_refused", lambda: encoder.measure(seal_record(modified), blobs), "MNLM_VISION_BUDGET")
        modified["channels"]["perception"] = absent("explicitly_withheld")
        c = encoder.measure(seal_record(modified), blobs)
        check("absent_image_remains_absent", c["kind"] == "Absent" and c["shape"] == [0, 384])
        check("encoder_parameters_unchanged", state == shell.state_digest())
    finally:
        if encoder is not None:
            shell.lens = None
            del encoder
            import torch
            torch.cuda.empty_cache()
        release_lease()
    check("source_images_unchanged", before == file_hash(blobs_path)
          and hashlib.sha256(blobs.read(original["sha256"], original["media_type"])).hexdigest() == original["sha256"])
