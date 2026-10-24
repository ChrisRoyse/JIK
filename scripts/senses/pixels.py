"""Frozen, source-neutral RGB pixel-patch encoder, preserving every decoded pixel.

This is an explicit observation encoder, not a pretrained semantic vision model.
O(decoded pixels), no resize/pool/fit; 8x8 RGB patches plus x/y/width/height.
The image's dimensions must fit the fixed per-observation budget before decode.
"""
import hashlib
import io
import platform
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as PIL_VERSION, features

from .blobs import MAX_BYTES
from .contract import SenseError, digest, require, validate_record

PATCH = 8
DIM = PATCH * PATCH * 3 + 4
MAX_PIXELS = 512 * 512


def contract():
    base = Path(__file__).resolve().parent
    code = {n: hashlib.sha256((base / n).read_bytes()).hexdigest() for n in ("pixels.py", "blobs.py", "contract.py")}
    c = {"name": "ucr-rgb-patches-v1", "channel": "perception", "perspective": "perception",
         "role": "universal_instrument", "kind": "encoder", "inputs": "ucr_blobs", "dtype": "f32",
         "shape": [None, DIM], "norm": "RGB / 255; absolute pixel coordinates", "corpus_hash": "none",
         "weights_sha256": digest(code), "frozen": True, "patch": PATCH, "max_pixels": MAX_PIXELS,
         "runtime": {"device": "algorithmic-cpu", "python": platform.python_version(), "numpy": np.__version__,
                     "pillow": PIL_VERSION, "jpeg": features.version_codec("jpg"), "platform": platform.platform()},
         "cpu_fallback": False, "semantic_pretraining": False, "code_sha256": code}
    return {**c, "lens_id": digest(c)}


def decode_rgb(raw, media_type, *, alignment=PATCH):
    require(alignment in (1, PATCH), "MNLM_PIXEL_LAYOUT", "unsupported patch alignment")
    require(0 < len(raw) <= MAX_BYTES, "MNLM_PIXEL_BYTES", "image exceeds fixed retained-byte budget")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            require(media_type in ("image/jpeg", "image/png") and im.format == {"image/jpeg": "JPEG", "image/png": "PNG"}[media_type],
                    "MNLM_PIXEL_TYPE", "decoded format differs from media type")
            w, h = im.size
            require(im.mode == "RGB" and getattr(im, "n_frames", 1) == 1,
                    "MNLM_PIXEL_LAYOUT", "supply a single RGB image; no silent conversion")
            require(w > 0 and h > 0 and w * h <= MAX_PIXELS and w % alignment == h % alignment == 0,
                    "MNLM_PIXEL_BUDGET", "split into explicitly aligned retained patches; no truncation or resize")
            im.load()
            return np.asarray(im, dtype=np.uint8).copy()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SenseError("MNLM_PIXEL_DECODE", "image could not be decoded safely") from exc


def encode_pixels(rgb):
    h, w, _ = rgb.shape
    patches = rgb.reshape(h // PATCH, PATCH, w // PATCH, PATCH, 3).transpose(0, 2, 1, 3, 4)
    color = patches.reshape(-1, PATCH * PATCH * 3).astype(np.float32) / np.float32(255)
    y, x = np.mgrid[0:h:PATCH, 0:w:PATCH]
    coords = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, w), np.full(x.size, h)]).astype(np.float32)
    return np.column_stack([color, coords])  # one instrument; no cross-lens fusion


def measure(record, blobs):
    validate_record(record)
    parts, regions, start = [], [], 0
    items = [p for p in record["channels"]["perception"]["items"] if p["modality"] == "image"]
    require(len(items) <= 4, "MNLM_PIXEL_BUDGET", "at most four separate images per bounded observation")
    for item in items:
        raw = blobs.read(item["sha256"], item["media_type"])
        rgb = decode_rgb(raw, item["media_type"])
        fields = {f["name"]: f["value"] for f in item["measurements"]}
        require(len(fields) == len(item["measurements"]) and fields.get("width") == rgb.shape[1]
                and fields.get("height") == rgb.shape[0] and fields.get("channels") == 3 and fields.get("layout") == "RGB",
                "MNLM_PIXEL_GEOMETRY", "declared image geometry differs from retained pixels")
        values = encode_pixels(rgb)
        parts.append(values)
        regions.append({"sha256": item["sha256"], "token_start": start, "token_end": start + len(values),
                        "height": rgb.shape[0], "width": rgb.shape[1], "order": "row_major_8x8_rgb; no padding"})
        start += len(values)
    values = np.concatenate(parts) if parts else np.empty((0, DIM), np.float32)
    return {"kind": "Multi" if parts else "Absent", "channel": "perception", "tokens": values,
            "shape": list(values.shape), "reason": None if parts else "no_image_observation", "regions": regions}
