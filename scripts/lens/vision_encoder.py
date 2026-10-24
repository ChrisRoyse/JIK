"""Frozen DINOv2 visual instrument: CLS + every native spatial patch, no pooling.

One image per forward makes measurements independent of batch neighbors. RGB is
right/bottom padded to a 14-pixel grid; nothing is resized, cropped or discarded.
Padding and positional interpolation are declared parts of this instrument, not
a claim to reproduce upstream's crop-based evaluation or establish useful bits.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from learned import check_gpu
from senses.contract import SenseError, digest, require, validate_record
from vision_assets import REPO, REVISION, FILES, snapshot, read_validated

SPEC = {"name": "ucr-dinov2-small-image", "repo": REPO, "revision": REVISION,
        "kind": "multi", "inputs": "ucr", "channel": "perception", "perspective": "perception",
        "role": "universal_instrument", "module": "vision_ucr", "group": "universal",
        "sim": "maxsim", "sparse": False}
PATCH, MAX_IMAGES = 14, 4
MAX_PIXELS = 512 * 512
# Area alone would allow an extremely long edge, and padding could explode tokens.
MAX_EDGE, MAX_TOKENS = 512, 1370


def bind_checkpoint(model, state):
    """Attested HF attention-key rename only; reject ALL drift before mutation."""
    expected = model.state_dict()
    mapping = {name: name for name in state}
    layout = "original-attention-keys"
    if set(state) != set(expected):
        layout = "bijective-attention-projection-rename-v1"
        aliases = {"attention.attention.query.": "attention.q_proj.",
                   "attention.attention.key.": "attention.k_proj.",
                   "attention.attention.value.": "attention.v_proj.",
                   "attention.output.dense.": "attention.o_proj."}
        for old in state:
            new = old
            for source, target in aliases.items():
                new = new.replace(source, target)
            mapping[old] = new
    require(len(set(mapping.values())) == len(state) and set(mapping.values()) == set(expected),
            "MNLM_VISION_CHECKPOINT_LAYOUT", "unsupported or incomplete checkpoint key schema; no partial load")
    mapped = {mapping[n]: t for n, t in state.items()}
    require(all(t.shape == expected[n].shape and t.dtype == expected[n].dtype for n, t in mapped.items()),
            "MNLM_VISION_CHECKPOINT_LAYOUT", "checkpoint shape/dtype differs; no partial load")
    model.load_state_dict(mapped, strict=True)
    return {"schema": layout, "mapping_sha256": digest(mapping), "tensors": len(mapped),
            "renamed": sum(a != b for a, b in mapping.items())}


def image_array(item, blobs):
    from senses.pixels import decode_rgb
    rgb = decode_rgb(blobs.read(item["sha256"], item["media_type"]), item["media_type"], alignment=1)
    h, w = rgb.shape[:2]
    require(max(h, w) <= MAX_EDGE, "MNLM_VISION_BUDGET", "image edge exceeds frozen 512-pixel ceiling")
    fields = {f["name"]: f["value"] for f in item["measurements"]}
    require(len(fields) == len(item["measurements"]) and fields.get("width") == w
            and fields.get("height") == h and fields.get("channels") == 3 and fields.get("layout") == "RGB",
            "MNLM_PIXEL_GEOMETRY", "declared geometry differs from retained pixels")
    ph, pw = (h + PATCH - 1) // PATCH * PATCH, (w + PATCH - 1) // PATCH * PATCH
    require(1 + ph * pw // PATCH ** 2 <= MAX_TOKENS, "MNLM_VISION_BUDGET", "padded image exceeds native token ceiling")
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:h, :w] = rgb
    return padded, {"sha256": item["sha256"], "height": h, "width": w,
                    "padded_height": ph, "padded_width": pw, "grid": [ph // PATCH, pw // PATCH],
                    "patch": PATCH, "order": "CLS then row-major spatial patches",
                    "padding": "right/bottom zero RGB before normalization"}


class VisionEncoder:
    dim = 384

    def __init__(self, spec=SPEC):
        import torch
        import transformers
        from PIL import __version__ as pil_version, features
        from safetensors.torch import load
        from transformers import Dinov2Config, Dinov2Model
        require(spec == SPEC, "MNLM_LENS_FROZEN_VIOLATION", "unsupported vision instrument")
        runtime = check_gpu()
        try:
            snap = snapshot()
            data = read_validated(snap)
        except (OSError, ValueError) as exc:
            raise SenseError("MNLM_VISION_ASSET_DRIFT", "acquire/check the exact pinned visual assets in HF_HUB_CACHE") from exc
        sha, files = FILES["model.safetensors"][1], ["model.safetensors"]
        assets = {name: pin[1] for name, pin in FILES.items()}
        config = Dinov2Config.from_dict(json.loads(data["config.json"]))
        config._attn_implementation = "eager"
        require((config.hidden_size, config.patch_size, config.num_channels, config.num_hidden_layers) == (384, 14, 3, 12),
                "MNLM_VISION_MODEL_SHAPE", "unsupported visual checkpoint geometry")
        processor = json.loads(data["preprocessor_config.json"])
        self.mean = np.asarray(processor["image_mean"], np.float32)
        self.std = np.asarray(processor["image_std"], np.float32)
        self.model = Dinov2Model(config)
        checkpoint_layout = bind_checkpoint(self.model, load(data["model.safetensors"]))
        del data
        self.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        runtime = check_gpu()  # attest effective execution settings, not pre-configuration state
        self.identity = {**SPEC, "shape": "Multi(var,384)", "dtype": "f32", "modality": "image", "norm": "none",
            "weights_sha256": sha, "weight_files": files, "assets": assets,
            "checkpoint_layout": checkpoint_layout,
            "corpus_hash": None, "pretraining_data": "upstream training bytes unavailable; overlap unassessed",
            "preprocessing": {"scale": "RGB/255", "mean": processor["image_mean"], "std": processor["image_std"],
                "crop": False, "resize": False, "padding": "right/bottom zero RGB to patch grid before normalization",
                "patch": PATCH, "max_pixels": MAX_PIXELS, "max_edge": MAX_EDGE, "max_images": MAX_IMAGES,
                "max_tokens_per_image": MAX_TOKENS, "output": "last_hidden_state CLS and all patch tokens; no pooling",
                "position": "native DINOv2 bicubic interpolation", "execution_unit": "one image, float32 eager CUDA"},
            "runtime": {**runtime, "python": platform.python_version(), "torch": torch.__version__,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "transformers": transformers.__version__, "numpy": np.__version__, "pillow": pil_version,
                "jpeg": features.version_codec("jpg"), "platform": platform.platform()},
            "code_sha256": {n: hashlib.sha256((Path(__file__).parents[1] / n).read_bytes()).hexdigest()
                for n in ("lens/vision_encoder.py", "lens/vision_assets.py", "lens/vision_ucr.py", "lens/image_ucr.py",
                          "lens/learned.py", "senses/pixels.py", "senses/blobs.py", "senses/contract.py", "senses/views.py")}}

    def measure(self, record, blobs):
        import torch
        require(torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32 and not torch.backends.cudnn.benchmark,
                "MNLM_VISION_RUNTIME_DRIFT", "restore the pinned deterministic float32 execution settings")
        validate_record(record)
        items = [p for p in record["channels"]["perception"]["items"] if p["modality"] == "image"]
        require(len(items) <= MAX_IMAGES, "MNLM_VISION_BUDGET", "at most four images per observation")
        parts, regions, start = [], [], 0
        for item in items:
            padded, region = image_array(item, blobs)
            x = (padded.astype(np.float32) / np.float32(255) - self.mean) / self.std
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to("cuda")
            with torch.inference_mode():
                values = self.model(pixel_values=x).last_hidden_state[0].cpu().numpy().copy()
            expected = 1 + region["grid"][0] * region["grid"][1]
            require(values.shape == (expected, self.dim) and values.dtype == np.float32 and np.isfinite(values).all(),
                    "MNLM_VISION_OUTPUT", "visual encoder emitted an invalid native slot")
            parts.append(values)
            regions.append({**region, "token_start": start, "token_end": start + len(values)})
            start += len(values)
        tokens = np.concatenate(parts) if parts else np.empty((0, self.dim), np.float32)
        return {"kind": "Multi" if parts else "Absent", "channel": "perception", "tokens": tokens,
                "shape": list(tokens.shape), "reason": None if parts else "no_image_observation", "regions": regions}
