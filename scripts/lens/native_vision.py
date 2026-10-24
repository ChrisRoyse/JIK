"""Frozen native visual candidate instruments; every spatial row stays typed.

Bounded RGB inputs are neither resized nor cropped. Explicit padding and native
position interpolation are part of each instrument, not upstream benchmark parity.
"""
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np

from learned import check_gpu
from native_assets import PLAN_PIN, bind, read_assets, state_sha
from senses.contract import require, validate_record

SPECS = {
    "siglip2": dict(repo="google/siglip2-base-patch16-224", dim=768, stride=16,
                   cls=False, canvas=None, output="all final spatial patches"),
    "mae": dict(repo="facebook/vit-mae-base", dim=768, stride=16,
               cls=True, canvas=None, output="CLS then every unmasked spatial patch"),
    "sam": dict(repo="facebook/sam-vit-base", dim=256, stride=16,
               cls=False, canvas=1024, output="all neck spatial cells on fixed canvas"),
    "convnext": dict(repo="facebook/convnext-tiny-224", dim=768, stride=32,
                    cls=False, canvas=None, output="all final convolutional spatial cells"),
}
MAX_EDGE, MAX_IMAGES = 512, 4


def geometry(name, height, width):
    require(name in SPECS and type(height) is int and type(width) is int
            and 0 < height <= MAX_EDGE and 0 < width <= MAX_EDGE,
            "MNLM_NATIVE_VISION_BUDGET", "bounded positive RGB geometry required")
    spec = SPECS[name]
    stride = spec["stride"]
    ph = spec["canvas"] or ((height+stride-1)//stride*stride)
    pw = spec["canvas"] or ((width+stride-1)//stride*stride)
    grid = [ph//stride, pw//stride]
    return dict(height=height, width=width, padded_height=ph, padded_width=pw,
                grid=grid, stride=stride, cls=spec["cls"],
                shape=[grid[0]*grid[1]+int(spec["cls"]), spec["dim"]],
                order="CLS then row-major cells" if spec["cls"] else "row-major cells",
                padding="right/bottom zero RGB before normalization")


def construct(name, config, tensors):
    if name == "siglip2":
        from transformers import SiglipVisionConfig, SiglipVisionModel
        cfg = SiglipVisionConfig.from_dict(config["vision_config"])
        cfg._attn_implementation = "eager"
        model = SiglipVisionModel(cfg)
        selected = lambda k: k if k.startswith("vision_model.") else None
        excluded = lambda k: k.startswith("text_model.") or k in ("logit_scale", "logit_bias")
    elif name == "mae":
        from transformers import ViTMAEConfig, ViTMAEModel
        cfg = ViTMAEConfig.from_dict(config)
        cfg.mask_ratio = 0.0
        cfg._attn_implementation = "eager"
        model = ViTMAEModel(cfg)
        selected = lambda k: k.removeprefix("vit.") if k.startswith("vit.") else None
        excluded = lambda k: k.startswith("decoder.")
    elif name == "sam":
        from transformers import SamVisionConfig, SamVisionModel
        cfg = SamVisionConfig.from_dict(config["vision_config"])
        cfg._attn_implementation = "eager"
        model = SamVisionModel(cfg)
        selected = lambda k: k if k.startswith("vision_encoder.") else None
        excluded = lambda k: k.startswith(("prompt_encoder.", "mask_decoder.", "shared_image_embedding."))
    else:
        from transformers import ConvNextConfig, ConvNextModel
        cfg = ConvNextConfig.from_dict(config)
        model = ConvNextModel(cfg)
        selected = lambda k: k.removeprefix("convnext.") if k.startswith("convnext.") else None
        excluded = lambda k: k in ("classifier.weight", "classifier.bias")
    family = {"siglip2": "siglip-vision", "mae": "vit"}.get(name)
    layout = bind(model, tensors, selected, excluded, rename_family=family)
    return model, cfg, layout


class NativeVisionEncoder:
    def __init__(self, name, *, asset_root, plan_path):
        import torch
        import transformers
        from PIL import __version__ as pillow_version, features
        require(name in SPECS, "MNLM_NATIVE_VISION_NAME", "unknown visual instrument")
        check_gpu()
        self.name, self.spec = name, SPECS[name]
        started = time.perf_counter()
        row, config, processor, tensors = read_assets(asset_root, plan_path, self.spec["repo"])
        self.load_phases = dict(asset_read=time.perf_counter()-started)
        t = time.perf_counter()
        self.model, cfg, layout = construct(name, config, tensors)
        del tensors
        self.mean = np.asarray(processor["image_mean"], dtype=np.float32)
        self.std = np.asarray(processor["image_std"], dtype=np.float32)
        require(self.mean.shape == self.std.shape == (3,) and bool((self.std > 0).all()),
                "MNLM_NATIVE_VISION_NORMALIZATION", "invalid pinned channel normalization")
        self.dim = self.spec["dim"]
        self.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        self.load_phases["construct_bind_cuda"] = time.perf_counter()-t
        self.identity = dict(name="ucr-native-"+name+"-image", modality="image",
            shape=f"Multi(var,{self.dim})", dtype="f32", norm="none", repo=row["repo"],
            revision=row["revision"], assets=row["files"], asset_plan=PLAN_PIN,
            checkpoint_layout=layout, effective_config=json.loads(cfg.to_json_string(use_diff=False)), frozen=True,
            preprocessing=dict(max_edge=MAX_EDGE, max_images=MAX_IMAGES, **self.spec,
                mean=self.mean.tolist(), std=self.std.tolist(), scale="RGB/255",
                crop=False, resize=False, padding="right/bottom zero RGB before normalization",
                mae_masking="zero masking; ordered arange noise; preserve row-major patch order",
                position="native interpolation for SigLIP/MAE; original fixed SAM; none ConvNeXT",
                pooling=False, execution="one image float32 eager CUDA"),
            runtime={**check_gpu(), "python": platform.python_version(),
                     "transformers": transformers.__version__,
                     "numpy": np.__version__, "pillow": pillow_version,
                     "jpeg": features.version_codec("jpg"), "platform": platform.platform()},
            dependency_sha256={"model": hashlib.sha256(
                Path(inspect.getfile(type(self.model))).read_bytes()).hexdigest()},
            code_sha256={n: hashlib.sha256((Path(__file__).parents[1]/n).read_bytes()).hexdigest()
                         for n in ("lens/native_vision.py", "lens/native_assets.py", "lens/learned.py",
                                   "senses/pixels.py", "senses/blobs.py", "senses/contract.py")},
            pretraining_overlap="upstream training bytes unavailable; unassessed",
            scientific_admission=False)
        self.identity["state_sha256"] = state_sha(self.model)

    def measure(self, record, blobs):
        import torch
        from senses.pixels import decode_rgb
        validate_record(record)
        require(torch.are_deterministic_algorithms_enabled()
                and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32 and not torch.backends.cudnn.benchmark,
                "MNLM_NATIVE_VISION_RUNTIME", "restore declared deterministic CUDA settings")
        items = [x for x in record["channels"]["perception"]["items"] if x["modality"] == "image"]
        require(len(items) <= MAX_IMAGES, "MNLM_NATIVE_VISION_BUDGET", "too many images")
        parts, regions, offset = [], [], 0
        phases = dict(frontend=0.0, forward=0.0, materialize=0.0)
        for item in items:
            t = time.perf_counter()
            rgb = decode_rgb(blobs.read(item["sha256"], item["media_type"]), item["media_type"], alignment=1)
            h, w, channels = rgb.shape
            fields = {x["name"]: x["value"] for x in item["measurements"]}
            require(len(fields) == len(item["measurements"]) and channels == fields.get("channels") == 3
                    and fields.get("height") == h and fields.get("width") == w and fields.get("layout") == "RGB",
                    "MNLM_NATIVE_VISION_GEOMETRY", "declared geometry differs from retained RGB")
            g = geometry(self.name, h, w)
            padded = np.zeros((g["padded_height"], g["padded_width"], 3), dtype=np.uint8)
            padded[:h, :w] = rgb
            array = (padded.astype(np.float32)/np.float32(255)-self.mean)/self.std
            x = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to("cuda")
            kwargs = {}
            if self.name in ("mae", "siglip2"):
                kwargs["interpolate_pos_encoding"] = True
            if self.name == "mae":
                kwargs["noise"] = torch.arange(g["grid"][0]*g["grid"][1],
                                               dtype=torch.float32, device="cuda").unsqueeze(0)
            torch.cuda.synchronize()
            phases["frontend"] += time.perf_counter()-t
            t = time.perf_counter()
            with torch.inference_mode():
                output = self.model(pixel_values=x, **kwargs)
                values = output.last_hidden_state[0]
                if self.name == "mae":
                    require(bool((output.mask == 0).all())
                            and torch.equal(output.ids_restore[0], kwargs["noise"][0].long()),
                            "MNLM_NATIVE_VISION_PATCH_ORDER", "MAE masked or reordered source patches")
                if self.name in ("sam", "convnext"):
                    require(list(values.shape) == [self.dim, *g["grid"]],
                            "MNLM_NATIVE_VISION_GRID", "unexpected native feature grid")
                    values = values.permute(1, 2, 0).reshape(-1, self.dim)
            torch.cuda.synchronize()
            phases["forward"] += time.perf_counter()-t
            t = time.perf_counter()
            array = values.cpu().numpy().copy()
            require(list(array.shape) == g["shape"] and array.dtype == np.float32
                    and bool(np.isfinite(array).all()), "MNLM_NATIVE_VISION_OUTPUT", "invalid spatial readings")
            parts.append(array)
            regions.append(dict(**g, sha256=item["sha256"], token_start=offset, token_end=offset+len(array)))
            offset += len(array)
            phases["materialize"] += time.perf_counter()-t
        self.last_phases = phases
        return dict(kind="Multi" if parts else "Absent", channel="perception",
                    tokens=np.concatenate(parts) if parts else np.empty((0, self.dim), np.float32),
                    regions=regions, shape=[offset, self.dim],
                    reason=None if parts else "no_image_observation")
