"""Immutable, content-bound consumer checkpoints; no change to frozen encoders."""
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from senses.contract import canonical, digest, require
from production.publication import sync_dir
from model.contracts import Instrument, ModelConfig, HEX
from model.network import SensoryLanguageModel

SCHEMA = "sensory-surface-checkpoint-v1"


def code_hashes():
    root = Path(__file__).resolve().parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in ("contracts.py", "native.py", "attention.py", "network.py", "checkpoint.py", "publication_input.py")}


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(canonical([name, list(array.shape), array.dtype.str]))
        h.update(array.tobytes())
    return h.hexdigest()


def runtime_contract(device):
    device = torch.device(device)
    return {"torch": str(torch.__version__), "device_type": device.type,
            "cuda": torch.version.cuda if device.type == "cuda" else None,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "default_dtype": str(torch.get_default_dtype())}


def save(model, path, *, step, input_sha256, parent=None):
    require(type(step) is int and step >= 0 and HEX.fullmatch(input_sha256)
            and (parent is None or HEX.fullmatch(parent)), "MNLM_MODEL_LINEAGE", "step and exact input/parent identities required")
    require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),
            "MNLM_MODEL_NONFINITE", "refuse to checkpoint nonfinite parameters")
    require(sum(t.numel() * t.element_size() for t in model.state_dict().values()) < model.config.checkpoint_bytes,
            "MNLM_MODEL_CHECKPOINT_BUDGET", "weight bytes exceed fixed checkpoint budget")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    require(len(raw) <= model.config.checkpoint_bytes, "MNLM_MODEL_CHECKPOINT_BUDGET", "serialized weights exceed budget")
    manifest = {"schema": SCHEMA, "purpose": "engineering_surface_decoder", "meaning_asserted": False,
                "config": asdict(model.config), "instruments": [asdict(s) for s in model.specs],
                "trainable": [n for n, p in model.named_parameters() if p.requires_grad],
                "state_sha256": state_hash(model), "weights_sha256": hashlib.sha256(raw).hexdigest(),
                "weights_bytes": len(raw), "code_sha256": code_hashes(), "torch_version": str(torch.__version__),
                "device_type": model.tokens.weight.device.type, "step": step, "input_sha256": input_sha256,
                "runtime": runtime_contract(model.tokens.weight.device),
                "parent": parent, "optimizer_resume_supported": False}
    manifest_raw = canonical(manifest)
    require(len(manifest_raw) <= 1 << 20, "MNLM_MODEL_CHECKPOINT_BUDGET", "metadata exceeds fixed 1 MiB envelope")
    root = Path(path)
    require(not root.exists(), "MNLM_MODEL_CHECKPOINT_EXISTS", "checkpoint versions are immutable; choose a new destination")
    root.mkdir(parents=True, exist_ok=False)
    for name, payload in (("weights.pt", raw), ("manifest.json", manifest_raw)):
        with (root / name).open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    # A crash before this marker leaves a visibly incomplete, unreadable version.
    with (root / "COMMITTED").open("xb") as f:
        f.write(digest(manifest).encode())
        f.flush()
        os.fsync(f.fileno())
    sync_dir(root)
    sync_dir(root.parent)
    return digest(manifest)


def load(path, device):
    root = Path(path)
    require((root / "COMMITTED").is_file() and (root / "manifest.json").is_file(),
            "MNLM_MODEL_CHECKPOINT_INCOMPLETE", "missing committed checkpoint manifest")
    require((root / "manifest.json").stat().st_size <= 1 << 20 and (root / "COMMITTED").stat().st_size == 64,
            "MNLM_MODEL_CHECKPOINT_BUDGET", "invalid manifest/marker size")
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    require(canonical(manifest) == raw and digest(manifest) == (root / "COMMITTED").read_text(),
            "MNLM_MODEL_CHECKPOINT_CORRUPT", "manifest content differs from commit marker")
    require(manifest.get("schema") == SCHEMA and manifest.get("purpose") == "engineering_surface_decoder"
            and manifest.get("meaning_asserted") is False, "MNLM_MODEL_CHECKPOINT_SCHEMA", "unrecognized checkpoint purpose/schema")
    require(manifest["code_sha256"] == code_hashes(), "MNLM_MODEL_CODE_DRIFT", "restore exact consumer code or perform explicit migration")
    require(manifest["torch_version"] == str(torch.__version__) and manifest["device_type"] == torch.device(device).type,
            "MNLM_MODEL_RUNTIME_DRIFT", "replay requires the declared runtime/device; migration is a separate experiment")
    require(manifest.get("runtime") == runtime_contract(device), "MNLM_MODEL_RUNTIME_DRIFT",
            "restore the declared GPU, precision and determinism settings before replay")
    config = ModelConfig(**manifest["config"])
    config.validate()
    require((root / "weights.pt").is_file() and (root / "weights.pt").stat().st_size == manifest["weights_bytes"]
            and manifest["weights_bytes"] <= config.checkpoint_bytes, "MNLM_MODEL_CHECKPOINT_CORRUPT", "missing/wrong weight bytes")
    raw = (root / "weights.pt").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest["weights_sha256"],
            "MNLM_MODEL_CHECKPOINT_CORRUPT", "weights differ from the immutable manifest")
    specs = tuple(Instrument(**s) for s in manifest["instruments"])
    model = SensoryLanguageModel(config, specs).to(device)
    state = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    names = {n for n, _ in model.named_parameters()}
    require(len(manifest["trainable"]) == len(set(manifest["trainable"])) and set(manifest["trainable"]) <= names,
            "MNLM_MODEL_CHECKPOINT_CORRUPT", "invalid trainable parameter roster")
    for n, p in model.named_parameters():
        p.requires_grad_(n in manifest["trainable"])
    require(state_hash(model) == manifest["state_sha256"], "MNLM_MODEL_CHECKPOINT_CORRUPT", "cold state readback differs")
    return model, manifest
