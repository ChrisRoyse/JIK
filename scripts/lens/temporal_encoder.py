"""A trainable universal history encoder, frozen before candidate measurement.

Source adapters supply native history tokens; neither source names nor labels
enter the network. Every input token retains its own output. This candidate is
not scientifically admitted, even when its training loss decreases.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn
from safetensors.torch import save as serialize, load as deserialize

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from senses.contract import SenseError, canonical, digest, require
from senses.forecast import put
from senses.instruments import contracts, measure
from senses.pipeline import file_hash
from model.attention import Attention, Linear
from model.checkpoint import state_hash, runtime_contract
from production.publication import sync_dir


@dataclass(frozen=True)
class Config:
    dims: int = 128
    width: int = 32
    heads: int = 4
    max_tokens: int = 256
    max_batch: int = 16
    max_attention_elements: int = 1 << 22


CONFIG = Config()
ENCODER_PARAMETERS = 16896
SCHEMA = "frozen-temporal-candidate-v1"


def execution_code():
    root = Path(__file__).resolve().parents[1]
    return {name: file_hash(root / name) for name in
            ("lens/temporal_encoder.py", "model/attention.py", "model/checkpoint.py", "senses/contract.py", "senses/instruments.py")}


def bounded_read(path, limit=1 << 20):
    try:
        with path.open("rb") as f:
            raw = f.read(limit + 1)
    except OSError as exc:
        raise SenseError("MNLM_TEMPORAL_CHECKPOINT", "missing/unreadable checkpoint member") from exc
    require(len(raw) <= limit, "MNLM_TEMPORAL_CHECKPOINT", "checkpoint member exceeds declared bound")
    return raw


def runtime():
    return {**runtime_contract("cuda"), "python": platform.python_version(), "numpy": np.__version__,
            "system": platform.system(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def configure():
    require(torch.cuda.is_available(), "MNLM_TEMPORAL_CUDA", "CUDA is required; no CPU fallback")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "MNLM_TEMPORAL_RUNTIME", "restore declared cuBLAS workspace")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.get_default_dtype() == torch.float32, "MNLM_TEMPORAL_RUNTIME", "float32 default required")
    return runtime()


def positions(n, device):
    scale = torch.exp(torch.arange(0, CONFIG.width, 2, device=device) * (-np.log(10000.) / CONFIG.width))
    phase = torch.arange(n, device=device, dtype=torch.float32)[:, None] * scale
    result = torch.empty((n, CONFIG.width), device=device)
    result[:, 0::2], result[:, 1::2] = phase.sin(), phase.cos()
    return result


class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(CONFIG.dims, CONFIG.width)
        self.attention = Attention(CONFIG)
        self.norm1 = nn.LayerNorm(CONFIG.width)
        self.norm2 = nn.LayerNorm(CONFIG.width)
        self.ff = nn.Sequential(Linear(CONFIG.width, 4 * CONFIG.width), nn.GELU(), Linear(4 * CONFIG.width, CONFIG.width))
        self.output = nn.LayerNorm(CONFIG.width)
        require(sum(p.numel() for p in self.parameters()) == ENCODER_PARAMETERS,
                "MNLM_TEMPORAL_COST", "parameter preflight changed")

    def forward(self, values, lengths):
        require(values.dtype == torch.float32 and values.ndim == 3 and values.device.type == "cuda"
                and 0 < values.shape[0] <= CONFIG.max_batch and 0 < values.shape[1] <= CONFIG.max_tokens
                and values.shape[2] == CONFIG.dims and bool(torch.isfinite(values).all()),
                "MNLM_TEMPORAL_INPUT", "bounded finite float32 CUDA native tokens required")
        require(lengths.dtype == torch.int64 and lengths.shape == (len(values),) and lengths.device == values.device
                and bool(((lengths > 0) & (lengths <= values.shape[1])).all()),
                "MNLM_TEMPORAL_LENGTH", "explicit nonempty native lengths required")
        padding = torch.arange(values.shape[1], device=values.device)[None] >= lengths[:, None]
        h = self.input(values) + positions(values.shape[1], values.device)
        q = self.norm1(h)
        h = h + self.attention(q, q, padding=padding, causal=True)
        h = h + self.ff(self.norm2(h))
        return self.output(h).masked_fill(padding[..., None], 0)


def freeze(model, head, out, lineage):
    require(model.input.weight.device.type == "cuda", "MNLM_TEMPORAL_CUDA", "candidate must use declared CUDA runtime")
    require(all(bool(torch.isfinite(t).all()) for t in (*model.state_dict().values(), *head.state_dict().values())),
            "MNLM_TEMPORAL_NONFINITE", "refuse nonfinite trained parameters")
    model.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    weights = serialize({n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()})
    aux = serialize({n: t.detach().cpu().contiguous() for n, t in head.state_dict().items()})
    require(len(weights) <= 1 << 20 and len(aux) <= 1 << 20, "MNLM_TEMPORAL_WEIGHT_BUDGET", "unexpected checkpoint size")
    upstream = next(c for c in contracts() if c["channel"] == "history")
    manifest = {"schema": SCHEMA, "config": asdict(CONFIG), "encoder_parameters": ENCODER_PARAMETERS,
                "weights_sha256": hashlib.sha256(weights).hexdigest(), "weights_bytes": len(weights),
                "head_sha256": hashlib.sha256(aux).hexdigest(), "head_bytes": len(aux),
                "encoder_state_sha256": state_hash(model), "head_state_sha256": state_hash(head),
                "input_weights_sha256": upstream["weights_sha256"], "input_contract": upstream["contract"],
                "training_lineage": lineage, "training_runtime": runtime(), "execution_code_sha256": execution_code(),
                "frozen": True, "production_admission": False}
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in (("encoder.safetensors", weights), ("head.safetensors", aux), ("manifest.json", canonical(manifest))):
        put(out / name, raw)
    put(out / "COMMITTED", digest(manifest).encode())
    sync_dir(out)
    sync_dir(out.parent)
    return digest(manifest)


class FrozenTemporal:
    dim = CONFIG.width

    def __init__(self, root, expected_id):
        root = Path(root)
        raw = bounded_read(root / "manifest.json")
        require(hashlib.sha256(raw).hexdigest() == expected_id
                and bounded_read(root / "COMMITTED", 64) == expected_id.encode(),
                "MNLM_TEMPORAL_CHECKPOINT", "incomplete/unbound checkpoint")
        try:
            m = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise SenseError("MNLM_TEMPORAL_CHECKPOINT", "malformed checkpoint manifest") from exc
        require(isinstance(m, dict) and canonical(m) == raw and m.get("schema") == SCHEMA
                and m.get("config") == asdict(CONFIG) and m.get("frozen") is True
                and m.get("production_admission") is False and m.get("execution_code_sha256") == execution_code(),
                "MNLM_TEMPORAL_CODE_DRIFT", "restore exact frozen definition or commission a new version")
        require({"weights_bytes", "weights_sha256", "encoder_state_sha256", "training_lineage",
                 "input_contract", "input_weights_sha256"} <= m.keys(),
                "MNLM_TEMPORAL_CHECKPOINT", "incomplete frozen instrument contract")
        upstream = next(c for c in contracts() if c["channel"] == "history")
        require(m["input_contract"] == upstream["contract"] and m["input_weights_sha256"] == upstream["weights_sha256"],
                "MNLM_TEMPORAL_CODE_DRIFT", "history preprocessor changed")
        self.runtime = configure()
        weights = bounded_read(root / "encoder.safetensors")
        require(len(weights) == m["weights_bytes"] <= 1 << 20 and hashlib.sha256(weights).hexdigest() == m["weights_sha256"],
                "MNLM_TEMPORAL_CHECKPOINT", "frozen encoder bytes changed")
        # Safe tensor-only decoding, complete shape/dtype validation before mutation.
        try:
            state = deserialize(weights)
        except Exception as exc:
            raise SenseError("MNLM_TEMPORAL_CHECKPOINT_LAYOUT", "invalid safe tensor encoding") from exc
        model = TemporalNet()
        expected = model.state_dict()
        require(set(state) == set(expected) and all(state[n].shape == t.shape and state[n].dtype == t.dtype
                and bool(torch.isfinite(state[n]).all()) for n, t in expected.items()),
                "MNLM_TEMPORAL_CHECKPOINT_LAYOUT", "incomplete or incompatible tensor layout")
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False).to("cuda")
        require(state_hash(model) == m["encoder_state_sha256"], "MNLM_TEMPORAL_CHECKPOINT", "cold parameter hash mismatch")
        self.versions = tuple(p._version for p in model.parameters())
        self.manifest = m
        self.contract = {"name": "ucr-temporal-learned-v1", "channel": "history", "perspective": "history", "inputs": "ucr",
                         "kind": "learned", "role": "universal_instrument", "dtype": "f32", "shape": [None, CONFIG.width],
                         "norm": "none", "sim": "maxsim", "cpu_fallback": False, "runtime": self.runtime,
                         "weights_sha256": m["weights_sha256"], "corpus_hash": lineage_hash(m), "checkpoint_id": expected_id,
                         "training_lineage": m["training_lineage"], "execution_code_sha256": execution_code(),
                         "preprocessing": {"input_contract": m["input_contract"], "input_weights_sha256": m["input_weights_sha256"],
                                           "order": "one frozen CUDA forward per observation; all native token rows retained"}}
        self.contract["lens_id"] = digest(self.contract)

    def encode(self, values):
        require(runtime() == self.runtime and all(not m.training for m in self.model.modules())
                and all(not p.requires_grad for p in self.model.parameters())
                and tuple(p._version for p in self.model.parameters()) == self.versions,
                "MNLM_TEMPORAL_RUNTIME_DRIFT", "frozen parameters, precision or inference state changed")
        require(isinstance(values, np.ndarray) and values.dtype == np.float32 and values.ndim == 2
                and values.shape[1] == CONFIG.dims and values.shape[0] <= CONFIG.max_tokens
                and bool(np.isfinite(values).all()), "MNLM_TEMPORAL_INPUT", "finite native float32 history required; never truncate")
        if not len(values):
            return np.empty((0, CONFIG.width), np.float32)
        x = torch.from_numpy(np.array(values, copy=True)).to("cuda")[None]
        with torch.inference_mode():
            result = self.model(x, torch.tensor([len(values)], device="cuda"))[0].cpu().numpy()
        require(np.isfinite(result).all(), "MNLM_TEMPORAL_NONFINITE", "encoder emitted nonfinite values")
        return result

    def encode_record(self, record, context):
        """Source-neutral intake from the existing UCR history channel adapter."""
        slot = measure(record, "history", context)
        encoded = self.encode(slot["tokens"])
        return {**slot, "tokens": encoded, "shape": list(encoded.shape)}


def lineage_hash(manifest):
    return digest(manifest["training_lineage"])
