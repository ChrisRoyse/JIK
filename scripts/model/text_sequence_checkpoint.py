"""Native identity/world plus exact diagnostic optimizer recovery; never serving."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from claims.codata_source import check, sha
from model import native_world_checkpoint
from model.constellation_learning import frozen_pin, trainable
from model.constellation_training_checkpoint import optimizer_for, validate
from model.loop_training_checkpoint import finite, fingerprint, put
from model.text_sequence_data import FLAGS, PLAN, PURPOSE, code_pins
from production.publication import sync_dir
from senses.contract import canonical


def save(root, model, optimizer, data, *, step, parent, evidence, frozen):
    root = Path(root)
    data.inputs.fresh()
    validate(model, optimizer, SimpleNamespace(plan=PLAN, frozen=frozen, binding=data.bind), step)
    check(not root.exists() and type(step) is int and 0 <= step <= 1520, "TEXT_CHECKPOINT_CURSOR")
    root.mkdir(parents=True)
    pin = native_world_checkpoint.save(root/"model", model,
        tuple(model.start(g["context"]) for g in data.groups), data.bind)
    payload = dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(model.core.initial.device))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    size = sum(p.stat().st_size for p in (root/"model").iterdir() if p.is_file())
    check(size+len(raw) <= model.config.checkpoint_bytes, "TEXT_CHECKPOINT_BUDGET")
    manifest = dict(schema="native-text-diagnostic-update-v1", model=pin, binding=data.bind,
        step=step, parent=parent, evidence=evidence, frozen=frozen, code=code_pins(),
        bytes=len(raw), sha256=sha(raw), fingerprint=fingerprint(model, optimizer),
        names=[n for n, _ in trainable(model)], torch_version=str(torch.__version__), flags=FLAGS)
    put(root/"training.pt", raw)
    raw = canonical(manifest)
    put(root/"manifest.json", raw)
    put(root/"COMMITTED", sha(raw).encode())
    sync_dir(root)
    sync_dir(root.parent)
    return sha(raw)


def load(root, model, data, *, expected, evidence, purpose):
    check(purpose == PURPOSE, "TEXT_SEQUENCE_ADMISSION_REQUIRED")
    root = Path(root)
    data.inputs.fresh()
    check((root/"manifest.json").is_file() and (root/"COMMITTED").is_file(), "TEXT_CHECKPOINT_INCOMPLETE")
    check((root/"manifest.json").stat().st_size <= 1 << 20, "TEXT_CHECKPOINT_BUDGET")
    raw = (root/"manifest.json").read_bytes()
    manifest = json.loads(raw)
    check(sha(raw) == expected == (root/"COMMITTED").read_text() and canonical(manifest) == raw,
          "TEXT_CHECKPOINT_PIN")
    check(manifest["schema"] == "native-text-diagnostic-update-v1" and manifest["binding"] == data.bind
          and manifest["code"] == code_pins() and manifest["evidence"] == evidence
          and manifest["torch_version"] == str(torch.__version__) and manifest["flags"] == FLAGS,
          "TEXT_CHECKPOINT_CONTRACT")
    size = sum(p.stat().st_size for p in (root/"model").iterdir() if p.is_file())
    check(size+manifest["bytes"] <= model.config.checkpoint_bytes
          and (root/"training.pt").stat().st_size == manifest["bytes"], "TEXT_CHECKPOINT_BUDGET")
    raw = (root/"training.pt").read_bytes()
    check(sha(raw) == manifest["sha256"], "TEXT_OPTIMIZER_BYTES")
    check(type(manifest["step"]) is int and 0 <= manifest["step"] <= 1520, "TEXT_CHECKPOINT_CURSOR")
    native_world_checkpoint.load(root/"model", model, expected=manifest["model"], binding=data.bind)
    optimizer = optimizer_for(model, PLAN)
    check(manifest["names"] == [n for n, _ in trainable(model)] and frozen_pin(model) == manifest["frozen"],
          "TEXT_CHECKPOINT_FROZEN")
    payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    check(set(payload) == {"optimizer", "cpu_rng", "cuda_rng"} and finite(payload), "TEXT_OPTIMIZER_STATE")
    optimizer.load_state_dict(payload["optimizer"])
    validate(model, optimizer, SimpleNamespace(plan=PLAN, frozen=manifest["frozen"], binding=data.bind), manifest["step"])
    check(fingerprint(model, optimizer) == manifest["fingerprint"], "TEXT_OPTIMIZER_READBACK")
    torch.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], model.core.initial.device)
    return optimizer, manifest
