"""Exact sensory-dynamics AdamW continuation with an immutable identity standard."""
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from model import native_world_checkpoint
from model.constellation_learning import trainable
from model.loop_training_checkpoint import finite, fingerprint, put, validate_optimizer
from model.text_dynamics import PLAN, code_pins, frozen_pin, optimizer_for
from model.text_sequence_data import FLAGS, PURPOSE
from production.publication import sync_dir
from senses.contract import canonical


def validate(model, optimizer, step, frozen):
    check(type(step) is int and 0 <= step <= PLAN["steps"], "DYNAMICS_CURSOR")
    named = trainable(model)
    check([n for n, _ in named] == [n for n, _ in model.named_parameters()
          if not n.startswith("core.identity.")], "DYNAMICS_PARAMETER_SCOPE")
    # The common validator checks parameter identities and exact AdamW state.
    class Parameters:
        def parameters(self):
            return iter(p for _, p in named)
    validate_optimizer(Parameters(), optimizer, PLAN, step)
    check(len(optimizer.state) == (len(named) if step else 0), "DYNAMICS_OPTIMIZER_COVERAGE")
    model.check_reference()
    check(frozen_pin(model) == frozen, "DYNAMICS_FROZEN_DRIFT")


def save(root, model, optimizer, data, *, step, parent, evidence, frozen):
    root = Path(root)
    data.inputs.fresh()
    validate(model, optimizer, step, frozen)
    check(not root.exists() and data.evidence == evidence["examples"], "DYNAMICS_SAVE_EVIDENCE")
    root.mkdir(parents=True)
    pin = native_world_checkpoint.save(root/"model", model,
        tuple(model.start(g["context"]) for g in data.groups), data.bind)
    payload = dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(model.core.initial.device))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    size = sum(p.stat().st_size for p in (root/"model").iterdir() if p.is_file())
    check(size+len(raw) <= model.config.checkpoint_bytes, "DYNAMICS_CHECKPOINT_BUDGET")
    manifest = dict(schema="native-text-dynamics-update-v1", model=pin, binding=data.bind,
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
    check(purpose == PURPOSE, "DYNAMICS_ADMISSION_REQUIRED")
    root = Path(root)
    data.inputs.fresh()
    check((root/"manifest.json").is_file() and (root/"COMMITTED").is_file(), "DYNAMICS_CHECKPOINT_INCOMPLETE")
    check((root/"manifest.json").stat().st_size <= 1 << 20, "DYNAMICS_CHECKPOINT_BUDGET")
    raw = (root/"manifest.json").read_bytes()
    m = json.loads(raw)
    check(sha(raw) == expected == (root/"COMMITTED").read_text() and canonical(m) == raw,
          "DYNAMICS_CHECKPOINT_PIN")
    check(m["schema"] == "native-text-dynamics-update-v1" and m["binding"] == data.bind
          and m["code"] == code_pins() and m["evidence"] == evidence
          and evidence["examples"] == data.evidence and m["torch_version"] == str(torch.__version__)
          and m["flags"] == FLAGS, "DYNAMICS_CHECKPOINT_CONTRACT")
    size = sum(p.stat().st_size for p in (root/"model").iterdir() if p.is_file())
    check(type(m["bytes"]) is int and m["bytes"] > 0 and size+m["bytes"] <= model.config.checkpoint_bytes
          and (root/"training.pt").stat().st_size == m["bytes"], "DYNAMICS_CHECKPOINT_BUDGET")
    raw = (root/"training.pt").read_bytes()
    check(sha(raw) == m["sha256"], "DYNAMICS_OPTIMIZER_BYTES")
    native_world_checkpoint.load(root/"model", model, expected=m["model"], binding=data.bind)
    optimizer = optimizer_for(model)
    check(m["names"] == [n for n, _ in trainable(model)], "DYNAMICS_PARAMETER_ORDER")
    payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    check(set(payload) == {"optimizer", "cpu_rng", "cuda_rng"} and finite(payload), "DYNAMICS_OPTIMIZER_STATE")
    optimizer.load_state_dict(payload["optimizer"])
    validate(model, optimizer, m["step"], m["frozen"])
    check(fingerprint(model, optimizer) == m["fingerprint"], "DYNAMICS_OPTIMIZER_READBACK")
    torch.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], model.core.initial.device)
    return optimizer, m
