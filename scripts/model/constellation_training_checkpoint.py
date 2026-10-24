"""Exact readout AdamW recovery tied to an eligible scientific-ledger prefix."""
import io
import json
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from model import world_checkpoint
from model.constellation_learning import code_pins, configure, trainable, frozen_pin
from model.constellation_world import ConstellationWorld
from model.loop_training_checkpoint import put, finite, fingerprint
from production.publication import sync_dir


def optimizer_for(model, plan):
    configure(model)
    return torch.optim.AdamW([p for _, p in trainable(model)], lr=plan["learning_rate"],
                             weight_decay=plan["weight_decay"], foreach=False, fused=False)


def optimizer_steps(data, step):
    offset = getattr(data, "optimizer_offset", 0)
    check(type(offset) is int and offset >= 0
          and offset == data.binding.get("continuation", {}).get("optimizer_steps", 0),
          "LEARNING_CURSOR")
    return offset+step


def validate(model, optimizer, data, step):
    check(type(step) is int and 0 <= step <= data.plan["steps"], "LEARNING_CURSOR")
    cumulative = optimizer_steps(data, step)
    check(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
          "LEARNING_OPTIMIZER")
    names = [n for n, _ in trainable(model)]
    check(names == [n for n, _ in model.named_parameters() if n.startswith(("query.", "place."))],
          "LEARNING_PARAMETER_SCOPE")
    group = optimizer.param_groups[0]
    params = [p for _, p in trainable(model)]
    check(len(params) == len(group["params"]) and all(a is b for a, b in zip(params, group["params"])),
          "LEARNING_PARAMETER_ORDER")
    expected = dict(lr=data.plan["learning_rate"], weight_decay=data.plan["weight_decay"],
        betas=(.9, .999), eps=1e-8, amsgrad=False, maximize=False, foreach=False,
        capturable=False, differentiable=False, fused=False)
    check(all(group[k] == v for k, v in expected.items()), "LEARNING_OPTIMIZER")
    check(len(optimizer.state) == (len(params) if cumulative else 0), "LEARNING_OPTIMIZER_COVERAGE")
    for p, state in optimizer.state.items():
        check(any(p is q for q in params) and set(state) == {"step", "exp_avg", "exp_avg_sq"}
              and float(state["step"]) == cumulative, "LEARNING_CURSOR")
        check(all(state[k].shape == p.shape and state[k].dtype == p.dtype and state[k].device == p.device
                  for k in ("exp_avg", "exp_avg_sq")) and finite(state), "LEARNING_OPTIMIZER_STATE")
    model.check_reference()
    check(frozen_pin(model) == data.frozen, "LEARNING_FROZEN_DRIFT")


def save(root, model, optimizer, data, *, step, parent):
    root = Path(root)
    data.fresh()
    validate(model, optimizer, data, step)
    check(not root.exists(), "LEARNING_CHECKPOINT_EXISTS")
    root.mkdir(parents=True)
    pin = world_checkpoint.save(root/"model", model,
        tuple(model.start(c) for c in sorted(data.session.states)))
    payload = dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(model.core.initial.device))
    buf = io.BytesIO()
    torch.save(payload, buf)
    raw = buf.getvalue()
    size = sum(p.stat().st_size for p in (root/"model").rglob("*") if p.is_file())
    check(size+len(raw) <= model.config.checkpoint_bytes, "LEARNING_CHECKPOINT_BUDGET")
    m = dict(schema="constellation-readout-learning-v2", model=pin, binding=data.binding,
        optimizer_steps=optimizer_steps(data, step),
        frozen=data.frozen, step=step, parent=parent, names=[n for n, _ in trainable(model)],
        bytes=len(raw), sha256=sha(raw), fingerprint=fingerprint(model, optimizer),
        torch_version=str(torch.__version__), code=code_pins(), serving=False)
    put(root/"training.pt", raw)
    raw = canonical(m)
    put(root/"manifest.json", raw)
    put(root/"COMMITTED", sha(raw).encode())
    sync_dir(root)
    sync_dir(root.parent)
    return sha(raw)


def load(root, data, *, expected):
    root = Path(root)
    data.fresh()
    check((root/"manifest.json").is_file() and (root/"COMMITTED").is_file(), "LEARNING_CHECKPOINT_INCOMPLETE")
    check((root/"manifest.json").stat().st_size <= 1 << 20, "LEARNING_CHECKPOINT_BUDGET")
    raw = (root/"manifest.json").read_bytes()
    m = json.loads(raw)
    check(sha(raw) == expected == (root/"COMMITTED").read_text() and raw == canonical(m),
          "LEARNING_CHECKPOINT_PIN")
    check(m["schema"] == "constellation-readout-learning-v2" and m["binding"] == data.binding
          and m["frozen"] == data.frozen and m["code"] == code_pins()
          and m["torch_version"] == str(torch.__version__) and m["serving"] is False,
          "LEARNING_CHECKPOINT_CONTRACT")
    check(type(m.get("optimizer_steps")) is int
          and m["optimizer_steps"] == optimizer_steps(data, m["step"]), "LEARNING_CURSOR")
    model, _, _ = world_checkpoint.load(root/"model", expected=m["model"],
        standard=data.model.standard_id, world=data.model.world_id, device=data.session.device)
    model = ConstellationWorld.from_world(model)
    optimizer = optimizer_for(model, data.plan)
    check(m["names"] == [n for n, _ in trainable(model)], "LEARNING_PARAMETER_ORDER")
    size = sum(p.stat().st_size for p in (root/"model").rglob("*") if p.is_file())
    check(type(m["bytes"]) is int and 0 < m["bytes"]+size <= model.config.checkpoint_bytes
          and (root/"training.pt").stat().st_size == m["bytes"], "LEARNING_CHECKPOINT_BUDGET")
    raw = (root/"training.pt").read_bytes()
    check(sha(raw) == m["sha256"], "LEARNING_OPTIMIZER_PIN")
    payload = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    check(set(payload) == {"optimizer", "cpu_rng", "cuda_rng"} and finite(payload["optimizer"]),
          "LEARNING_OPTIMIZER_STATE")
    optimizer.load_state_dict(payload["optimizer"])
    validate(model, optimizer, data, m["step"])
    check(fingerprint(model, optimizer) == m["fingerprint"], "LEARNING_CHECKPOINT_READBACK")
    torch.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], model.core.initial.device)
    return model, optimizer, m
