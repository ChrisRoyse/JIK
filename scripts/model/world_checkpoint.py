"""Immutable world-loop snapshots preserve separate identity and world-bank pins."""
import io
import json
from pathlib import Path

import torch

from model import loop_checkpoint
from model.loop_training_checkpoint import put, finite
from model.native import NativeInput
from model.world_loop import WorldLoop
from production.publication import sync_dir
from senses.contract import canonical, require


def code_pins():
    root = Path(__file__).parent
    return {n: loop_checkpoint.sha((root/n).read_bytes()) for n in ("world_loop.py", "world_checkpoint.py")}


def save(root, model, states):
    root = Path(root)
    require(not root.exists(), "MNLM_WORLD_CHECKPOINT_EXISTS", "snapshots are immutable")
    model.check_reference()
    require(all(s.world_id == model.world_id for s in states), "MNLM_WORLD_CONTEXT_BANK", "restore matching contexts")
    extra = {k: v for k, v in model.state_dict().items() if not k.startswith("core.")}
    size = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    require(size <= model.config.checkpoint_bytes and finite(extra),
            "MNLM_WORLD_CHECKPOINT_BUDGET", "invalid or oversized snapshot")
    root.mkdir(parents=True)
    core_pin = loop_checkpoint.save(model.core, states, root/"core")
    buf = io.BytesIO()
    torch.save(extra, buf)
    raw = buf.getvalue()
    core_bytes = sum(p.stat().st_size for p in (root/"core").iterdir())
    require(core_bytes+len(raw) <= model.config.checkpoint_bytes,
            "MNLM_WORLD_CHECKPOINT_BUDGET", "serialized snapshot exceeds contract")
    manifest = dict(schema="intrinsic-world-checkpoint-v1", core=core_pin, standard=model.standard_id,
                    world=model.world_id, world_source=model.world_source_id, world_rows=model.reference_rows,
                    payload_sha256=loop_checkpoint.sha(raw), payload_bytes=len(raw), code=code_pins(),
                    scientific_admission=False, useful_capability_gain_demonstrated=False)
    put(root/"world.pt", raw)
    payload = canonical(manifest)
    put(root/"manifest.json", payload)
    pin = loop_checkpoint.sha(payload)
    put(root/"COMMITTED", pin.encode())
    sync_dir(root)
    sync_dir(root.parent)
    return pin


def load(root, *, expected, standard, world, device):
    root = Path(root)
    require((root/"COMMITTED").is_file() and (root/"manifest.json").is_file(),
            "MNLM_WORLD_CHECKPOINT_INCOMPLETE", "restore the committed snapshot")
    require((root/"manifest.json").stat().st_size <= 1 << 20,
            "MNLM_WORLD_CHECKPOINT_BUDGET", "invalid metadata size")
    raw = (root/"manifest.json").read_bytes()
    require(loop_checkpoint.sha(raw) == expected == (root/"COMMITTED").read_text(),
            "MNLM_WORLD_CHECKPOINT_PIN", "supply retained snapshot pin")
    m = json.loads(raw)
    require(m["schema"] == "intrinsic-world-checkpoint-v1" and m["standard"] == standard
            and m["world"] == world and m["code"] == code_pins()
            and m["scientific_admission"] is False and m["useful_capability_gain_demonstrated"] is False,
            "MNLM_WORLD_CHECKPOINT_CONTRACT", "world, identity or code contract changed")
    core, states, _ = loop_checkpoint.load(root/"core", expected_checkpoint=m["core"],
                                          expected_standard=standard, device=device)
    require(m["payload_bytes"] <= core.config.checkpoint_bytes
            and (root/"world.pt").stat().st_size == m["payload_bytes"],
            "MNLM_WORLD_CHECKPOINT_BUDGET", "payload size differs")
    payload = (root/"world.pt").read_bytes()
    require(loop_checkpoint.sha(payload) == m["payload_sha256"],
            "MNLM_WORLD_CHECKPOINT_BYTES", "restore original world bytes")
    extra = torch.load(io.BytesIO(payload), weights_only=True, map_location=device)
    refs = {s.name: NativeInput(s.identity, tuple(m["world_rows"][s.name]), extra[f"world_values_{i}"],
                                extra[f"world_padding_{i}"], extra[f"world_absent_{i}"])
            for i, s in enumerate(core.specs)}
    model = WorldLoop(core, refs, world_source_id=m["world_source"]).to(device)
    require(model.world_id == world and finite(extra), "MNLM_WORLD_REFERENCE_DRIFT", "restore original world bank")
    combined = {**{"core."+k: v for k, v in core.state_dict().items()}, **extra}
    model.load_state_dict(combined, strict=True)
    model.check_reference()
    return model, tuple(model.bind_state(s) for s in states), m
