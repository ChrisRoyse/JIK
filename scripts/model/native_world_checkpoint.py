"""Immutable complete sensory/identity model snapshots with external source pins."""
import io
import json
from dataclasses import asdict
from pathlib import Path

import torch

from model.loop_training_checkpoint import put, finite
from model.native_identity_world import SensoryState
from production.publication import sync_dir
from senses.contract import canonical, digest, require
from senses.perception_corpus import sha


def code():
    root = Path(__file__).parent
    names = ("native_world_checkpoint.py", "native_identity_world.py", "identity_loop.py",
             "constellation_world.py", "world_loop.py", "window_maxsim.py", "stream_attention.py",
             "attention.py", "native.py", "constellation.py", "contracts.py", "timed_world_input.py",
             "event_temporal_input.py", "episode_input.py", "text_episode_input.py")
    return {n: sha(root/n) for n in names}


def save(root, model, states, binding):
    root = Path(root)
    require(not root.exists(), "MNLM_NATIVE_WORLD_EXISTS", "snapshots are immutable")
    model.check_reference()
    require(states and len({s.context for s in states}) == len(states)
            and all(isinstance(s, SensoryState) and s.world_id == model.world_id
                    and s.standard_id == model.standard_id and type(s.sequence) is int and s.sequence >= 0
                    and s.hidden.shape == model.core.initial.shape
                    and s.identity_hidden is not None
                    and s.identity_hidden.shape == model.core.identity.initial.shape for s in states),
            "MNLM_NATIVE_WORLD_CONTEXT", "matching contexts required")
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "world": torch.stack([s.hidden.detach().cpu() for s in states]),
               "identity": torch.stack([s.identity_hidden.detach().cpu() for s in states])}
    require(finite(payload) and sum(t.numel()*t.element_size() for t in payload["model"].values())
            + payload["world"].numel()*4 + payload["identity"].numel()*4 <= model.config.checkpoint_bytes,
            "MNLM_NATIVE_WORLD_BUDGET", "snapshot exceeds fixed budget or is nonfinite")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    require(len(raw) <= model.config.checkpoint_bytes, "MNLM_NATIVE_WORLD_BUDGET",
            "serialized snapshot exceeds fixed budget")
    import hashlib
    manifest = dict(schema="native-identity-world-v2", binding=binding, standard=model.standard_id,
        temporal=json.loads(model.core._temporal), temporal_id=model.core.temporal_id,
        world=model.world_id, code=code(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
        config=asdict(model.config), instruments=[asdict(s) for s in model.specs],
        world_rows=model.reference_rows, identity_rows=model.core.identity.reference_rows,
        identity_instruments=[asdict(s) for s in model.core.identity.specs],
        identity_source=model.core.identity.source_id,
        contexts=[dict(context=s.context, sequence=s.sequence, last_event=s.last_event) for s in states],
        scientific_admission=False, serving=False, learned_understanding=False)
    root.mkdir(parents=True)
    put(root/"state.pt", raw)
    put(root/"manifest.json", canonical(manifest))
    pin = digest(manifest)
    put(root/"COMMITTED", pin.encode())
    sync_dir(root)
    sync_dir(root.parent)
    return pin


def load(root, model, *, expected, binding):
    root = Path(root)
    require((root/"COMMITTED").is_file() and (root/"manifest.json").is_file(),
            "MNLM_NATIVE_WORLD_INCOMPLETE", "restore complete snapshot")
    require((root/"manifest.json").stat().st_size <= 1 << 20, "MNLM_NATIVE_WORLD_BUDGET",
            "invalid manifest size")
    m = json.loads((root/"manifest.json").read_bytes())
    require(digest(m) == expected == (root/"COMMITTED").read_text(),
            "MNLM_NATIVE_WORLD_PIN", "supply retained snapshot pin")
    require(m["schema"] == "native-identity-world-v2" and m["code"] == code()
            and m["temporal_id"] == model.core.temporal_id
            and m["temporal"] == json.loads(model.core._temporal)
            and digest(m["temporal"]) == m["temporal_id"]
            and m["binding"] == binding and m["standard"] == model.standard_id
            and m["world"] == model.world_id
            and m["config"] == asdict(model.config)
            and m["instruments"] == [asdict(s) for s in model.specs]
            and m["identity_instruments"] == [asdict(s) for s in model.core.identity.specs]
            and m["identity_source"] == model.core.identity.source_id
            and m["world_rows"] == {n: list(rows) for n, rows in model.reference_rows.items()}
            and m["identity_rows"] == {n: list(rows) for n, rows in model.core.identity.reference_rows.items()}
            and all(m[k] is False for k in ("scientific_admission", "serving", "learned_understanding")),
            "MNLM_NATIVE_WORLD_CONTRACT", "source/standard/model contract changed")
    require(m["bytes"] <= model.config.checkpoint_bytes and (root/"state.pt").stat().st_size == m["bytes"]
            and sha(root/"state.pt") == m["sha256"], "MNLM_NATIVE_WORLD_BYTES", "restore original bytes")
    payload = torch.load(root/"state.pt", map_location=model.core.initial.device, weights_only=True)
    require(finite(payload), "MNLM_NATIVE_WORLD_BYTES", "nonfinite snapshot")
    model.load_state_dict(payload["model"], strict=True)
    model.check_reference()
    states = tuple(SensoryState(s["context"], model.standard_id, s["sequence"], payload["world"][i],
        s["last_event"], model.world_id, payload["identity"][i]) for i, s in enumerate(m["contexts"]))
    require(len(states) == len({s.context for s in states}) == payload["world"].shape[0]
            == payload["identity"].shape[0] and all(type(s.sequence) is int and s.sequence >= 0
            and s.hidden.shape == model.core.initial.shape
            and s.identity_hidden.shape == model.core.identity.initial.shape for s in states),
            "MNLM_NATIVE_WORLD_CONTEXT", "invalid context roster or shapes")
    return states
