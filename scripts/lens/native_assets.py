"""Read pinned native candidate assets locally; acquisition is a separate stage."""
import hashlib
import io
import json
from pathlib import Path

from senses.contract import require

PLAN_PIN = "a86b32363076447e2cad9af8cf1fb87284aada1f9efeb1a251a4cef102079356"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_assets(root, plan_path, repo):
    raw = Path(plan_path).read_bytes()
    require(sha(raw) == PLAN_PIN, "MNLM_NATIVE_ASSET_PLAN", "use the pinned candidate acquisition plan")
    plan = json.loads(raw)
    rows = [r for r in plan["models"] if r["repo"] == repo]
    require(len(rows) == 1, "MNLM_NATIVE_ASSET_REPO", "instrument is outside the pinned candidate roster")
    row = rows[0]
    base = Path(root)/repo.replace("/", "--")/row["revision"]
    files = {}
    for name, entry in row["files"].items():
        path = base/name
        require(path.is_file() and path.stat().st_size == entry["bytes"],
                "MNLM_NATIVE_ASSET_SIZE", "acquire the exact original artifact: "+name)
        value = path.read_bytes()
        require(sha(value) == entry["sha256"], "MNLM_NATIVE_ASSET_HASH", "asset drift: "+name)
        files[name] = value
    weights = [n for n, v in row["files"].items() if v["role"] == "original_weight_artifact"]
    require(len(weights) == 1, "MNLM_NATIVE_ASSET_WEIGHTS", "one original weight artifact required")
    name = weights[0]
    if name.endswith(".safetensors"):
        from safetensors.torch import load
        tensors = load(files[name])
    else:
        import torch
        tensors = torch.load(io.BytesIO(files[name]), map_location="cpu", weights_only=True)
    require(isinstance(tensors, dict) and all(isinstance(k, str) for k in tensors),
            "MNLM_NATIVE_ASSET_TENSORS", "expected tensor state dictionary")
    return row, json.loads(files["config.json"]), json.loads(files["preprocessor_config.json"]), tensors


def renamed_key(name, family):
    """Documented Transformers renames only; never alter tensor contents."""
    if family == "siglip-vision":
        return name.removeprefix("vision_model.")
    if family == "vit":
        for old, new in (
            ("encoder.layer.", "layers."),
            ("attention.attention.query.", "attention.q_proj."),
            ("attention.attention.key.", "attention.k_proj."),
            ("attention.attention.value.", "attention.v_proj."),
            ("attention.output.dense.", "attention.o_proj."),
            ("intermediate.dense.", "mlp.fc1."),
            ("output.dense.", "mlp.fc2."),
        ):
            name = name.replace(old, new)
        return name
    require(False, "MNLM_NATIVE_WEIGHT_SCHEMA", "unsupported checkpoint rename family")


def bind(model, tensors, select, exclude, *, rename_family=None):
    """Every original key is selected or explicitly excluded; selected keys match exactly."""
    import torch
    selected, excluded, mapping = {}, {}, {}
    for name, value in tensors.items():
        target = select(name)
        if target is not None:
            require(target not in selected, "MNLM_NATIVE_WEIGHT_COLLISION", name)
            selected[target] = value
            mapping[name] = target
        else:
            require(exclude(name), "MNLM_NATIVE_WEIGHT_UNACCOUNTED", name)
            excluded[name] = list(value.shape)
    expected = model.state_dict()
    schema = "original-selected-keys"
    if set(selected) != set(expected) and rename_family is not None:
        converted = {}
        for name, value in selected.items():
            target = renamed_key(name, rename_family)
            require(target not in converted, "MNLM_NATIVE_WEIGHT_COLLISION", name)
            converted[target] = value
        selected = converted
        mapping = {name: renamed_key(target, rename_family) for name, target in mapping.items()}
        schema = "transformers-" + rename_family + "-renames-v1"
    require(set(selected) == set(expected), "MNLM_NATIVE_WEIGHT_LAYOUT",
            "checkpoint keys differ: missing="+str(sorted(set(expected)-set(selected))[:5])+
            " extra="+str(sorted(set(selected)-set(expected))[:5]))
    require(all(selected[k].shape == v.shape and selected[k].dtype == v.dtype
                for k, v in expected.items()), "MNLM_NATIVE_WEIGHT_SHAPE", "shape or dtype differs")
    model.load_state_dict(selected, strict=True)
    # Read the materialized model, not load_state_dict's return status. Comparing
    # uint8 views also catches signed-zero or other byte changes hidden by float equality.
    loaded = model.state_dict()
    for name, value in selected.items():
        require(torch.equal(loaded[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)),
                "MNLM_NATIVE_WEIGHT_READBACK", "loaded bytes differ: "+name)
    return dict(selected_tensors=len(selected), excluded_tensors=excluded,
                schema=schema, byte_readback=True,
                selected_bytes=sum(v.numel()*v.element_size() for v in selected.values()),
                mapping_sha256=sha(json.dumps(mapping, sort_keys=True).encode()))


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
