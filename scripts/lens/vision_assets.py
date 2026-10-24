"""Bounded, pinned DINOv2 safetensors acquisition; no remote executable code.

All four assets are checked on acquisition AND every model load. Runtime loading
never downloads. The SHA-256 pins come from revision-bound upstream bytes (#15).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REPO = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
FILES = {
    "model.safetensors": (88249960, "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"),
    "config.json": (547, "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"),
    "preprocessor_config.json": (436, "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
    "README.md": (3033, "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c"),
}


def check(path, size, expected):
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if path.stat().st_size != size or actual != expected:
        raise ValueError(f"MNLM_VISION_ASSET_DRIFT: {path.name}")
    return actual


def snapshot():
    # Exact revision, independent of mutable refs/main. Bulk data has no C: default.
    root = os.environ.get("HF_HUB_CACHE")
    if not root:
        raise ValueError("MNLM_VISION_CACHE_REQUIRED: set HF_HUB_CACHE explicitly")
    return Path(root) / "models--facebook--dinov2-small" / "snapshots" / REVISION


def validate(path):
    return {name: check(path / name, *pin) for name, pin in FILES.items()}


def read_validated(path):
    """Hash and load the SAME bounded bytes, once per resident; no path-swap gap."""
    result = {}
    for name, (size, expected) in FILES.items():
        with (path / name).open("rb") as f:
            raw = f.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"MNLM_VISION_ASSET_DRIFT: {name}")
        result[name] = raw
    return result


def acquire(root):
    path = Path(root).resolve() / "models--facebook--dinov2-small" / "snapshots" / REVISION
    path.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        target = path / name
        if target.exists():
            check(target, size, expected)
            continue
        temp = path / (name + ".partial")
        if temp.exists():
            check(temp, size, expected)
            temp.rename(target)
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("xb") as f:
            count = 0
            while chunk := response.read(min(1 << 20, size - count + 1)):
                count += len(chunk)
                if count > size:
                    raise ValueError("MNLM_VISION_ASSET_SIZE: download exceeds pin")
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        check(temp, size, expected)
        # Single acquisition process; existing destinations are never repaired silently.
        if target.exists():
            raise FileExistsError(target)
        temp.rename(target)
    hashes = validate(path)
    receipt = {"repo": REPO, "revision": REVISION, "files": hashes,
               "bytes": sum(s for s, _ in FILES.values()), "training_overlap": "unassessed",
               "license": "Apache-2.0 (upstream model card)", "path": str(path)}
    (path / "acquisition.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    print(json.dumps(acquire(ap.parse_args().cache)))
