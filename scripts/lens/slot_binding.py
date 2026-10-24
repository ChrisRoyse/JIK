"""A measurement cache binds lens identity, calibration bytes and slot digest.

Legacy files have no such binding: they are remeasured, never silently relabelled.
Context readers are remeasured until their external dependencies have manifests.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import blake3
import numpy as np

from lenscommon import LensError, write_json
from slotio import slot_digest
from slot_layout import check_seal, SCHEMA


def view_binding(path, contract):
    if not contract.get("segmentation"):
        return {}
    from text_windows import validate_slot
    view = path.with_suffix(".views.json")
    if not view.is_file():
        raise LensError("MNLM_LENS_WINDOW_PROVENANCE", "segmented slot has no retained window provenance")
    raw = view.read_bytes()
    validate_slot(path, contract, json.loads(raw))
    return {"views_sha256": hashlib.sha256(raw).hexdigest()}


def current(path, contract):
    cal = path.parent.parent.parent / "cal" / f"{path.parent.name}.jsonl"
    return {"calibration_blake3": blake3.blake3(cal.read_bytes()).hexdigest(),
            "contract_blake3": blake3.blake3(json.dumps(contract, sort_keys=True).encode()).hexdigest()}


def presence_digest(path):
    with np.load(path, allow_pickle=False) as z:
        absent = z["absent"]
    if absent.dtype != np.bool_ or absent.ndim != 1:
        raise LensError("MNLM_SLOT_LAYOUT", f"{path}: absent must be a boolean vector")
    return blake3.blake3(absent.tobytes()).hexdigest()


def reuse(path, contract):
    binding_path = path.with_suffix(".binding.json")
    if contract.get("inputs") == "context" or not binding_path.exists():
        return None
    old = json.loads(binding_path.read_text(encoding="utf-8"))
    expected = current(path, contract)
    if any(old.get(k) != v for k, v in expected.items()):
        return None
    if old.get("layout_schema") != SCHEMA:
        raise LensError("MNLM_SLOT_LAYOUT_UNSEALED", f"{path}: value-only historical cache; use a new measurement generation")
    measured = slot_digest(path, norm=contract.get("norm"))
    check_seal(measured, old, path)
    views = view_binding(path, contract)
    if any(old.get(k) != v for k, v in views.items()):
        raise LensError("MNLM_SLOT_STALE", f"{path}: retained tokenizer windows changed")
    return {"slot": str(path), "reused": True, **measured, **expected, "absent_blake3": presence_digest(path)}


def seal(path, contract, record):
    measured = slot_digest(path, norm=contract.get("norm"))
    if any(measured[k] != record[k] for k in ("digest", "n", "dims", "absent")):
        raise LensError("MNLM_SLOT_STALE", f"{path}: writer output differs from cold native readback")
    record.update({k: measured[k] for k in ("layout_schema", "layout_digest", "kind")})
    record.update(current(path, contract))
    record["absent_blake3"] = presence_digest(path)
    views = view_binding(path, contract)
    record.update(views)
    write_json(path.with_suffix(".binding.json"), {**{k: record[k] for k in (
        "digest", "layout_schema", "layout_digest", "n", "dims", "absent", "calibration_blake3", "contract_blake3")}, **views})
