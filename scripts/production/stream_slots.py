"""Byte-bounded native shard measurement and exact replay (#16, PC-35)."""
import hashlib
import json
import os
from pathlib import Path

import blake3
import numpy as np

from senses.contract import canonical, digest, require
from production.publication import sync_dir
from production.session import batch_digest
from slotio import TokenSpool, save_slot, slot_digest
from slot_layout import check_seal
from learned_ucr import combine
from text_windows import validate_slot


def durable(path, value, limit):
    raw = canonical(value) + b"\n"
    require(len(raw) <= limit, "MNLM_STREAM_METADATA_BUDGET", "metadata exceeds frozen reservation")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    require(not path.exists() and not tmp.exists(), "MNLM_STREAM_IMMUTABLE", str(path))
    try:
        with tmp.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def measure(session, rows, raw, policy, path, plan):
    """One bounded shard. Multi tokens stream; Dense/Sparse retain <= slot quota."""
    size, metadata = plan["slot_budget_bytes"], plan["metadata_budget_bytes"]
    if hasattr(session, "begin_shard"):
        session.begin_shard()
    batch = plan["batch_rows"]
    probe_rows = rows[:batch]
    golden = batch_digest(session.batch(probe_rows, policy))
    spool = TokenSpool(path, session.dim) if session.kind == "multi" else None
    parts, bindings, details, windows, payload = [], [], [], [], 4096
    metadata_bytes = len(canonical(policy)) + 1024
    try:
        for start in range(0, len(rows), batch):
            result = session.batch(rows[start:start + batch], policy)
            arrays, bound, extra = result
            if start == 0:
                require(batch_digest(result) == golden, "MNLM_STREAM_REPLAY", "first batch differs from golden probe")
            payload += sum(a.nbytes for a in arrays.values())
            require(payload <= size, "MNLM_STREAM_SLOT_BUDGET", "native slot exceeds frozen reservation; no truncation")
            bindings.extend(bound)
            details.extend(extra)
            windows.extend(arrays["windows"].tolist())
            # Bound metadata before a byte of this batch enters the spool.
            metadata_bytes += len(canonical([bound, extra])) + 4
            require(metadata_bytes <= metadata,
                    "MNLM_STREAM_METADATA_BUDGET", "view provenance exceeds reservation")
            if spool:
                spool.add(arrays["tokens"], arrays["offsets"])
            else:
                parts.append(arrays)
        require(batch_digest(session.batch(probe_rows, policy)) == golden,
                "MNLM_STREAM_REPLAY", "shard-boundary replay differs; shard not accepted")
        if spool:
            spool.finish(np.asarray(windows, np.int64))
        else:
            save_slot(path, combine(parts, session.kind, session.dim))
    finally:
        if spool:
            spool.fh.close()
            spool.raw.unlink(missing_ok=True)
    require(path.stat().st_size <= size, "MNLM_STREAM_SLOT_BUDGET", "container exceeds reserved bytes")
    native = slot_digest(path, norm=session.contract.get("norm"))
    require(native["n"] == len(rows) and native["dims"] == session.dim, "MNLM_STREAM_SLOT", "native count/dim mismatch")
    provenance = {"generation": session.reader.generation, "policy": policy, "bindings": bindings, "details": details}
    if session.blobs is not None and session.module in ("audio_ucr", "image_ucr", "vision_ucr"):
        provenance["blobs_manifest_sha256"] = session.blobs.sha256
    validate_slot(path, session.contract, provenance)
    durable(path.with_suffix(".views.json"), provenance, metadata)
    seal = {**native, "calibration_blake3": blake3.blake3(raw).hexdigest(),
            "contract_blake3": blake3.blake3(json.dumps(session.contract, sort_keys=True).encode()).hexdigest(),
            "ucr_generation": session.reader.generation, "view_bindings_sha256": digest(bindings),
            "policy_sha256": digest(policy), "window_provenance_sha256": digest(provenance)}
    return {"contract": session.contract, "lens_id": session.lens_id, "seal": seal,
            "replay": {"rows": len(probe_rows), "golden_sha256": golden, "max_abs": 0.0, "byte_identical": True},
            "replay_mode": getattr(session, "replay_mode", "forward"),
            "sha256": file_sha(path), "views_sha256": file_sha(path.with_suffix(".views.json"))}


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_checkpoint(checkpoint, path, plan, key, name, raw):
    require(checkpoint["plan_sha256"] == digest(plan) and checkpoint["shard"] == key and checkpoint["instrument"] == name,
            "MNLM_STREAM_CHECKPOINT", "checkpoint belongs to another input plan")
    slot = checkpoint["slot"]
    require(slot["contract"]["weights_sha256"] == plan["instruments"][name]["weights_sha256"]
            and slot["contract"]["execution_code_sha256"] == plan["code_sha256"]
            and slot["contract"]["environment"] == plan["environment"]
            and slot["contract"]["ucr_generation"] == plan["generation"]
            and slot["contract"]["policies_sha256"] == plan["policies_sha256"],
            "MNLM_STREAM_CHECKPOINT", "checkpoint instrument/source identity drift")
    require(path.is_file() and path.with_suffix(".views.json").is_file(),
            "MNLM_STREAM_CHECKPOINT", "completed checkpoint content is missing; restore exact files")
    require(path.stat().st_size <= plan["instruments"][name]["slot_budget_bytes"]
            and path.with_suffix(".views.json").stat().st_size <= plan["metadata_budget_bytes"]
            and file_sha(path) == slot["sha256"] and file_sha(path.with_suffix(".views.json")) == slot["views_sha256"],
            "MNLM_STREAM_CHECKPOINT", "completed checkpoint bytes changed; restore exact files")
    check_seal(slot_digest(path, norm=slot["contract"].get("norm")), slot["seal"], name)
    require(slot["seal"]["calibration_blake3"] == blake3.blake3(raw).hexdigest()
            and slot["seal"]["contract_blake3"] == blake3.blake3(json.dumps(slot["contract"], sort_keys=True).encode()).hexdigest(),
            "MNLM_STREAM_CHECKPOINT", "checkpoint source/contract binding changed")
    views = json.loads(path.with_suffix(".views.json").read_bytes())
    validate_slot(path, slot["contract"], views)
    require(digest(views) == slot["seal"]["window_provenance_sha256"]
            and digest(views["bindings"]) == slot["seal"]["view_bindings_sha256"]
            and digest(views["policy"]) == slot["seal"]["policy_sha256"]
            and views["generation"] == plan["generation"], "MNLM_STREAM_CHECKPOINT", "provenance changed")
    require(slot["replay"]["byte_identical"] is True and slot["replay"]["max_abs"] == 0,
            "MNLM_STREAM_CHECKPOINT", "exact replay missing")
    return slot
