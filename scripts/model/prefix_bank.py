"""Bind separate prefix instruments and learned windows to a native model batch.

Selections are explicit observation identities; no source name enters the model.
Native rows remain intact, and the existing model allocation limits stay binding.
"""
from pathlib import Path

from senses.contract import require, digest
from senses.pipeline import file_hash
from production.native_rows import mapped_rows, one, joined, value_hash
from model.contracts import Instrument
from model.native import from_arrays
from prefix_panel import load as load_panel
from windowed_temporal import load as load_windowed


def select(path, spec, rows, row_ids, config):
    require(row_ids and len(row_ids) == len(set(row_ids)), "MNLM_MODEL_ROWS", "unique explicit observation selection required")
    index = {r["atom"]: i for i, r in enumerate(rows)}
    require(len(index) == len(rows) and all(k in index for k in row_ids), "MNLM_MODEL_ROWS", "missing/duplicate observation")
    positions = [index[k] for k in row_ids]
    with mapped_rows(path) as arrays:
        width = max(1, max(int(arrays["offsets"][i + 1] - arrays["offsets"][i]) for i in positions))
        require(width <= config.max_native_tokens and len(positions) * width * spec.dims <= config.max_native_elements,
                "MNLM_MODEL_INPUT_BUDGET", "selected native payload exceeds fixed budget; never truncate")
        native = joined([one(arrays, i) for i in positions], "multi", spec.dims)
    return from_arrays(spec, native, row_ids, config), {
        "instrument": spec.identity, "selected_rows_sha256": digest([rows[i] for i in positions]),
        "native_value_sha256": value_hash(native), "tokens": len(native["tokens"])}


def read_bank(root, expected_sha256, row_ids, channels, config):
    require(channels and len(channels) == len(set(channels)), "MNLM_MODEL_CHANNEL", "unique explicit channel selection required")
    m, rows = load_panel(root, expected_sha256)
    require(all(c in m["slots"] for c in channels), "MNLM_MODEL_CHANNEL", "unknown native channel")
    specs, inputs, receipts = [], {}, {}
    for c in channels:
        slot = m["slots"][c]
        spec = Instrument.from_slot(slot["contract"]["name"], slot)
        native, receipt = select(root / (c + ".npz"), spec, rows, row_ids, config)
        specs.append(spec)
        inputs[spec.name], receipts[spec.name] = native, receipt
    require(sum(x.values.numel() for x in inputs.values()) <= config.max_native_elements,
            "MNLM_MODEL_INPUT_BUDGET", "combined native batch exceeds fixed budget")
    return tuple(specs), inputs, {"schema": "prefix-bank-model-input-v1", "manifest_sha256": expected_sha256,
        "row_ids": list(row_ids), "reader_code_sha256": file_hash(Path(__file__)), "slots": receipts,
        "stage4_accepted": False}


def read_windowed(root, expected_sha256, expected_source_sha256, row_ids, config):
    m, rows = load_windowed(root, expected_sha256)
    require(m["source_panel_manifest_sha256"] == expected_source_sha256,
            "MNLM_MODEL_INPUT_BINDING", "learned history belongs to a different source publication")
    spec = Instrument.from_slot(m["contract"]["name"], {"contract": m["contract"], "seal": m["slot"]})
    native, receipt = select(root / "history.npz", spec, rows, row_ids, config)
    return spec, native, {"schema": "windowed-model-input-v1", "manifest_sha256": expected_sha256,
        "source_panel_manifest_sha256": expected_source_sha256, "row_ids": list(row_ids), "slot": receipt,
        "reader_code_sha256": file_hash(Path(__file__)), "stage4_accepted": False}
