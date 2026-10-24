"""Read a bounded, explicit row selection from a frozen temporal native publication."""
from pathlib import Path

from senses.contract import require, digest
from senses.pipeline import file_hash
from temporal_slots import load
from production.native_rows import mapped_rows, one, joined, value_hash
from model.contracts import Instrument
from model.native import from_arrays


def read_temporal(root, expected_manifest_sha256, row_ids, config):
    require(row_ids and len(set(row_ids)) == len(row_ids), "MNLM_MODEL_ROWS", "unique explicit temporal row selection required")
    manifest, rows = load(root, expected_manifest_sha256)
    index = {r["atom"]: i for i, r in enumerate(rows)}
    require(len(index) == len(rows) and all(k in index for k in row_ids),
            "MNLM_MODEL_ROWS", "missing/duplicate observation in temporal publication")
    spec = Instrument.from_slot(manifest["contract"]["name"], {"contract": manifest["contract"], "seal": manifest["slot"]})
    with mapped_rows(root / "history.npz") as arrays:
        positions = [index[k] for k in row_ids]
        width = max(1, max(int(arrays["offsets"][i+1] - arrays["offsets"][i]) for i in positions))
        require(width <= config.max_native_tokens and len(positions) * width * spec.dims <= config.max_native_elements,
                "MNLM_MODEL_INPUT_BUDGET", "selected temporal payload exceeds bounded batch; never truncate")
        native = joined([one(arrays, i) for i in positions], "multi", spec.dims)
    result = from_arrays(spec, native, row_ids, config)
    receipt = {"schema": "temporal-model-input-v1", "manifest_sha256": expected_manifest_sha256,
               "reader_code_sha256": file_hash(Path(__file__)), "instrument": spec.identity,
               "selected_rows_sha256": digest([rows[i] for i in positions]), "row_ids": list(row_ids),
               "native_value_sha256": value_hash(native), "stage4_accepted": False}
    return spec, result, receipt
