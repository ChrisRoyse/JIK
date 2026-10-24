"""Bounded row slicing/packing of native Dense/Sparse/Multi measurements."""
import io
import zipfile
from contextlib import contextmanager

import numpy as np

from senses.contract import digest, require
from slotio import arrays_digest
from slot_layout import members, mapped, validate, unit_norm_error
from learned_ucr import combine
import text_windows


def one(arrays, i):
    out = {k: arrays[k][i:i + 1].copy() for k in ("absent", "windows")}
    if "X" in arrays:
        out["X"] = arrays["X"][i:i + 1].copy()
    else:
        off = "offsets" if "tokens" in arrays else "indptr"
        lo, hi = map(int, arrays[off][i:i + 2])
        out[off] = np.asarray([0, hi - lo], np.int64)
        for k in (("tokens",) if off == "offsets" else ("data", "indices")):
            out[k] = arrays[k][lo:hi].copy()
        if off == "indptr":
            out["shape"] = np.asarray([1, arrays["shape"][1]], np.int64)
    return out


@contextmanager
def mapped_rows(path):
    with zipfile.ZipFile(path) as z:
        _, meta = members(z)
    arrays = mapped(path, meta)
    try:
        yield arrays
    finally:
        for a in arrays.values():
            if getattr(a, "_mmap", None) is not None:
                a._mmap.close()


def joined(rows, kind, dim):
    if kind != "multi":
        return combine(rows, kind, dim)
    return {"tokens": np.concatenate([r["tokens"] for r in rows]),
            "offsets": np.r_[0, np.cumsum([len(r["tokens"]) for r in rows], dtype=np.int64)],
            "absent": np.concatenate([r["absent"] for r in rows]),
            "windows": np.concatenate([r["windows"] for r in rows])}


def value_hash(arrays):
    return digest({"layout": {k: [a.dtype.str, list(a.shape)] for k, a in arrays.items()},
                   "values": arrays_digest(arrays)})


def pack(arrays):
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def unpack(raw, contract, detail, dim):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        kind, _ = members(z)
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = validate(arrays, kind)
    require(meta["n"] == 1 and meta["dims"] == dim and "windows" in arrays,
            "MNLM_CACHE_CORRUPT", "cache row has wrong native dimensions")
    unit_norm_error(arrays, kind, contract.get("norm"))
    if contract.get("repo") and contract.get("modality") == "text":
        require(contract.get("execution_unit") == "atom-local-v1", "MNLM_CACHE_CONTRACT", "text cache requires atom-local execution")
        trace = detail.get("text_windows")
        require(isinstance(trace, dict) and trace.get("prefix") == contract.get("prefix", ""),
                "MNLM_CACHE_CORRUPT", "cached tokenizer prefix changed")
        text_windows.validate(trace, contract["kind"], int(arrays["windows"][0]),
                              native_rows=len(arrays["tokens"]) if kind == "multi" else None)
    return arrays
