"""Exact native-panel states for the #12 CMI commissioning experiment.

This serializes typed slots; it never pools tokens, projects vectors, rounds
values or treats an absent slot as a zero vector. Full bytes are equality keys.
Offsets locate a row's data; absolute corpus offsets are not extra variables.
One pinned file descriptor is checked before and after materialization. Retain
the original slot files and row/unit mapping with the experiment's evidence.
O(native bytes + typed key bytes), with one file mapped at a time. CPU I/O only.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from finite_cmi import canonical, require
import slot_layout


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def member(a):
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "bytes": base64.b64encode(a.tobytes(order="C")).decode("ascii")}


def native_states(descriptors, expected_ids, n):
    require(type(n) is int and n > 0 and isinstance(descriptors, list) and isinstance(expected_ids, list),
            "MNLM_CMI_NATIVE_PLAN", "declare complete ordered panel IDs and row count")
    require(all(isinstance(d, dict) and set(d) == {"id", "path", "sha256"} for d in descriptors),
            "MNLM_CMI_NATIVE_PLAN", "each slot needs its frozen ID, path and physical SHA256")
    ids = [d["id"] for d in descriptors]
    require(all(isinstance(i, str) and bool(i) for i in ids) and
            ids == expected_ids and len(set(ids)) == len(ids),
            "MNLM_CMI_PANEL_INCOMPLETE", "missing, extra, repeated or reordered conditioning slot")
    rows = [[] for _ in range(n)]
    for d in descriptors:
        arrays = {}
        with Path(d["path"]).open("rb") as fh:
            before = hashlib.file_digest(fh, "sha256").hexdigest()
            require(before == d["sha256"], "MNLM_CMI_NATIVE_CHANGED", "slot bytes differ from frozen input")
            fh.seek(0)
            try:
                with zipfile.ZipFile(fh) as zf:
                    kind, meta = slot_layout.members(zf)
                    arrays = slot_layout.mapped(fh, meta)
                    info = slot_layout.validate(arrays, kind)
                    require(info["n"] == n, "MNLM_CMI_NATIVE_ROWS", "slot and unit manifest have different rows")
                    for i in range(n):
                        parts = {"absent": member(arrays["absent"][i:i + 1])}
                        if "windows" in arrays:
                            parts["windows"] = member(arrays["windows"][i:i + 1])
                        if kind == "dense":
                            parts["X"] = member(arrays["X"][i])
                        else:
                            offsets = arrays["offsets" if kind == "multi" else "indptr"]
                            start, end = int(offsets[i]), int(offsets[i + 1])
                            key = "tokens" if kind == "multi" else "data"
                            parts[key] = member(arrays[key][start:end])
                            if kind == "sparse":
                                parts["indices"] = member(arrays["indices"][start:end])
                                parts["vocabulary_extent"] = member(arrays["shape"][1:2])
                        rows[i].append({"id": d["id"], "kind": kind, "members": parts})
            finally:
                for a in arrays.values():
                    if isinstance(a, np.memmap):
                        a._mmap.close()
            fh.seek(0)
            require(hashlib.file_digest(fh, "sha256").hexdigest() == before,
                    "MNLM_CMI_NATIVE_CHANGED", "slot changed during exact-state readback; rerun from immutable inputs")
    return [canonical(r) for r in rows]
