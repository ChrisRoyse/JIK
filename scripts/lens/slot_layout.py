"""Native slot layout checks, bounded independently of the payload (PC-04/07).

The legacy digest deliberately remains comparable to old evidence. The additional
seal binds the meaning of those bytes: member names, shapes, dtypes and C order.
No layout-only seal is trusted without the value digest and native validation.
"""
from __future__ import annotations

import json
import math
import zipfile

import blake3
import numpy as np
from numpy.lib import format as npformat

from lenscommon import LensError

SCHEMA = "slot-layout-v1"
CHUNK = 1 << 26
KINDS = {"dense": {"X", "absent"}, "multi": {"tokens", "offsets", "absent"},
         "sparse": {"data", "indices", "indptr", "shape", "absent"}}


def require(ok, message, code="MNLM_SLOT_LAYOUT"):
    if not ok:
        raise LensError(code, message)


def header(fh):
    version = npformat.read_magic(fh)
    require(version in ((1, 0), (2, 0)), f"unsupported npy version {version}")
    read = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
    shape, fortran, dtype = read(fh)
    require(not fortran and not dtype.hasobject and dtype.fields is None,
            "slots require C-order primitive numeric arrays")
    require(all(type(d) is int and d >= 0 for d in shape), "invalid array extent")
    return shape, dtype


def members(zf):
    """Read headers once; refuse duplicate/extra names and ambiguous ZIP layouts."""
    infos = zf.infolist()
    names = [i.filename for i in infos]
    require(len(names) == len(set(names)), "duplicate ZIP member")
    keys = {n[:-4] for n in names if n.endswith(".npy")}
    require(len(keys) == len(names), "unexpected non-array member")
    kind = next((k for k, required in KINDS.items() if keys in (required, required | {"windows"})), None)
    require(kind is not None, f"unknown slot members {sorted(keys)}")
    result = {}
    for info in sorted(infos, key=lambda x: x.filename):
        require(info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 1,
                "slots must be uncompressed and unencrypted")
        # Read through ZipFile so local and central names are checked by Python.
        with zf.open(info) as fh:
            shape, dtype = header(fh)
            header_bytes = fh.tell()
        require(header_bytes + math.prod(shape) * dtype.itemsize == info.file_size,
                f"{info.filename}: array size differs from member size", "MNLM_SLOT_TORN")
        zf.fp.seek(info.header_offset)
        local = zf.fp.read(30)
        require(len(local) == 30 and local[:4] == b"PK\x03\x04", "invalid local ZIP header")
        require(int.from_bytes(local[8:10], "little") == zipfile.ZIP_STORED,
                "local compression differs from the accepted layout")
        start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
        require(start + info.file_size <= zf.start_dir, "member overlaps central directory", "MNLM_SLOT_TORN")
        result[info.filename[:-4]] = {"shape": shape, "dtype": dtype,
                                     "offset": start + header_bytes, "info": info}
    return kind, result


def layout_digest(meta, value_digest):
    descriptor = {k: {"shape": list(v["shape"]), "dtype": v["dtype"].str, "order": "C"}
                  for k, v in sorted(meta.items())}
    body = {"schema": SCHEMA, "members": descriptor, "value_digest": value_digest}
    return blake3.blake3(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mapped(file, meta):
    return {k: (np.memmap(file, dtype=v["dtype"], mode="r", offset=v["offset"], shape=v["shape"])
                if math.prod(v["shape"]) else np.empty(v["shape"], dtype=v["dtype"]))
            for k, v in meta.items()}


def chunks(a):
    flat = a.reshape(-1)
    size = max(1, CHUNK // max(a.dtype.itemsize, 1))
    for i in range(0, flat.size, size):
        yield flat[i:i + size]


def monotone(a):
    size = CHUNK // a.dtype.itemsize
    return all(np.all(a[i + 1:min(len(a), i + size + 1)] >= a[i:min(len(a) - 1, i + size)])
               for i in range(0, len(a) - 1, size))


def validate(arrays, kind):
    """Read the complete persisted value boundaries without materializing a slot."""
    a = arrays["absent"]
    require(a.ndim == 1 and a.dtype == np.bool_, "absence must be a boolean vector")
    require(all(np.all(c.view(np.uint8) <= 1) for c in chunks(a)), "noncanonical absence byte")
    n = len(a)
    if "windows" in arrays:
        w = arrays["windows"]
        require(w.shape == (n,) and w.dtype.str == "<i8", "windows must be an int64 vector matching observations")
        require(all(np.all(c >= 0) for c in chunks(w)), "negative window count")
    values = arrays[{"dense": "X", "multi": "tokens", "sparse": "data"}[kind]]
    require(values.dtype.str == "<f4", "values must be little-endian f32")
    require(all(np.isfinite(c).all() for c in chunks(values)), "nonfinite slot value", "MNLM_SLOT_NONFINITE")
    if kind == "dense":
        require(values.ndim == 2 and values.shape[0] == n and values.shape[1] > 0,
                "dense rows differ from absence length or lack dimensions")
        d = values.shape[1]
        rows = max(1, CHUNK // (4 * d))
        require(all(not np.any(values[i:i + rows][a[i:i + rows]]) for i in range(0, n, rows)),
                "absent dense observations contain values")
    else:
        o = arrays["offsets" if kind == "multi" else "indptr"]
        require(o.ndim == 1 and o.dtype.str == "<i8" and o.shape == (n + 1,), "invalid native offsets")
        require(o[0] == 0 and o[-1] == len(values) and monotone(o), "offsets do not partition the payload")
        size = CHUNK // 8
        require(all(np.array_equal(o[i + 1:min(n, i + size) + 1] == o[i:min(n, i + size)], a[i:min(n, i + size)])
                    for i in range(0, n, size)), "absence differs from native offsets")
        if kind == "multi":
            require(values.ndim == 2 and values.shape[1] > 0, "Multi must retain its token axis")
            d = values.shape[1]
        else:
            shape, ix = arrays["shape"], arrays["indices"]
            require(shape.dtype.str == "<i8" and shape.shape == (2,) and shape[0] == n and shape[1] > 0,
                    "sparse shape differs from observation count")
            require(values.ndim == 1 and ix.shape == values.shape and ix.dtype.str == "<i4", "invalid CSR arrays")
            d = int(shape[1])
            require(all(np.all((c >= 0) & (c < d)) for c in chunks(ix)), "sparse index outside vocabulary")
    return {"n": n, "dims": d, "absent": sum(int(c.sum()) for c in chunks(a)), "kind": kind}


def check_seal(measured, expected, where):
    require(expected.get("layout_schema") == SCHEMA and expected.get("layout_digest"),
            f"{where}: historical value-only seal; remeasure into a new generation", "MNLM_SLOT_LAYOUT_UNSEALED")
    require(all(measured.get(k) == expected.get(k) for k in ("digest", "layout_digest", "layout_schema", "n", "dims", "absent")),
            f"{where}: persisted slot differs from its measurement seal", "MNLM_SLOT_STALE")


def unit_norm_error(arrays, kind, norm):
    if norm not in ("unit", "per-token L2"):
        return None
    require(kind in ("dense", "multi"), "unit norm requires a declared dense or Multi contract")
    values = arrays["X" if kind == "dense" else "tokens"]
    size, worst = max(1, CHUNK // (16 * values.shape[1])), 0.0
    for i in range(0, len(values), size):
        block = values[i:i + size]
        if kind == "dense":
            block = block[~arrays["absent"][i:i + size]]
        if len(block):
            error = np.abs(np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64)) - 1)
            worst = max(worst, float(error.max()))
    require(worst <= 1e-5, f"declared unit norm violated: max error {worst}", "MNLM_SLOT_NORM")
    return worst
