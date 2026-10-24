"""Slot files in bounded memory (issue #12 chain 11, issue #5 pipeline cost). measure-run15 was killed by the kernel OOM killer
at 110 GB resident: a multi-vector lens (bge-m3-colbert) over whole court opinions writes about 8.5 M token rows × 1024 f32
(35 GB) per question, and the old path held the batches, their concatenation, and a `.tobytes()` copy for the digest, and it
re-loaded every reused slot whole to re-digest it. Cost now: the digest hashes array buffers in place (no copy); a reused
slot is digested by streaming each .npy member of the .npz in CHUNK pieces; a new multi-vector slot streams each batch's token
rows to a raw file as it is produced and copies that file into the .npz member, so resident memory is O(batch + CHUNK), not
O(tokens). The digest is the same function as before (BLAKE3 over sorted keys, each key's name then its C-order bytes), so
every digest already on the ledger stays comparable."""
from __future__ import annotations

import zipfile
import os
from pathlib import Path

import blake3
import numpy as np
from numpy.lib import format as npformat
from lenscommon import LensError
import slot_layout

CHUNK = 1 << 26  # 64 MiB per read while streaming a member


def _buf(a: np.ndarray) -> memoryview:
    a = np.ascontiguousarray(a)
    return memoryview(a.reshape(-1).view(np.uint8)) if a.size else memoryview(b"")


def arrays_digest(arrays: dict) -> str:
    h = blake3.blake3()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(_buf(arrays[k]))
    return h.hexdigest()


def slot_digest(path: Path, norm=None) -> dict:
    """Cold native checks and a new shape/dtype seal alongside the legacy digest.

    Historical value digests remain comparable. Acceptance and cache reuse also
    require the layout seal; a reader cannot upgrade an old publication. Headers,
    memmaps and payload hashing share one open descriptor. Memory is O(CHUNK).
    """
    h, arrays = blake3.blake3(), {}
    try:
        with zipfile.ZipFile(path) as zf:
            kind, meta = slot_layout.members(zf)
            arrays = slot_layout.mapped(zf.fp, meta)
            counts = slot_layout.validate(arrays, kind)
            if norm in ("unit", "per-token L2"):
                counts["norm_max_error"] = slot_layout.unit_norm_error(arrays, kind, norm)
            for key, info in meta.items():
                h.update(key.encode())
                with zf.open(info["info"]) as fh:
                    slot_layout.header(fh)
                    while b := fh.read(CHUNK):
                        h.update(b)
            value_digest = h.hexdigest()
            return {"digest": value_digest, **counts, "layout_schema": slot_layout.SCHEMA,
                    "layout_digest": slot_layout.layout_digest(meta, value_digest)}
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise LensError("MNLM_SLOT_LAYOUT", f"{path}: invalid native slot: {exc}") from exc
    finally:
        for array in arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()


def exclusive(path):
    try:
        return path.open("xb")
    except FileExistsError as exc:
        raise LensError("MNLM_SLOT_SCRATCH_EXISTS", f"{path}: scratch exists; inspect the owning run before recovery") from exc


def publish(tmp, path):
    """A slot is visible only after cold validation and a durability barrier."""
    with tmp.open("r+b") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_slot(path: Path, arrays: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Exclusive creation prevents concurrent/stale scratch from being overwritten.
    fh = exclusive(tmp)
    try:
        with fh:
            np.savez(fh, **arrays)
        expected = arrays_digest(arrays)
        if slot_digest(tmp)["digest"] != expected:
            raise LensError("MNLM_SLOT_STALE", f"{tmp}: writer bytes differ from cold readback")
        publish(tmp, path)
        return expected
    finally:
        tmp.unlink(missing_ok=True)


class TokenSpool:
    """Token rows of a multi-vector lens written to a raw file batch by batch; `finish` writes the .npz (absent, offsets,
    tokens, windows) and returns its digest, hashing in sorted-key order while it copies."""

    def __init__(self, path: Path, dim: int):
        self.path, self.dim = path, dim
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = path.with_name(path.name + ".tokens.raw")
        self.fh = exclusive(self.raw)
        self.offsets, self.rows = [0], 0

    def add(self, toks: np.ndarray, offs: np.ndarray) -> None:
        toks = np.ascontiguousarray(toks)
        if toks.dtype != np.float32:
            raise LensError("MNLM_SLOT_LAYOUT", "Multi tokens must already be f32; no silent cast")
        if toks.ndim != 2 or toks.shape[1] != self.dim:
            raise RuntimeError(f"MNLM_SLOT_SHAPE {self.path}: token batch {toks.shape}, lens dim {self.dim}")
        offs = np.asarray(offs)
        if (offs.dtype != np.int64 or offs.ndim != 1 or not len(offs) or offs[0] != 0
                or offs[-1] != len(toks) or np.any(offs[1:] < offs[:-1]) or not np.isfinite(toks).all()):
            raise LensError("MNLM_SLOT_LAYOUT", "invalid Multi batch offsets or nonfinite tokens")
        self.fh.write(_buf(toks))
        self.offsets.extend((np.asarray(offs[1:], np.int64) + self.offsets[-1]).tolist())
        self.rows += len(toks)

    def finish(self, windows: np.ndarray) -> tuple[str, dict]:
        self.fh.close()
        offs = np.asarray(self.offsets, np.int64)
        if offs[-1] != self.rows:
            raise LensError("MNLM_SLOT_SHAPE", f"{self.path}: offsets do not match token rows")
        absent = np.diff(offs) == 0
        windows = np.asarray(windows)
        if windows.dtype != np.int64 or windows.shape != absent.shape or np.any(windows < 0):
            raise LensError("MNLM_SLOT_LAYOUT", "windows must be nonnegative int64 counts matching observations")
        small = {"absent": absent, "offsets": offs, "windows": windows}
        h = blake3.blake3()
        tmp = self.path.with_name(self.path.name + ".tmp")
        fh = exclusive(tmp)
        try:
            with fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for key in sorted([*small, "tokens"]):
                    h.update(key.encode())
                    with zf.open(f"{key}.npy", "w", force_zip64=True) as out:
                        if key != "tokens":
                            npformat.write_array(out, small[key], allow_pickle=False)
                            h.update(_buf(small[key]))
                            continue
                        npformat.write_array_header_2_0(out, {"descr": npformat.dtype_to_descr(np.dtype(np.float32)), "fortran_order": False, "shape": (self.rows, self.dim)})
                        with self.raw.open("rb") as src:
                            while b := src.read(CHUNK):
                                out.write(b)
                                h.update(b)
            measured = slot_digest(tmp)
            if measured["digest"] != h.hexdigest():
                raise LensError("MNLM_SLOT_STALE", f"{tmp}: spooled bytes differ from cold readback")
            publish(tmp, self.path)
            self.raw.unlink()
            return h.hexdigest(), {"absent": int(absent.sum()), "n": int(len(absent))}
        finally:
            tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ Multi slots: members by memmap, the native late-interaction view
LANDMARKS = 64
LANDMARK_SEED = 7200


def npz_member(path: Path, key: str) -> np.ndarray:
    """Read-only native memmap after header validation; no payload-sized copy."""
    with zipfile.ZipFile(path) as zf:
        _, meta = slot_layout.members(zf)
        slot_layout.require(key in meta, f"{path}: missing member {key}")
        return slot_layout.mapped(zf.fp, {key: meta[key]})[key]


def landmark_rows(absent: np.ndarray, family: str, k: int = LANDMARKS) -> np.ndarray:
    """k present rows chosen by a label-free BLAKE3 order of (seed, family, row)."""
    present = np.flatnonzero(~absent)
    key = [blake3.blake3(f"{LANDMARK_SEED}|{family}|{i}".encode()).digest(8) for i in present]
    return np.sort(present[np.argsort(key, kind="stable")[:k]])


def maxsim_landmarks(path: Path, absent: np.ndarray, family: str, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """The Multi slot's native view for a probe: for every atom, the symmetric late-interaction score (mean over its tokens of
    the max over a landmark's tokens, and the converse, averaged) against LANDMARKS label-free landmark atoms. Token rows are
    read by memmap; scores are fp32 on the GPU with TF32 off. Returns (X [n, LANDMARKS] with absent rows zero, landmark rows).
    Cost: O(Σ T_atom × Σ T_landmark × d) multiply-adds. Each pair tiles both axes;
    device storage O(chunk² + chunk*d + max T_landmark), independent of total landmark tokens."""
    from multi_view import maxsim_pair, validate_multi

    toks, offs = npz_member(path, "tokens"), np.asarray(npz_member(path, "offsets"))
    validate_multi(toks, offs, absent)
    lm = landmark_rows(absent, family)
    X = np.zeros((len(absent), len(lm)), np.float32)
    for i in np.flatnonzero(~absent):
        for col, j in enumerate(lm):
            X[i, col] = maxsim_pair(toks[offs[i]:offs[i + 1]], toks[offs[j]:offs[j + 1]], device=device)
    return X, lm
