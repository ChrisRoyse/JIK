"""Cold Stage 1 acceptance at the raw, canonical and sidecar boundaries (#19).

No persistent stat/mtime cache: these checks certify the bytes read in this pass.
New shards seal sidecar hashes; legacy manifests replay sidecars against content.
"""
import hashlib
import io
import json
from pathlib import Path
import re

import blake3
import numpy as np
import zstandard

from contracts import ContractError

HEX = re.compile(r"[0-9a-f]{64}\Z")
SUFFIXES = ("b3.bin", "rows.npy", "n.npy")


def require(ok, code, message):
    if not ok:
        raise ContractError(code, message)


def file_digest(path, algorithm="sha256"):
    h = blake3.blake3() if algorithm == "blake3" else hashlib.sha256()
    n = 0
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def attest_raw(plan):
    files, hashes = plan.get("files"), plan.get("sha256")
    require(isinstance(files, list) and bool(files) and isinstance(hashes, list)
            and len(files) == len(hashes) and all(isinstance(h, str) and HEX.fullmatch(h) for h in hashes),
            "MNLM_CANON_PLAN", "one SHA256 required for every named raw input")
    require(len({str(Path(f).resolve()) for f in files}) == len(files),
            "MNLM_CANON_PLAN", "duplicate raw paths in one plan")
    total = 0
    for f, expected in zip(files, hashes):
        require(Path(f).is_file(), "MNLM_CANON_RAW_MISSING", str(f))
        actual, size = file_digest(f)
        require(actual == expected, "MNLM_CANON_RAW_DIGEST_MISMATCH", f"{f}: raw bytes differ from receipt")
        total += size
    return total


def seal_sidecars(directory, shard):
    result = {}
    for suffix in SUFFIXES:
        h, n = file_digest(Path(directory) / f"{shard}.{suffix}")
        result[suffix] = {"sha256": h, "bytes": n}
    return result


def legacy_sidecars(directory, row):
    """Replay legacy metadata from its digest-bound canonical stream, O(bytes)."""
    shard, n = row["shard"], row["docs"]
    try:
        b3 = np.memmap(directory / f"{shard}.b3.bin", mode="r", dtype=np.uint8, shape=(n, 16)) if n else []
        rows = np.load(directory / f"{shard}.rows.npy", allow_pickle=False, mmap_mode="r")
        sizes = np.load(directory / f"{shard}.n.npy", allow_pickle=False, mmap_mode="r")
        require(rows.dtype == np.dtype("int32") and sizes.dtype == np.dtype("int64")
                and rows.shape == sizes.shape == (n,), "MNLM_CANON_SIDECAR_MISMATCH", shard)
        i, byte_count = 0, 0
        with (directory / f"{shard}.jsonl.zst").open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as z:
            for line in io.TextIOWrapper(z, encoding="utf-8"):
                r = json.loads(line)
                data = r["text"].encode("utf-8")
                h = blake3.blake3(data).digest(length=16)
                require(i < n and r["row"] == int(rows[i]) and r["n"] == int(sizes[i]) == len(data)
                        and r["b3"] == h.hex() and h == bytes(b3[i]),
                        "MNLM_CANON_SIDECAR_MISMATCH", f"{shard}: canonical row {i} differs")
                byte_count += len(data)
                i += 1
        require(i == n and byte_count == row["canonical_bytes"], "MNLM_CANON_SIDECAR_MISMATCH", shard)
    except ContractError:
        raise
    except (OSError, ValueError, KeyError, TypeError, zstandard.ZstdError) as exc:
        raise ContractError("MNLM_CANON_SIDECAR_MISMATCH", f"{shard}: {exc}") from exc


def attest_shard(directory, row):
    directory = Path(directory)
    shard = row.get("shard")
    require(isinstance(shard, str) and shard and Path(shard).name == shard and shard not in (".", ".."),
            "MNLM_CANON_MANIFEST", "a named local shard is required")
    require(not row.get("error"), "MNLM_CANON_SHARD_UNACCEPTED", f"{shard}: latest manifest records a failure")
    require(type(row.get("docs")) is int and row["docs"] >= 0
            and type(row.get("out_bytes")) is int and row["out_bytes"] > 0
            and isinstance(row.get("out_blake3"), str) and HEX.fullmatch(row["out_blake3"]),
            "MNLM_CANON_MANIFEST", f"{shard}: incomplete count/digest binding")
    path = directory / f"{shard}.jsonl.zst"
    require(path.is_file(), "MNLM_CANON_SHARD_MISSING", str(path))
    got, size = file_digest(path, "blake3")
    require(got == row["out_blake3"] and size == row["out_bytes"],
            "MNLM_CANON_SHARD_DIGEST_MISMATCH", str(path))
    for suffix in SUFFIXES:
        require((directory / f"{shard}.{suffix}").is_file(), "MNLM_CANON_SIDECAR_MISSING", f"{shard}.{suffix}")
    require((directory / f"{shard}.b3.bin").stat().st_size == row["docs"] * 16,
            "MNLM_CANON_SIDECAR_MISMATCH", f"{shard}: digest row count")
    if "sidecars" in row:
        require(isinstance(row["sidecars"], dict) and set(row["sidecars"]) == set(SUFFIXES),
                "MNLM_CANON_MANIFEST", f"{shard}: complete sidecar seals required")
        require(seal_sidecars(directory, shard) == row["sidecars"],
                "MNLM_CANON_SIDECAR_MISMATCH", f"{shard}: sidecar digest differs")
    else:
        legacy_sidecars(directory, row)
    return row
