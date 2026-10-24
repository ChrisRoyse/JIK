"""Content-checked index publication. Cost O(cache bytes) once per load (PC-04/07).

Cache is regenerable, never authority. An incomplete or legacy cache refuses with
remediation. Each array is atomically replaced; the hashed manifest is committed last.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import blake3
import numpy as np

from assertion import ClaimError

SCHEMA = "claim-index-v2"


def file_hash(p):
    h = blake3.blake3()
    with Path(p).open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def load_cache(cache, meta, names, shards):
    required = {f"{n}.npy" for n in names} | {f"shard-{s}-{c}.npy" for s in shards for c in "rspo"}
    if meta.get("cache_schema") != SCHEMA or set(meta.get("cache_blake3", {})) != required:
        raise ClaimError("MNLM_CLAIM_CACHE_UNSEALED", "legacy/incomplete cache; build in a fresh cache directory")
    for name in sorted(required):
        p = cache / name
        if not p.is_file() or file_hash(p) != meta["cache_blake3"][name]:
            raise ClaimError("MNLM_CLAIM_CACHE_CORRUPT", f"{name} differs from manifest; rebuild in a fresh cache directory")
    arrays = {n: np.load(cache / f"{n}.npy", allow_pickle=False) for n in names}
    per = {s: tuple(np.load(cache / f"shard-{s}-{c}.npy", allow_pickle=False) for c in "rspo") for s in shards}
    return arrays, per


def write_cache(cache, arrays, per, meta):
    payload = {f"{n}.npy": a for n, a in arrays.items()}
    payload.update({f"shard-{s}-{c}.npy": a for s, vals in per.items() for c, a in zip("rspo", vals)})
    digests = {}
    for name, arr in payload.items():
        p = cache / name
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
        with tmp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        digests[name] = file_hash(tmp)
        os.replace(tmp, p)
    meta.update(cache_schema=SCHEMA, cache_blake3=digests)
    tmp = cache / f"meta.json.tmp.{os.getpid()}"
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache / "meta.json")
