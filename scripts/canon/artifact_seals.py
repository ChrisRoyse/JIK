"""Complete, parent-bound Stage 1 artifacts (#21). Readers hash bytes, never sizes alone.

One audit context per consumer boundary memoizes the dependency DAG (PC-04/07).
Seals publish last; a failed build has no accepted seal. Historical unsealed files
require a real rebuild in a new version, not a seal synthesized by a reader.
"""
from pathlib import Path
import json
import os

from common import digest, digest_file
from training_scope import POLICY_SHA256


def identity(body):
    return digest(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())

SCHEMA = "dedup-artifacts-v1"
STAGES = {
    "population": ("dedup", (), ("population.json", "population.npz")),
    "exact": ("dedup", ("population",), ("exact.removed.npy", "exact.kept_of.npy", "exact.jsonl.zst", "exact.summary.json")),
    "near": ("dedup", ("exact",), ("near.removed.npy", "near.kept_of.npy", "near.est_jaccard.npy",
        "near.links.npy", "near.links_est.npy", "near.links_removed.npy", "near.jsonl.zst", "near.summary.json")),
    "unique": ("dedup", ("near",), ("unique.npy", "unique.summary.json")),
    "corpus": ("sa", ("unique",), ("corpus.bin", "docs.npy", "offsets.npy", "corpus.json")),
    "table": ("sa", ("corpus",), ("corpus.bin.table.bin", "table.json")),
    "similar": ("sa", ("table",), ("ranges.txt", "collect.txt", "similar.json")),
    "spans": ("dedup", ("similar",), ("spans.npy", "spans.jsonl.zst", "spans.summary.json")),
}


def refuse(code, detail):
    raise SystemExit(f"{code}: {detail}; rebuild the affected generation from accepted inputs")


def invalidate(directory, name):
    (directory / (name + ".seal.json")).unlink(missing_ok=True)


def publish(directory, name, files, parents):
    directory = Path(directory)
    entries = {}
    for file in files:
        path = directory / file
        # Flush producer writes before the complete marker becomes durable.
        with path.open("r+b") as f:
            os.fsync(f.fileno())
        h, size = digest_file(path)
        entries[file] = {"blake3": h, "bytes": size}
    body = {"schema": SCHEMA, "name": name, "policy": POLICY_SHA256,
            "parents": parents, "files": entries}
    obj = {**body, "id": identity(body)}
    dest = directory / (name + ".seal.json")
    tmp = dest.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return obj["id"]


def accept(directory, name, files, parents):
    directory = Path(directory)
    seal = directory / (name + ".seal.json")
    try:
        obj = json.loads(seal.read_text(encoding="utf-8"))
        body = {k: v for k, v in obj.items() if k != "id"}
        if obj.get("id") != identity(body) or obj.get("schema") != SCHEMA or obj.get("name") != name:
            refuse("MNLM_DEDUP_SEAL_INVALID", seal)
        if obj.get("policy") != POLICY_SHA256 or obj.get("parents") != parents:
            refuse("MNLM_DEDUP_PARENT_MISMATCH", seal)
        if set(obj["files"]) != set(files):
            refuse("MNLM_DEDUP_SEAL_INVALID", f"{seal}: file inventory")
        for file in files:
            h, size = digest_file(directory / file)
            if obj["files"][file] != {"blake3": h, "bytes": size}:
                refuse("MNLM_DEDUP_BYTES_MISMATCH", directory / file)
        return obj["id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        refuse("MNLM_DEDUP_UNSEALED", f"{seal}: {exc}")


class Audit:
    def __init__(self, paths):
        self.paths, self.accepted = paths, {}

    def require(self, stage):
        if stage not in self.accepted:
            location, deps, files = STAGES[stage]
            parents = {dep: self.require(dep) for dep in deps}
            self.accepted[stage] = accept(getattr(self.paths, location), stage, files, parents)
        return self.accepted[stage]

    def begin(self, stage):
        location, deps, _ = STAGES[stage]
        parents = {dep: self.require(dep) for dep in deps}
        invalidate(getattr(self.paths, location), stage)
        self.accepted.pop(stage, None)
        return parents

    def finish(self, stage, parents):
        location, _, files = STAGES[stage]
        identity = publish(getattr(self.paths, location), stage, files, parents)
        self.accepted[stage] = identity
        return identity
