"""Shared helpers for Stage 1 (issue #8): paths, manifests, sidecars, population tables. No policy lives here."""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "census"))
from ledger import Ledger, digest, digest_file  # noqa: E402,F401

HOST = socket.gethostname()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict]:
    """A line that is not JSON refuses with a typed error naming the line: a manifest torn by a crash must be seen, not
    raised as a bare JSONDecodeError and worked around."""
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, l in enumerate(f, 1):
            if not l.strip():
                continue
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"MNLM_JSONL_TORN {path} line {i}: {exc}") from None
    return out


def append_jsonl(path: Path, row: dict) -> None:
    """The manifest row is the commit point of a shard (the data file was renamed into place before it), so it is
    fsynced like a ledger entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_npy(path: Path, arr) -> None:
    """np.save through a temporary file and an atomic rename: a crash never leaves a truncated .npy under the final name."""
    tmp = path.with_name(path.name + ".tmp.npy")
    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sidecars_complete(canon_dir: Path, shard: str, docs: int) -> bool:
    """The three sidecars of a sealed shard exist with the sizes the doc count implies (b3.bin is 16 bytes per doc)."""
    b3 = canon_dir / f"{shard}.b3.bin"
    return b3.exists() and b3.stat().st_size == 16 * docs and (canon_dir / f"{shard}.rows.npy").exists() and (canon_dir / f"{shard}.n.npy").exists()


def write_json(path: Path, obj) -> None:
    """Atomic: a summary or population table is never seen half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Paths:
    """<out>/canon/<v>/<src>/, <out>/sig/<v>/<src>/, <out>/sa/<v>/, <out>/dedup/<v>/ ; ledger and evidence under <root>."""

    def __init__(self, root: Path, out: Path, version: str):
        self.root, self.out, self.version = Path(root), Path(out), version
        self.ledger = self.root / "ledger" / "ledger.jsonl"
        # Canonical shards and signatures are pure functions of (raw sha256, contract), not of the population version:
        # MNLM_CANON_VERSION lets a larger population (p0-v1) reuse the shards sealed for a smaller one (p0micro-v1), PC-01.
        self.canon_version = os.environ.get("MNLM_CANON_VERSION") or version

    def canon(self, src: str) -> Path:
        return self.out / "canon" / self.canon_version / src

    def sig(self, src: str) -> Path:
        return self.out / "sig" / self.canon_version / src

    @property
    def sa(self) -> Path:
        return self.out / "sa" / self.version

    @property
    def dedup(self) -> Path:
        return self.out / "dedup" / self.version

    def evidence(self, issue: str, run: str) -> Path:
        return self.root / "evidence" / "fsv" / issue / run


def manifest_rows(canon_dir: Path) -> dict[str, dict]:
    """Latest manifest row per shard (the manifest is append-only; a later row supersedes)."""
    out: dict[str, dict] = {}
    for r in read_jsonl(canon_dir / "manifest.jsonl"):
        out[r["shard"]] = r
    return out


def sealed_shards(canon_dir: Path) -> list[dict]:
    """Cold-attested latest shards; an incomplete source is never a smaller success."""
    from attestation import attest_shard
    from contracts import ContractError
    from plan_seals import accepted, group_rows, read_seals
    rows = []
    seals = read_seals(canon_dir)
    for plan, history in sorted(group_rows(read_jsonl(canon_dir / "manifest.jsonl")).items()):
        complete = accepted(history, seals.get(plan))
        if complete is None:
            raise ContractError("MNLM_CANON_PLAN_UNSEALED", f"{plan}: complete extraction/recovery seal required")
        rows.extend(attest_shard(canon_dir, r) for r in complete)
    if len({r["shard"] for r in rows}) != len(rows):
        raise ContractError("MNLM_CANON_SHARD_NAME_COLLISION", "multiple plans claim the same physical shard")
    return sorted(rows, key=lambda r: r["shard"])


def load_sidecars(canon_dir: Path, shard: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(b3 uint8[n,16], rows int32[n], nbytes int64[n]) for the docs of one sealed shard."""
    b3 = np.fromfile(canon_dir / f"{shard}.b3.bin", dtype=np.uint8).reshape(-1, 16)
    rows = np.load(canon_dir / f"{shard}.rows.npy")
    n = np.load(canon_dir / f"{shard}.n.npy")
    if not (len(b3) == len(rows) == len(n)):
        raise RuntimeError(f"MNLM_CANON_SIDECAR_MISMATCH {canon_dir}/{shard}: {len(b3)} {len(rows)} {len(n)}")
    return b3, rows, n


def iter_canon_docs(canon_dir: Path, shard: str, expected_blake3=None):
    """Streams the canonical records of one shard: dicts with row, b3, n, kind, text, f."""
    import io

    import zstandard
    import blake3
    class BoundReader(io.RawIOBase):
        def __init__(self, raw):
            self.raw, self.hash = raw, blake3.blake3()

        def readable(self):
            return True

        def readinto(self, buffer):
            n = self.raw.readinto(buffer)
            self.hash.update(memoryview(buffer)[:n])
            return n

    with (canon_dir / f"{shard}.jsonl.zst").open("rb") as fh:
        bound = BoundReader(fh)
        with zstandard.ZstdDecompressor().stream_reader(bound, closefd=False) as zr:
            for line in io.TextIOWrapper(zr, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)
        if expected_blake3 is not None and bound.hash.hexdigest() != expected_blake3:
            raise RuntimeError(f"MNLM_ATOMS_CANON_CHANGED: {canon_dir}/{shard}; rebuild from accepted canonical bytes")


def ledger_append(paths: Paths, actor: str, kind: str, subject: str, payload: dict) -> dict:
    return Ledger(paths.ledger, actor).append(kind, subject, payload)
