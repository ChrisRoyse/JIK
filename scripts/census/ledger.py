"""Append-only, hash-chained ledger (spec Appendix B: LedgerEntry {seq, prev_hash, kind, subject, payload_digests,
actor, ts, hash}; Merkle checkpoint every 1000 entries).

One JSONL file. Every entry's `hash` is BLAKE3 over the canonical JSON of the entry without its `hash` field; `prev_hash`
is the previous entry's hash (64 zeros for the genesis entry). `verify_chain` re-derives every hash and refuses with a
typed error on the first break. Nothing is ever rewritten; a wrong entry is superseded by a later entry that says so.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any

import blake3

GENESIS = "0" * 64
CHECKPOINT_EVERY = 1000


class LedgerError(Exception):
    def __init__(self, code: str, message: str, **detail: Any):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.detail = code, message, detail

    def as_dict(self) -> dict:
        return {"ok": False, "code": self.code, "message": self.message, **self.detail}


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def digest_file(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    h, n = blake3.blake3(), 0
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return digest(canonical(body))


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return GENESIS
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [blake3.blake3(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


class Ledger:
    def __init__(self, path: Path, actor: str):
        self.path = Path(path)
        self.actor = actor
        if not isinstance(actor, str) or not actor:
            raise LedgerError("MNLM_LEDGER_ENTRY_INVALID", "a nonempty actor is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._seq, self._head, self._since_ckpt = self._load_tail()

    def _load_tail(self) -> tuple[int, str, list[str]]:
        if not self.path.exists():
            return 0, GENESIS, []
        return _scan(self.path)

    @property
    def head(self) -> str:
        return self._head

    @property
    def seq(self) -> int:
        return self._seq

    def _locked(self):
        """Exclusive lock on <ledger>.lock for the whole read-tail + append critical section. Two writers that each read
        the tail and then append would both write seq N+1 and the chain could never be repaired (append-only); the lock
        makes the append serial on one host. POSIX: fcntl.flock; Windows: msvcrt.locking on the first byte."""
        import contextlib

        @contextlib.contextmanager
        def cm():
            lock = self.path.with_suffix(".jsonl.lock")
            with lock.open("a+b") as lf:
                try:
                    import fcntl
                except ImportError:
                    fcntl = None
                    import msvcrt
                # Do not catch exceptions raised *inside* the critical section
                # as platform selection or lock-acquisition failures.
                try:
                    lf.seek(0)
                    if fcntl is not None:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise LedgerError("MNLM_LEDGER_BUSY", "another writer owns this ledger; retry after it finishes") from exc
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                    else:
                        lf.seek(0)
                        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        return cm()

    def append(self, kind: str, subject: str, payload_digests: dict) -> dict:
        if kind == "ledger.checkpoint":
            raise LedgerError("MNLM_LEDGER_RESERVED_KIND", "checkpoints are generated inside the writer lock")
        _validate_content(kind, subject, payload_digests, self.actor)
        with self._locked():
            self._seq, self._head, self._since_ckpt = self._load_tail()
            # A crash after a fully durable boundary entry may leave its valid
            # checkpoint pending. Complete it before accepting another event.
            self._checkpoint()
            entry = self._write(kind, subject, payload_digests)
            self._since_ckpt.append(entry["hash"])
            self._checkpoint()
        return entry

    def _write(self, kind, subject, payload):
        """Caller holds the writer lock and has validated the entire prefix."""
        entry = {"seq": self._seq + 1, "prev_hash": self._head, "kind": kind,
                 "subject": subject, "payload_digests": payload, "actor": self.actor,
                 "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        entry["hash"] = entry_hash(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq, self._head = entry["seq"], entry["hash"]
        return entry

    def _checkpoint(self):
        if len(self._since_ckpt) >= CHECKPOINT_EVERY:
            self._write("ledger.checkpoint", f"seq<={self._seq}",
                        {"merkle_root": merkle_root(self._since_ckpt), "entries": len(self._since_ckpt)})
            self._since_ckpt = []


def _validate_content(kind, subject, payload, actor):
    if any(not isinstance(v, str) or not v for v in (kind, subject, actor)) or not isinstance(payload, dict):
        raise LedgerError("MNLM_LEDGER_ENTRY_INVALID", "kind, subject and actor must be nonempty strings; payload must be an object")
    try:
        canonical({"kind": kind, "subject": subject, "payload_digests": payload, "actor": actor})
    except (ValueError, TypeError, UnicodeError) as exc:
        raise LedgerError("MNLM_LEDGER_ENTRY_INVALID", "payload must be finite, serializable JSON") from exc


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON object key")
        out[key] = value
    return out


def _scan(path):
    prev, seq, since = GENESIS, 0, []
    keys = {"seq", "prev_hash", "kind", "subject", "payload_digests", "actor", "ts", "hash"}
    with Path(path).open("rb") as f:
        for lineno, line in enumerate(f, 1):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete final line; preserve it for recovery")
                e = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(e, dict) or set(e) != keys or type(e["seq"]) is not int:
                    raise ValueError("invalid entry schema")
                _validate_content(e["kind"], e["subject"], e["payload_digests"], e["actor"])
                if not isinstance(e["ts"], str) or not e["ts"]:
                    raise ValueError("missing timestamp")
                if _dt.datetime.fromisoformat(e["ts"]).utcoffset() is None:
                    raise ValueError("timestamp must declare its timezone")
                if any(not isinstance(e[k], str) or not re.fullmatch("[0-9a-f]{64}", e[k]) for k in ("hash", "prev_hash")):
                    raise ValueError("invalid hash")
                if e["seq"] != seq + 1 or e["prev_hash"] != prev or entry_hash(e) != e["hash"]:
                    raise ValueError("sequence, previous hash or entry digest mismatch")
                if e["kind"] == "ledger.checkpoint":
                    payload = e["payload_digests"]
                    if type(payload.get("entries")) is not int or payload["entries"] != len(since) or not since:
                        raise ValueError("checkpoint entry count mismatch")
                    if merkle_root(since) != payload.get("merkle_root"):
                        raise ValueError("checkpoint root mismatch")
                    since = []
                else:
                    since.append(e["hash"])
                prev, seq = e["hash"], e["seq"]
            except (ValueError, TypeError, KeyError, UnicodeError, LedgerError) as exc:
                raise LedgerError("MNLM_LEDGER_CHAIN_BROKEN", str(exc), line=lineno,
                                  remediation="Preserve the damaged ledger and restore its sealed bytes; never append over or truncate it.") from exc
    return seq, prev, since


def verify_chain(path: Path) -> dict:
    """Re-derive every hash and link. Returns {ok, seq, head} or raises LedgerError(MNLM_LEDGER_CHAIN_BROKEN)."""
    path = Path(path)
    if not path.exists():
        raise LedgerError("MNLM_LEDGER_MISSING", f"{path} does not exist")
    seq, head, _ = _scan(path)
    return {"ok": True, "seq": seq, "head": head, "entries": seq}


def read_entries(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
