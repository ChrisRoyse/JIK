"""Digest-pinned local percept bytes. Never resolve arbitrary network locations.

Read into a bounded immutable snapshot before decoding, so a path replacement
cannot swap the bytes after verification. Per-atom cost O(blob bytes); no corpus
waveform cache or unbounded decoded allocation (PC-04/07).
"""
import hashlib
import json
from pathlib import Path

from .contract import SenseError, is_sha256, require

MAX_BYTES = 64 * 1024 * 1024


class Blobs:
    def __init__(self, path, sha256):
        require(path and is_sha256(sha256), "MNLM_UCR_BLOBS_REQUIRED", "supply a local blob manifest and its SHA-256")
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise SenseError("MNLM_UCR_BLOB_MANIFEST", "cannot read the pinned local blob manifest") from exc
        require(hashlib.sha256(raw).hexdigest() == sha256, "MNLM_UCR_BLOB_MANIFEST", "blob manifest digest differs")
        try:
            body = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise SenseError("MNLM_UCR_BLOB_MANIFEST", "blob manifest is not valid JSON") from exc
        require(isinstance(body, dict) and set(body) == {"schema", "blobs"} and body["schema"] == "ucr-blobs-v1"
                and isinstance(body["blobs"], list), "MNLM_UCR_BLOB_MANIFEST", "expected ucr-blobs-v1")
        self.sha256, self.entries = sha256, {}
        for item in body["blobs"]:
            require(isinstance(item, dict) and set(item) == {"sha256", "path", "bytes", "media_type"},
                    "MNLM_UCR_BLOB_MANIFEST", "explicit content address, path, length and media type required")
            key = item["sha256"]
            require(is_sha256(key) and key not in self.entries and isinstance(item["path"], str)
                    and Path(item["path"]).is_absolute() and type(item["bytes"]) is int and item["bytes"] > 0
                    and isinstance(item["media_type"], str), "MNLM_UCR_BLOB_MANIFEST", "invalid or duplicate blob identity")
            self.entries[key] = item

    def read(self, sha256, media_type):
        require(sha256 in self.entries, "MNLM_UCR_BLOB_MISSING", "percept absent from pinned manifest")
        item = self.entries[sha256]
        require(media_type == item["media_type"], "MNLM_UCR_BLOB_TYPE", "percept and blob media type differ")
        require(item["bytes"] <= MAX_BYTES, "MNLM_UCR_BLOB_BUDGET", "split the source into bounded retained blobs")
        try:
            with Path(item["path"]).open("rb") as src:
                raw = src.read(item["bytes"] + 1)
        except OSError as exc:
            raise SenseError("MNLM_UCR_BLOB_MISSING", "restore the exact retained bytes named by the blob manifest") from exc
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == sha256,
                "MNLM_UCR_BLOB_DIGEST", "retained percept bytes differ from their identity")
        return raw
