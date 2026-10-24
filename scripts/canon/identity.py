"""Contract `identity-v1` readers (issue #10): the authored identity partition built by scripts/identity/build_identity.py.
One canonical record per authored record, per speaker span, per scenario transcript, and per section of the profile; the
record's `text` is the training text and every other key is an explicit field (spec §18). `canon-v1` normalisation is
applied by contracts.open_shard. A record without a resolved anchor cannot exist in the raw file (the builder refuses),
and the Stage 1 FSV re-checks every anchor against the sealed scripture canon (fsv_identity.py). Cost: O(bytes).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

CONTRACT_ID = "identity-v1"
PROSE = "prose"
SOURCES = {"identity-v1"}
SHARDS = {"records.jsonl": "records", "exemplars.jsonl": "exemplars", "sft.jsonl": "sft", "profile.md": "profile"}
HEADING = re.compile(r"^#{1,3} ")
LIST_FIELDS = ("refs", "refs_expanded", "rules", "corrections", "standards", "messages", "anchors")


class IdentityError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def _accepted(receipts: dict) -> dict:
    return {k: v for k, v in receipts.items() if v.get("verified") is True}


def list_shards(source_id: str, raw_dir: Path, receipts: dict) -> list[dict]:
    acc = _accepted(receipts)
    plan = []
    for rel, shard in SHARDS.items():
        r = acc.get(rel)
        if r is None:
            raise IdentityError("MNLM_IDENTITY_RECEIPT_MISSING", f"{source_id}: no verified receipt for {rel}")
        plan.append({"shard": shard, "files": [raw_dir / rel], "sha256": [r["sha256"]], "kind": PROSE})
    return plan


def jsonl_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    with path.open("r", encoding="utf-8") as f:
        for row, line in enumerate(f):
            if not line.strip():
                yield row, None, {}, PROSE
                continue
            d = json.loads(line)
            if "text" not in d or "id" not in d:
                raise IdentityError("MNLM_CANON_SCHEMA_MISMATCH", f"{path.name} row {row} lacks id/text")
            fields = {}
            for k, v in d.items():
                if k == "text":
                    continue
                if k == "anchors":
                    fields["anchor_refs"] = [a["ref"] for a in v]
                elif k == "messages":
                    fields["roles"] = [m["role"] for m in v]
                elif isinstance(v, dict):
                    fields[k] = json.dumps(v, sort_keys=True, ensure_ascii=False)
                else:
                    fields[k] = v
            yield row, d["text"], fields, PROSE


def profile_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) per markdown heading of level 1 to 3; the preamble before the first heading is section 0."""
    out, head, buf = [], "", []
    for ln in text.replace("\r\n", "\n").split("\n"):
        if HEADING.match(ln):
            if head or buf:
                out.append((head, "\n".join(buf)))
            head, buf = ln.lstrip("#").strip(), [ln]
        else:
            buf.append(ln)
    if head or buf:
        out.append((head, "\n".join(buf)))
    return out


def profile_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    for row, (head, body) in enumerate(profile_sections(path.read_text(encoding="utf-8"))):
        yield row, body, {"id": f"PROFILE.{row}", "type": "profile_section", "enforce": "standard", "tag": "profile", "section": head, "title": head[:120]}, PROSE


def open_shard(source_id: str, shard: dict) -> Iterator[tuple[str, str, Iterator]]:
    path: Path = shard["files"][0]
    if shard["shard"] == "profile":
        yield shard["shard"], PROSE, profile_docs(path)
    else:
        yield shard["shard"], PROSE, jsonl_docs(path)
