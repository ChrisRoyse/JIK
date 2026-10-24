"""Stage 1 FSV for the identity partition (issue #10): independent re-extraction of a sealed record from the raw jsonl /
profile.md (B1 path for fsv_stage1.raw_row_text), and the boundary that makes the partition a partition of the verifier:
every anchor reference carried by every sealed identity record must resolve to a verse of the sealed scripture-v1 World
English Bible shard and carry that verse's text byte for byte. Negative controls: a forged reference must not resolve;
a record with a flipped anchor byte must not match. Cost: O(canon bytes) once.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import zstandard

from common import iter_canon_docs, sealed_shards

HEADING = re.compile(r"^#{1,3} ")
REQUIRED = {"identity": 12, "standard": 10, "rule": 12, "episode_rule": 12, "hard_rule": 36, "correction": 22, "scenario": 17, "audience": 9, "anchor": 8,
            "canon_frame": 9, "self": 33, "surprise": 10, "proposition": 4, "voice": 1}


def reparse(raw: Path, shard: str, target: int) -> str | None:
    """Row `target` of the raw file, re-read without the contract module."""
    if shard == "profile":
        secs, head, buf = [], None, []
        for ln in raw.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"):
            if HEADING.match(ln):
                if head is not None or buf:
                    secs.append("\n".join(buf))
                head, buf = ln, [ln]
            else:
                buf.append(ln)
        if head is not None or buf:
            secs.append("\n".join(buf))
        return secs[target] if target < len(secs) else None
    with raw.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == target:
                return json.loads(line)["text"] if line.strip() else None
    return None


def web_verses(out: Path) -> dict[str, str]:
    p = out / "canon" / "scripture-v1" / "ebible-english" / "eng-web.jsonl.zst"
    verses = {}
    with p.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        for ln in io.TextIOWrapper(zr, encoding="utf-8"):
            r = json.loads(ln)
            if r["f"].get("role") == "verse":
                verses[r["f"]["ref"]] = r["text"]
    return verses


def checks(paths, check) -> None:
    """Every sealed identity record: anchor refs resolve and the inlined verse text equals the sealed WEB verse."""
    cdir = paths.canon("identity-v1")
    verses = web_verses(paths.out)
    check("B4.identity.web_verses_loaded", len(verses) > 30000, verses=len(verses))
    by_type: dict[str, int] = {}
    recs = anchors = bad_ref = bad_text = no_anchor = 0
    exemplars = bad_span = 0
    for sh in sealed_shards(cdir):
        for rec in iter_canon_docs(cdir, sh["shard"]):
            f = rec["f"]
            if sh["shard"] == "records":
                recs += 1
                by_type[f["type"]] = by_type.get(f["type"], 0) + 1
                refs = f.get("anchor_refs") or []
                if not refs:
                    no_anchor += 1
                for ref in refs:
                    anchors += 1
                    if ref not in verses:
                        bad_ref += 1
                # the anchors block must be exactly the sealed verses in order (a poetry verse spans several lines, so the
                # block is compared whole, never line by line)
                block = rec["text"].split(f"\nAnchors (World English Bible, eng-web):\n", 1)
                expected = "\n".join(f"{ref} {verses[ref]}" for ref in refs if ref in verses)
                if len(block) != 2 or block[1] != expected:
                    bad_text += len(refs)
            elif sh["shard"] == "exemplars":
                exemplars += 1
                v = verses.get(f["ref"])
                if v is None or rec["text"] not in v:
                    bad_span += 1
    check("B4.identity.anchors_resolve_in_sealed_web", recs > 0 and bad_ref == 0 and no_anchor == 0, records=recs, anchors=anchors, unresolved=bad_ref, without_anchor=no_anchor)
    check("B4.identity.anchor_text_equals_web_verse", anchors > 0 and bad_text == 0, anchors=anchors, mismatched=bad_text)
    check("B4.identity.exemplar_spans_inside_web_verse", exemplars > 0 and bad_span == 0, exemplars=exemplars, outside=bad_span)
    check("B4.identity.record_families_complete", all(by_type.get(k, 0) == n for k, n in REQUIRED.items()), by_type=by_type, required=REQUIRED)
    forged = "MAT.99.1"
    check("B4.identity.neg.forged_ref_does_not_resolve", forged not in verses, ref=forged)
    sample = next(iter(verses.items()))
    flipped = sample[1][:-1] + ("x" if not sample[1].endswith("x") else "y")
    check("B4.identity.neg.flipped_anchor_text_differs", flipped != verses[sample[0]], ref=sample[0])
