"""Versioned source views and structural relations (#27), never semantic evidence.

Keep sentence atoms, pairs within paragraphs, paragraphs and attested pages.
Each present level tiles the source. Source offsets are Unicode code points.
Fixed-level interval sweeps cost O(characters + nodes + emitted edges).
"""
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import segment, conserved, SEGMENTER_ID

CONTRACT = "source-multiscale-v3"
LEVELS = ("sentence", "sentence_pair", "paragraph", "page")
BREAKS = re.compile(r"(?:\r?\n[ \t]*){2,}|\u2029+")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def require(ok, code):
    if not ok:
        raise ValueError(code)


def tile(spans, length):
    return (bool(spans) and spans[0][0] == 0 and spans[-1][1] == length
            and all(type(a) is int and type(b) is int and 0 <= a < b <= length for a, b in spans)
            and all(a[1] == b[0] for a, b in zip(spans, spans[1:])))


def build(text, source_id, group, layout=None):
    require(isinstance(text, str) and bool(text.strip()), "MNLM_MULTISCALE_EMPTY")
    require(isinstance(source_id, str) and bool(source_id) and isinstance(group, str) and bool(group),
            "MNLM_MULTISCALE_LINEAGE")
    text_hash = digest(text.encode())
    pages = []
    if layout is not None:
        require(isinstance(layout, dict) and layout.get("text_sha256") == text_hash,
                "MNLM_MULTISCALE_LAYOUT_SOURCE")
        require(isinstance(layout.get("authority"), str) and bool(layout["authority"].strip()),
                "MNLM_MULTISCALE_LAYOUT_AUTHORITY")
        pages = layout.get("pages")
        require(isinstance(pages, list) and all(isinstance(p, list) and len(p) == 2 for p in pages)
                and tile(pages, len(text)), "MNLM_MULTISCALE_PAGE_COVERAGE")
    namespace = identity([CONTRACT, SEGMENTER_ID, source_id, text_hash, layout])
    by_level = {level: [] for level in LEVELS}
    def add(level, start, end, **fields):
        node = {"id": identity([namespace, level, start, end]), "level": level,
                "start": start, "end": end, "ordinal": len(by_level[level]),
                "content_sha256": digest(text[start:end].encode()), **fields}
        by_level[level].append(node)
        return node
    ends = [m.end() for m in BREAKS.finditer(text)]
    paragraph_status = ("inferred:blank-line-or-U+2029" if ends else
                        "unsegmented-text-block:no-paragraph-boundaries")
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    start = 0
    for end in ends:
        add("paragraph", start, end, boundary="blank-line-or-U+2029-v1")
        # Within a paragraph, a single newline may be a printed line wrap.
        # Replace it only in the boundary detector's same-length view; all
        # persisted spans and measurements still address the untouched source.
        detection = text[start:end].translate({ord(c): " " for c in "\r\n\u2028"})
        spans = segment(detection)
        require(conserved(detection, spans), "MNLM_MULTISCALE_SEGMENT")
        pending, sentence_count = [], 0
        for a, b, role in spans:
            node = add("sentence", start+a, start+b, role=role)
            pending.append(node)
            sentence_count += int(bool(text[start+a:start+b].strip()))
            if sentence_count == 2:
                add("sentence_pair", pending[0]["start"], pending[-1]["end"], sentence_count=2)
                pending, sentence_count = [], 0
        if pending:
            add("sentence_pair", pending[0]["start"], pending[-1]["end"], sentence_count=sentence_count)
        start = end
    for a, b in pages:
        add("page", a, b, boundary="source-attested-layout")
    edges = []
    def edge(relation, a, b, start=None, end=None):
        rec = {"relation": relation, "from": a["id"], "to": b["id"]}
        if start is not None:
            rec.update(start=start, end=end)
        edges.append(rec)
    for nodes in by_level.values():
        if nodes:
            require(tile([(n["start"], n["end"]) for n in nodes], len(text)), "MNLM_MULTISCALE_COVERAGE")
        for a, b in zip(nodes, nodes[1:]):
            edge("next", a, b)
    # Each pair of levels comprises two ordered disjoint partitions. An interval
    # sweep emits every actual intersection, including paragraph/page crossings.
    for left, right in itertools.combinations(LEVELS, 2):
        a_nodes, b_nodes = by_level[left], by_level[right]
        i = j = 0
        while i < len(a_nodes) and j < len(b_nodes):
            a, b = a_nodes[i], b_nodes[j]
            lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
            if lo < hi:
                if a["start"] == b["start"] and a["end"] == b["end"]:
                    edge("coextensive", a, b, lo, hi)
                elif a["start"] <= b["start"] and a["end"] >= b["end"]:
                    edge("contains", a, b, lo, hi)
                elif b["start"] <= a["start"] and b["end"] >= a["end"]:
                    edge("contains", b, a, lo, hi)
                else:
                    edge("overlaps", a, b, lo, hi)
            if a["end"] <= b["end"]:
                i += 1
            if b["end"] <= a["end"]:
                j += 1
    return {"contract": CONTRACT, "segmenter_id": SEGMENTER_ID, "source_id": source_id,
            "source_text_sha256": text_hash, "source_characters": len(text), "split_group": group,
            "layout": layout, "page_status": "present" if pages else "absent:no-source-pagination",
            "paragraph_status": paragraph_status,
            "nodes": [n for level in LEVELS for n in by_level[level]], "edges": edges,
            "measurement_status": "pending:frozen-instrument-readings", "served": False}


def causal_view(record, cutoff, split_group):
    """Read only fully observed units. Do not expose parent text or future IDs.

    The caller must supply the evaluation population's group; a group mismatch
    refuses. Entity grouping alone does not establish near-duplicate separation.
    """
    require(record["split_group"] == split_group, "MNLM_MULTISCALE_SPLIT_GROUP")
    require(type(cutoff) is int and 0 <= cutoff <= record["source_characters"], "MNLM_MULTISCALE_CUTOFF")
    nodes = [n for n in record["nodes"] if n["end"] <= cutoff]
    visible = {n["id"] for n in nodes}
    edges = [e for e in record["edges"] if e["from"] in visible and e["to"] in visible]
    return {"nodes": nodes, "edges": edges}
