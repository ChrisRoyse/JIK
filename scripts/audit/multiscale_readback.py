"""Cold readback of source spans, complete relations and lineage (#27).

No multiscale producer imports. Reference edge enumeration uses interval bisection,
not the producer's two-pointer sweep. O(source bytes + nodes log nodes + edges).
"""
import argparse
import bisect
from collections import Counter
import hashlib
import itertools
import io
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ident(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def check(ok, code):
    if not ok:
        raise ValueError(code)


def audit_document(record, text, source, entity, layout=None):
    check(record["entity"] == entity, "MNLM_MULTI_READBACK_LINEAGE")
    return audit_spans(record, text, source, "entity:"+entity, layout)


def audit_spans(record, text, source, split_group, layout=None, source_id=None):
    check(record["source_text_sha256"] == sha(text.encode()) == source["text_sha256"]
          and record["source_characters"] == len(text)
          and record["source_id"] == (ident(source) if source_id is None else source_id),
          "MNLM_MULTI_READBACK_SOURCE")
    check(record["split_group"] == split_group, "MNLM_MULTI_READBACK_LINEAGE")
    check(record["layout"] == layout and record["page_status"] ==
          ("present" if layout else "absent:no-source-pagination"), "MNLM_MULTI_READBACK_LAYOUT")
    check(record["contract"] == "source-multiscale-v3" and record["served"] is False
          and record["measurement_status"] == "pending:frozen-instrument-readings", "MNLM_MULTI_READBACK_CLAIM")
    levels = {k: [] for k in ("sentence", "sentence_pair", "paragraph", "page")}
    namespace = ident([record["contract"], record["segmenter_id"], record["source_id"], record["source_text_sha256"], layout])
    seen = set()
    for node in record["nodes"]:
        level, start, end = node["level"], node["start"], node["end"]
        check(level in levels and type(start) is int and type(end) is int and 0 <= start < end <= len(text),
              "MNLM_MULTI_READBACK_SPAN")
        check(node["id"] == ident([namespace, level, start, end]) and node["id"] not in seen,
              "MNLM_MULTI_READBACK_ID")
        check(node["content_sha256"] == sha(text[start:end].encode()), "MNLM_MULTI_READBACK_CONTENT")
        seen.add(node["id"])
        levels[level].append(node)
    for level, nodes in levels.items():
        if level == "page" and layout is None:
            check(not nodes, "MNLM_MULTI_READBACK_INVENTED_PAGE")
            continue
        cursor = 0
        for ordinal, n in enumerate(nodes):
            check(n["start"] == cursor and n["ordinal"] == ordinal, "MNLM_MULTI_READBACK_TILING")
            cursor = n["end"]
        check(nodes and cursor == len(text), "MNLM_MULTI_READBACK_TILING")
    if layout:
        check([[n["start"], n["end"]] for n in levels["page"]] == layout["pages"], "MNLM_MULTI_READBACK_PAGE_SOURCE")
    ends = [m.end() for m in re.finditer(r"(?:\r?\n[ \t]*){2,}|\u2029+", text)]
    check(record["paragraph_status"] == ("inferred:blank-line-or-U+2029" if ends else
          "unsegmented-text-block:no-paragraph-boundaries"), "MNLM_MULTI_READBACK_PARAGRAPH_STATUS")
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    check([n["end"] for n in levels["paragraph"]] == ends, "MNLM_MULTI_READBACK_PARAGRAPHS")
    sentences = levels["sentence"]
    sentence_starts = [n["start"] for n in sentences]
    para_starts = [n["start"] for n in levels["paragraph"]]
    for pair in levels["sentence_pair"]:
        i = bisect.bisect_left(sentence_starts, pair["start"])
        j = bisect.bisect_left(sentence_starts, pair["end"])
        members = sentences[i:j]
        check(members and members[0]["start"] == pair["start"] and members[-1]["end"] == pair["end"],
              "MNLM_MULTI_READBACK_PAIR_MEMBERS")
        count = sum(bool(text[n["start"]:n["end"]].strip()) for n in members)
        para = levels["paragraph"][bisect.bisect_right(para_starts, pair["start"])-1]
        check(count == pair["sentence_count"] and 0 <= count <= 2 and pair["end"] <= para["end"]
              and (count == 2 or pair["end"] == para["end"]), "MNLM_MULTI_READBACK_PAIR_COUNT")
    expected = set()
    def edge(kind, a, b, lo=None, hi=None):
        expected.add((kind, a["id"], b["id"], lo, hi))
    for nodes in levels.values():
        for a, b in zip(nodes, nodes[1:]):
            edge("next", a, b)
    for left, right in itertools.combinations(levels, 2):
        right_nodes = levels[right]
        right_ends = [n["end"] for n in right_nodes]
        for a in levels[left]:
            j = bisect.bisect_right(right_ends, a["start"])
            while j < len(right_nodes) and right_nodes[j]["start"] < a["end"]:
                b = right_nodes[j]
                lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
                if (a["start"], a["end"]) == (b["start"], b["end"]):
                    edge("coextensive", a, b, lo, hi)
                elif a["start"] <= b["start"] and b["end"] <= a["end"]:
                    edge("contains", a, b, lo, hi)
                elif b["start"] <= a["start"] and a["end"] <= b["end"]:
                    edge("contains", b, a, lo, hi)
                else:
                    edge("overlaps", a, b, lo, hi)
                j += 1
    actual = [(e["relation"], e["from"], e["to"], e.get("start"), e.get("end")) for e in record["edges"]]
    check(len(actual) == len(set(actual)) and set(actual) == expected, "MNLM_MULTI_READBACK_RELATIONS")
    out = Counter(documents=1, source_characters=len(text), nodes=len(record["nodes"]), edges=len(actual))
    out.update(n["level"] for n in record["nodes"])
    out[record["page_status"]] += 1
    out[record["paragraph_status"]] += 1
    return out


def audit(root, source_root, layouts=None):
    report_raw = (root/"report.json").read_bytes()
    report = json.loads(report_raw)
    source_raw = (source_root/"report.json").read_bytes()
    source = json.loads(source_raw)
    check(report["source_report_sha256"] == sha(source_raw) and source["complete"], "MNLM_MULTI_READBACK_MANIFEST")
    check(report["complete"] and not report["errors"] and report["served"] is False
          and report["scientific_admission"] is False and set(report["files"]) == set(source["families"]),
          "MNLM_MULTI_READBACK_INCOMPLETE")
    layout_map = json.loads(layouts.read_bytes()) if layouts else {}
    check(report["layouts_sha256"] == (sha(layouts.read_bytes()) if layouts else None), "MNLM_MULTI_READBACK_LAYOUT_PIN")
    counts = {}
    for family, manifest in report["files"].items():
        check(manifest["file"] == family+".jsonl", "MNLM_MULTI_READBACK_FILENAME")
        raw = (root/manifest["file"]).read_bytes()
        check(sha(raw) == manifest["sha256"] and len(raw) == manifest["bytes"], "MNLM_MULTI_READBACK_HASH")
        row_raw = (source_root/"records"/family/"rows.jsonl").read_bytes()
        anchor_raw = (source_root/"chunked"/family/"anchors.json").read_bytes()
        check(sha(row_raw) == manifest["source_rows_sha256"] == source["families"][family]["rows_sha256"]
              and sha(anchor_raw) == manifest["anchors_sha256"], "MNLM_MULTI_READBACK_INPUT_HASH")
        rows = [json.loads(l) for l in row_raw.decode().split("\n") if l]
        selected = sorted({i for q in json.loads(anchor_raw).values() for i in q["rows"]})
        chunk_raw = (source_root/"chunked"/family/"chunks/atoms.parquet").read_bytes()
        check(sha(chunk_raw) == manifest["source_chunks_sha256"] == source["families"][family]["chunks_sha256"],
              "MNLM_MULTI_READBACK_INPUT_HASH")
        members = pq.read_table(io.BytesIO(chunk_raw), columns=["row"]).column("row").to_pylist()
        check(sorted(set(members)) == selected, "MNLM_MULTI_READBACK_POPULATION")
        records = [json.loads(l) for l in raw.decode().split("\n") if l]
        check([r["row"] for r in records] == selected, "MNLM_MULTI_READBACK_POPULATION")
        count = Counter()
        for r in records:
            src = rows[r["row"]]
            check(r["source"] == src["text_source"], "MNLM_MULTI_READBACK_SOURCE_REFERENCE")
            count.update(audit_document(r, src["text"], src["text_source"], src["entity"], layout_map.get(src["entity"])))
        check(dict(count) == report["counts"][family], "MNLM_MULTI_READBACK_COUNTS")
        counts[family] = dict(count)
    return {"acceptance": "author-checked", "counts": counts, "all_spans_and_relations_equal": True,
            "source_report_sha256": sha(source_raw), "multiscale_report_sha256": sha(report_raw)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--layouts", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.publication, a.source, a.layouts)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps(result))
