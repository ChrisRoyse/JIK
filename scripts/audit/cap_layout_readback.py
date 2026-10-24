"""Cold CAP audit with an XML tree walk independent of the streaming mapper."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import html.entities
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_CAP_READBACK_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def raw_audit(capture, source):
    report_raw = (capture/"report.json").read_bytes(); report = json.loads(report_raw)
    source_raw = (source/"report.json").read_bytes(); parent = json.loads(source_raw)
    check(report["schema"] == "cap-evaluation-originals-v1" and report["complete"] is True
          and report["errors"] == [] and parent["complete"] is True and parent["errors"] == [], "RAW_COMPLETE")
    check(sha(source_raw) == report["source_report_sha256"] and file_sha(source/"legal.jsonl") ==
          report["legal_sha256"] == parent["sources"]["legal"]["sha256"], "RAW_PARENT")
    with (source/"legal.jsonl").open("rb") as f:
        rows = [json.loads(line) for line in f][:report["requested_cases"]]
    check(len(rows) == report["requested_cases"] and [r["id"] for r in rows] == report["selected_document_ids"]
          and len(report["cases"]) == len(rows), "RAW_COUNT")
    with (capture/"cases.jsonl").open("rb") as f:
        check([json.loads(line) for line in f] == report["cases"], "RAW_LEDGER")
    count = Counter(); loaded = []
    for index, (row, item) in enumerate(zip(rows, report["cases"]), 1):
        check(row["id"] == item["document_id"] and row["source"] == item["captured_source"], "RAW_IDENTITY")
        check(set(item["artifacts"]) == {"json", "html"}, "RAW_SIBLINGS")
        source_id = row["source"]["document_id"]
        volume, path = source_id.split("/html/"); volume = volume.removeprefix("f2d_")
        stem = path.removesuffix(".html"); values = {}
        for kind in ("json", "html"):
            rec = item["artifacts"][kind]; name = row["id"]+"."+kind
            value = (capture/"raw"/name).read_bytes()
            folder = "cases" if kind == "json" else "html"
            check(rec["file"] == name and rec["status"] == 200 and len(value) == rec["bytes"]
                  and sha(value) == rec["sha256"] and rec["url"] == f"https://static.case.law/f2d/{volume}/{folder}/{stem}.{kind}", "RAW_BYTES")
            check(0 < len(value) <= 8*1024*1024 and 1 <= rec["attempts"] <= 2, "RAW_BUDGET")
            count.update(files=1, bytes=len(value)); count["reused_bytes" if rec["reused"] else "received_bytes"] += len(value)
            values[kind] = value
        checkpoint = json.loads((capture/"checkpoints"/f"{index:04d}.json").read_bytes())
        check(checkpoint["source_report_sha256"] == sha(source_raw) and checkpoint["completed_cases"] == index
              and checkpoint["errors"] == 0 and checkpoint["last_document_id"] == row["id"]
              and checkpoint["totals"]["reused_bytes"] == count["reused_bytes"]
              and checkpoint["totals"]["received_bytes"] >= count["received_bytes"], "RAW_CHECKPOINT")
        loaded.append((row, item, json.loads(values["json"]), values["html"].decode()))
    check(count["bytes"] <= 256*1024*1024 and report["totals"]["reused_bytes"] == count["reused_bytes"]
          and report["totals"]["received_bytes"] >= count["received_bytes"], "RAW_TOTALS")
    check(all(report[k] is False for k in ("layout_acceptance", "eligibility_acceptance", "sealed_evaluation", "served", "scientific_admission")), "RAW_CLAIMS")
    return dict(report_sha256=sha(report_raw), counts=dict(count), cases=len(rows)), loaded


def tree_assignment(value):
    # XML and HTML named entities differ; retain entity values as numeric XML references.
    value = re.sub(r"&([A-Za-z][A-Za-z0-9]+);", lambda m: "".join("&#%d;" % ord(c)
                   for c in html.entities.html5[m[1]+";"]), value)
    root = ET.fromstring(value)
    check(root.tag == "section" and "casebody" in root.get("class", "").split(), "TREE_ROOT")
    chars = []; orders = []; blocks = []; labels = {}

    def text(s, state):
        for c in s or "":
            if not c.isspace():
                check(state is not None, "TREE_UNATTRIBUTED")
                chars.append(c); orders.append(state["pages"][state["index"]])

    def walk(node, state=None):
        classes = node.get("class", "").split()
        if node.tag == "a":
            if "page-label" in classes:
                check(state is not None, "TREE_MARKER")
                if len(chars) > state["start"]:
                    state["index"] += 1
                check(state["index"] < len(state["pages"]), "TREE_TRANSITIONS")
                page = state["pages"][state["index"]]; label = node.attrib["data-label"]
                check(page not in labels or labels[page] == label, "TREE_LABEL")
                labels[page] = label; state["markers"].append([len(chars), label]); return
            if set(classes) & {"bracketnum", "footnotemark"} or node.get("href", "").startswith("#ref_footnote_"):
                return
        own = "data-blocks" in node.attrib
        if own:
            check(state is None, "TREE_NESTED")
            refs = json.loads(node.attrib["data-blocks"]); pages = []
            for ref in refs:
                if not pages or ref[1] != pages[-1]:
                    pages.append(ref[1])
            check(pages and pages == sorted(set(pages)), "TREE_PAGE_ORDER")
            state = dict(pages=pages, index=0, start=len(chars), markers=[])
        text(node.text, state)
        for child in node:
            walk(child, state); text(child.tail, state)
        if own:
            check(state["index"] == len(state["pages"])-1, "TREE_TRANSITIONS")
            blocks.append(dict(element_id=node.get("id"), start=state["start"], end=len(chars),
                               references=refs, page_orders=pages, markers=state["markers"]))
    walk(root)
    return root, "".join(chars), orders, blocks, labels


def audit_case(result, row, metadata, html):
    field = row["fields"][0]; text = field["text"]
    check(sha(text.encode()) == field["sha256"] and result["text_sha256"] == field["sha256"]
          and result["source_characters"] == len(text), "TEXT_HASH")
    volume, name = row["source"]["document_id"].split("/html/")
    check(metadata["file_name"]+".html" == name and any(c["type"] == "official" and c["cite"] ==
          volume.removeprefix("f2d_")+" F.2d "+metadata["first_page"] for c in metadata["citations"]), "CASE_IDENTITY")
    for key, expected in (("document_id", row["id"]), ("source_document_id", row["source"]["document_id"]),
                          ("group_keys", row["group_keys"]), ("case_id", metadata["id"]), ("case_name", metadata["name"]),
                          ("court", metadata["court"]), ("jurisdiction", metadata["jurisdiction"])):
        check(result[key] == expected, "METADATA")
    value = metadata.get("decision_date")
    if value:
        check(re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", value) is not None, "DATE")
        components = list(map(int, value.split("-"))); date(*(components+[1]*(3-len(components))))
        expected = dict(value=value, precision=["year", "month", "day"][len(components)-1], status="source-attested:decision-date")
    else:
        expected = dict(value=value, precision="missing", status="unavailable")
    check(result["decision_date"] == expected, "DATE")
    root, normalized, char_pages, blocks, labels = tree_assignment(html)
    positions = [i for i, c in enumerate(text) if not c.isspace()]
    check(normalized == "".join(text[i] for i in positions) and result["nonwhitespace_characters"] == len(positions), "CONTENT")
    first, last = metadata["first_page_order"], metadata["last_page_order"]
    check(root.get("data-firstpage") == metadata["first_page"] and root.get("data-lastpage") == metadata["last_page"], "PAGE_METADATA")
    for page, label in ((first, metadata["first_page"]), (last, metadata["last_page"])):
        check(page not in labels or labels[page] == label, "PAGE_LABEL")
        labels[page] = label
    check(set(char_pages) == set(range(first, last+1)) == set(labels), "PAGE_COVERAGE")
    spans = {p: [] for p in labels}; start = 0
    for i in range(1, len(char_pages)+1):
        if i == len(char_pages) or char_pages[i] != char_pages[start]:
            spans[char_pages[start]].append([0 if start == 0 else positions[start], len(text) if i == len(char_pages) else positions[i]])
            start = i
    expected = [dict(page_order=p, label=labels[p], source_spans=spans[p],
                     part_sha256=[sha(text[a:b].encode()) for a, b in spans[p]]) for p in sorted(labels)]
    check(result["pages"] == expected, "PAGE_SPANS")
    check(result["block_evidence"] == blocks, "BLOCK_EVIDENCE")
    check(result["contract"] == "cap-physical-page-spans-v1" and result["layout_accepted"] is True
          and all(result[k] is False for k in ("eligibility_acceptance", "sealed_evaluation", "scientific_admission")), "CLAIMS")
    return Counter(accepted=1, pages=len(expected), source_characters=len(text), spans=sum(len(v) for v in spans.values()),
                   discontinuous_pages=sum(len(v) > 1 for v in spans.values()))


def audit(root, capture, source):
    raw, cases = raw_audit(capture, source)
    report_bytes = (root/"report.json").read_bytes(); report = json.loads(report_bytes)
    check(report["capture_report_sha256"] == raw["report_sha256"] and report["source_report_sha256"] == file_sha(source/"report.json"), "PARENT")
    dest = root/"layouts.jsonl"; rec = report["file"]
    check(rec["name"] == dest.name and rec["sha256"] == file_sha(dest) and rec["bytes"] == dest.stat().st_size, "OUTPUT_HASH")
    counts = Counter(cases=len(cases)); errors = []
    with dest.open("rb") as f:
        for row, item, metadata, html in cases:
            line = next(f, None); check(line is not None, "MISSING_ROW")
            result = json.loads(line)
            check(result["document_id"] == row["id"], "ROW_ORDER")
            if result["layout_accepted"]:
                counts.update(audit_case(result, row, metadata, html))
                check(result["artifacts"] == {k: {n: v[n] for n in ("url", "sha256", "bytes")} for k, v in item["artifacts"].items()}, "OUTPUT_PROVENANCE")
            else:
                check(isinstance(result.get("error"), str) and result["error"], "REFUSAL")
                counts["refused"] += 1; errors.append(result)
        check(next(f, None) is None, "EXTRA_ROW")
    check(report["counts"] == dict(counts) and report["errors"] == errors and report["complete"] == (not errors), "COUNTS")
    check(all(report[k] is False for k in ("eligibility_acceptance", "sealed_evaluation", "scientific_admission")), "REPORT_CLAIMS")
    return dict(acceptance="author-checked", raw=raw, counts=dict(counts), refused=errors,
                all_accepted_character_assignments_equal=True, report_sha256=sha(report_bytes), sealed_evaluation=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "capture", "publication", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); result = audit(a.publication, a.capture, a.source)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps(result))
