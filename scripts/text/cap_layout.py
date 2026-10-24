"""Bind retained CAP text to original physical pages without flattening footnotes.

Page spans address unchanged Unicode source text. Every non-whitespace character
must align exactly; HTML block page references, not marker order, assign pages.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

CONTRACT = "cap-physical-page-spans-v1"
VOID = {"br", "hr", "img", "input", "meta", "link", "wbr", "area", "base", "col", "embed", "param", "source", "track"}


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_CAP_LAYOUT_"+code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def date_record(value):
    check(isinstance(value, str) or value is None, "DATE_TYPE")
    if not value:
        return dict(value=value, precision="missing", status="unavailable")
    check(re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", value) is not None, "DATE_FORMAT")
    parts = [int(x) for x in value.split("-")]
    try:
        date(*(parts+[1]*(3-len(parts))))
    except ValueError as exc:
        raise ValueError("MNLM_CAP_LAYOUT_DATE_VALUE") from exc
    return dict(value=value, precision={1: "year", 2: "month", 3: "day"}[len(parts)], status="source-attested:decision-date")


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []; self.blocks = []; self.parts = []; self.n = 0; self.root = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs); classes = set(a.get("class", "").split())
        if not self.stack:
            check(self.root is None and tag == "section" and "casebody" in classes, "HTML_ROOT")
            self.root = a
        skipped = bool(self.stack and self.stack[-1][1])
        block = self.stack[-1][2] if self.stack else None
        marker = tag == "a" and "page-label" in classes
        if marker and not skipped:
            check(block is not None and a.get("data-label"), "MARKER_BLOCK")
            block["markers"].append([self.n, a["data-label"]])
        skipped |= marker or (tag == "a" and (bool(classes & {"bracketnum", "footnotemark"})
                                              or a.get("href", "").startswith("#ref_footnote_")))
        if "data-blocks" in a and not skipped:
            check(block is None, "NESTED_BLOCK")
            refs = json.loads(a["data-blocks"])
            check(isinstance(refs, list) and refs and all(isinstance(x, list) and len(x) == 3
                  and isinstance(x[0], str) and type(x[1]) is int and isinstance(x[2], list) for x in refs), "BLOCK_REFERENCES")
            orders = list(dict.fromkeys(x[1] for x in refs))
            check(orders == sorted(orders), "BLOCK_PAGE_ORDER")
            block = dict(element_id=a.get("id"), start=self.n, end=None, page_orders=orders, references=refs, markers=[])
            self.blocks.append(block)
        if tag not in VOID:
            self.stack.append((tag, skipped, block))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        check(bool(self.stack) and self.stack[-1][0] == tag, "HTML_BALANCE")
        _, _, block = self.stack.pop()
        if block is not None and (not self.stack or self.stack[-1][2] is not block):
            block["end"] = self.n

    def handle_data(self, data):
        value = "".join(c for c in data if not c.isspace())
        if not value or (self.stack and self.stack[-1][1]):
            return
        check(bool(self.stack) and self.stack[-1][2] is not None, "UNATTRIBUTED_TEXT")
        self.parts.append(value); self.n += len(value)


def map_case(row, metadata, html):
    match = re.fullmatch(r"f2d_([0-9]+)/html/([0-9]+-[0-9]+)\.html", row["source"]["document_id"])
    check(match is not None and metadata["file_name"] == match[2], "CASE_FILENAME")
    check(any(x["type"] == "official" and x["cite"] == f"{match[1]} F.2d {metadata['first_page']}"
              for x in metadata["citations"]), "CASE_CITATION")
    check(len(row["fields"]) == 1 and row["fields"][0]["name"] == "text", "SOURCE_FIELD")
    field = row["fields"][0]; text = field["text"]
    check(sha(text.encode()) == field["sha256"] and len(text) == field["characters"], "SOURCE_TEXT_HASH")
    parser = Blocks(); parser.feed(html); parser.close()
    check(not parser.stack and parser.root is not None, "HTML_COMPLETE")
    check(parser.root["data-firstpage"] == metadata["first_page"]
          and parser.root["data-lastpage"] == metadata["last_page"], "PAGE_METADATA")
    positions = [i for i, c in enumerate(text) if not c.isspace()]
    check("".join(parser.parts) == "".join(text[i] for i in positions), "CONTENT_ALIGNMENT")
    first, last = metadata["first_page_order"], metadata["last_page_order"]
    check(type(first) is int and type(last) is int and 0 < first <= last, "PAGE_RANGE")
    labels = {first: metadata["first_page"], last: metadata["last_page"]}
    segments = []; end = 0
    for block in parser.blocks:
        start, stop = block["start"], block["end"]; orders = block["page_orders"]
        check(start == end and type(stop) is int and stop > start, "BLOCK_COVERAGE")
        check(all(first <= p <= last for p in orders), "BLOCK_OUTSIDE_CASE")
        marks = block["markers"]; transitions = [m for m in marks if m[0] != start]
        check(len(transitions) == len(orders)-1 and all(start < m[0] < stop for m in transitions)
              and all(a[0] < b[0] for a, b in zip(transitions, transitions[1:])), "PAGE_TRANSITIONS")
        initial = [m for m in marks if m[0] == start]
        check(len(initial) <= 1, "PAGE_START_MARKERS")
        for order, marker in ([(orders[0], initial[0])] if initial else [])+list(zip(orders[1:], transitions)):
            check(order not in labels or labels[order] == marker[1], "PAGE_LABEL_CONFLICT")
            labels[order] = marker[1]
        cuts = [start]+[m[0] for m in transitions]+[stop]
        for a, b, order in zip(cuts, cuts[1:], orders):
            lo = 0 if a == 0 else positions[a]
            hi = len(text) if b == len(positions) else positions[b]
            if segments and segments[-1]["page_order"] == order:
                segments[-1]["end"] = hi
            else:
                segments.append(dict(start=lo, end=hi, page_order=order))
        end = stop
    check(end == len(positions) and set(s["page_order"] for s in segments) == set(range(first, last+1))
          and set(labels) == set(range(first, last+1)), "PAGE_COVERAGE")
    check(len(set(labels.values())) == len(labels), "PAGE_LABEL_DUPLICATE")
    pages = []; page_spans = {order: [] for order in labels}
    for segment in segments:
        page_spans[segment["page_order"]].append([segment["start"], segment["end"]])
    for order in sorted(labels):
        spans = page_spans[order]
        pages.append(dict(page_order=order, label=labels[order], source_spans=spans,
                          part_sha256=[sha(text[a:b].encode()) for a, b in spans]))
    return dict(document_id=row["id"], source_document_id=row["source"]["document_id"],
                text_sha256=field["sha256"], source_characters=len(text), case_id=metadata["id"],
                case_name=metadata["name"], court=metadata["court"], jurisdiction=metadata["jurisdiction"],
                decision_date=date_record(metadata.get("decision_date")), group_keys=row["group_keys"],
                contract=CONTRACT, pages=pages, block_evidence=parser.blocks,
                nonwhitespace_characters=len(positions), layout_accepted=True,
                eligibility_acceptance=False, sealed_evaluation=False, scientific_admission=False)


def run(source, capture, pin, output):
    start = time.perf_counter(); check(not output.exists(), "EXISTS")
    raw = (capture/"report.json").read_bytes(); report = json.loads(raw)
    check(sha(raw) == pin and report["complete"] is True and not report["errors"], "CAPTURE_PIN")
    source_raw = (source/"report.json").read_bytes(); source_report = json.loads(source_raw)
    check(sha(source_raw) == report["source_report_sha256"] and source_report["complete"] is True
          and not source_report["errors"], "SOURCE_PIN")
    check(file_sha(source/"legal.jsonl") == report["legal_sha256"] == source_report["sources"]["legal"]["sha256"], "SOURCE_HASH")
    with (source/"legal.jsonl").open("rb") as f:
        rows = [json.loads(line) for line in f][:report["requested_cases"]]
    check([r["id"] for r in rows] == report["selected_document_ids"] == [r["document_id"] for r in report["cases"]], "CASE_ORDER")
    output.mkdir(parents=True); errors = []; counts = Counter(); phases = Counter()
    dest = output/"layouts.jsonl"
    with dest.open("xb") as f:
        for row, item in zip(rows, report["cases"]):
            t = time.perf_counter(); counts["cases"] += 1
            try:
                check(item["captured_source"] == row["source"] and set(item["artifacts"]) == {"json", "html"}, "CASE_SOURCE")
                artifacts = {}
                match = re.fullmatch(r"f2d_([0-9]+)/html/([0-9]+-[0-9]+)\.html", row["source"]["document_id"])
                for kind, folder in (("json", "cases"), ("html", "html")):
                    rec = item["artifacts"][kind]; path = capture/"raw"/(row["id"]+"."+kind)
                    value = path.read_bytes()
                    check(match is not None and rec["file"] == path.name and len(value) == rec["bytes"] and sha(value) == rec["sha256"]
                          and rec["url"] == f"https://static.case.law/f2d/{match[1]}/{folder}/{match[2]}.{kind}", "ARTIFACT_HASH_URL")
                    artifacts[kind] = value
                phases["input_seconds"] += time.perf_counter()-t; t = time.perf_counter()
                result = map_case(row, json.loads(artifacts["json"]), artifacts["html"].decode())
                result["artifacts"] = {k: {n: v[n] for n in ("url", "sha256", "bytes")} for k, v in item["artifacts"].items()}
                counts.update(accepted=1, pages=len(result["pages"]), source_characters=result["source_characters"],
                              spans=sum(len(p["source_spans"]) for p in result["pages"]),
                              discontinuous_pages=sum(len(p["source_spans"]) > 1 for p in result["pages"]))
                phases["mapping_seconds"] += time.perf_counter()-t
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = dict(document_id=row["id"], layout_accepted=False, error=str(exc))
                errors.append(result); counts["refused"] += 1
                phases["refusal_seconds"] += time.perf_counter()-t
            t = time.perf_counter(); f.write(canonical(result)+b"\n"); phases["write_seconds"] += time.perf_counter()-t
    result = dict(contract=CONTRACT, capture_report_sha256=pin, source_report_sha256=sha(source_raw),
                  file=dict(name=dest.name, sha256=file_sha(dest), bytes=dest.stat().st_size),
                  counts=dict(counts), errors=errors, complete=not errors and len(rows) == report["requested_cases"],
                  eligibility_acceptance=False, sealed_evaluation=False, scientific_admission=False,
                  code_sha256=file_sha(Path(__file__)), timing=dict(phases))
    wall = time.perf_counter()-start
    result["timing"].update(wall_seconds=wall, other_seconds=wall-sum(phases.values()))
    publish(output/"report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "capture", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--capture-report-sha256", required=True)
    a = p.parse_args(); result = run(a.source, a.capture, a.capture_report_sha256, a.output)
    print(json.dumps(result))
    if not result["complete"]:
        raise SystemExit(2)
