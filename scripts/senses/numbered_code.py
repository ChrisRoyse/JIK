"""Rendered numbered file views -> typed syntax, without executing source.

The filename declares a grammar, not an authenticated repository file. The
numbered display is retained byte-for-byte; extracted code is explicitly a
rendered excerpt. No dedenting, missing-line repair or language guessing.
"""
import copy
import hashlib
from pathlib import Path, PurePosixPath
import re

from canon.fences import lines
from .code_syntax import contract as syntax_contract, syntax_fields, MAX_BYTES
from .contract import digest, field, require, seal_record, validate_record

SCHEMA = "numbered-file-view-v1"
HEADER = re.compile(r"\[File: (.+) \(([0-9]+) lines total\)\]")
NUMBERED = re.compile(r"([0-9]+):(.*)")
OMITTED = re.compile(r"\(([0-9]+) more lines (above|below)\)")
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def contract():
    return {"schema": SCHEMA, "code_sha256": CODE_SHA256, "syntax": syntax_contract(),
            "source": "rendered-numbered-excerpt", "python_suffixes": [".py", ".pyi"],
            "dedent": False, "execution": False, "whole_repository_file_attested": False}


CONTRACT = contract()
CONTRACT_ID = digest(CONTRACT)


def parse(text):
    require(isinstance(text, str), "MNLM_FILE_VIEW_INPUT", "retained response text required")
    iterator = iter(lines(text))
    first = next(iterator, None)
    if first is None:
        return None
    header = HEADER.fullmatch(text[first[0]:first[1]])
    if header is None:
        return None
    path, declared = header.groups()
    receipt = {"schema": SCHEMA, "contract": CONTRACT_ID, "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "path": path, "declared_total": declared, "line_map": [], "rendered_text": None,
               "full_display": False, "whole_repository_file_attested": False}
    prefix = [field("file_view.contract", CONTRACT_ID, "reference"),
              field("file_view.response", receipt["response_sha256"], "reference"),
              field("file_view.path", path, "reference")]

    def fail(code):
        receipt["status"] = code
        receipt["fields"] = prefix + [field("file_view.status", code, "category")]
        # No partial syntax or reconstructed code is published on a bad boundary.
        receipt["line_map"], receipt["rendered_text"] = [], None
        return receipt

    if len(text.encode()) > MAX_BYTES:
        return fail("MNLM_FILE_VIEW_BYTE_BUDGET")
    if len(declared) > 9:
        return fail("MNLM_FILE_VIEW_LINE_RANGE")
    total = int(declared)
    receipt["declared_total"] = total
    if total < 1:
        return fail("MNLM_FILE_VIEW_LINE_RANGE")
    row = next(iterator, None)
    above = 0
    if row is not None:
        omitted = OMITTED.fullmatch(text[row[0]:row[1]])
        if omitted:
            if omitted[2] != "above" or len(omitted[1]) > 9:
                return fail("MNLM_FILE_VIEW_OMISSION")
            above = int(omitted[1])
            row = next(iterator, None)
    pieces, mapping = [], []
    expected = above + 1
    while row is not None:
        start, end, line_end = row
        match = NUMBERED.fullmatch(text[start:end])
        if match is None:
            break
        if len(match[1]) > 9 or int(match[1]) != expected or expected > total:
            return fail("MNLM_FILE_VIEW_LINE_RANGE")
        content_start = start + len(match[1]) + 1
        pieces.append(text[content_start:line_end])
        mapping.append({"line": expected, "response_chars": [content_start, line_end]})
        expected += 1
        row = next(iterator, None)
    if not pieces:
        return fail("MNLM_FILE_VIEW_NO_LINES")
    below = total - expected + 1
    omitted = OMITTED.fullmatch(text[row[0]:row[1]]) if row else None
    if below and (omitted is None or omitted[2] != "below" or len(omitted[1]) > 9 or int(omitted[1]) != below):
        return fail("MNLM_FILE_VIEW_OMISSION")
    if not below and omitted is not None:
        return fail("MNLM_FILE_VIEW_OMISSION")
    rendered = "".join(pieces)
    language = "python" if PurePosixPath(path).suffix.lower() in (".py", ".pyi") else "unsupported"
    fs = syntax_fields(rendered, language)
    state = next(f["value"] for f in fs if f["name"] == "syntax.status")
    receipt.update(status=state, line_map=mapping, rendered_text=rendered, full_display=above == below == 0,
                   above=above, below=below, language=language)
    receipt["fields"] = prefix + [field("file_view.status", state, "category"),
        field("file_view.extent", "full_display" if receipt["full_display"] else "partial_display", "category"),
        field("file_view.coordinates", "rendered-excerpt-relative-utf8-bytes", "reference"),
        field("file_view.mapping", digest(mapping), "reference"),
        field("file_view.first_line", above + 1, "integer", "line"),
        field("file_view.last_line", expected - 1, "integer", "line"),
        field("file_view.declared_total", total, "integer", "line"), *fs]
    return receipt


def enrich(record):
    """Adapter only: existing source-neutral encoder consumes the returned fields."""
    validate_record(record)
    require(not any(f["name"].startswith("file_view.") for c in ("structure", "provenance")
                    for f in record["channels"][c]["items"]),
            "MNLM_FILE_VIEW_ALREADY_DERIVED", "reopen the retained parent; do not append the adapter twice")
    out, receipts, fields = copy.deepcopy(record), [], []
    for i, event in enumerate(record["channels"]["behaviour"]["items"]):
        for j, item in enumerate(event["measurements"]):
            if item["type"] != "text":
                continue
            view = parse(item["value"])
            if view is None:
                continue
            fields += [field("file_view.ordinal", len(receipts), "integer", "view"), *view["fields"]]
            receipts.append({"event_index": i, "field_index": j, "field_name": item["name"], **view})
    if receipts:
        # All original observed channels and bytes remain, with a new derived ID.
        from .contract import present
        out["channels"]["structure"] = present(out["channels"]["structure"]["items"] + fields)
        out["channels"]["provenance"] = present(out["channels"]["provenance"]["items"] + [
            field("file_view.parent_atom", record["atom_id"], "reference"),
            field("file_view.adapter", CONTRACT_ID, "reference")])
        out = seal_record(out)
    return out, receipts
