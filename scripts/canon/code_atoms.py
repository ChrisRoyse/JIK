"""Stage 2 (issue #11, spec §18 "Operations, code" and "structural role flag"): atoms for fenced code blocks and markdown
headings inside prose records. Outside a fence the text is sentence-segmented (segment.py); a fence line and a heading line
are structural atoms (ROLE_STRUCT); the body of a fence declared as Python is split into top-level units by the CPython
`ast` (functions, classes, statement blocks; decorators and the comments and blank lines that precede a unit attach to it),
any other fence body is one code atom. A body that does not parse is one code atom tagged UNIT_UNPARSED (the text is the
text; a parse failure is recorded, never a refusal). Spans tile the record exactly, so conservation holds by construction.
Deterministic given the interpreter: the unit boundaries depend on CPython's grammar, so the manifest records the Python
version beside the segmenter id. Cost: O(chars) per record; one ast.parse per declared-Python fence body.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import ROLE_SENTENCE, ROLE_WS, segment  # noqa: F401
from fences import CODE_SHA256 as FENCES_SHA256, LINE_END, fences, lines

CODE_ATOMIZER_ID = hashlib.sha256(Path(__file__).read_bytes() + bytes.fromhex(FENCES_SHA256)).hexdigest()

ROLE_CODE, ROLE_STRUCT = 6, 7
UNIT_NONE, UNIT_FUNCTION, UNIT_CLASS, UNIT_STATEMENT, UNIT_UNPARSED, UNIT_BLOCK = 0, 1, 2, 3, 4, 5
UNIT_NAMES = {UNIT_NONE: "none", UNIT_FUNCTION: "function", UNIT_CLASS: "class", UNIT_STATEMENT: "statement", UNIT_UNPARSED: "unparsed", UNIT_BLOCK: "block"}
PYTHON_INFO = frozenset({"python", "py", "python3", "py3"})
PY_VERSION = "cpython-" + ".".join(map(str, sys.version_info[:3]))

_HEADING = re.compile(r"^#{1,6} \S")


def _units_python(body: str) -> list[tuple[int, int]] | None:
    """Top-level unit kinds and char starts for a Python fence body; None when it does not parse."""
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if not tree.body:
        return None
    line_start = [0, *(m.end() for m in LINE_END.finditer(body))]
    units = []
    for node in tree.body:
        lineno = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        kind = UNIT_FUNCTION if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else UNIT_CLASS if isinstance(node, ast.ClassDef) else UNIT_STATEMENT
        units.append((line_start[lineno - 1], kind))
    return units


def _code_spans(body_start: int, body: str, info: str) -> list[tuple[int, int, int, int]]:
    """(start, end, role, unit) spans tiling body (absolute offsets)."""
    if not body:
        return []
    lang = info.split()[0].lower() if info.strip() else ""
    if lang in PYTHON_INFO:
        units = _units_python(body)
        if units is None:
            return [(body_start, body_start + len(body), ROLE_CODE, UNIT_UNPARSED)]
        starts = [0] + [s for s, _ in units[1:]]  # the first unit takes any leading comment or blank lines
        ends = starts[1:] + [len(body)]
        return [(body_start + s, body_start + e, ROLE_CODE, k) for (s, e), (_, k) in zip(zip(starts, ends), units) if e > s]
    return [(body_start, body_start + len(body), ROLE_CODE, UNIT_BLOCK)]


def _prose_spans(start: int, text: str) -> list[tuple[int, int, int, int]]:
    """Sentence atoms with heading lines flagged structural (absolute offsets)."""
    out: list[tuple[int, int, int, int]] = []
    pos = 0
    for a, _, b in lines(text):
        line = text[a:b]
        if _HEADING.match(line):
            if a > pos:
                out += [(start + pos + s, start + pos + e, r, UNIT_NONE) for s, e, r in segment(text[pos:a])]
            out.append((start + a, start + b, ROLE_STRUCT, UNIT_NONE))
            pos = b
    if pos < len(text):
        out += [(start + pos + s, start + pos + e, r, UNIT_NONE) for s, e, r in segment(text[pos:])]
    return out


def atomise_text(text: str) -> tuple[list[tuple[int, int, int, int]], dict]:
    """(start, end, role, unit) spans tiling `text`; stats: fences, unclosed, unparsed."""
    out: list[tuple[int, int, int, int]] = []
    stats = {"fences": 0, "unclosed": 0, "unparsed": 0}
    pos = 0  # start of the prose region not yet emitted
    n = len(text)
    for i, body_start, body_end, close_end, info, closed in fences(text):
        if not closed:
            stats["unclosed"] += 1
        stats["fences"] += 1
        if i > pos:
            out += _prose_spans(pos, text[pos:i])
        out.append((i, body_start, ROLE_STRUCT, UNIT_NONE))
        spans = _code_spans(body_start, text[body_start:body_end], info)
        stats["unparsed"] += sum(1 for s in spans if s[3] == UNIT_UNPARSED)
        out += spans
        if closed:
            out.append((body_end, close_end, ROLE_STRUCT, UNIT_NONE))
        pos = close_end
    if pos < n:
        out += _prose_spans(pos, text[pos:])
    return out, stats


if __name__ == "__main__":
    sample = "Intro line. Second sentence.\n# Heading\n```python\n# comment\nimport os\n\n@dec\ndef f(x):\n    return x\n\nclass C:\n    pass\nprint(f(1))\n```\nAfter. Done.\n```\nraw block\n```\n```py\ndef broken(:\n```\n"
    spans, st = atomise_text(sample)
    assert "".join(sample[s:e] for s, e, _, _ in spans) == sample and spans[0][0] == 0 and spans[-1][1] == len(sample)
    assert all(spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1))
    for s, e, r, u in spans:
        print(r, UNIT_NAMES[u], repr(sample[s:e]))
    print(st, PY_VERSION)
    two = "A first paragraph.\n\nSecond paragraph here.\n# Late heading\nTail. End.\n## Another\n"  # a heading after the first region (the p0-v1 refusal of 2026-10-15)
    sp, _ = atomise_text(two)
    assert sp[0][0] == 0 and sp[-1][1] == len(two) and all(sp[k][1] == sp[k + 1][0] for k in range(len(sp) - 1)), sp
    assert [two[s:e] for s, e, r, _ in sp if r == ROLE_STRUCT] == ["# Late heading\n", "## Another\n"]
    print("late-heading tiling ok")
