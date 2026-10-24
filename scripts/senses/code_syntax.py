"""Declared code -> typed, atom-local syntax for the existing structure sense.

This adapter never executes code, resolves names, reads test labels or claims
semantic equivalence. AST node order, parent field/index, operators and UTF-8
source spans survive; identifiers and literal values remain in the surface.
Unparsed/unsupported/budget-limited inputs have diagnostics, never a partial AST.
Cost: one O(bytes) fence scan per record, then O(atom bytes + nodes) per code atom.
"""
from __future__ import annotations

import ast
from bisect import bisect_right
import hashlib
from pathlib import Path
import platform
import re
from typing import NamedTuple

from .contract import digest, field, require
from canon.fences import CODE_SHA256 as FENCES_SHA256, fences

VERSION = "atom-syntax-v2"
MAX_BYTES = 262144
MAX_NODES = 4096
PYTHON = frozenset({"python", "py", "python3", "py3"})
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def contract():
    return {"schema": VERSION, "parser": "stdlib.ast.parse(mode=exec)",
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "code_sha256": CODE_SHA256,
            "fences_sha256": FENCES_SHA256,
            "max_utf8_bytes": MAX_BYTES, "max_nodes": MAX_NODES,
            "line_endings": ["LF", "CRLF", "CR"], "source_normalisation": False,
            "coordinates": "atom-relative UTF-8 byte offsets", "execution": False}


CONTRACT_ID = digest(contract())


class SyntaxSource:
    """Record-scoped declaration index; content outside the atom is never parsed."""

    def __init__(self, text):
        self.text = text
        self.binding = CONTRACT_ID
        self.regions = [(a, b, language, closed) for _, a, b, _, language, closed in fences(text)]
        self.starts = [r[0] for r in self.regions]

    def for_span(self, start, end):
        require(0 <= start < end <= len(self.text), "MNLM_SYNTAX_SPAN", "atom outside retained text")
        index = bisect_right(self.starts, start) - 1
        if index < 0:
            return []
        lo, hi, language, closed = self.regions[index]
        if not (lo <= start < end <= hi):
            return []
        return syntax_fields(self.text[start:end], language, self.binding) + [
            field("syntax.fence_closed", closed, "boolean")]


class SyntaxAnalysis(NamedTuple):
    fields: list
    nodes: list
    node_fields: list
    line_starts: list


def syntax_fields(text, language, binding=None):
    """A source adapter may also supply an explicit language for a bare code atom."""
    return analyse(text, language, binding).fields


def analyse(text, language, binding=None):
    """One parse/traversal; prepared node occurrences can serve source-bound units."""
    require(isinstance(text, str) and isinstance(language, str), "MNLM_SYNTAX_INPUT", "text and declared language required")
    prefix = [field("syntax.contract", binding or CONTRACT_ID, "reference"),
              field("syntax.source", hashlib.sha256(text.encode()).hexdigest(), "reference"),
              field("syntax.language", language, "category")]

    def failure(code, remediation):
        return SyntaxAnalysis(prefix + [field("syntax.status", code, "category"),
                              field("syntax.remediation", remediation, "reference")], [], [], [])

    if language not in PYTHON:
        return failure("MNLM_SYNTAX_LANGUAGE_UNSUPPORTED", "Register a versioned parser for the declared language; do not guess it.")
    raw = text.encode("utf-8")
    if len(raw) > MAX_BYTES:
        return failure("MNLM_SYNTAX_BYTE_BUDGET", "Keep the source; split at declared structural boundaries before syntax extraction.")
    try:
        tree = ast.parse(text, mode="exec")
    except (SyntaxError, ValueError):
        return failure("MNLM_SYNTAX_UNPARSED", "Retain original bytes; review syntax or the declared grammar. No AST was emitted.")
    except (RecursionError, MemoryError):
        return failure("MNLM_SYNTAX_PARSER_RESOURCE", "Retain original bytes; use a bounded structural unit. No AST was emitted.")
    # Iterative traversal: parser limits do not justify a second recursive walk.
    # Operators/context objects may be singletons; each occurrence is a node.
    stack, nodes = [(tree, -1, "root", -1)], []
    while stack:
        if len(nodes) == MAX_NODES:
            return failure("MNLM_SYNTAX_NODE_BUDGET", "Keep the source; split at structural boundaries. No partial AST was emitted.")
        node, parent, edge, position = stack.pop()
        ordinal = len(nodes)
        nodes.append((node, parent, edge, position))
        children = []
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                children.append((value, ordinal, name, -1))
            elif isinstance(value, list):
                children.extend((child, ordinal, name, i) for i, child in enumerate(value) if isinstance(child, ast.AST))
        stack.extend(reversed(children))
    # CPython columns are UTF-8 byte offsets, not Python string indices.
    line_starts = [0, *(m.end() for m in re.finditer(rb"\r\n|\r|\n", raw))]
    output = prefix + [field("syntax.status", "parsed", "category"),
                       field("syntax.nodes", len(nodes), "integer", "node")]
    emissions = []
    for ordinal, (node, parent, edge, position) in enumerate(nodes):
        fields = [field("syntax.node", ordinal, "integer", "node"),
                       field("syntax.kind", type(node).__name__, "category"),
                       field("syntax.parent", parent, "integer", "node"),
                       field("syntax.edge", edge, "category"),
                       field("syntax.index", position, "integer", "ordinal")]
        has_span = all(getattr(node, k, None) is not None for k in ("lineno", "col_offset", "end_lineno", "end_col_offset"))
        fields.append(field("syntax.has_span", has_span, "boolean"))
        if has_span:
            start = line_starts[node.lineno - 1] + node.col_offset
            end = line_starts[node.end_lineno - 1] + node.end_col_offset
            require(0 <= start < end <= len(raw), "MNLM_SYNTAX_SPAN", "parser span exceeds atom bytes")
            fields.extend([field("syntax.start", start, "integer", "utf8_byte"),
                           field("syntax.end", end, "integer", "utf8_byte")])
        for name in ("is_async", "simple", "conversion", "level"):
            value = getattr(node, name, None)
            if type(value) is int:
                fields.append(field("syntax." + name, value, "integer"))
        emissions.append(fields)
        output.extend(fields)
    return SyntaxAnalysis(output, nodes, emissions, line_starts)
