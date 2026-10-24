"""Lossless Python file partitions with a complete, source-bound syntax graph.

Units start at module statements and class members. Indented units are never
dedented or reparsed. Every AST occurrence belongs to exactly one byte partition;
its full-file coordinates and parent edge survive even when the parent is outside.
"""
import ast
from bisect import bisect_right
from collections import Counter
import hashlib
import io
from pathlib import Path
import tokenize

from .code_syntax import analyse, contract as syntax_contract
from .contract import digest, field, require

CODE = "MNLM_CODE_UNITS"
VERSION = "source-code-units-v1"
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def contract():
    return {"schema": VERSION, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "syntax": syntax_contract(), "coordinates": "file-relative UTF-8 bytes",
            "boundaries": "module statements, class members and namespace collection entries; functions stay whole",
            "encoding": "declared file codec must decode to the exact observed UTF-8 text; no normalization",
            "coverage": "exact byte tiling and every AST occurrence once", "execution": False}


def partition(raw, language):
    text = raw.decode("utf-8")
    require(bool(raw), CODE, "empty files have no nonempty units")
    encoding, encoding_ok = None, True
    if language == "python":
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            encoding_ok = raw.decode(encoding) == text
        except (SyntaxError, LookupError, UnicodeError):
            encoding_ok = False
    a = analyse(text, language if encoding_ok else "unsupported")
    if not encoding_ok:
        replacements = {"syntax.language": language, "syntax.status": "MNLM_CODE_UNIT_ENCODING",
                        "syntax.remediation": "Retain bytes; use a parser with explicit declared-encoding offsets. Do not reinterpret."}
        for f in a.fields:
            if f["name"] in replacements:
                f["value"] = replacements[f["name"]]
    file_sha = hashlib.sha256(raw).hexdigest()
    if not a.nodes:
        unit = {"span": [0, len(raw)], "kind": "unparsed_or_unsupported_file",
                "key": ["exact", file_sha], "nodes": [], "parents": [], "sha256": file_sha,
                "syntax": a.fields, "history_reason": "exact_bytes_only"}
        context = {"schema": VERSION, "file_sha256": file_sha, "nodes": [], "units": [summary(unit)],
                   "syntax": syntax_contract(), "declared_encoding": encoding, "parsed": False, "admitted": False}
        return [unit], context

    starts, scopes, keys = {0: []}, [], {}
    for i, (node, parent, edge, _) in enumerate(a.nodes):
        ancestors = scopes[parent] if parent >= 0 else ()
        scope = (*ancestors, (type(node).__name__, node.name)) if isinstance(node, DEFINITIONS) else ancestors
        scopes.append(scope)
        if isinstance(node, DEFINITIONS):
            keys[i] = scope
        in_function = any(kind in ("FunctionDef", "AsyncFunctionDef") for kind, _ in ancestors)
        if parent >= 0 and edge == "body" and isinstance(a.nodes[parent][0], (ast.Module, ast.ClassDef)) and not in_function:
            line = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
            starts.setdefault(a.line_starts[line - 1], []).append(i)
        if parent >= 0 and not in_function and edge in ("elts", "keys", "values"):
            collection, assignment = a.nodes[parent][0], a.nodes[parent][1]
            if isinstance(collection, (ast.Tuple, ast.List, ast.Set, ast.Dict)) and assignment >= 0:
                statement, namespace = a.nodes[assignment][0], a.nodes[assignment][1]
                if (isinstance(statement, (ast.Assign, ast.AnnAssign)) and namespace >= 0
                        and isinstance(a.nodes[namespace][0], (ast.Module, ast.ClassDef))
                        and collection.end_lineno > collection.lineno and node.lineno > collection.lineno):
                    starts.setdefault(a.line_starts[node.lineno - 1], []).append(i)
    bounds = sorted(starts)
    ends = bounds[1:] + [len(raw)]
    counts = Counter(keys.values())
    units = []
    for lo, hi in zip(bounds, ends):
        require(0 <= lo < hi <= len(raw), CODE, "nonempty, ordered byte partition required")
        declarations = [i for i in starts[lo] if i in keys]
        names = [type(a.nodes[i][0]).__name__ for i in starts[lo]]
        key, reason = None, "no_unique_structural_identity"
        if len(declarations) == 1:
            scope = keys[declarations[0]]
            if all(counts[scope[:j]] == 1 for j in range(1, len(scope) + 1)):
                key, reason = ["definition", list(map(list, scope))], "unique_qualified_definition"
            else:
                reason = "ambiguous_definition_or_ancestor"
        elif not declarations:
            # Exact bytes in the same enclosing declaration, never fuzzy matching.
            enclosing = scopes[a.nodes[starts[lo][0]][1]] if starts[lo] else ()
            key, reason = ["exact", hashlib.sha256(raw[lo:hi]).hexdigest(), list(map(list, enclosing))], "exact_scoped_bytes"
        units.append({"span": [lo, hi], "kind": "+".join(names) or "module_prefix", "key": key,
                      "nodes": [], "parents": [], "sha256": hashlib.sha256(raw[lo:hi]).hexdigest(),
                      "history_reason": reason})
    owners, graph = [], []
    for i, ((node, parent, edge, position), emitted) in enumerate(zip(a.nodes, a.node_fields)):
        fields = {f["name"]: f["value"] for f in emitted}
        anchor = fields.get("syntax.start")
        owner = bisect_right(bounds, anchor) - 1 if anchor is not None else (owners[parent] if parent >= 0 else 0)
        owners.append(owner)
        units[owner]["nodes"].append(i)
        graph.append({"id": i, "parent": parent, "field": edge, "index": position,
                      "kind": type(node).__name__, "owner": owner,
                      "span": [fields["syntax.start"], fields["syntax.end"]] if fields["syntax.has_span"] else None})
        if parent >= 0 and owners[parent] != owner:
            units[owner]["parents"].append(parent)
    for u in units:
        u["parents"] = sorted(set(u["parents"]))
        u["syntax"] = [*a.fields[:4], field("syntax.nodes", len(u["nodes"]), "integer", "node"),
                       field("syntax.file_nodes", len(a.nodes), "integer", "node"),
                       field("syntax.coordinates", "file_relative_utf8_bytes", "category"),
                       *[f for i in u["nodes"] for f in a.node_fields[i]]]
    context = {"schema": VERSION, "file_sha256": file_sha, "nodes": graph,
               "units": [summary(u) for u in units], "syntax": syntax_contract(), "declared_encoding": encoding,
               "parsed": True, "admitted": False}
    validate_partition(raw, units, context)
    return units, context


def summary(unit):
    return {k: v for k, v in unit.items() if k != "syntax"}


def validate_partition(raw, units, context):
    require(context["file_sha256"] == hashlib.sha256(raw).hexdigest()
            and context["units"] == [summary(u) for u in units], CODE, "source/context binding changed")
    cursor, assigned = 0, []
    for i, u in enumerate(units):
        lo, hi = u["span"]
        require(lo == cursor < hi <= len(raw) and hashlib.sha256(raw[lo:hi]).hexdigest() == u["sha256"],
                CODE, "unit bytes overlap, have a gap or changed")
        raw[lo:hi].decode("utf-8")
        cursor = hi
        assigned.extend(u["nodes"])
        require(all(context["nodes"][n]["owner"] == i for n in u["nodes"]), CODE, "node owner differs")
        expected = sorted({context["nodes"][n]["parent"] for n in u["nodes"]
                           if context["nodes"][n]["parent"] >= 0
                           and context["nodes"][context["nodes"][n]["parent"]]["owner"] != i})
        require(u["parents"] == expected, CODE, "cross-unit parent edge missing")
    require(cursor == len(raw) and sorted(assigned) == list(range(len(context["nodes"]))), CODE,
            "source byte/node conservation failed")
    for i, node in enumerate(context["nodes"]):
        require(node["id"] == i and -1 <= node["parent"] < i, CODE, "invalid syntax parent identity")
