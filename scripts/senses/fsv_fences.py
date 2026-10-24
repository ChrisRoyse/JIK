"""Real-prefix fence/atom/syntax boundary proof; controls never enter training.

One source pass O(bytes + atoms + AST nodes), no outcome reads, no GPU. Persist
every region and atom span, then separately cold-read and restate boundaries.
"""
import argparse
import ast
from collections import Counter, deque
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.code_syntax import SyntaxSource, contract, syntax_fields, PYTHON
from senses.contract import canonical, digest
from senses.instruments import tokens
from senses.pipeline import file_hash
from senses.trajectory_prefix import load
from canon.code_atoms import atomise_text, CODE_ATOMIZER_ID
from canon.fsv_stage2 import code_record_check, fences, tiles


def status(fields):
    return next(f["value"] for f in fields if f["name"] == "syntax.status")


def reference_regions(text):
    return [(a, b, info.split()[0].lower() if info.strip() else "unspecified", b < end)
            for _, a, b, end, info in fences(text)]


def audit_ast(text, fields):
    """BFS AST paths, independently recovered raw-byte offsets, no producer walk."""
    nodes = []
    for f in fields:
        if f["name"] == "syntax.node":
            nodes.append({})
        if nodes:
            nodes[-1][f["name"]] = f["value"]
    if status(fields) != "parsed":
        assert not nodes
        return 0
    raw, line_starts = text.encode(), [0]
    for i, b in enumerate(raw):
        if b == 10 or (b == 13 and raw[i + 1:i + 2] != b"\n"):
            line_starts.append(i + 1)
    expected, queue = {}, deque([(ast.parse(text), ())])
    while queue:
        node, path = queue.popleft()
        span = None
        if getattr(node, "end_lineno", None) is not None:
            span = (line_starts[node.lineno - 1] + node.col_offset,
                    line_starts[node.end_lineno - 1] + node.end_col_offset)
        flags = {k: getattr(node, k) for k in ("is_async", "simple", "conversion", "level")
                 if type(getattr(node, k, None)) is int}
        expected[path] = (type(node).__name__, span, flags)
        for name, value in ast.iter_fields(node):
            for index, child in (enumerate(value) if isinstance(value, list) else [(-1, value)]):
                if isinstance(child, ast.AST):
                    queue.append((child, path + ((name, index),)))
    actual, paths = {}, []
    for i, node in enumerate(nodes):
        assert node["syntax.node"] == i
        parent = node["syntax.parent"]
        assert parent == -1 if i == 0 else 0 <= parent < i
        path = () if parent == -1 else paths[parent] + ((node["syntax.edge"], node["syntax.index"]),)
        paths.append(path)
        assert path not in actual
        span = (node["syntax.start"], node["syntax.end"]) if node["syntax.has_span"] else None
        if span:
            assert 0 <= span[0] < span[1] <= len(raw)
            raw[slice(*span)].decode()
        flags = {k: node["syntax." + k] for k in ("is_async", "simple", "conversion", "level") if "syntax." + k in node}
        actual[path] = (node["syntax.kind"], span, flags)
    assert actual == expected
    assert len(nodes) == next(f["value"] for f in fields if f["name"] == "syntax.nodes")
    return len(nodes)


def controls():
    checks = []
    def check(name, ok):
        assert ok, name
        checks.append(name)
    for ending, name in [("\n", "LF"), ("\r\n", "CRLF"), ("\r", "CR")]:
        body = ending.join(["é = 1", "def f(x):", "    return x + é", ""])
        text = "Intro" + ending + "```python" + ending + body + "```" + ending + "After" + ending
        source = SyntaxSource(text)
        a, b, language, closed = source.regions[0]
        check(name + "_exact_bytes", text[a:b] == body and language == "python" and closed)
        fs = source.for_span(a, b)
        check(name + "_AST_offsets", status(fs) == "parsed" and audit_ast(body, fs) > 0)
        spans, _ = atomise_text(text)
        audit = code_record_check(text, spans)
        check(name + "_atoms", tiles(len(text), [(s, e) for s, e, _, _ in spans])
              and not any(v for k, v in audit.items() if k != "python_bodies"))
        check(name + "_cross_fence_absent", source.for_span(a, len(text)) == [])
    cases = [
        ("````python\nx=1\n```\ny=2\n````", 1, True),
        ("~~~python\nx=1\n```\ny=2\n~~~~", 1, True),
        ("```python\rx=1\r``` trailing\ry=2", 1, False),
        ("    ```python\nx=1\n    ```", 0, False),
        ("```python\nx=1", 1, False),
        ("```python", 1, False),
        ("```python\r\n```", 1, True),
        ("```python\r\nx=1\r```\n~~~py\ny=2\r\n~~~", 2, True),
        ("```python\nx='a\u2028b'\n```", 1, True),
        ("```python\tinfo\nx=1\n```", 1, True),
    ]
    for i, (text, count, closed) in enumerate(cases):
        source = SyntaxSource(text)
        check("fence_control_" + str(i), len(source.regions) == count
              and source.regions == reference_regions(text)
              and (not count or source.regions[-1][-1] == closed))
    text = "```python\r\nx=1\r\n```\r\nProse\r\n```py\r\ny=2\r\n```\r\n"
    a, b, _, _ = SyntaxSource(text).regions[0]
    check("sibling_isolation", SyntaxSource(text).for_span(a, b) == SyntaxSource(text.replace("y=2", "def broken(:")).for_span(a, b))
    bad = syntax_fields("def broken(:", "python")
    check("malformed_no_tree", status(bad) == "MNLM_SYNTAX_UNPARSED" and audit_ast("def broken(:", bad) == 0)
    check("unspecified_not_inferred", status(syntax_fields("x=1", "unspecified")) == "MNLM_SYNTAX_LANGUAGE_UNSUPPORTED")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--before-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay", type=Path)
    args = ap.parse_args()
    started = time.perf_counter()
    assert file_hash(args.before) == args.before_sha256
    spec = importlib.util.spec_from_file_location("senses._before_syntax", args.before)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    assert digest(before.contract()) != digest(contract())
    manifest, _, records = load(args.input, args.sha256)
    source_hash = file_hash(args.input / "inputs.ucr.jsonl")
    assert source_hash == manifest["files"]["inputs.ucr.jsonl"]["sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    counts, old, languages, statuses, changed = Counter(), Counter(), Counter(), Counter(), []
    checks = controls()
    with (args.out / "observations.jsonl").open("xb") as output:
        for atom, record in sorted(records.items()):
            text = record["channels"]["surface"]["items"][0]["text"]
            previous = before.SyntaxSource(text)
            source = SyntaxSource(text)
            assert source.regions == reference_regions(text), atom
            spans, stats = atomise_text(text)
            assert tiles(len(text), [(a, b) for a, b, _, _ in spans]), atom
            audit = code_record_check(text, spans)
            assert not any(v for k, v in audit.items() if k != "python_bodies"), (atom, audit)
            readings = []
            for a, b, lang, closed in source.regions:
                languages[lang] += 1
                counts.update(regions=1, closed=closed, code_bytes=len(text[a:b].encode()))
                fs = syntax_fields(text[a:b], lang)
                nodes = audit_ast(text[a:b], fs)
                statuses[status(fs)] += 1
                counts["ast_nodes"] += nodes
                values = tokens(fs, "structure")
                counts["native_value_bytes_projected"] += values.nbytes
                readings.append({"span": [a, b], "language": lang, "closed": closed, "fields": fs,
                                 "native_shape": list(values.shape), "native_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
            for a, b, lang, closed in previous.regions:
                old.update(regions=1, closed=closed)
                if lang in PYTHON:
                    old[status(before.syntax_fields(text[a:b], lang))] += 1
            if previous.regions != source.regions:
                changed.append(atom)
            counts.update(records=1, atoms=len(spans), surface_bytes=len(text.encode()), unclosed=stats["unclosed"])
            output.write(canonical({"atom": atom, "spans": spans, "regions": readings}) + b"\n")
    if args.replay:
        prior = json.loads((args.replay / "report.json").read_bytes())
        assert prior["observations_sha256"] == file_hash(args.replay / "observations.jsonl")
        assert file_hash(args.out / "observations.jsonl") == prior["observations_sha256"]
        checks.append("cold_complete_replay_exact")
    assert file_hash(args.input / "inputs.ucr.jsonl") == source_hash
    assert counts["records"] == manifest["record_count"]
    checks += ["bound_source_unchanged", "independent_fences_all_records", "conserved_atoms_all_records",
               "independent_AST_paths_and_spans", "native_reading_hashes_bound", "new_parser_identity"]
    base = Path(__file__).resolve().parents[1]
    code = {name: file_hash(base / name) for name in (
        "canon/fences.py", "canon/code_atoms.py", "canon/fsv_stage2.py", "canon/segment.py",
        "senses/code_syntax.py", "senses/fsv_fences.py", "senses/trajectory_prefix.py", "senses/instruments.py")}
    report = {"schema": "fence-boundary-fsv-v1", "acceptance": "author-checked only", "checks": checks,
              "input_sha256": args.sha256, "source_sha256": source_hash, "before_sha256": args.before_sha256,
              "syntax": contract(), "code_atomizer": CODE_ATOMIZER_ID, "code_sha256": code,
              "counts": dict(counts), "before": dict(old),
              "statuses": dict(statuses), "languages": dict(languages), "changed_records": changed,
              "observations_sha256": file_hash(args.out / "observations.jsonl"),
              "admitted": False, "stage4_production": False, "wall_seconds": time.perf_counter() - started}
    (args.out / "report.json").write_bytes(canonical(report))
    print(json.dumps({"counts": dict(counts), "statuses": dict(statuses), "changed_records": len(changed),
                      "checks": len(checks), "report_sha256": file_hash(args.out / "report.json")}))


if __name__ == "__main__":
    main()
