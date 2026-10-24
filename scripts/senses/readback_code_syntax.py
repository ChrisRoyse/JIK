"""Producer-independent readback of code syntax FSV: stdlib only, no MNLM imports.

Usage: python -I scripts/senses/readback_code_syntax.py RUN OUTPUT
This is a separate author readback, not independently adjudicated acceptance.
"""
import ast
from bisect import bisect_right
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys
import zlib


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def readback(root):
    root = Path(root)
    inventory = json.loads((root / "inventory.json").read_bytes())
    for name, h in inventory.items():
        p = (root / name).resolve()
        assert p.is_relative_to(root.resolve()) and sha(p) == h, ("inventory", name)
    report = json.loads((root / "report.json").read_bytes())
    assert sha(report["input"]) == report["input_sha256"]
    assert sha(report["span_proof"]) == report["span_proof_sha256"]
    assert sha(Path(__file__).with_name("code_syntax.py")) == report["syntax"]["code_sha256"]
    assert sha(Path(__file__).resolve().parents[1] / "canon/fences.py") == report["syntax"]["fences_sha256"]
    env = {r["record"]["row"]: r for r in map(json.loads, Path(report["input"]).read_bytes().splitlines())}
    proof = json.loads(Path(report["span_proof"]).read_bytes())
    retained = {(s["row"], s["start"], s["end"]) for s in proof["spans"] if s["role"] == 6}
    db = sqlite3.connect((root / "syntax.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    entries, seen, slots, total_nodes, counts, slot_receipts = [], set(), 0, 0, Counter(), []
    try:
        for atom, body in db.execute("SELECT atom,body FROM records ORDER BY atom"):
            r = json.loads(body)
            assert hashlib.sha256(canonical({k: v for k, v in r.items() if k != "atom_id"})).hexdigest() == atom == r["atom_id"]
            loc = r["locator"]
            assert loc["source"] == proof["scope"]["source"] != "cuyahoga-eighth-district"
            key = (loc["row"], *loc["span"])
            assert key in retained and key not in seen
            seen.add(key)
            text = env[loc["row"]]["record"]["text"][slice(*loc["span"])]
            assert r["channels"]["surface"]["items"] == [{"text": text}]
            fs = r["channels"]["structure"]["items"]
            state = next(f["value"] for f in fs if f["name"] == "syntax.status")
            counts[state] += 1
            nodes = []
            for f in fs:
                if f["name"] == "syntax.node":
                    nodes.append({})
                if nodes:
                    nodes[-1][f["name"]] = f["value"]
            if state == "parsed":
                tree = ast.parse(text)
                queue, expected = deque([(tree, ())]), {}
                while queue:  # breadth-first paths, distinct from producer ordinal traversal
                    parent, path = queue.popleft()
                    point = tuple(getattr(parent, k, None) for k in ("lineno", "col_offset", "end_lineno", "end_col_offset"))
                    flags = {k: getattr(parent, k) for k in ("is_async", "simple", "conversion", "level")
                             if type(getattr(parent, k, None)) is int}
                    expected[path] = (type(parent).__name__, point if None not in point else None, flags)
                    for name, value in ast.iter_fields(parent):
                        candidates = enumerate(value) if isinstance(value, list) else [(-1, value)]
                        for i, child in candidates:
                            if isinstance(child, ast.AST):
                                queue.append((child, path + ((name, i),)))
                raw = text.encode("utf-8")
                line_starts = [0] + [i + 1 for i, b in enumerate(raw)
                                     if b == 10 or (b == 13 and raw[i + 1:i + 2] != b"\n")]
                actual, paths = {}, []
                for i, node in enumerate(nodes):
                    assert node["syntax.node"] == i
                    parent = node["syntax.parent"]
                    assert (-1 == parent if i == 0 else 0 <= parent < i)
                    path = paths[parent] + ((node["syntax.edge"], node["syntax.index"]),) if parent >= 0 else ()
                    paths.append(path)
                    assert path not in actual
                    point = None
                    if node["syntax.has_span"]:
                        start, end = node["syntax.start"], node["syntax.end"]
                        assert 0 <= start < end <= len(raw)
                        raw[start:end].decode("utf-8")  # offsets must be codepoint boundaries
                        a, b = bisect_right(line_starts, start), bisect_right(line_starts, end)
                        point = (a, start - line_starts[a - 1], b, end - line_starts[b - 1])
                    flags = {k: node["syntax." + k] for k in ("is_async", "simple", "conversion", "level") if "syntax." + k in node}
                    actual[path] = (node["syntax.kind"], point, flags)
                assert actual == expected  # exact parent/field/index paths, not aggregate edge counts
                assert len(nodes) == next(f["value"] for f in fs if f["name"] == "syntax.nodes")
            else:
                assert not nodes
            total_nodes += len(nodes)
            hashes = {}
            for lens, meta, compressed, h in db.execute("SELECT lens,meta,values_blob,digest FROM slots WHERE atom=?", (atom,)):
                m, raw = json.loads(meta), zlib.decompress(compressed)
                assert len(raw) == math.prod(m["shape"]) * 4
                assert all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", raw))
                assert hashlib.sha256(canonical(m) + raw).hexdigest() == h
                hashes[lens] = h
                slots += 1
                if m["channel"] == "structure":
                    slot_receipts.append([list(key), h])
            entries.append([atom, hashes])
        assert seen == retained
        generations = db.execute("SELECT hash,body FROM generations ORDER BY seq").fetchall()
        assert len(generations) == 1
        head, body = generations[0]
        body = json.loads(body)
        assert hashlib.sha256(canonical(body)).hexdigest() == head == report["generation"]
        assert body["parent"] is None and body["entries_digest"] == hashlib.sha256(canonical(sorted(entries))).hexdigest()
    finally:
        db.close()
    assert len(seen) == report["records"] and total_nodes == report["nodes"] and counts == report["status_counts"]
    assert not any(report[k] for k in ("meaning_asserted", "stage3_admitted", "stage4_production"))
    return {"acceptance": "separate author readback only", "files": len(inventory), "records": len(seen),
            "slots": slots, "nodes": total_nodes, "statuses": dict(counts), "report_sha256": sha(root / "report.json"),
            "structure_receipt_sha256": hashlib.sha256(canonical(sorted(slot_receipts))).hexdigest(), "failures": []}


if __name__ == "__main__":
    result = readback(sys.argv[1])
    with Path(sys.argv[2]).open("xb") as out:
        out.write(canonical(result) + b"\n")
    print(json.dumps(result))
