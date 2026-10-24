"""Three-boundary FSV for real chunked inputs, plus explicit adversarial probes.

Run the actual rechunk CLI, then independently reconstruct every source observation
from cold Parquet/JSON reads. No encoder runs or synthetic scientific results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.chunk_spans import chunk_spans
from senses.trajectory_population import publish

SCRIPT = Path(__file__).parents[1] / "text/rechunk_anchor.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_read(output, source):
    rep = json.loads((output / "summary.json").read_bytes())
    assert sha(source / "rows.jsonl") == rep["parent_rows_sha256"]
    assert sha(output / "anchors.json") == rep["anchors_sha256"]
    assert sha(output / "chunks/atoms.parquet") == rep["chunks_sha256"]
    anchors = json.loads((output / "anchors.json").read_bytes())
    needed = {i for a in anchors.values() for i in a["rows"]}
    original = [json.loads(line) for line in (source / "rows.jsonl").read_text(encoding="utf-8").split("\n") if line]
    groups = {}
    data = pq.read_table(output / "chunks/atoms.parquet").to_pylist()
    for r in data:
        groups.setdefault(r["row"], []).append(r)
    assert set(groups) == needed
    chars = 0
    for i, parts in groups.items():
        text = original[i]["text"]
        end = 0
        for j, part in enumerate(parts):
            assert part["chunk"] == j and part["start"] == end
            assert part["end"] > end and part["end"] - end <= 600
            assert part["text"] == text[end:part["end"]]
            assert part["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            assert part["entity"] == original[i]["entity"]
            end = part["end"]
        assert end == len(text) and "".join(r["text"] for r in parts) == text
        chars += len(text)
    assert chars == rep["source_characters"] == rep["chunk_characters"]
    assert len(data) == rep["chunks"] and len(groups) == rep["entities"]
    assert rep["old_slot_reuse_allowed"] is False and rep["served"] is False
    return {"entities": len(groups), "chunks": len(data), "characters": chars}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("chunked", "source", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, ok):
        checks.append({"name": name, "pass": bool(ok)})
    def run(name, chunked=a.chunked, source=a.source):
        dest = a.output / name
        cp = subprocess.run([sys.executable, "-I", str(SCRIPT), "--chunked", str(chunked),
                             "--source", str(source), "--output", str(dest)], capture_output=True, text=True)
        (a.output / (name+".log")).write_text(cp.stdout+cp.stderr, encoding="utf-8")
        return cp, dest
    before = {str(p): sha(p) for p in (a.chunked/"anchors.json", a.chunked/"summary.json", a.source/"rows.jsonl")}
    cp, actual = run("real")
    check("B1 real CLI succeeds", cp.returncode == 0)
    counts = cold_read(actual, a.source)
    check("B2 every real source character survives cold readback", True)
    check("B2 anchors unchanged byte for byte", (actual/"anchors.json").read_bytes() == (a.chunked/"anchors.json").read_bytes())
    cp, _ = run("real")
    check("B3 refuses existing publication", cp.returncode != 0 and "MNLM_RECHUNK_EXISTS" in cp.stderr)
    for text in ("No.", "\n A \nB\n", "a"*1401, "αβγ "*701, "Do not proceed. " + "Longword "*200,
                 "Legal clause\u2028continued with original unicode separator."):
        parts = chunk_spans(text)
        check("B2 planted short/long/unicode boundary " + hashlib.sha256(text.encode()).hexdigest()[:10],
              "".join(r["text"] for r in parts) == text and all(0 < len(r["text"]) <= 600 for r in parts))
    for text in ("", " \n\t"):
        try:
            chunk_spans(text)
            good = False
        except ValueError as exc:
            good = "MNLM_CHUNK_EMPTY" in str(exc)
        check("B2 blank observation refuses", good)
    tampered_source = a.output / "source-negative"
    tampered_source.mkdir()
    (tampered_source/"rows.jsonl").write_bytes((a.source/"rows.jsonl").read_bytes()+b" ")
    cp, _ = run("changed-source", source=tampered_source)
    check("B1 changed source bytes refuse", cp.returncode != 0 and "MNLM_RECHUNK_SOURCE" in cp.stderr)
    bad = a.output / "anchor-negative"
    shutil.copytree(a.chunked, bad)
    anchors = json.loads((bad/"anchors.json").read_bytes())
    first = next(iter(anchors.values()))
    first["rows"][0] = first["rows"][1]
    (bad/"anchors.json").write_text(json.dumps(anchors), encoding="utf-8")
    cp, _ = run("changed-anchors", chunked=bad)
    check("B1 changed anchor digest refuses", cp.returncode != 0 and "MNLM_RECHUNK_ANCHORS" in cp.stderr)
    rep = json.loads((bad/"summary.json").read_bytes())
    rep["anchors_sha256"] = sha(bad/"anchors.json")
    (bad/"summary.json").write_text(json.dumps(rep), encoding="utf-8")
    cp, _ = run("duplicate-rows", chunked=bad)
    check("B1 duplicate independent entity row refuses", cp.returncode != 0 and "MNLM_RECHUNK_ROWS" in cp.stderr)
    corrupt = a.output / "publication-negative"
    shutil.copytree(actual, corrupt)
    table = pq.read_table(corrupt/"chunks/atoms.parquet").to_pylist()
    table[0]["text"] += "X"
    pq.write_table(pa.Table.from_pylist(table), corrupt/"chunks/atoms.parquet")
    # Even a tampered summary matching the corrupted bytes must fail source readback.
    rep = json.loads((corrupt/"summary.json").read_bytes())
    rep["chunks_sha256"] = sha(corrupt/"chunks/atoms.parquet")
    (corrupt/"summary.json").write_text(json.dumps(rep), encoding="utf-8")
    try:
        cold_read(corrupt, a.source)
        good = False
    except AssertionError:
        good = True
    check("B3 altered chunk with recomputed digest fails source reconstruction", good)
    check("B3 original source publications unchanged", all(sha(Path(p)) == d for p, d in before.items()))
    out = {"schema": "fsv-chunked-inputs-v1", "checks": checks, "passed": sum(r["pass"] for r in checks),
           "total": len(checks), "real_counts": counts, "source_hashes": before,
           "publication_sha256": sha(actual/"summary.json"), "acceptance": "author-checked"}
    publish(a.output/"fsv.json", out)
    print(json.dumps({k: out[k] for k in ("passed", "total", "real_counts")}))
    if out["passed"] != out["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
