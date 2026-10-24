"""Publish lossless chunks of the SAME retained anchor population (#12).

This changes the measurement input contract, not labels, sampling or thresholds.
Old slots cannot be reused. Source bytes are pinned before rows are interpreted.
O(source bytes + selected text characters), one source read per publication.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.chunk_spans import chunk_spans, CONTRACT, MAX_CHARS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, code, message):
    if not ok:
        raise ValueError(f"{code}: {message}")


def build(chunked, source, output):
    started = time.perf_counter()
    require(not output.exists(), "MNLM_RECHUNK_EXISTS", "new immutable generation required")
    summary_raw = (chunked / "summary.json").read_bytes()
    summary = json.loads(summary_raw)
    anchors_raw = (chunked / "anchors.json").read_bytes()
    require(sha(anchors_raw) == summary["anchors_sha256"], "MNLM_RECHUNK_ANCHORS", "anchor digest changed")
    source_raw = (source / "rows.jsonl").read_bytes()
    require(sha(source_raw) == summary["parent_rows_sha256"], "MNLM_RECHUNK_SOURCE", "parent digest changed")
    # JSONL uses LF record delimiters; Unicode line separators may occur in strings.
    rows = [json.loads(line) for line in source_raw.decode("utf-8").split("\n") if line]
    anchors = json.loads(anchors_raw)
    require(bool(anchors), "MNLM_RECHUNK_ANCHORS", "empty anchor roster")
    used = set()
    for name, a in anchors.items():
        ids, labels = a["rows"], a["labels"]
        require(len(ids) == len(labels) == len(set(ids)) and bool(ids)
                and all(type(i) is int and 0 <= i < len(rows) for i in ids),
                "MNLM_RECHUNK_ROWS", f"{name}: invalid or repeated row")
        require(all(type(y) is int and 0 <= y < len(a["classes"]) for y in labels),
                "MNLM_RECHUNK_LABELS", f"{name}: invalid class index")
        require(len({rows[i]["entity"] for i in ids}) == len(ids),
                "MNLM_RECHUNK_ENTITY", f"{name}: repeated independent entity")
        used.update(ids)
    require(len(used) == summary["entities"], "MNLM_RECHUNK_POPULATION", "entity union changed")
    loaded = time.perf_counter()
    cols = {k: [] for k in ("entity", "row", "chunk", "text", "start", "end", "role", "forced_cut", "source_sha256")}
    characters = forced = 0
    for i in sorted(used):
        text = rows[i]["text"]
        pieces = chunk_spans(text)
        source_hash = sha(text.encode())
        characters += len(text)
        for j, part in enumerate(pieces):
            rec = {**part, "row": i, "chunk": j, "entity": rows[i]["entity"], "source_sha256": source_hash}
            for k in cols:
                cols[k].append(rec[k])
            forced += part["forced_cut"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "chunks").mkdir()
    (output / "anchors.json").write_bytes(anchors_raw)
    table = pa.table(cols)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    chunk_raw = buf.getvalue()
    (output / "chunks" / "atoms.parquet").write_bytes(chunk_raw)
    report = {"schema": "a2-domain-chunked-lossless-v2", "family": summary["family"],
              "anchor_family": "A2", "parent_rows_sha256": sha(source_raw), "anchors_sha256": sha(anchors_raw),
              "parent_summary_sha256": sha(summary_raw), "chunks_sha256": sha(chunk_raw),
              "entities": len(used), "chunks": len(cols["text"]), "source_characters": characters,
              "chunk_characters": sum(map(len, cols["text"])), "forced_cuts": forced,
              "anchors": summary["anchors"], "rules": {"chunker": CONTRACT, "max_chars": MAX_CHARS,
              "offset_unit": "unicode-code-point", "minimum_length_filter": False, "head_truncation": False},
              "old_slot_reuse_allowed": False, "served": False,
              "code_sha256": sha(Path(__file__).read_bytes()),
              "chunker_sha256": sha(Path(__file__).with_name("chunk_spans.py").read_bytes()),
              "segmenter_sha256": sha(Path(__file__).parents[1].joinpath("canon/segment.py").read_bytes()),
              "timing": {"load_hash_seconds": loaded-started, "build_write_seconds": time.perf_counter()-loaded}}
    publish(output / "summary.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("chunked", "source", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    r = build(a.chunked, a.source, a.output)
    print(json.dumps({k: r[k] for k in ("entities", "chunks", "source_characters", "chunk_characters")}))


if __name__ == "__main__":
    main()
