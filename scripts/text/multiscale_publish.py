"""Publish multiscale source views of the unchanged full-source A2 sample (#27).

No new sampling, labels, slots, GPU jobs or admitted senses. One read/hash of
each named source and anchor file, streamed per-document output, no text copies.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.multiscale_spans import build, digest, identity, require, CONTRACT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layouts", type=Path)
    p.add_argument("--layouts-sha256")
    a = p.parse_args()
    start = time.perf_counter()
    require(not a.output.exists(), "MNLM_MULTISCALE_EXISTS")
    report_raw = (a.source/"report.json").read_bytes()
    require(digest(report_raw) == a.source_report_sha256, "MNLM_MULTISCALE_SOURCE_REPORT")
    source = json.loads(report_raw)
    require(source["complete"] and not source["errors"] and source["families"]
            and set(source["families"]) == set(source["families_requested"]), "MNLM_MULTISCALE_SOURCE_INCOMPLETE")
    layouts = {}
    require(bool(a.layouts) == bool(a.layouts_sha256), "MNLM_MULTISCALE_LAYOUT_PIN")
    if a.layouts:
        raw = a.layouts.read_bytes()
        require(digest(raw) == a.layouts_sha256, "MNLM_MULTISCALE_LAYOUT_PIN")
        layouts = json.loads(raw)
        require(isinstance(layouts, dict), "MNLM_MULTISCALE_LAYOUT_FORMAT")
    a.output.mkdir(parents=True, exist_ok=False)
    counters, manifests, failures, layout_seen = {}, {}, [], set()
    load_seconds = geometry_seconds = write_seconds = 0.0
    for family, metrics in sorted(source["families"].items()):
        try:
            at = time.perf_counter()
            rows_raw = (a.source/"records"/family/"rows.jsonl").read_bytes()
            rows_hash = digest(rows_raw)
            require(rows_hash == metrics["rows_sha256"], "MNLM_MULTISCALE_ROWS_HASH")
            rows = [json.loads(l) for l in rows_raw.decode().split("\n") if l]
            chunk_dir = a.source/"chunked"/family
            summary_raw = (chunk_dir/"summary.json").read_bytes()
            summary = json.loads(summary_raw)
            anchor_raw = (chunk_dir/"anchors.json").read_bytes()
            anchor_hash = digest(anchor_raw)
            require(anchor_hash == summary["anchors_sha256"]
                    and rows_hash == summary["parent_rows_sha256"], "MNLM_MULTISCALE_ANCHOR_HASH")
            anchors = json.loads(anchor_raw)
            selected = sorted({i for q in anchors.values() for i in q["rows"]})
            require(len(rows) == metrics["entities"] and len(selected) == summary["entities"]
                    and all(type(i) is int and 0 <= i < len(rows) for i in selected), "MNLM_MULTISCALE_POPULATION")
            # Bind membership to the chunks in the pinned restoration report,
            # not only a mutable sidecar's self-reported anchor hash.
            chunk_raw = (chunk_dir/"chunks/atoms.parquet").read_bytes()
            require(digest(chunk_raw) == metrics["chunks_sha256"], "MNLM_MULTISCALE_CHUNKS_HASH")
            members = pq.read_table(io.BytesIO(chunk_raw), columns=["row", "entity"]).to_pylist()
            require({r["row"] for r in members} == set(selected)
                    and all(type(r["row"]) is int and 0 <= r["row"] < len(rows)
                            and rows[r["row"]]["entity"] == r["entity"] for r in members), "MNLM_MULTISCALE_POPULATION")
            load_seconds += time.perf_counter()-at
            count = Counter()
            dest = a.output/(family+".jsonl")
            with dest.open("xb") as out:
                for index in selected:
                    try:
                        at = time.perf_counter()
                        row = rows[index]
                        require(digest(row["text"].encode()) == row["text_source"]["text_sha256"],
                                "MNLM_MULTISCALE_TEXT_HASH")
                        entity = row["entity"]
                        if entity in layouts:
                            layout_seen.add(entity)
                        record = build(row["text"], identity(row["text_source"]), "entity:"+entity, layouts.get(entity))
                        record.update(row=index, entity=entity, source=row["text_source"])
                        geometry_seconds += time.perf_counter()-at
                        at = time.perf_counter()
                        out.write((json.dumps(record, sort_keys=True, separators=(",", ":"))+"\n").encode())
                        write_seconds += time.perf_counter()-at
                        count.update(documents=1, source_characters=len(row["text"]), nodes=len(record["nodes"]),
                                     edges=len(record["edges"]))
                        count.update(n["level"] for n in record["nodes"])
                        count[record["page_status"]] += 1
                        count[record["paragraph_status"]] += 1
                    except (ValueError, KeyError, TypeError) as exc:
                        failures.append({"family": family, "row": index, "error": str(exc)})
            counters[family] = dict(count)
            manifests[family] = {"file": dest.name, "sha256": digest(dest.read_bytes()), "bytes": dest.stat().st_size,
                                 "source_rows_sha256": rows_hash, "anchors_sha256": anchor_hash,
                                 "source_chunks_sha256": metrics["chunks_sha256"],
                                 "chunk_summary_sha256": digest(summary_raw)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"family": family, "error": str(exc)})
    if set(layouts)-layout_seen:
        failures.append({"error": "MNLM_MULTISCALE_UNUSED_LAYOUT", "entities": sorted(set(layouts)-layout_seen)})
    finish = time.perf_counter()
    result = {"schema": CONTRACT, "source_report_sha256": digest(report_raw), "source": str(a.source),
              "layouts_sha256": a.layouts_sha256, "counts": counters, "files": manifests, "errors": failures,
              "complete": not failures, "served": False, "scientific_admission": False,
              "measurement_status": "pending:frozen-instrument-readings", "old_slot_reuse_allowed": False,
              "label_scope": "unchanged document-level anchors; no child labels inferred",
              "evaluation_status": "entity groups inherited; near-duplicate closure still required",
              "code_sha256": digest(Path(__file__).read_bytes()),
              "geometry_sha256": digest(Path(__file__).with_name("multiscale_spans.py").read_bytes()),
              "cost": "O(source bytes + characters + nodes + edges); four levels; no all-pairs corpus",
              "timing": {"load_hash_seconds": load_seconds, "geometry_seconds": geometry_seconds,
                         "write_seconds": write_seconds, "other_seconds": finish-start-load_seconds-geometry_seconds-write_seconds,
                         "wall_seconds": finish-start}}
    publish(a.output/"report.json", result)
    print(json.dumps({k: result[k] for k in ("counts", "complete", "errors", "timing")}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
