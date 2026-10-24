"""Fixed-question >=500-entity calibration, with unavailable questions named (#12).

No class changes, resampling toward outcomes, old-slot reuse or test-set claims.
O(source bytes + rows log rows + selected characters); one ordering per domain.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from math import ceil
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.chunk_spans import chunk_spans, CONTRACT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, code):
    if not ok:
        raise ValueError(code)


def load_family(source, family, pin):
    raw = (source/"records"/family/"rows.jsonl").read_bytes()
    summary = (source/"records"/family/"summary.json").read_bytes()
    old = (source/"chunked"/family/"anchors.json").read_bytes()
    require(sha(raw) == pin["rows_sha256"] and sha(summary) == pin["summary_sha256"]
            and sha(old) == pin["original_anchors_sha256"], "MNLM_CALPOP_INPUT_HASH")
    rows = [json.loads(l) for l in raw.decode().split("\n") if l]
    meta, original = json.loads(summary), json.loads(old)
    require({q: a["classes"] for q,a in original.items()} == pin["questions"], "MNLM_CALPOP_QUESTION_CHANGED")
    require(len(rows) == meta["rows"] and len({r["entity"] for r in rows}) == len(rows), "MNLM_CALPOP_ENTITY")
    require(all(sha(r["text"].encode()) == r["text_source"]["text_sha256"] for r in rows), "MNLM_CALPOP_TEXT_HASH")
    return rows, meta, original


def select(rows, meta, original, seed, target):
    order = sorted(range(len(rows)), key=lambda i: (sha(f"{seed}|chunked|{rows[i]['entity']}".encode()), rows[i]["entity"]))
    anchors, power = {}, {}
    for prop, old in sorted(original.items()):
        classes = old["classes"]
        require(len(classes) >= 2 and len(classes) == len(set(classes)), "MNLM_CALPOP_CLASSES")
        values = meta["values"][prop]
        require(all(c in values for c in classes), "MNLM_CALPOP_VOCABULARY")
        value_to_label = {values.index(c): j for j,c in enumerate(classes)}
        per = ceil(target/len(classes))
        available = Counter(r["y"][prop] for r in rows)
        counts = [available[values.index(c)] for c in classes]
        ready = min(counts) >= per
        power[prop] = {"classes": classes, "available_per_class": counts, "required_per_class": per,
                       "planned_n": per*len(classes), "available": ready,
                       "code": None if ready else "MNLM_CALPOP_UNDERPOWERED"}
        if not ready:
            continue
        chosen, labels, taken = [], [], Counter()
        for i in order:
            label = value_to_label.get(rows[i]["y"][prop])
            if label is not None and taken[label] < per:
                chosen.append(i); labels.append(label); taken[label] += 1
        anchors[prop] = {"classes": classes, "rows": chosen, "labels": labels, "per_class": per,
                         "overlap_with_historical_sample": len(set(chosen).intersection(old["rows"]))}
    return anchors, power


def build(source, contract_path, output):
    start = time.perf_counter()
    require(not output.exists(), "MNLM_CALPOP_EXISTS")
    config_raw = contract_path.read_bytes()
    config = json.loads(config_raw)
    require(config["schema"] == "a2-calibration-contract-v1" and config["sample_target"] == 500
            and config["seed"] == 25, "MNLM_CALPOP_CONTRACT")
    raw = (source/"report.json").read_bytes()
    require(sha(raw) == config["source_report_sha256"], "MNLM_CALPOP_SOURCE_REPORT")
    root = json.loads(raw)
    require(root["complete"] and not root["errors"] and set(root["families"]) == set(config["families"]),
            "MNLM_CALPOP_SOURCE_INCOMPLETE")
    output.mkdir(parents=True, exist_ok=False)
    (output/"contract.json").write_bytes(config_raw)
    families, errors = {}, []
    load_s = sample_s = publish_s = 0.0
    for family, pin in sorted(config["families"].items()):
        try:
            tick = time.perf_counter()
            rows, meta, old = load_family(source, family, pin)
            load_s += time.perf_counter()-tick
            tick = time.perf_counter()
            anchors, power = select(rows, meta, old, config["seed"], config["sample_target"])
            used = sorted({i for a in anchors.values() for i in a["rows"]})
            sample_s += time.perf_counter()-tick
            tick = time.perf_counter()
            dest = output/family
            dest.mkdir()
            anchor_raw = json.dumps(anchors, sort_keys=True, separators=(",", ":")).encode()
            (dest/"anchors.json").write_bytes(anchor_raw)
            chunks, chars = [], 0
            for i in used:
                row = rows[i]
                chars += len(row["text"])
                for j, span in enumerate(chunk_spans(row["text"])):
                    chunks.append({**span, "row": i, "entity": row["entity"], "chunk": j,
                                   "source_sha256": row["text_source"]["text_sha256"]})
            chunk_hash = None
            if chunks:
                (dest/"chunks").mkdir()
                buf = io.BytesIO()
                pq.write_table(pa.Table.from_pylist(chunks), buf, compression="zstd")
                body = buf.getvalue()
                (dest/"chunks/atoms.parquet").write_bytes(body)
                chunk_hash = sha(body)
            summary = {"schema": "a2-calibration-population-v1", "family": family, "anchor_family": "A2",
                       "source_pin": pin, "parent_rows_sha256": pin["rows_sha256"], "anchors_sha256": sha(anchor_raw),
                       "chunks_sha256": chunk_hash, "entities": len(used), "chunks": len(chunks),
                       "source_characters": chars, "chunk_characters": sum(len(c["text"]) for c in chunks),
                       "power": power, "available_questions": sorted(anchors), "calibration_only": True,
                       "old_slot_reuse_allowed": False, "served": False, "chunker": CONTRACT,
                       "chunker_sha256": sha(Path(__file__).with_name("chunk_spans.py").read_bytes()),
                       "anchors": {q: {"n": len(a["rows"]), "classes": a["classes"]} for q,a in anchors.items()}}
            publish(dest/"summary.json", summary)
            families[family] = {"summary_sha256": sha((dest/"summary.json").read_bytes()),
                                "entities": len(used), "chunks": len(chunks), "source_characters": chars,
                                "ready": sorted(anchors), "unavailable": sorted(set(old)-set(anchors))}
            publish_s += time.perf_counter()-tick
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"family": family, "error": str(exc)})
    elapsed = time.perf_counter()-start
    report = {"schema": "a2-calibration-publication-v1", "contract_sha256": sha(config_raw), "families": families,
              "errors": errors, "complete": not errors, "all_questions_available": not errors and all(not f["unavailable"] for f in families.values()),
              "scientific_admission": False, "served": False, "calibration_only": True,
              "code_sha256": sha(Path(__file__).read_bytes()), "source_report_sha256": sha(raw),
              "timing": {"load_hash_seconds": load_s, "sample_seconds": sample_s, "publish_seconds": publish_s,
                         "other_seconds": elapsed-load_s-sample_s-publish_s, "wall_seconds": elapsed}}
    publish(output/"report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "contract", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    r = build(a.source, a.contract, a.output)
    print(json.dumps(r))
    if not r["complete"]:
        raise SystemExit(2)
