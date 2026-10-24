"""Lossless multiscale views of complete captured document fields (#27).

Each field retains its source and document/entity group. Empty fields have an
explicit coverage record. No source fields or instrument readings are pooled.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.multiscale_spans import build, identity, require, CONTRACT
from senses.trajectory_population import publish

SCHEMA = "evaluation-multiscale-v1"


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def field_record(row, field):
    text = field["text"]
    require(len(text) == field["characters"] and len(text.encode()) == field["utf8_bytes"]
            and hashlib.sha256(text.encode()).hexdigest() == field["sha256"], "MNLM_EVAL_MULTI_FIELD_HASH")
    require(row["page_status"] == "absent:no-attested-page-offsets" and row["sealed_evaluation"] is False,
            "MNLM_EVAL_MULTI_SOURCE_CLAIM")
    source = dict(document_id=row["id"], source_record_sha256=row["source"]["record_sha256"],
                  field_name=field["name"], text_sha256=field["sha256"])
    group = row["group_keys"][-1]
    status = "nonempty" if text.strip() else ("whitespace-only" if text else "empty")
    graph = build(text, identity(source), group) if status == "nonempty" else None
    return dict(document_id=row["id"], domain=row["domain"], field_name=field["name"],
                source=source, source_characters=len(text), group_keys=row["group_keys"],
                split_group=group, field_status=status, graph=graph)


def run(source, pin, output):
    started = time.perf_counter(); require(not output.exists(), "MNLM_EVAL_MULTI_EXISTS")
    raw = (source/"report.json").read_bytes(); report = json.loads(raw)
    require(hashlib.sha256(raw).hexdigest() == pin, "MNLM_EVAL_MULTI_SOURCE_PIN")
    require(report["schema"] == "evaluation-source-capture-v1" and report["complete"] is True
            and not report["errors"] and report["sealed_evaluation"] is False
            and report["scientific_admission"] is False and report["served"] is False, "MNLM_EVAL_MULTI_SOURCE_CLAIM")
    output.mkdir(parents=True); files = {}; errors = []; phase = Counter()
    for domain, src in report["sources"].items():
        count = Counter(); start = time.perf_counter()
        try:
            path = source/(domain+".jsonl")
            require(src["file"] == path.name and path.stat().st_size == src["bytes"]
                    and file_sha(path) == src["sha256"], "MNLM_EVAL_MULTI_SOURCE_HASH")
            phase["load_hash_seconds"] += time.perf_counter()-start
            dest = output/path.name
            with path.open("rb") as f, dest.open("xb") as out:
                for line in f:
                    start = time.perf_counter(); row = json.loads(line)
                    count["documents"] += 1
                    phase["load_hash_seconds"] += time.perf_counter()-start
                    for field in row["fields"]:
                        try:
                            start = time.perf_counter(); result = field_record(row, field)
                            phase["geometry_seconds"] += time.perf_counter()-start
                            start = time.perf_counter(); out.write(canonical(result)+b"\n")
                            phase["write_seconds"] += time.perf_counter()-start
                            count.update(fields=1, source_characters=len(field["text"]))
                            count[result["field_status"]] += 1
                            if result["graph"]:
                                graph = result["graph"]
                                count.update(nodes=len(graph["nodes"]), edges=len(graph["edges"]))
                                count.update(n["level"] for n in graph["nodes"])
                                count[graph["paragraph_status"]] += 1
                                count[graph["page_status"]] += 1
                        except (ValueError, KeyError, TypeError) as exc:
                            errors.append(dict(domain=domain, document_id=row["id"], field=field.get("name"), error=str(exc)))
            start = time.perf_counter()
            files[domain] = dict(file=dest.name, sha256=file_sha(dest), bytes=dest.stat().st_size,
                                 source_sha256=src["sha256"], counts=dict(count))
            phase["write_seconds"] += time.perf_counter()-start
            require(count["documents"] == src["counts"]["selected_documents"]
                    and count["fields"] == src["counts"]["selected_fields"]
                    and count["source_characters"] == src["counts"]["selected_characters"], "MNLM_EVAL_MULTI_COVERAGE")
            print(json.dumps(dict(domain=domain, **count)), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(domain=domain, error=str(exc)))
    wall = time.perf_counter()-started
    phase["other_seconds"] = wall-sum(phase.values()); phase["wall_seconds"] = wall
    result = dict(schema=SCHEMA, geometry_contract=CONTRACT, source_report_sha256=pin,
                  files=files, errors=errors, complete=not errors and set(files) == set(report["sources"]),
                  sealed_evaluation=False, served=False, scientific_admission=False,
                  measurement_status="pending:frozen-instrument-readings", old_slot_reuse_allowed=False,
                  semantic_edges="pending:structural relations only", timing=dict(phase),
                  code_sha256=file_sha(Path(__file__)), geometry_sha256=file_sha(Path(__file__).with_name("multiscale_spans.py")))
    publish(output/"report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); result = run(a.source, a.source_report_sha256, a.output)
    print(json.dumps({k: result[k] for k in ("complete", "errors", "timing")}))
    if not result["complete"]:
        raise SystemExit(2)
