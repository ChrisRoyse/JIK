"""Audit captured fields to complete multiscale spans and relations (#27)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.multiscale_readback import audit_spans, check, sha


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit_field(record, row, field):
    text = field["text"]
    check(sha(text.encode()) == field["sha256"] and len(text) == field["characters"]
          and len(text.encode()) == field["utf8_bytes"], "MNLM_EVAL_MULTI_READBACK_FIELD")
    source = dict(document_id=row["id"], source_record_sha256=row["source"]["record_sha256"],
                  field_name=field["name"], text_sha256=field["sha256"])
    check(record["document_id"] == row["id"] and record["domain"] == row["domain"]
          and record["field_name"] == field["name"] and record["source"] == source
          and record["source_characters"] == len(text), "MNLM_EVAL_MULTI_READBACK_LINEAGE")
    check(record["group_keys"] == row["group_keys"] and record["split_group"] == row["group_keys"][-1],
          "MNLM_EVAL_MULTI_READBACK_GROUP")
    status = "nonempty" if text.strip() else ("whitespace-only" if text else "empty")
    check(record["field_status"] == status, "MNLM_EVAL_MULTI_READBACK_STATUS")
    count = Counter(fields=1, source_characters=len(text)); count[status] += 1
    if status == "nonempty":
        detail = audit_spans(record["graph"], text, source, row["group_keys"][-1])
        detail.pop("documents"); detail.pop("source_characters")
        count.update(detail)
    else:
        check(record["graph"] is None, "MNLM_EVAL_MULTI_READBACK_EMPTY_GRAPH")
    return count


def audit_sequence(rows, graphs):
    count = Counter()
    for row in rows:
        count["documents"] += 1
        for field in row["fields"]:
            record = next(graphs, None)
            check(record is not None, "MNLM_EVAL_MULTI_READBACK_MISSING_FIELD")
            count.update(audit_field(record, row, field))
    check(next(graphs, None) is None, "MNLM_EVAL_MULTI_READBACK_EXTRA_FIELD")
    return count


def audit(root, source):
    raw = (root/"report.json").read_bytes(); report = json.loads(raw)
    source_raw = (source/"report.json").read_bytes(); source_report = json.loads(source_raw)
    check(report["source_report_sha256"] == sha(source_raw) and source_report["complete"] is True
          and source_report["errors"] == [], "MNLM_EVAL_MULTI_READBACK_SOURCE")
    check(report["schema"] == "evaluation-multiscale-v1" and report["geometry_contract"] == "source-multiscale-v3"
          and report["complete"] is True and report["errors"] == []
          and set(report["files"]) == set(source_report["sources"]), "MNLM_EVAL_MULTI_READBACK_INCOMPLETE")
    check(all(report[k] is False for k in ("sealed_evaluation", "served", "scientific_admission", "old_slot_reuse_allowed"))
          and report["measurement_status"] == "pending:frozen-instrument-readings"
          and report["semantic_edges"] == "pending:structural relations only", "MNLM_EVAL_MULTI_READBACK_CLAIM")
    totals = Counter(); summaries = {}
    for domain, sm in report["files"].items():
        path = root/(domain+".jsonl"); src = source_report["sources"][domain]
        check(sm["file"] == path.name and path.stat().st_size == sm["bytes"] and file_sha(path) == sm["sha256"],
              "MNLM_EVAL_MULTI_READBACK_HASH")
        source_path = source/path.name
        check(file_sha(source_path) == sm["source_sha256"] == src["sha256"], "MNLM_EVAL_MULTI_READBACK_INPUT_HASH")
        with source_path.open("rb") as f, path.open("rb") as g:
            count = audit_sequence((json.loads(line) for line in f), (json.loads(line) for line in g))
        check(dict(count) == sm["counts"] and count["documents"] == src["counts"]["selected_documents"]
              and count["fields"] == src["counts"]["selected_fields"]
              and count["source_characters"] == src["counts"]["selected_characters"], "MNLM_EVAL_MULTI_READBACK_COVERAGE")
        summaries[domain] = dict(count); totals.update(count)
    return dict(acceptance="author-checked", all_spans_and_relations_equal=True, counts=summaries, totals=dict(totals),
                source_report_sha256=sha(source_raw), multiscale_report_sha256=sha(raw), sealed_evaluation=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "publication", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); result = audit(a.publication, a.source)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True)
    print(json.dumps(result))
