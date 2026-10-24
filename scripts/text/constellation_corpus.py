"""Lossless source-ordered text for constellation training, never admission.

PC-07/28/35: hash each source once; stream records and publish one document at
a time. Sentence boundaries tile original fields. No length filter or atom cap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_WS, conserved, segment, SEGMENTER_ID
from senses.trajectory_population import publish
from text.atomise_pilot import records

SCHEMA = "lossless-text-constellation-corpus-v1"
FLAGS = dict(scientific_admission=False, served=False, training_ready=False,
             source_reports_are_world_facts=False, sealed_evaluation=False)
FINANCE_FIELDS = tuple("section_"+s for s in
    ("1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B",
     "10", "11", "12", "13", "14", "15"))
ATOM_SCHEMA = pa.schema([(n, t) for n, t in (
    ("atom_id", pa.string()), ("doc", pa.string()), ("domain", pa.string()),
    ("source_ordinal", pa.int64()), ("field", pa.string()), ("field_index", pa.int64()),
    ("sequence", pa.string()), ("pos", pa.int64()), ("char_start", pa.int64()),
    ("char_end", pa.int64()), ("byte_start", pa.int64()), ("byte_end", pa.int64()),
    ("kind", pa.string()), ("text", pa.string()), ("text_sha256", pa.string()),
    ("previous", pa.string()), ("next", pa.string()))])


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_TEXT_CORPUS_"+code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for raw in iter(lambda: f.read(8 << 20), b""):
            h.update(raw)
    return h.hexdigest()


def fields(domain, record):
    names = FINANCE_FIELDS if domain == "finance" else ("text",)
    if domain == "finance":
        need(not {n for n in record if n.startswith("section_")} - set(names), "UNKNOWN_SECTION")
    output = []
    for name in names:
        value = record.get(name)
        need(value is None or isinstance(value, str), "FIELD_TYPE")
        output.append((name, value))
    return output


def spans(text, domain):
    # Boundary detection alone sees joined hard wraps; replacement is one-for-one
    # in codepoints. Readings and byte addresses always use ORIGINAL text.
    view = text
    if domain in ("finance", "legal"):
        view = re.sub(r"(?<!\n)\r?\n(?!\r?\n)", lambda m: " "*len(m[0]), text)
    result = segment(view)
    need(conserved(text, result), "SEGMENT_COVERAGE")
    # Layout travels with adjacent content, including internal whitespace-only
    # lines. An entirely blank field stays explicit layout, never a fake sense.
    merged = []
    for start, end, role in result:
        if merged and not text[start:end].strip():
            a, _, previous_role = merged[-1]
            merged[-1] = (a, end, previous_role)
        elif merged and not text[merged[-1][0]:merged[-1][1]].strip():
            merged[-1] = (merged[-1][0], end, role)
        else:
            merged.append((start, end, role))
    result = merged
    need(conserved(text, result), "SEGMENT_COVERAGE")
    return result


def document(source, ordinal, record):
    selected = fields(source["domain"], record)
    identity = dict(source_sha256=source["sha256"], domain=source["domain"], ordinal=ordinal)
    doc = digest(canonical(identity))
    rows, field_meta = [], []
    for fi, (name, text) in enumerate(selected):
        if text is None or not text:
            field_meta.append(dict(name=name, status="missing" if text is None else "empty",
                                   characters=0, bytes=0, atoms=0))
            continue
        sequence = digest(canonical(dict(document=doc, field=name)))
        group, byte_cursor = [], 0
        for pos, (start, end, _) in enumerate(spans(text, source["domain"])):
            surface = text[start:end]
            raw = surface.encode("utf-8")
            atom = digest(canonical(dict(sequence=sequence, span=[start, end], sha256=digest(raw))))
            group.append(dict(atom_id=atom, doc=doc, domain=source["domain"], source_ordinal=ordinal,
                field=name, field_index=fi, sequence=sequence, pos=pos, char_start=start, char_end=end,
                byte_start=byte_cursor, byte_end=byte_cursor+len(raw), text=surface,
                kind="text" if surface.strip() else "layout", text_sha256=digest(raw),
                previous=None, next=None))
            byte_cursor += len(raw)
        for a, b in zip(group, group[1:]):
            a["next"], b["previous"] = b["atom_id"], a["atom_id"]
        need(byte_cursor == len(text.encode("utf-8")), "BYTE_COVERAGE")
        rows.extend(group)
        field_meta.append(dict(name=name, status="present", characters=len(text), bytes=byte_cursor,
            sha256=digest(text.encode("utf-8")), sequence=sequence, atoms=len(group)))
    metadata = {k: v for k, v in record.items() if k not in {n for n, _ in selected}}
    return rows, dict(id=doc, **identity, provider_metadata=metadata, fields=field_meta,
        source_licence=source["licence"], source_repo=source["repo"], source_revision=source["revision"],
        source_path=source["path"], clock="source-field-codepoint-order; no physical timestamp",
        field_order="declared source schema; field boundaries are not sentence succession")


def build(raw, output, per_source):
    start = time.perf_counter()
    need(type(per_source) is int and per_source > 0, "DOCUMENT_LIMIT")
    need(not output.exists(), "EXISTS")
    manifest_raw = (raw/"manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    need(manifest["schema"] == "text-pilot-acquisition-v1", "SOURCE_SCHEMA")
    need(len({s["domain"] for s in manifest["sources"]}) == len(manifest["sources"]), "SOURCE_ROSTER")
    output.mkdir(parents=True)
    phases = dict(source_pin_seconds=0., segmentation_write_seconds=0.)
    counts, summaries, failures = dict(documents=0, fields=0, atoms=0, characters=0, bytes=0,
        transitions=0, short_spans=0, long_spans=0, beyond_old_document_cap=0), [], []
    with pq.ParquetWriter(output/"atoms.parquet", ATOM_SCHEMA, compression="zstd") as writer, \
            (output/"documents.jsonl").open("xb") as docs:
        for source in manifest["sources"]:
            before = counts.copy()
            try:
                path = (raw/source["path"]).resolve()
                need(path.is_relative_to(raw.resolve()), "SOURCE_PATH")
                tick = time.perf_counter()
                need(path.stat().st_size == source["bytes"] and file_sha(path) == source["sha256"], "SOURCE_PIN")
                phases["source_pin_seconds"] += time.perf_counter()-tick
                tick = time.perf_counter()
                selected, scanned = 0, 0
                for ordinal, record in enumerate(records(path)):
                    scanned += 1
                    if not any(isinstance(t, str) and t.strip() for _, t in fields(source["domain"], record)):
                        continue
                    rows, meta = document(source, ordinal, record)
                    need(rows, "EMPTY_DOCUMENT")
                    writer.write_table(pa.Table.from_pylist(rows, schema=ATOM_SCHEMA))
                    docs.write(canonical(meta)+b"\n")
                    selected += 1
                    counts["documents"] += 1
                    counts["atoms"] += len(rows)
                    counts["fields"] += sum(f["status"] == "present" for f in meta["fields"])
                    counts["characters"] += sum(len(r["text"]) for r in rows)
                    counts["bytes"] += sum(r["byte_end"]-r["byte_start"] for r in rows)
                    counts["transitions"] += sum(r["next"] is not None for r in rows)
                    counts["short_spans"] += sum(len(r["text"].strip()) < 24 for r in rows)
                    counts["long_spans"] += sum(len(r["text"].strip()) > 600 for r in rows)
                    counts["beyond_old_document_cap"] += max(0, len(rows)-48)
                    if selected == per_source:
                        break
                phases["segmentation_write_seconds"] += time.perf_counter()-tick
                need(selected == per_source, "SOURCE_EXHAUSTED")
                summaries.append(dict(domain=source["domain"], scanned=scanned,
                    counts={k: counts[k]-before[k] for k in counts}))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                failures.append(dict(domain=source["domain"], error=str(exc)))
    tick = time.perf_counter()
    files = {n: dict(bytes=(output/n).stat().st_size, sha256=file_sha(output/n))
             for n in ("atoms.parquet", "documents.jsonl")}
    phases["artifact_pin_seconds"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema=SCHEMA, flags=FLAGS, complete=not failures, errors=failures,
        source_manifest_sha256=digest(manifest_raw), source_bytes=sum(s["bytes"] for s in manifest["sources"]),
        sources=manifest["sources"], per_source=per_source, selection="first nonempty complete source records",
        segmenter=SEGMENTER_ID, code_sha256=file_sha(Path(__file__)), files=files, counts=counts,
        domains=summaries, contiguous=True, full_fields=True, atom_cap=None, character_filter=None,
        clock="source-field-order", native_window_policy="all tokenizer windows; separate typed lanes",
        splits_assigned=False, phases_seconds=phases, wall_seconds=wall,
        phase_coverage=sum(phases.values())/wall, acceptance="author-checked only")
    publish(output/"report.json", report)
    print(json.dumps(dict(report_sha256=file_sha(output/"report.json"), complete=report["complete"],
                         counts=counts, errors=failures, seconds=wall)), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--documents-per-source", type=int, required=True)
    a = p.parse_args()
    raise SystemExit(0 if build(a.raw, a.output, a.documents_per_source)["complete"] else 2)
