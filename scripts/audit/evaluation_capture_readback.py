"""Rebuild evaluation capture from pinned source bytes, without publisher imports.

The source audit sorts all eligible IDs rather than using the producer's heap.
Its persisted truth bundle can then audit mutations without rescanning 7.3 GB.
The bundle is author-checked evidence, never an independent adjudication.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

SCHEMA = "evaluation-source-capture-v1"
SECTIONS = ["section_"+s for s in
            "1 1A 1B 2 3 4 5 6 7 7A 8 9 9A 9B 10 11 12 13 14 15".split()]


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_EVAL_READBACK_"+code)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("xb") as f:
        f.write(encode(value)+b"\n")


def read_rows(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            for line in f:
                yield json.loads(line)


def expected_record(src, row, ordinal, document, rank):
    domain = src["domain"]
    names = SECTIONS if domain == "finance" else ["text"]
    identity = dict(repo=src["repo"], revision=src["revision"], file=src["file"],
                    source_sha256=src["sha256"], ordinal=ordinal,
                    document_id=document, record_sha256=digest(encode(row)))
    field_values = []
    for name in names:
        content = row[name]; raw = content.encode("utf-8")
        field_values.append(dict(name=name, text=content, characters=len(content),
                                 utf8_bytes=len(raw), sha256=digest(raw)))
    groups = ["document:"+digest(encode([src["repo"], document]))]
    entity = "document-only:world-referent-unresolved"
    if domain == "finance" and isinstance(row.get("cik"), str) and row["cik"]:
        groups += ["sec-cik:"+row["cik"]]; entity = "source-attested:sec-cik"
    clocks = {
        "biomedical": ("created", "provider-created:semantics-unadjudicated"),
        "science": ("created", "provider-created:semantics-unadjudicated"),
        "finance": ("year", "provider-year:semantics-unadjudicated"),
        "legal": ("created", "ingestion-time:not-decision-time"),
    }
    field, status = clocks.get(domain, (None, "absent:document-event-time"))
    return dict(id=digest(encode(identity)), domain=domain, source=identity,
                selection_rank_hex=rank, fields=field_values,
                metadata={k: v for k, v in row.items() if k not in names},
                source_licence=src["licence"], group_keys=groups,
                entity_resolution=entity, clock=dict(field=field, value=row.get(field), status=status),
                page_status="absent:no-attested-page-offsets", sealed_evaluation=False)


def build_truth(raw_root, history_root, contract, output, truth):
    """Full cold source scan. No selection decisions are taken from output scores."""
    started = time.perf_counter()
    check(not truth.exists(), "TRUTH_EXISTS")
    cfg_raw = contract.read_bytes(); cfg = json.loads(cfg_raw)
    check(cfg["schema"] == SCHEMA and cfg["sealed_evaluation"] is False, "CONTRACT")
    raw_manifest = (raw_root/"manifest.json").read_bytes()
    check(digest(raw_manifest) == cfg["raw_manifest_sha256"], "SOURCE_MANIFEST")
    sources = json.loads(raw_manifest)["sources"]
    check(len(sources) == len(cfg["domains"]) and
          {s["domain"] for s in sources} == set(cfg["domains"]), "SOURCE_INVENTORY")
    seen = set(); histories = {}
    for name, pin in cfg["history"].items():
        root = history_root/name; raw = (root/"manifest.json").read_bytes()
        meta = json.loads(raw); path = root/"atoms.parquet"
        check(digest(raw) == pin and meta["raw_manifest_sha256"] == digest(raw_manifest), "HISTORY_PIN")
        check(file_hash(path) == meta["atoms_sha256"], "HISTORY_BYTES")
        documents = set(); count = 0
        table = pq.read_table(path, columns=["domain", "doc"])
        for batch in table.to_batches(max_chunksize=32768):
            values = batch.to_pydict(); count += batch.num_rows
            documents.update(zip(values["domain"], values["doc"]))
        check(count == meta["atoms"] and {d for d, _ in documents} == set(cfg["domains"]), "HISTORY_COUNT")
        seen |= documents
        histories[name] = dict(manifest_sha256=pin, atoms_sha256=meta["atoms_sha256"],
                               atoms=count, documents=len(documents))
    truth.mkdir(parents=True)
    write(truth/"contract.json", cfg)
    summaries = {}
    for src in sources:
        domain = src["domain"]; path = (raw_root/src["path"]).resolve()
        check(path.is_relative_to(raw_root.resolve()), "SOURCE_PATH")
        check(path.stat().st_size == src["bytes"] and file_hash(path) == src["sha256"], "SOURCE_BYTES")
        observed = list(read_rows(output/(domain+".jsonl")))
        ordinals = {r["source"]["ordinal"] for r in observed}
        check(len(ordinals) == len(observed), "DUPLICATE_ORDINAL")
        eligible = []; identifiers = set(); expected = {}; counts = Counter(); schemas = Counter()
        names = SECTIONS if domain == "finance" else ["text"]
        for ordinal, row in enumerate(read_rows(path)):
            counts["raw_records"] += 1
            schemas[digest(encode(sorted(row)))] += 1
            document = next((str(row[k]) for k in ("id", "filename", "url") if row.get(k)), f"{domain}:{ordinal}")
            if document in identifiers:
                counts["duplicate_document_ids"] += 1; continue
            identifiers.add(document)
            if (domain, document) in seen:
                counts["historically_exposed_documents"] += 1; continue
            check(all(isinstance(row.get(n), str) for n in names), "SOURCE_TEXT_FIELDS")
            if domain == "finance":
                check({n for n in row if n.startswith("section_")} == set(names), "SECTION_INVENTORY")
            if not any(row[n].strip() for n in names):
                counts["empty_documents"] += 1; continue
            counts["eligible_documents"] += 1
            rank = digest(encode([SCHEMA, cfg["selection_seed"], domain, document]))
            eligible.append((rank, ordinal))
            if ordinal in ordinals:
                expected[ordinal] = expected_record(src, row, ordinal, document, rank)
        chosen = sorted(eligible)[:cfg["target_documents_per_source"]]
        check([r["source"]["ordinal"] for r in observed] == [o for _, o in chosen], "SELECTION")
        rows = [expected[o] for _, o in chosen]
        counts["selected_documents"] = len(rows)
        counts["selected_fields"] = sum(len(r["fields"]) for r in rows)
        counts["selected_characters"] = sum(len(f["text"]) for r in rows for f in r["fields"])
        dest = truth/(domain+".jsonl")
        with dest.open("xb") as f:
            for row in rows:
                f.write(encode(row)+b"\n")
        summaries[domain] = dict(counts=dict(counts), schema_counts=dict(schemas),
                                truth_sha256=file_hash(dest), source_sha256=src["sha256"],
                                source_bytes=src["bytes"], requested_documents=cfg["target_documents_per_source"],
                                available=len(rows) == cfg["target_documents_per_source"])
        print(json.dumps(dict(domain=domain, source_readback=True, **counts)), flush=True)
    certificate = dict(schema="evaluation-capture-truth-v1", contract_sha256=digest(cfg_raw),
                       raw_manifest_sha256=digest(raw_manifest), history=histories, sources=summaries,
                       historically_exposed_document_ids=len(seen), audit_code_sha256=file_hash(Path(__file__)),
                       wall_seconds=time.perf_counter()-started, adjudication="author-checked")
    write(truth/"report.json", certificate)
    return validate(truth, output)


def validate(truth, output):
    """Compare publication against separately rebuilt, hash-bound source truth."""
    cert_raw = (truth/"report.json").read_bytes(); cert = json.loads(cert_raw)
    raw = (output/"report.json").read_bytes(); report = json.loads(raw)
    check(report["schema"] == SCHEMA and report["complete"] is True and report["errors"] == [], "INCOMPLETE")
    check(all(report[k] is False for k in ("sealed_evaluation", "scientific_admission", "served")), "CLAIM")
    for key in ("contract_sha256", "raw_manifest_sha256", "history", "historically_exposed_document_ids"):
        check(report[key] == cert[key], "PROVENANCE_"+key.upper())
    check(set(report["sources"]) == set(cert["sources"]), "SOURCE_INVENTORY")
    totals = Counter()
    for domain, expected in cert["sources"].items():
        sm = report["sources"][domain]; path = output/(domain+".jsonl")
        check(sm["file"] == path.name, "OUTPUT_PATH")
        check(path.stat().st_size == sm["bytes"] and file_hash(path) == sm["sha256"], "OUTPUT_BYTES")
        truth_path = truth/path.name
        check(file_hash(truth_path) == expected["truth_sha256"], "TRUTH_BYTES")
        observed = list(read_rows(path)); rows = list(read_rows(truth_path))
        check(observed == rows, "SOURCE_CONTENT")
        for key in ("counts", "schema_counts", "source_sha256", "source_bytes", "requested_documents", "available"):
            check(sm[key] == expected[key], "SUMMARY_"+key.upper())
        errors = (output/(domain+".errors.jsonl")).read_bytes()
        check(errors == b"" and digest(errors) == sm["errors_sha256"], "ERROR_LEDGER")
        for key in ("selected_documents", "selected_fields", "selected_characters", "raw_records"):
            totals[key] += expected["counts"][key]
    check(report["all_requested_source_counts_available"] is all(s["available"] for s in cert["sources"].values()), "AVAILABILITY")
    return dict(ok=True, report_sha256=digest(raw), truth_report_sha256=digest(cert_raw),
                totals=dict(totals), adjudication="author-checked", sealed_evaluation=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "truth", "result"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("raw", "history", "contract"):
        p.add_argument("--"+name, type=Path)
    a = p.parse_args()
    if a.raw is not None:
        check(a.history is not None and a.contract is not None, "ARGUMENTS")
        result = build_truth(a.raw, a.history, a.contract, a.output, a.truth)
    else:
        result = validate(a.truth, a.output)
    write(a.result, result)
    print(json.dumps(result))
