"""Pinned lossless text sequences; source order is an ordinal, never a timestamp."""
import json
from pathlib import Path

import pyarrow.parquet as pq

from text.constellation_corpus import SCHEMA, FLAGS, canonical, digest, file_sha, need


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def validate_rows(rows, metadata):
    """Check serialized topology/coordinates independently of segmentation."""
    need(rows and len(rows) == sum(f["atoms"] for f in metadata["fields"]), "ROW_COVERAGE")
    expected_doc = digest(canonical({k: metadata[k] for k in ("source_sha256", "domain", "ordinal")}))
    need(metadata["id"] == expected_doc, "DOCUMENT_ID")
    cursor = 0
    for fi, field in enumerate(metadata["fields"]):
        group = rows[cursor:cursor+field["atoms"]]
        cursor += field["atoms"]
        if field["status"] != "present":
            need(not group and field["characters"] == field["bytes"] == 0, "FIELD_STATUS")
            continue
        need(group, "FIELD_COVERAGE")
        sequence = digest(canonical(dict(document=expected_doc, field=field["name"])))
        need(field["sequence"] == sequence, "SEQUENCE_ID")
        cp, bp, parts = 0, 0, []
        for pos, row in enumerate(group):
            raw = row["text"].encode("utf-8")
            need(row["doc"] == expected_doc and row["domain"] == metadata["domain"]
                 and row["source_ordinal"] == metadata["ordinal"] and row["sequence"] == sequence
                 and row["field"] == field["name"] and row["field_index"] == fi, "ROW_BINDING")
            need(row["pos"] == pos and row["char_start"] == cp and row["char_end"] == cp+len(row["text"])
                 and row["byte_start"] == bp and row["byte_end"] == bp+len(raw) and raw, "SPAN_COVERAGE")
            pin = digest(raw)
            need(row["text_sha256"] == pin and row["atom_id"] == digest(canonical(dict(
                sequence=sequence, span=[cp, cp+len(row["text"])], sha256=pin))), "ATOM_ID")
            need(row["kind"] == ("text" if row["text"].strip() else "layout"), "ATOM_KIND")
            need(row["previous"] == (group[pos-1]["atom_id"] if pos else None)
                 and row["next"] == (group[pos+1]["atom_id"] if pos+1 < len(group) else None), "SUCCESSOR")
            cp, bp = row["char_end"], row["byte_end"]
            parts.append(raw)
        need(cp == field["characters"] and bp == field["bytes"]
             and digest(b"".join(parts)) == field["sha256"], "FIELD_BYTES")
    return rows


class Corpus:
    """One cold pin check; at most one complete document resident per caller."""
    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        need(file_sha(self.root/"report.json") == pin, "REPORT_PIN")
        self.report = r = json.loads((self.root/"report.json").read_bytes())
        need(r["schema"] == SCHEMA and r["flags"] == FLAGS and r["complete"] and not r["errors"]
             and r["contiguous"] and r["full_fields"] and r["atom_cap"] is None
             and r["character_filter"] is None, "CONTRACT")
        self.stamps = {}
        for name in ("report.json", "atoms.parquet", "documents.jsonl"):
            path = self.root/name
            if name != "report.json":
                need(path.stat().st_size == r["files"][name]["bytes"]
                     and file_sha(path) == r["files"][name]["sha256"], "ARTIFACT_PIN")
            self.stamps[name] = stamp(path)
        with (self.root/"documents.jsonl").open("rb") as f:
            self.documents = [json.loads(line) for line in f]
        self.table = pq.ParquetFile(self.root/"atoms.parquet")
        need(len(self.documents) == r["counts"]["documents"] == self.table.num_row_groups
             and self.table.metadata.num_rows == r["counts"]["atoms"], "DOCUMENT_COVERAGE")
        self.fresh()

    def fresh(self):
        need(all(stamp(self.root/n) == s for n, s in self.stamps.items()), "SOURCE_CHANGED")

    def document(self, index):
        need(type(index) is int and 0 <= index < len(self.documents), "DOCUMENT_INDEX")
        self.fresh()
        rows = self.table.read_row_group(index).to_pylist()
        validate_rows(rows, self.documents[index])
        self.fresh()
        return rows

    def batches(self, count):
        need(type(count) is int and count > 0, "BATCH_SIZE")
        pending = []
        for i in range(len(self.documents)):
            for row in self.document(i):
                pending.append(row)
                if len(pending) == count:
                    yield pending
                    pending = []
        if pending:
            yield pending

    def next_pairs(self, index):
        rows = self.document(index)
        for left, right in zip(rows, rows[1:]):
            if left["next"] is not None:
                need(left["next"] == right["atom_id"] and left["sequence"] == right["sequence"], "SUCCESSOR")
                yield left, right

    def training_authority(self):
        raise ValueError("MNLM_TEXT_CORPUS_ADMISSION_REQUIRED")
