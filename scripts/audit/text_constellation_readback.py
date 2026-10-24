"""Original source bytes -> independent lossless span/topology readback."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import Corpus, validate_rows


def source_rows(path):
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()
    else:
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def run(a):
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.pin)
    manifest = json.loads((a.raw/"manifest.json").read_bytes())
    need(file_sha(a.raw/"manifest.json") == corpus.report["source_manifest_sha256"], "RAW_MANIFEST")
    sources = {s["domain"]: s for s in manifest["sources"]}
    phase = dict(input_seconds=time.perf_counter()-start)
    tick = time.perf_counter()
    original, indices = {}, {}
    for meta in corpus.documents:
        indices.setdefault(meta["domain"], set()).add(meta["ordinal"])
    for domain, wanted in indices.items():
        src = sources[domain]
        path = a.raw/src["path"]
        need(path.stat().st_size == src["bytes"] and file_sha(path) == src["sha256"], "RAW_SOURCE")
        for ordinal, row in enumerate(source_rows(path)):
            if ordinal in wanted:
                original[(domain, ordinal)] = row
            if ordinal >= max(wanted):
                break
    phase["independent_raw_read_seconds"] = time.perf_counter()-tick
    tick = time.perf_counter()
    counts = dict(documents=0, fields=0, atoms=0, characters=0, bytes=0, transitions=0,
                  cross_field_edges=0, source_text_mismatches=0)
    all_rows, controls = [], []
    for i, meta in enumerate(corpus.documents):
        rows = corpus.document(i)
        raw = original[(meta["domain"], meta["ordinal"])]
        names = [f["name"] for f in meta["fields"]]
        need(set(names) == ({n for n in raw if n.startswith("section_")} if meta["domain"] == "finance"
                           else {"text"}), "RAW_FIELD_ROSTER")
        need(meta["provider_metadata"] == {k: v for k, v in raw.items() if k not in names}, "RAW_METADATA")
        for field in meta["fields"]:
            name = field["name"]
            group = [r for r in rows if r["field"] == name]
            expected = raw.get(name)
            expected_bytes = (expected or "").encode()
            need("".join(r["text"] for r in group) == (expected or ""), "RAW_TEXT")
            for r in group:
                need(r["text"] == expected[r["char_start"]:r["char_end"]]
                     and r["text"].encode() == expected_bytes[r["byte_start"]:r["byte_end"]], "RAW_SPAN")
            counts["fields"] += bool(expected)
        counts["documents"] += 1
        counts["atoms"] += len(rows)
        counts["characters"] += sum(len(r["text"]) for r in rows)
        counts["bytes"] += sum(len(r["text"].encode()) for r in rows)
        counts["transitions"] += sum(1 for _ in corpus.next_pairs(i))
        all_rows.append(rows)
    for k in ("documents", "fields", "atoms", "characters", "bytes", "transitions"):
        need(counts[k] == corpus.report["counts"][k], "COUNT_READBACK")
    phase["source_spans_and_consumer_seconds"] = time.perf_counter()-tick
    tick = time.perf_counter()
    def deny(name, fn, expected):
        try:
            fn()
        except ValueError as exc:
            need(str(exc) == expected, "CONTROL_CODE:"+name+":"+str(exc))
            controls.append(dict(name=name, error=str(exc)))
        else:
            raise ValueError("MNLM_TEXT_CORPUS_CONTROL_ACCEPTED:"+name)
    base = all_rows[0]
    bad = copy.deepcopy(base)
    bad.pop(1)
    deny("omitted_source_atom", lambda: validate_rows(bad, corpus.documents[0]), "MNLM_TEXT_CORPUS_ROW_COVERAGE")
    bad = copy.deepcopy(base)
    bad[0], bad[1] = bad[1], bad[0]
    deny("reordered_atom", lambda: validate_rows(bad, corpus.documents[0]), "MNLM_TEXT_CORPUS_SPAN_COVERAGE")
    bad = copy.deepcopy(base)
    bad[0]["next"] = all_rows[1][0]["atom_id"]
    deny("cross_document_successor", lambda: validate_rows(bad, corpus.documents[0]), "MNLM_TEXT_CORPUS_SUCCESSOR")
    bad = copy.deepcopy(base)
    bad[0]["byte_end"] += 1
    deny("wrong_utf8_span", lambda: validate_rows(bad, corpus.documents[0]), "MNLM_TEXT_CORPUS_SPAN_COVERAGE")
    bad = copy.deepcopy(base)
    bad[0]["text"] = "X"+bad[0]["text"][1:]
    deny("changed_reading_surface", lambda: validate_rows(bad, corpus.documents[0]), "MNLM_TEXT_CORPUS_ATOM_ID")
    deny("unadmitted_training", corpus.training_authority, "MNLM_TEXT_CORPUS_ADMISSION_REQUIRED")
    phase["negative_controls_seconds"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema="text-constellation-source-readback-v1", corpus=a.pin, counts=counts,
        controls=controls, flags=corpus.report["flags"], source_records_independently_read=True,
        code_sha256=file_sha(Path(__file__)), phases_seconds=phase, wall_seconds=wall,
        phase_coverage=sum(phase.values())/wall, acceptance="author-checked only")
    publish(a.output/"report.json", report)
    print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), counts=counts,
                         controls=len(controls), seconds=wall)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("raw", "corpus", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--pin", required=True)
    run(p.parse_args())
