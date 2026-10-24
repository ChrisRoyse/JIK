"""Independent full-source restoration readback and real-row FSV extraction.

No restoration, historical-cut or chunking producer imports. One second source
stream checks every published full description, while retaining a small real
source subset for adversarial boundary probes on both hosts.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import blake3
import pyarrow.parquet as pq
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError(code)


def cold_publications(restored, old_records, old_chunks):
    report = json.loads((restored/"report.json").read_bytes())
    check(report["complete"] and not report["errors"] and report["served"] is False
          and set(report["families"]) == set(report["families_requested"]),
          "MNLM_FULL_READBACK_INCOMPLETE")
    expected, fixture, stats = {}, {}, {}
    for family, metrics in report["families"].items():
        old = [json.loads(l) for l in (old_records/family/"rows.jsonl").read_text(encoding="utf-8").split("\n") if l]
        rows_raw = (restored/"records"/family/"rows.jsonl").read_bytes()
        rows = [json.loads(l) for l in rows_raw.decode("utf-8").split("\n") if l]
        summary = json.loads((restored/"records"/family/"summary.json").read_bytes())
        check(sha(rows_raw) == metrics["rows_sha256"] == summary["rows_sha256"], "MNLM_FULL_READBACK_ROWS_HASH")
        check(len(old) == len(rows) == metrics["entities"], "MNLM_FULL_READBACK_ROWS")
        check(sha((old_records/family/"rows.jsonl").read_bytes()) == summary["original_rows_sha256"],
              "MNLM_FULL_READBACK_PARENT_HASH")
        for a, b in zip(old, rows):
            check({k: v for k, v in a.items() if k != "text"} ==
                  {k: v for k, v in b.items() if k not in ("text", "text_source")}, "MNLM_FULL_READBACK_LABELS")
            check(b["entity"] not in expected, "MNLM_FULL_READBACK_DUPLICATE")
            expected[b["entity"]] = (b["text"], b["text_source"])
        dest = restored/"chunked"/family
        chunk_sum = json.loads((dest/"summary.json").read_bytes())
        raw = (dest/"chunks/atoms.parquet").read_bytes()
        check(sha(raw) == metrics["chunks_sha256"] == chunk_sum["chunks_sha256"], "MNLM_FULL_READBACK_CHUNK_HASH")
        check((dest/"anchors.json").read_bytes() == (old_chunks/family/"anchors.json").read_bytes(),
              "MNLM_FULL_READBACK_ANCHORS")
        anchors = json.loads((dest/"anchors.json").read_bytes())
        selected = {i for a in anchors.values() for i in a["rows"]}
        cursors, chunks = {}, pq.read_table(io.BytesIO(raw)).to_pylist()
        for r in chunks:
            i = r["row"]
            check(i in selected, "MNLM_FULL_READBACK_UNEXPECTED_ROW")
            end, part = cursors.get(i, (0, 0))
            check(r["start"] == end and r["chunk"] == part and r["end"] > end,
                  "MNLM_FULL_READBACK_OFFSETS")
            check(r["entity"] == rows[i]["entity"] and r["text"] == rows[i]["text"][end:r["end"]]
                  and len(r["text"]) <= 600, "MNLM_FULL_READBACK_TEXT")
            cursors[i] = (r["end"], part+1)
        check(set(cursors) == selected and all(cursors[i][0] == len(rows[i]["text"]) for i in selected),
              "MNLM_FULL_READBACK_COVERAGE")
        stats[family] = {"entities": len(rows), "sample_entities": len(selected), "chunks": len(chunks),
                         "full_characters": sum(len(r["text"]) for r in rows)}
        # Real minimum proof corpus: up to three members of one existing question.
        prop = next(iter(anchors))
        keep = anchors[prop]["rows"][:3]
        labels = anchors[prop]["labels"][:3]
        anchor_n = len(keep)
        # Include real Unicode separators and recovered tails even if those rows
        # are outside this small question sample; JSONL readers must still parse them.
        for predicate in (lambda r: "\u2028" in r["text"] or "\u2029" in r["text"],
                          lambda r: len(r["text"]) > 1500):
            extra = next((i for i, r in enumerate(rows) if predicate(r) and i not in keep), None)
            if extra is not None:
                keep.append(extra)
        fixture[family] = {"rows": [old[i] for i in keep], "parent": summary,
                           "anchors": {prop: {**anchors[prop], "rows": list(range(anchor_n)), "labels": labels}}}
    return report, expected, fixture, stats


class DigestReader(io.RawIOBase):
    def __init__(self, path):
        self.file = path.open("rb")
        self.digest = blake3.blake3()
    def readable(self):
        return True
    def readinto(self, buffer):
        n = self.file.readinto(buffer)
        self.digest.update(memoryview(buffer)[:n])
        return n
    def close(self):
        self.file.close()
        super().close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("canon", "restored", "old-records", "old-chunks", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    a = p.parse_args()
    check(not a.output.exists(), "MNLM_FULL_READBACK_EXISTS")
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    report, expected, fixture, stats = cold_publications(a.restored, a.old_records, a.old_chunks)
    wanted = {r["entity"] for f in fixture.values() for r in f["rows"]}
    check(sha((a.canon/"manifest.jsonl").read_bytes()) == report["canonical_manifest_sha256"],
          "MNLM_FULL_READBACK_MANIFEST")
    source = DigestReader(a.canon/"wikidata5m_text.jsonl.zst")
    seen, proof, n = set(), [], 0
    with source, zstandard.ZstdDecompressor().stream_reader(source, closefd=False) as dec:
        for line in io.TextIOWrapper(dec, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            q = r["f"].get("entity_id")
            if q in expected:
                text, ref = expected[q]
                check(q not in seen and r["text"] == text and r["row"] == ref["canonical_row"],
                      "MNLM_FULL_READBACK_CANONICAL")
                check(sha(text.encode()) == ref["text_sha256"], "MNLM_FULL_READBACK_TEXT_HASH")
                seen.add(q)
            if q in wanted:
                proof.append(line.encode("utf-8"))
    check(n == report["source_publication"]["docs"] and source.digest.hexdigest() == report["source_publication"]["out_blake3"],
          "MNLM_FULL_READBACK_SOURCE_STREAM")
    check(seen == set(expected) and len(proof) == len(wanted), "MNLM_FULL_READBACK_MISSING")
    root = a.output/"real-subset"
    (root/"canon").mkdir(parents=True)
    raw = zstandard.ZstdCompressor().compress(b"".join(proof))
    (root/"canon/wikidata5m_text.jsonl.zst").write_bytes(raw)
    publication = {**report["source_publication"], "docs": len(proof), "out_bytes": len(raw),
                   "canonical_bytes": sum(json.loads(line)["n"] for line in proof),
                   "out_blake3": blake3.blake3(raw).hexdigest(), "proof_subset_of": report["source_publication"]["out_blake3"],
                   "purpose": "FSV real retained source rows only; never scientific calibration"}
    (root/"canon/manifest.jsonl").write_text(json.dumps(publication)+"\n", encoding="utf-8")
    for family, f in fixture.items():
        recdir, chdir = root/"records"/family, root/"chunked"/family
        recdir.mkdir(parents=True); chdir.mkdir(parents=True)
        raw = b"".join((json.dumps(r, sort_keys=True)+"\n").encode() for r in f["rows"])
        (recdir/"rows.jsonl").write_bytes(raw)
        summary = {**f["parent"], "rows": len(f["rows"]), "rows_sha256": sha(raw)}
        publish(recdir/"summary.json", summary)
        anchors = json.dumps(f["anchors"], sort_keys=True).encode()
        (chdir/"anchors.json").write_bytes(anchors)
        publish(chdir/"summary.json", {"family": family, "parent_rows_sha256": sha(raw), "anchors_sha256": sha(anchors),
                "entities": len({i for a in f["anchors"].values() for i in a["rows"]}),
                "anchors": {k: {"classes": v["classes"], "n": len(v["rows"])} for k,v in f["anchors"].items()}})
    result = {"schema": "full-source-independent-readback-v1", "source_rows": n, "selected_entities": len(seen),
              "families": stats, "all_source_text_equal": True, "all_labels_equal": True,
              "all_selected_documents_tiled": True, "acceptance": "author-checked", "wall_seconds": time.perf_counter()-started,
              "source_blake3": source.digest.hexdigest(), "restoration_report_sha256": sha((a.restored/"report.json").read_bytes()),
              "real_subset_manifest_sha256": sha((root/"canon/manifest.jsonl").read_bytes())}
    publish(a.output/"readback.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
