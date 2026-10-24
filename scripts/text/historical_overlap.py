"""Exact observed-history/capture overlap and containment exposure (#14).

All historical versions are unioned per document. No shingles cross atom or
captured-field boundaries. Sparse intersections, never a dense document matrix.
These lexical flags cannot certify absence of memorization or entity exposure.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

SCHEMA = "historical-capture-overlap-v1"
RULES = dict(words=5, near_numerator=3, near_denominator=5,
             containment_numerator=9, containment_denominator=10, containment_min_shared=50,
             historical_boundary="each retained atom", query_boundary="each captured field")
WORD = re.compile(r"\w+")


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_HISTORY_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_query(root, pin, capture):
    require(sha(root/"report.json") == pin, "QUERY_PIN")
    rep = json.loads((root/"report.json").read_bytes())
    require(rep["schema"] == "evaluation-exact-overlap-v1" and rep["complete"] and not rep["errors"], "QUERY_INCOMPLETE")
    require(rep["source_report_sha256"] == sha(capture/"report.json"), "CAPTURE_BINDING")
    require(rep["threshold"] == dict(numerator=3, denominator=5) and rep["shingle_words"] == 5, "QUERY_RULES")
    for name, meta in rep["files"].items():
        require(Path(name).name == name and sha(root/name) == meta["sha256"] and (root/name).stat().st_size == meta["bytes"], "QUERY_FILE")
    words = json.loads((root/"words.json").read_bytes()); vocab = {w: i for i, w in enumerate(words)}
    tuples = np.load(root/"shingles.npy", allow_pickle=False)
    lookup = {tuple(map(int, row)): i for i, row in enumerate(tuples)}
    require(len(lookup) == len(tuples) and len(vocab) == len(words), "QUERY_COLLISION")
    ptr = np.load(root/"indptr.npy"); flat = np.load(root/"members.npy")
    documents = [json.loads(s) for s in (root/"documents.jsonl").read_bytes().splitlines()]
    groups = [json.loads(s) for s in (root/"groups.jsonl").read_bytes().splitlines()]
    require(ptr.shape == (len(documents)+1,) and ptr[0] == 0 and ptr[-1] == len(flat), "QUERY_OFFSETS")
    sets = [set(map(int, flat[ptr[i]:ptr[i+1]])) for i in range(len(documents))]
    postings = defaultdict(list)
    for i, values in enumerate(sets):
        for sid in sorted(values):
            postings[sid].append(i)
    return rep, vocab, lookup, documents, groups, sets, postings


def load_history(root, capture_report):
    documents = {}; pins = {}; stats = {}
    for version, expected in sorted(capture_report["history"].items()):
        folder = root/version; manifest = folder/"manifest.json"; path = folder/"atoms.parquet"
        require(sha(manifest) == expected["manifest_sha256"] and sha(path) == expected["atoms_sha256"], "HISTORY_PIN")
        report = json.loads(manifest.read_bytes())
        require(report["raw_manifest_sha256"] == capture_report["raw_manifest_sha256"], "HISTORY_RAW")
        rows = 0; local = set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["domain", "doc", "text"]):
            cols = batch.to_pydict(); rows += len(batch)
            for domain, doc, text in zip(cols["domain"], cols["doc"], cols["text"]):
                require(all(isinstance(x, str) and x for x in (domain, doc, text)), "HISTORY_ROW")
                key = (domain, doc); local.add(key)
                if key not in documents:
                    documents[key] = dict(texts=set(), sources=Counter())
                documents[key]["texts"].add(text)
                documents[key]["sources"][version] += 1
        require(rows == expected["atoms"] == report["atoms"] and len(local) == expected["documents"], "HISTORY_COUNT")
        pins[version] = dict(manifest=sha(manifest), atoms=sha(path), folder=str(folder))
        stats[version] = dict(rows=rows, documents=len(local))
    require(len(documents) == capture_report["historically_exposed_document_ids"], "HISTORY_UNION")
    return documents, pins, stats


def lexical(texts, vocab, lookup):
    # Negative IDs are local unknown words, disjoint from the query vocabulary.
    unknown = {}; values = set(); words = chars = 0
    for text in texts:
        tokens = WORD.findall(text.lower()); words += len(tokens); chars += len(text)
        ids = []
        for token in tokens:
            wid = vocab.get(token)
            if wid is None:
                if token not in unknown:
                    unknown[token] = -len(unknown)-1
                wid = unknown[token]
            ids.append(wid)
        values.update(zip(*(ids[offset:] for offset in range(5))))
    matched = sorted({lookup[v] for v in values if v in lookup})
    return len(values), matched, words, chars


def flags(intersection, left, right):
    near = intersection > 0 and 5*intersection >= 3*(left+right-intersection)
    contained = intersection >= 50 and 10*intersection >= 9*min(left, right)
    return near, contained


def exposure_rows(groups, documents, links):
    direct = defaultdict(list)
    for row in links:
        direct[row["query"]].append(row["history"])
    out = []
    for group in groups:
        evidence = sorted(set(h for i in group["members"] for h in direct[i]))
        out.append(dict(component=group["id"], document_ids=group["document_ids"], members=group["members"],
                        historical_link_detected=bool(evidence), historical_documents=evidence,
                        directly_linked_members=[i for i in group["members"] if direct[i]],
                        absence_is_clean_certificate=False))
    return out


def run(a):
    require(not a.output.exists(), "OUTPUT_EXISTS")
    started = time.perf_counter(); phases = {}
    require(sha(a.capture/"report.json") == a.capture_sha256, "CAPTURE_PIN")
    capture = json.loads((a.capture/"report.json").read_bytes())
    require(capture["complete"] and not capture["errors"] and capture["sealed_evaluation"] is False, "CAPTURE_STATE")
    query, vocab, lookup, qdocs, groups, qsets, postings = load_query(a.overlap, a.overlap_sha256, a.capture)
    histories, pins, version_stats = load_history(a.history, capture)
    phases["source_validation_and_indexes"] = time.perf_counter()-started
    a.output.mkdir(parents=True); links = []; summary = []; ptr = [0]; all_stats = Counter()
    phases["lexical"] = phases["sparse_intersection"] = phases["observed_text_write"] = 0
    with (a.output/"matched.bin").open("xb") as matched_file, (a.output/"pairs.bin").open("xb") as pairs_file, \
            (a.output/"observed.jsonl.gz").open("xb") as underlying:
        with gzip.GzipFile(filename="", mode="wb", fileobj=underlying, mtime=0) as observed:
            for hi, ((domain, doc), value) in enumerate(sorted(histories.items())):
                tick = time.perf_counter(); texts = sorted(value["texts"])
                total, matched, nwords, nchars = lexical(texts, vocab, lookup)
                require(total < 2**32 and hi < 2**32 and len(qdocs) < 2**32, "CARDINALITY_OVERFLOW")
                phases["lexical"] += time.perf_counter()-tick
                tick = time.perf_counter(); counts = Counter()
                for sid in matched:
                    counts.update(postings[sid]); all_stats["posting_visits"] += len(postings[sid])
                triples = []
                for qi, intersection in sorted(counts.items()):
                    triples.append((hi, qi, intersection))
                    near, contained = flags(intersection, total, len(qsets[qi]))
                    if near or contained:
                        links.append(dict(history=hi, query=qi, intersection=intersection,
                                          history_shingles=total, query_shingles=len(qsets[qi]),
                                          near_jaccard=near, containment_exposure=contained))
                np.asarray(triples, dtype="<u4").reshape(-1, 3).tofile(pairs_file)
                np.asarray(matched, dtype="<u4").tofile(matched_file); ptr.append(ptr[-1]+len(matched))
                phases["sparse_intersection"] += time.perf_counter()-tick
                tick = time.perf_counter()
                record = dict(history=hi, domain=domain, doc=doc, sources=dict(value["sources"]),
                              unique_atoms=len(texts), words=nwords, characters=nchars,
                              shingles=total, matched_shingles=len(matched), text_sha256=hashlib.sha256(encode(texts)).hexdigest())
                summary.append(record); observed.write(encode(dict(record, texts=texts))+b"\n")
                phases["observed_text_write"] += time.perf_counter()-tick
                all_stats.update(historical_documents=1, unique_atoms=len(texts), words=nwords, characters=nchars,
                                 historical_shingle_memberships=total, matched_memberships=len(matched), shared_pairs=len(triples),
                                 historical_below_50_shingles=int(total < 50), historical_empty_shingles=int(total == 0))
                if hi % 10000 == 0:
                    print(json.dumps(dict(history_documents=hi+1, pairs=all_stats["shared_pairs"], seconds=time.perf_counter()-started)), flush=True)
    tick = time.perf_counter()
    np.save(a.output/"matched-indptr.npy", np.asarray(ptr, np.uint64))
    exposure = exposure_rows(groups, qdocs, links)
    for name, values in (("historical.jsonl", summary), ("links.jsonl", links), ("component-exposure.jsonl", exposure)):
        (a.output/name).write_bytes(b"".join(encode(r)+b"\n" for r in values))
    all_stats.update(query_documents=len(qdocs), query_below_50_shingles=sum(len(s) < 50 for s in qsets),
                     near_links=sum(r["near_jaccard"] for r in links), containment_links=sum(r["containment_exposure"] for r in links),
                     direct_exposed_documents=len({r["query"] for r in links}),
                     exposed_components=sum(g["historical_link_detected"] for g in exposure),
                     grouped_exposed_documents=sum(len(g["members"]) for g in exposure if g["historical_link_detected"]))
    files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    phases["publication_and_hashes"] = time.perf_counter()-tick
    tick = time.perf_counter()
    for value in pins.values():
        require(sha(Path(value["folder"])/"manifest.json") == value["manifest"] and
                sha(Path(value["folder"])/"atoms.parquet") == value["atoms"], "HISTORY_DRIFT")
    require(sha(a.capture/"report.json") == a.capture_sha256 and sha(a.overlap/"report.json") == a.overlap_sha256, "QUERY_DRIFT")
    phases["final_source_recheck"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    require(.9 <= sum(phases.values())/wall <= 1.001, "TIMING_COVERAGE")
    report = dict(schema=SCHEMA, rules=RULES, capture_sha256=a.capture_sha256, overlap_sha256=a.overlap_sha256,
                  history_pins=pins, version_stats=version_stats, files=files, counts=dict(all_stats),
                  phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
                  complete=True, errors=[], sealed_evaluation=False, scientific_admission=False, served=False,
                  absence_is_clean_certificate=False, source_code_sha256=sha(Path(__file__)),
                  limitations=["historical retained atoms only; not every source character", "no shingles across atom boundaries",
                               "lexical thresholds cannot rule out all memorization", "world entities and instrument exposure unresolved",
                               "event-time/composition partitions and final seal pending"])
    publish(a.output/"report.json", report)
    print(json.dumps(dict(counts=dict(all_stats), wall_seconds=wall, phase_coverage=report["phase_coverage"])))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "overlap", "history", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--capture-sha256", required=True)
    p.add_argument("--overlap-sha256", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
