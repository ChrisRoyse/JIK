"""Independent literal-shingle, sparse-count and component exposure readback."""
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

RULES = dict(words=5, near_numerator=3, near_denominator=5,
             containment_numerator=9, containment_denominator=10, containment_min_shared=50,
             historical_boundary="each retained atom", query_boundary="each captured field")


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_HISTORY_READ_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for line in f:
            yield json.loads(line)


def header(root, pin, capture, overlap):
    check(sha(root/"report.json") == pin, "REPORT_PIN")
    r = json.loads((root/"report.json").read_bytes())
    check(r["schema"] == "historical-capture-overlap-v1" and r["rules"] == RULES and r["complete"] is True and r["errors"] == [], "CONTRACT")
    check(all(r.get(k) is False for k in ("sealed_evaluation", "scientific_admission", "served", "absence_is_clean_certificate")), "FALSE_CLAIM")
    check(sha(capture/"report.json") == r["capture_sha256"] and sha(overlap/"report.json") == r["overlap_sha256"], "SOURCE_PIN")
    names = {"historical.jsonl", "observed.jsonl.gz", "matched.bin", "matched-indptr.npy", "pairs.bin", "links.jsonl", "component-exposure.jsonl"}
    check(set(r["files"]) == names, "FILE_SET")
    for name, meta in r["files"].items():
        check(sha(root/name) == meta["sha256"] and (root/name).stat().st_size == meta["bytes"], "FILE_"+name)
    check(.9 <= r["phase_coverage"] <= 1.001 and abs(sum(r["phases_seconds"].values())/r["wall_seconds"]-r["phase_coverage"]) < 1e-10, "TIMING")
    return r


def original_history(capture, history):
    cr = json.loads((capture/"report.json").read_bytes()); unions = {}; stats = {}
    for version, meta in sorted(cr["history"].items()):
        root = history/version
        check(sha(root/"manifest.json") == meta["manifest_sha256"] and sha(root/"atoms.parquet") == meta["atoms_sha256"], "ORIGINAL_HISTORY_PIN")
        n = 0; docs = set()
        for batch in pq.ParquetFile(root/"atoms.parquet").iter_batches(batch_size=32768, columns=["domain", "doc", "text"]):
            c = batch.to_pydict(); n += len(batch)
            for d, key, text in zip(c["domain"], c["doc"], c["text"]):
                identity = d, key; docs.add(identity)
                value = unions.setdefault(identity, dict(texts=set(), sources=Counter()))
                value["texts"].add(text); value["sources"][version] += 1
        check(n == meta["atoms"] and len(docs) == meta["documents"], "ORIGINAL_HISTORY_COUNTS")
        stats[version] = dict(rows=n, documents=len(docs))
    check(len(unions) == cr["historically_exposed_document_ids"], "ORIGINAL_UNION")
    return unions, stats


def reference(root, pin, capture, overlap, history=None):
    started = time.perf_counter(); r = header(root, pin, capture, overlap)
    qr = json.loads((overlap/"report.json").read_bytes())
    check(qr["source_report_sha256"] == r["capture_sha256"], "QUERY_CAPTURE")
    for name, meta in qr["files"].items():
        check(sha(overlap/name) == meta["sha256"], "QUERY_FILE")
    cr = json.loads((capture/"report.json").read_bytes())
    for version, meta in cr["history"].items():
        check(r["history_pins"][version]["atoms"] == meta["atoms_sha256"] and
              r["history_pins"][version]["manifest"] == meta["manifest_sha256"], "HISTORY_BINDING")
    originals = None
    if history:
        originals, stats = original_history(capture, history)
        check(stats == r["version_stats"], "VERSION_COUNTS")
    words = json.loads((overlap/"words.json").read_bytes())
    ids = np.load(overlap/"shingles.npy", allow_pickle=False)
    # Literal word tuples, independent of producer's negative unknown-word IDs.
    lexicon = {tuple(words[int(i)] for i in row): sid for sid, row in enumerate(ids)}
    qp = np.load(overlap/"indptr.npy"); qv = np.load(overlap/"members.npy")
    nq = len(qp)-1; qlength = np.diff(qp).astype(np.int64)
    owner = np.repeat(np.arange(nq, dtype=np.uint32), qlength)
    order = np.argsort(qv, kind="stable")
    posting_values = owner[order]
    posting_ptr = np.r_[0, np.cumsum(np.bincount(qv, minlength=len(ids)), dtype=np.int64)]
    observed_matched = np.fromfile(root/"matched.bin", dtype="<u4")
    hp = np.load(root/"matched-indptr.npy", allow_pickle=False)
    saved_pairs = np.fromfile(root/"pairs.bin", dtype="<u4").reshape(-1, 3)
    documents = list(rows(root/"historical.jsonl"))
    check(hp.shape == (len(documents)+1,) and hp.dtype == np.uint64 and hp[0] == 0 and hp[-1] == len(observed_matched) and np.all(hp[1:] >= hp[:-1]), "MATCH_OFFSETS")
    stats = Counter(); links = []; observed_docs = []; cursor = 0; prior = None
    for hi, record in enumerate(rows(root/"observed.jsonl.gz")):
        key = record["domain"], record["doc"]
        check(prior is None or prior < key, "DOCUMENT_ORDER"); prior = key
        texts = record["texts"]
        check(texts == sorted(set(texts)) and hashlib.sha256(canonical(texts)).hexdigest() == record["text_sha256"], "OBSERVED_TEXT")
        if originals is not None:
            check(key in originals and originals[key]["texts"] == set(texts) and dict(originals[key]["sources"]) == record["sources"], "ORIGINAL_TEXT_UNION")
        lexical = set(); nw = nc = 0
        for text in texts:
            tokens = re.findall(r"\w+", text.lower()); nw += len(tokens); nc += len(text)
            lexical.update(tuple(tokens[i:i+5]) for i in range(len(tokens)-4))
        matched = sorted(lexicon[t] for t in lexical if t in lexicon)
        check(np.array_equal(np.asarray(matched, dtype=np.uint32), observed_matched[hp[hi]:hp[hi+1]]), "MATCHED_VALUES")
        n = len(lexical)
        meta = {k: v for k, v in record.items() if k != "texts"}
        check(meta == documents[hi] and meta["history"] == hi and meta["unique_atoms"] == len(texts) and
              meta["words"] == nw and meta["characters"] == nc and meta["shingles"] == n and meta["matched_shingles"] == len(matched), "HISTORY_CARDINALITY")
        postings = [posting_values[posting_ptr[s]:posting_ptr[s+1]] for s in matched]
        flat = np.concatenate(postings) if postings else np.empty(0, np.uint32)
        intersections = np.bincount(flat, minlength=nq)
        positive = np.flatnonzero(intersections)
        expected = np.column_stack((np.full(len(positive), hi), positive, intersections[positive])).astype(np.uint32)
        check(np.array_equal(expected, saved_pairs[cursor:cursor+len(expected)]), "SPARSE_PAIRS")
        cursor += len(expected)
        for qi in positive:
            inter = int(intersections[qi]); size = int(qlength[qi])
            near = 5*inter >= 3*(n+size-inter)
            contain = inter >= 50 and 10*inter >= 9*min(n, size)
            if near or contain:
                links.append(dict(history=hi, query=int(qi), intersection=inter, history_shingles=n,
                                  query_shingles=size, near_jaccard=near, containment_exposure=contain))
        observed_docs.append(meta)
        stats.update(historical_documents=1, unique_atoms=len(texts), words=nw, characters=nc,
                     historical_shingle_memberships=n, matched_memberships=len(matched), shared_pairs=len(expected),
                     historical_below_50_shingles=int(n < 50), historical_empty_shingles=int(n == 0), posting_visits=len(flat))
        if hi % 10000 == 0:
            print(json.dumps(dict(replayed_documents=hi+1, seconds=time.perf_counter()-started)), flush=True)
    check(len(observed_docs) == len(documents) == cr["historically_exposed_document_ids"] and cursor == len(saved_pairs), "TOTAL_ROWS")
    groups = list(rows(overlap/"groups.jsonl")); exposure = []
    direct = defaultdict(set)
    for link in links:
        direct[link["query"]].add(link["history"])
    for group in groups:
        hit_members = [i for i in group["members"] if direct[i]]
        evidence = sorted(set().union(*(direct[i] for i in hit_members)))
        exposure.append(dict(component=group["id"], document_ids=group["document_ids"], members=group["members"],
                             historical_link_detected=bool(evidence), historical_documents=evidence,
                             directly_linked_members=hit_members, absence_is_clean_certificate=False))
    stats.update(query_documents=nq, query_below_50_shingles=int((qlength < 50).sum()),
                 near_links=sum(v["near_jaccard"] for v in links), containment_links=sum(v["containment_exposure"] for v in links),
                 direct_exposed_documents=len({v["query"] for v in links}), exposed_components=sum(v["historical_link_detected"] for v in exposure),
                 grouped_exposed_documents=sum(len(v["members"]) for v in exposure if v["historical_link_detected"]))
    check(sha(capture/"report.json") == r["capture_sha256"] and sha(overlap/"report.json") == r["overlap_sha256"], "SOURCE_DRIFT")
    if history:
        for version, meta in cr["history"].items():
            check(sha(history/version/"manifest.json") == meta["manifest_sha256"] and
                  sha(history/version/"atoms.parquet") == meta["atoms_sha256"], "ORIGINAL_HISTORY_DRIFT")
    expected = dict(counts=dict(stats), links=links, exposure=exposure,
                    original_histories_replayed=history is not None, wall_seconds=time.perf_counter()-started)
    validate_derived(root, expected)
    return expected


def validate_derived(root, expected):
    report = json.loads((root/"report.json").read_bytes())
    check(report["counts"] == expected["counts"], "COUNTS")
    check(list(rows(root/"links.jsonl")) == expected["links"], "LINKS")
    check(list(rows(root/"component-exposure.jsonl")) == expected["exposure"], "COMPONENT_EXPOSURE")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("publication", "capture", "overlap", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--history", type=Path)
    p.add_argument("--report-sha256", required=True)
    return p


if __name__ == "__main__":
    a = parser().parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS")
    expected = reference(a.publication, a.report_sha256, a.capture, a.overlap, a.history)
    result = {k: v for k, v in expected.items() if k not in ("links", "exposure")}
    publish(a.output, dict(result, report_sha256=a.report_sha256, acceptance="author-checked", scientific_admission=False))
    print(json.dumps(result))
