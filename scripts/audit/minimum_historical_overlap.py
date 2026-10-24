"""Bounded real-source and declared-truth controls before the historical scan."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.historical_overlap import lexical, flags, sha, encode, require
from senses.trajectory_population import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); require(not a.output.exists(), "MINIMUM_EXISTS")
    a.output.mkdir(parents=True); qrows = []
    for domain in ("biomedical", "finance", "legal", "science", "general", "encyclopedic"):
        with (a.capture/(domain+".jsonl")).open("rb") as f:
            qrows.extend(json.loads(next(f)) for _ in range(4))
    vocab = {}; lookup = {}; qsets = []; native = []
    for doc in qrows:
        tuples = set()
        for field in doc["fields"]:
            tokens = re.findall(r"\w+", field["text"].lower())
            tuples.update(tuple(tokens[i:i+5]) for i in range(len(tokens)-4))
        native.append(tuples)
        for tup in sorted(tuples):
            for word in tup:
                if word not in vocab:
                    vocab[word] = len(vocab)
            ids = tuple(vocab[w] for w in tup)
            if ids not in lookup:
                lookup[ids] = len(lookup)
        qsets.append({lookup[tuple(vocab[w] for w in t)] for t in tuples})
    histories = {}
    for batch in pq.ParquetFile(a.history/"text-pilot-v3"/"atoms.parquet").iter_batches(batch_size=8192, columns=["domain", "doc", "text"]):
        cols = batch.to_pydict()
        for domain, doc, text in zip(cols["domain"], cols["doc"], cols["text"]):
            if len(histories) >= 24 and (domain, doc) not in histories:
                break
            histories.setdefault((domain, doc), set()).add(text)
        if len(histories) >= 24:
            break
    postings = defaultdict(list)
    for i, s in enumerate(qsets):
        for v in s:
            postings[v].append(i)
    checks = []; comparisons = []
    for (domain, doc), texts in histories.items():
        n, matched, _, _ = lexical(sorted(texts), vocab, lookup)
        hset = set()
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            hset.update(tuple(words[i:i+5]) for i in range(len(words)-4))
        counts = Counter(q for s in matched for q in postings[s])
        checks.append(dict(name="source-cardinality:"+doc, passed=n == len(hset)))
        for qi, qs in enumerate(native):
            actual = len(hset.intersection(qs))
            comparisons.append(dict(history=doc, domain=domain, query=qrows[qi]["id"], intersection=actual))
            require(counts[qi] == actual, "MINIMUM_SPARSE_INTERSECTION")
    checks.append(dict(name="all_576_real_pairs_exact", passed=len(comparisons) == 576))
    base = sorted(next(s for s in qsets if len(s) >= 100))[:100]
    # These controls are declared transformations of real source shingle IDs.
    cases = [("copy", set(base), set(base), (True, True)),
             ("exact_jaccard_boundary", set(base[:80]), set(base[20:]), (True, False)),
             ("below_jaccard", set(base[:80]), set(base[21:]), (False, False)),
             ("contained", set(base[:50]), set(base), (False, True)),
             ("below_containment", set(base[:50]), set(base[6:]), (False, False)),
             ("too_short_containment", set(base[:49]), set(base), (False, False)),
             ("empty", set(), set(base), (False, False))]
    controls = []
    for name, left, right, expected in cases:
        value = flags(len(left & right), len(left), len(right))
        checks.append(dict(name=name, passed=value == expected))
        controls.append(dict(name=name, left=sorted(left), right=sorted(right), expected=list(expected), actual=list(value)))
    publish(a.output/"real-pairs.json", comparisons); publish(a.output/"controls.json", controls)
    result = dict(checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  real_query_documents=len(qrows), real_history_documents=len(histories),
                  source_capture_sha256=sha(a.capture/"report.json"), scientific_admission=False)
    publish(a.output/"report.json", result)
    require(all(c["passed"] for c in checks), "MINIMUM_FAILED")
    print(json.dumps(dict(passed=result["passed"], total=result["total"])))


if __name__ == "__main__":
    main()
