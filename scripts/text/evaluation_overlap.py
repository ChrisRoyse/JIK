"""Exact surface-overlap components for the bounded evaluation capture (#14).

Lowercase word 5-grams do not cross fields. Exact tuple interning avoids hash
collisions. Prefix filtering is complete for Jaccard >= 3/5, not a semantic test.
Reference: Xiao et al. WWW 2008, section 3, prefix-filtering principle.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

CONTRACT = "evaluation-exact-overlap-v1"
WORD = re.compile(r"\w+")
NUM, DEN, WIDTH = 3, 5, 5


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_OVERLAP_"+code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Features:
    def __init__(self):
        self.words = {}; self.shingles = {}; self.sets = []; self.documents = []
        self.counts = Counter()

    def add(self, row):
        values = set(); fields = []
        for field in row["fields"]:
            text = field["text"]
            require(sha(text.encode()) == field["sha256"] and len(text) == field["characters"]
                    and len(text.encode()) == field["utf8_bytes"], "SOURCE_FIELD")
            words = WORD.findall(text.lower()); ids = []
            for word in words:
                if word not in self.words:
                    self.words[word] = len(self.words)
                ids.append(self.words[word])
            for i in range(len(ids)-WIDTH+1):
                key = tuple(ids[i:i+WIDTH])
                if key not in self.shingles:
                    self.shingles[key] = len(self.shingles)
                values.add(self.shingles[key])
            fields.append(dict(name=field["name"], sha256=field["sha256"], characters=len(text), words=len(words)))
        self.documents.append(dict(id=row["id"], domain=row["domain"], group_keys=row["group_keys"],
                                   fields=fields, exact_sha256=sha(canonical([f["text"] for f in row["fields"]]))))
        self.sets.append(values)
        self.counts.update(documents=1, fields=len(fields), characters=sum(f["characters"] for f in fields),
                           words=sum(f["words"] for f in fields), shingle_memberships=len(values))
        if not values:
            self.counts["no_five_word_shingles"] += 1

    def save(self, output):
        require(len(self.words) < 2**32 and len(self.shingles) < 2**32, "VOCABULARY_OVERFLOW")
        (output/"words.json").write_bytes(canonical(list(self.words)))
        tuples = np.fromiter((v for key in self.shingles for v in key), dtype=np.uint32,
                             count=len(self.shingles)*WIDTH).reshape(-1, WIDTH)
        np.save(output/"shingles.npy", tuples)
        lengths = np.array([len(s) for s in self.sets], dtype=np.uint64)
        ptr = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(lengths)])
        np.save(output/"indptr.npy", ptr)
        flat = np.fromiter((v for s in self.sets for v in sorted(s)), dtype=np.uint32, count=int(ptr[-1]))
        np.save(output/"members.npy", flat)
        with (output/"documents.jsonl").open("xb") as f:
            for row in self.documents:
                f.write(canonical(row)+b"\n")


def prefix_join(sets):
    frequency = Counter(v for s in sets for v in s)
    prefixes = [sorted(s, key=lambda v: (frequency[v], v))[:len(s)-(NUM*len(s)+DEN-1)//DEN+1]
                if s else [] for s in sets]
    index = defaultdict(deque); candidates = []; links = []
    counts = Counter(expired_postings=0, posting_visits=0, prefix_entries=0)
    for i in sorted(range(len(sets)), key=lambda i: (len(sets[i]), i)):
        if not sets[i]:
            continue
        minimum = (NUM*len(sets[i])+DEN-1)//DEN
        found = set()
        for token in prefixes[i]:
            posting = index[token]
            while posting and len(sets[posting[0]]) < minimum:
                posting.popleft(); counts["expired_postings"] += 1
            counts["posting_visits"] += len(posting)
            found.update(posting)
            posting.append(i); counts["prefix_entries"] += 1
        for j in sorted(found):
            a, b = sorted((i, j)); candidates.append((a, b))
            overlap = len(sets[i] & sets[j]); union = len(sets[i])+len(sets[j])-overlap
            if DEN*overlap >= NUM*union:
                links.append(dict(left=a, right=b, kind="near_exact_jaccard", intersection=overlap, union=union))
    candidates.sort(); links.sort(key=lambda r: (r["left"], r["right"]))
    counts.update(candidate_pairs=len(candidates), near_links=len(links))
    return candidates, links, dict(counts)


def components(documents, near):
    parents = list(range(len(documents))); links = list(near); first = {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i

    for i, row in enumerate(documents):
        keys = [("exact_fields", row["exact_sha256"])] + [("source_group", g) for g in row["group_keys"]]
        for kind, value in keys:
            key = kind, value
            if key in first:
                links.append(dict(left=first[key], right=i, kind=kind, value=value))
            else:
                first[key] = i
    links.sort(key=lambda r: (r["left"], r["right"], r["kind"]))
    for link in links:
        a, b = sorted((root(link["left"]), root(link["right"])))
        parents[b] = a
    groups = defaultdict(list)
    for i in range(len(documents)):
        groups[root(i)].append(i)
    result = []
    for members in groups.values():
        ids = sorted(documents[i]["id"] for i in members)
        result.append(dict(id=sha(canonical([CONTRACT, ids])), members=members,
                           document_ids=ids, domains=sorted({documents[i]["domain"] for i in members})))
    return links, sorted(result, key=lambda r: r["members"][0])


def run(source, pin, output):
    started = time.perf_counter(); require(not output.exists(), "EXISTS")
    raw = (source/"report.json").read_bytes(); source_report = json.loads(raw)
    require(sha(raw) == pin and source_report["schema"] == "evaluation-source-capture-v1", "SOURCE_PIN")
    require(source_report["complete"] is True and source_report["errors"] == []
            and source_report["sealed_evaluation"] is False, "SOURCE_INCOMPLETE")
    output.mkdir(parents=True); features = Features(); failures = []; phases = Counter()
    for domain, sm in sorted(source_report["sources"].items()):
        try:
            start = time.perf_counter(); path = source/(domain+".jsonl")
            require(path.name == sm["file"] and path.stat().st_size == sm["bytes"] and file_sha(path) == sm["sha256"], "SOURCE_BYTES")
            phases["load_hash_seconds"] += time.perf_counter()-start
            with path.open("rb") as f:
                for ordinal, line in enumerate(f):
                    start = time.perf_counter(); row = json.loads(line)
                    phases["load_hash_seconds"] += time.perf_counter()-start
                    try:
                        start = time.perf_counter(); features.add(row)
                        phases["lexical_seconds"] += time.perf_counter()-start
                    except (ValueError, KeyError, TypeError) as exc:
                        failures.append(dict(domain=domain, ordinal=ordinal, error=str(exc)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(domain=domain, error=str(exc)))
    expected = sum(sm["counts"]["selected_documents"] for sm in source_report["sources"].values())
    if len(features.documents) != expected:
        failures.append(dict(error="MNLM_OVERLAP_MISSING_DOCUMENTS"))
    ids = [r["id"] for r in features.documents]
    require(len(set(ids)) == len(ids), "DUPLICATE_ID")
    start = time.perf_counter(); candidates, near, join_stats = prefix_join(features.sets)
    phases["join_seconds"] = time.perf_counter()-start
    start = time.perf_counter(); links, groups = components(features.documents, near)
    phases["component_seconds"] = time.perf_counter()-start
    start = time.perf_counter(); features.save(output)
    np.save(output/"candidates.npy", np.array(candidates, dtype=np.uint32).reshape(-1, 2))
    for name, rows in (("links.jsonl", links), ("groups.jsonl", groups)):
        with (output/name).open("xb") as f:
            for row in rows:
                f.write(canonical(row)+b"\n")
    files = {p.name: dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    phases["publish_seconds"] = time.perf_counter()-start
    wall = time.perf_counter()-started; phases["other_seconds"] = wall-sum(phases.values()); phases["wall_seconds"] = wall
    counts = dict(features.counts, unique_words=len(features.words), unique_shingles=len(features.shingles),
                  **join_stats, links=len(links), components=len(groups),
                  grouped_documents=sum(len(g["members"]) for g in groups if len(g["members"]) > 1),
                  largest_component=max((len(g["members"]) for g in groups), default=0))
    report = dict(schema=CONTRACT, source_report_sha256=pin, counts=counts, files=files, errors=failures,
                  complete=not failures, threshold=dict(numerator=NUM, denominator=DEN), shingle_words=WIDTH,
                  lexical_rule="Unicode lowercase then regex word characters; separate fields; exact tuple IDs",
                  scope="complete overlap under declared lexical metric within this capture only",
                  sealed_evaluation=False, scientific_admission=False, served=False, observations_removed=0,
                  pending=["historical-text/entity overlap", "world referent resolution", "instrument exposure",
                           "source eligibility", "event-time and composition partitions", "anchors and final seal"],
                  timing=dict(phases), code_sha256=file_sha(Path(__file__)))
    publish(output/"report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); result = run(a.source, a.source_report_sha256, a.output)
    print(json.dumps({k: result[k] for k in ("counts", "complete", "errors", "timing")}))
    if not result["complete"]:
        raise SystemExit(2)
