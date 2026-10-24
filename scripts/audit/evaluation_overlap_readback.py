"""Independent exact lexical, candidate, edge and component readback (#14).

No producer imports. Reconstruct source shingles through persisted vocabularies;
enumerate candidates from static prefix buckets, unlike the streaming join.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np

CONTRACT = "evaluation-exact-overlap-v1"


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_OVERLAP_READBACK_"+code)


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path):
    with path.open("rb") as f:
        for line in f:
            yield json.loads(line)


def reference_candidates(sets):
    frequency = Counter(v for s in sets for v in s); buckets = defaultdict(list)
    for i, s in enumerate(sets):
        keep = len(s)-(3*len(s)+4)//5+1 if s else 0
        for v in sorted(s, key=lambda v: (frequency[v], v))[:keep]:
            buckets[v].append(i)
    pairs = set(); visits = expired = 0
    for members in buckets.values():
        largest = max(len(sets[i]) for i in members)
        expired += sum(5*len(sets[i]) < 3*largest for i in members)
        for i, j in itertools.combinations(members, 2):
            if 5*min(len(sets[i]), len(sets[j])) >= 3*max(len(sets[i]), len(sets[j])):
                pairs.add((min(i, j), max(i, j))); visits += 1
    return sorted(pairs), dict(prefix_entries=sum(map(len, buckets.values())), posting_visits=visits,
                               expired_postings=expired, candidate_pairs=len(pairs))


def reference_groups(documents, near):
    links = list(near); indexes = defaultdict(list)
    for i, row in enumerate(documents):
        indexes[("exact_fields", row["exact_sha256"])].append(i)
        for key in row["group_keys"]:
            indexes[("source_group", key)].append(i)
    for (kind, value), members in indexes.items():
        for i in members[1:]:
            links.append(dict(left=members[0], right=i, kind=kind, value=value))
    links.sort(key=lambda r: (r["left"], r["right"], r["kind"]))
    adjacency = defaultdict(set)
    for edge in links:
        adjacency[edge["left"]].add(edge["right"]); adjacency[edge["right"]].add(edge["left"])
    unseen = set(range(len(documents))); groups = []
    while unseen:
        seed = min(unseen); found = {seed}; todo = [seed]; unseen.remove(seed)
        while todo:
            i = todo.pop()
            for j in adjacency[i]-found:
                found.add(j); unseen.remove(j); todo.append(j)
        members = sorted(found); ids = sorted(documents[i]["id"] for i in members)
        groups.append(dict(id=sha(encode([CONTRACT, ids])), members=members, document_ids=ids,
                           domains=sorted({documents[i]["domain"] for i in members})))
    return links, groups


def audit(root, source):
    started = time.perf_counter(); raw = (root/"report.json").read_bytes(); report = json.loads(raw)
    source_raw = (source/"report.json").read_bytes(); sm = json.loads(source_raw)
    check(sha(source_raw) == report["source_report_sha256"] and sm["complete"] is True and sm["errors"] == [], "SOURCE")
    check(report["schema"] == CONTRACT and report["complete"] is True and report["errors"] == [], "INCOMPLETE")
    check(report["threshold"] == dict(numerator=3, denominator=5) and report["shingle_words"] == 5, "CONTRACT")
    check(all(report[k] is False for k in ("sealed_evaluation", "scientific_admission", "served"))
          and report["observations_removed"] == 0, "CLAIM")
    files = {"words.json", "shingles.npy", "indptr.npy", "members.npy", "documents.jsonl", "candidates.npy", "links.jsonl", "groups.jsonl"}
    check(set(report["files"]) == files, "INVENTORY")
    for name, info in report["files"].items():
        path = root/name
        check(path.stat().st_size == info["bytes"] and file_sha(path) == info["sha256"], "FILE_HASH")
    vocabulary = json.loads((root/"words.json").read_bytes())
    check(len(vocabulary) == len(set(vocabulary)) and all(isinstance(w, str) for w in vocabulary), "WORDS")
    word_index = {w: i for i, w in enumerate(vocabulary)}
    tuples = np.load(root/"shingles.npy", allow_pickle=False)
    check(tuples.dtype == np.dtype("uint32") and tuples.ndim == 2 and tuples.shape[1] == 5, "SHINGLE_SHAPE")
    shingle_index = {tuple(map(int, row)): i for i, row in enumerate(tuples)}
    check(len(shingle_index) == len(tuples) and (not tuples.size or int(tuples.max()) < len(vocabulary)), "SHINGLE_VOCAB")
    ptr = np.load(root/"indptr.npy", allow_pickle=False); flat = np.load(root/"members.npy", allow_pickle=False)
    check(ptr.dtype == np.dtype("uint64") and flat.dtype == np.dtype("uint32")
          and ptr.ndim == flat.ndim == 1 and len(ptr) > 0 and ptr[0] == 0 and ptr[-1] == len(flat)
          and np.all(ptr[1:] >= ptr[:-1]), "OFFSETS")
    documents = list(rows(root/"documents.jsonl")); sets = []; used_words = set(); used_shingles = set(); counts = Counter()
    check(len(ptr) == len(documents)+1 and len({r["id"] for r in documents}) == len(documents), "DOCUMENTS")
    i = 0
    for domain, source_meta in sorted(sm["sources"].items()):
        path = source/(domain+".jsonl")
        check(file_sha(path) == source_meta["sha256"], "SOURCE_BYTES")
        for row in rows(path):
            check(i < len(documents), "MISSING_DOCUMENT")
            actual = set(); fields = []
            for field in row["fields"]:
                text = field["text"]; check(sha(text.encode()) == field["sha256"], "SOURCE_FIELD")
                words = re.findall(r"\w+", text.lower())
                check(all(w in word_index for w in words), "MISSING_WORD")
                values = [word_index[w] for w in words]; used_words.update(values)
                for key in zip(values, values[1:], values[2:], values[3:], values[4:]):
                    check(key in shingle_index, "MISSING_SHINGLE"); actual.add(shingle_index[key])
                fields.append(dict(name=field["name"], sha256=field["sha256"], characters=len(text), words=len(words)))
            expected = dict(id=row["id"], domain=row["domain"], group_keys=row["group_keys"], fields=fields,
                            exact_sha256=sha(encode([f["text"] for f in row["fields"]])))
            check(documents[i] == expected, "DOCUMENT_LINEAGE")
            check(flat[int(ptr[i]):int(ptr[i+1])].tolist() == sorted(actual), "SHINGLE_MEMBERSHIP")
            sets.append(actual); used_shingles.update(actual); i += 1
            counts.update(documents=1, fields=len(fields), characters=sum(f["characters"] for f in fields),
                          words=sum(f["words"] for f in fields), shingle_memberships=len(actual))
            if not actual:
                counts["no_five_word_shingles"] += 1
    check(i == len(documents) and len(used_words) == len(vocabulary) and len(used_shingles) == len(tuples), "EXTRA_LEXICAL_DATA")
    pairs, stats = reference_candidates(sets)
    stored = np.load(root/"candidates.npy", allow_pickle=False)
    check(stored.dtype == np.dtype("uint32") and stored.shape == (len(pairs), 2)
          and stored.tolist() == [list(p) for p in pairs], "CANDIDATES")
    near = []
    for a, b in pairs:
        left = flat[int(ptr[a]):int(ptr[a+1])]; right = flat[int(ptr[b]):int(ptr[b+1])]
        common = len(np.intersect1d(left, right, assume_unique=True)); union = len(left)+len(right)-common
        if 5*common >= 3*union:
            near.append(dict(left=a, right=b, kind="near_exact_jaccard", intersection=common, union=union))
    links, groups = reference_groups(documents, near)
    check(list(rows(root/"links.jsonl")) == links, "LINKS")
    check(list(rows(root/"groups.jsonl")) == groups, "GROUPS")
    counts.update(unique_words=len(vocabulary), unique_shingles=len(tuples), **stats, near_links=len(near), links=len(links),
                  components=len(groups), grouped_documents=sum(len(g["members"]) for g in groups if len(g["members"]) > 1),
                  largest_component=max((len(g["members"]) for g in groups), default=0))
    check(dict(counts) == report["counts"], "COUNTS")
    return dict(ok=True, adjudication="author-checked", report_sha256=sha(raw), source_report_sha256=sha(source_raw),
                counts=dict(counts), all_source_shingles_equal=True, all_prefix_candidates_equal=True,
                all_exact_links_and_components_equal=True, sealed_evaluation=False, wall_seconds=time.perf_counter()-started)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "source", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); result = audit(a.publication, a.source)
    with a.output.open("xb") as f:
        f.write(encode(result)+b"\n")
    print(json.dumps(result))
