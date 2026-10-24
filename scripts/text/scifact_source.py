"""Capture a bounded official source-support corpus; never execute provider code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.evaluation_multiscale import file_sha

COMMIT = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
BASE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/"
MEMBERS = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")
LIMIT = 16 * 1024 * 1024


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_SCIFACT_" + code)


def capture(url, path, expected=None):
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response, path.open("xb") as out:
        header = response.headers.get("Content-Length")
        require(header is not None and 0 < int(header) <= LIMIT, "HTTP_SIZE")
        require(expected is None or int(header) == expected, "HTTP_CHANGED")
        count = 0
        while block := response.read(128 * 1024):
            count += len(block)
            require(count <= LIMIT and time.perf_counter()-started < 30, "HTTP_BOUND")
            out.write(block)
        require(count == int(header), "HTTP_TRUNCATED")
        receipt = dict(url=url, final_url=response.url, bytes=count,
                       etag=response.headers.get("ETag"), modified=response.headers.get("Last-Modified"))
    return {**receipt, "sha256": file_sha(path)}


def rows(path):
    with path.open("rb") as f:
        return [json.loads(line) for line in f]


def census(root):
    docs = rows(root/"corpus.jsonl")
    by_id = {d["doc_id"]: d for d in docs}
    require(len(by_id) == len(docs), "DUPLICATE_DOCUMENT")
    labels = Counter(); counts = Counter(documents=len(docs))
    for d in docs:
        require(type(d["doc_id"]) is int and isinstance(d["title"], str)
                and isinstance(d["abstract"], list) and d["abstract"]
                and all(isinstance(s, str) and s.strip() for s in d["abstract"]), "DOCUMENT_SCHEMA")
        counts.update(sentences=len(d["abstract"]), abstract_characters=sum(map(len, d["abstract"])))
    claim_ids = set()
    for split in ("train", "dev"):
        claims = rows(root/("claims_"+split+".jsonl")); counts[split+"_claims"] = len(claims)
        for c in claims:
            require(c["id"] not in claim_ids, "DUPLICATE_CLAIM"); claim_ids.add(c["id"])
            require(isinstance(c["claim"], str) and c["claim"].strip()
                    and isinstance(c["evidence"], dict) and c["cited_doc_ids"], "CLAIM_SCHEMA")
            counts["cited_pairs"] += len(c["cited_doc_ids"])
            for doc_id in c["cited_doc_ids"]:
                require(doc_id in by_id, "CITED_DOCUMENT")
            for doc_id, sets in c["evidence"].items():
                require(int(doc_id) in by_id and sets, "EVIDENCE_DOCUMENT")
                counts["evidence_pairs"] += 1
                counts["alternative_rationale_pairs"] += int(len(sets) > 1)
                for s in sets:
                    require(s["label"] in ("SUPPORT", "CONTRADICT") and s["sentences"]
                            and len(set(s["sentences"])) == len(s["sentences"])
                            and all(type(i) is int and 0 <= i < len(by_id[int(doc_id)]["abstract"])
                                    for i in s["sentences"]), "RATIONALE")
                    labels[s["label"]] += 1
                    counts["multi_sentence_rationales"] += int(len(s["sentences"]) > 1)
    contexts = rows(root/"claims_with_citances.jsonl")
    counts["citation_contexts"] = len(contexts)
    counts["context_claims"] = sum(len(r["claims"]) for r in contexts)
    return dict(counts=counts, rationale_labels=labels)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); started = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    phases = {}; receipts = {}; errors = []
    requests = [("data.tar.gz", BASE+"data.tar.gz", 3115079),
                ("claims_with_citances.jsonl", BASE+"claims_with_citances.jsonl", 4053606)]
    for name in ("LICENSE.md", "README.md", "doc/data.md", "doc/claims-with-citances.md"):
        requests.append((name.replace("/", "__"), "https://raw.githubusercontent.com/allenai/scifact/"+COMMIT+"/"+name, None))
    tick = time.perf_counter()
    for name, url, expected in requests:
        try:
            receipts[name] = capture(url, a.output/name, expected)
        except (OSError, ValueError) as exc:
            errors.append(dict(file=name, error=str(exc)))
    phases["capture"] = time.perf_counter()-tick
    publish(a.output/"capture.json", dict(files=receipts, errors=errors, provider_commit=COMMIT))
    require(not errors, "CAPTURE_INCOMPLETE")
    tick = time.perf_counter(); inventory = []
    with tarfile.open(a.output/"data.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        require(sum(m.size for m in members) <= 64*1024*1024, "ARCHIVE_BOUND")
        names = [m.name for m in members]
        require(len(set(names)) == len(names), "ARCHIVE_DUPLICATES")
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in member.name
                    and (member.isfile() or member.isdir()), "ARCHIVE_MEMBER")
            inventory.append(dict(name=member.name, bytes=member.size, kind="file" if member.isfile() else "directory"))
        for name in MEMBERS:
            matches = [m for m in members if m.name == "data/"+name and m.isfile()]
            require(len(matches) == 1 and 0 < matches[0].size <= LIMIT, "ARCHIVE_SELECTION")
            content = archive.extractfile(matches[0]).read()
            require(len(content) == matches[0].size, "ARCHIVE_TRUNCATED")
            with (a.output/name).open("xb") as out:
                out.write(content)
    phases["named_member_capture"] = time.perf_counter()-tick
    tick = time.perf_counter(); found = census(a.output)
    licence = (a.output/"LICENSE.md").read_text(encoding="utf-8")
    require("CC BY 4.0" in licence and "ODC-By 1.0" in licence, "LICENCE")
    phases["source_schema_census"] = time.perf_counter()-tick
    tick = time.perf_counter()
    files = {f.name:dict(bytes=f.stat().st_size, sha256=file_sha(f)) for f in sorted(a.output.iterdir()) if f.is_file()}
    phases["file_pins"] = time.perf_counter()-tick; wall = time.perf_counter()-started
    report = dict(schema="scifact-source-v1", files=files, archive_inventory=inventory, **found,
                  provider_commit=COMMIT, complete=True, test_labels_used=False, scientific_admission=False,
                  source_truth="provider-human-annotation-of-source-support", world_truth_adjudicated=False,
                  attribution="Wadden et al. 2020, Fact or Fiction: Verifying Scientific Claims, EMNLP",
                  phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall)
    publish(a.output/"report.json", report)
    print(json.dumps({k:report[k] for k in ("counts", "rationale_labels", "wall_seconds", "phase_coverage")}))


if __name__ == "__main__":
    main()
