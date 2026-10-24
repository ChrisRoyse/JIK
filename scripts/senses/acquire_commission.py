"""Training corpora for commissioned senses (commission-v1, #12/#25; docs/senses_of_a_machine.txt P1, §5.3, §11.1).

Each sense is an embedder we train from scratch on ONE corpus's associations and then freeze. Public embedders
share one web of associations (overlapping web corpora), which is why they saturate D1 (#12, 2026-10-18). These
corpora are chosen as different webs. Shards are disjoint from the pilot corpus (raw/text-pilot-v1): different
files of the same repositories, or repositories the pilot never read. Pinned revisions; files read-only; a
re-run verifies against the sealed manifest. Supply chain has no open corpus here: a named gap.
"""
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

UP, REG, BHL, PES = ("common-pile/uspto", "e8b975381547f026e2ccfe721f785dcb6a2fa9a0"), ("common-pile/regulations", "d1a63eef07bb893f834742f4738aeb1d545b7e01"), \
    ("common-pile/biodiversity_heritage_library", "ad1bbb00d5df579c0ef0a9dfe46e50e2bb1715b8"), ("common-pile/pes2o", "2caeba1585aaafa3c965410d80af85b09ed11cd9")
PUB, CAP, GH, EDGAR = ("common-pile/pubmed", "648b8cfc93953ca0663a9c96a8d842a91b98fb64"), ("common-pile/caselaw_access_project", "3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90"), \
    ("common-pile/github_archive", "60317859e1212d37268e2b55e31acd2aecd4ae52"), ("eloukas/edgar-corpus", "7e90f0f342569b35213445f809cfaf3b91f9964f")
SOURCES = {
    "patent": (UP, ["uspto_1.parquet"]),
    "regulatory": (REG, [f"v0/{i:05d}_regulations.jsonl.gz" for i in range(1, 5)]),
    "natural-history": (BHL, [f"v0/{i:05d}_bhl.jsonl.gz" for i in range(1, 4)]),
    "science": (PES, ["v0/documents/peS2o-0001.json.gz", "v0/documents/peS2o-0002.json.gz"]),
    "legal": (CAP, [f"cap_{i:05d}.jsonl.gz" for i in range(1, 5)]),
    "biomedical": (PUB, ["data/00001_pubmedcentral.jsonl.gz"]),
    "finance": (EDGAR, ["2016/train.jsonl"]),
    "software": (GH, [f"gharchive/raw/documents/{i:05d}_github-archive-threads.jsonl.gz" for i in range(1, 31)]),
}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sense, ((repo, rev), files) in SOURCES.items():
        for name in files:
            path = Path(hf_hub_download(repo, name, revision=rev, repo_type="dataset", local_dir=out / sense))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.chmod(0o444)
            rows.append({"sense": sense, "repo": repo, "revision": rev, "file": name, "path": str(path.relative_to(out)),
                         "bytes": path.stat().st_size, "sha256": digest})
            print(json.dumps(rows[-1]), flush=True)
    body = json.dumps({"schema": "commission-acquisition-v1", "sources": rows}, indent=1, sort_keys=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        assert manifest.read_text() == body, "MNLM_COMMISSION_DRIFT: acquired bytes differ from the sealed manifest"
    else:
        manifest.write_text(body)
    print("sealed", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
