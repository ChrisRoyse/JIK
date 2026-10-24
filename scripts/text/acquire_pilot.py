"""Multi-domain text pilot corpus v1 (#3, #25): one pinned shard per domain, sha256-sealed.

Domains: general (FineWeb-Edu), biomedical (PubMed Central, Common Pile open subset), legal (Caselaw
Access Project via Common Pile, open), finance (EDGAR 10-K corpus), science (arXiv abstracts), encyclopedic (Wikipedia).
Code is already present (SWE trajectories). Files are read-only once written; a re-run verifies.
"""
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

SOURCES = [
    ("general", "HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "sample/10BT/000_00000.parquet", "odc-by"),
    ("biomedical", "common-pile/pubmed", "648b8cfc93953ca0663a9c96a8d842a91b98fb64", "data/00000_pubmedcentral.jsonl.gz", "open subset (per-record licence)"),
    ("legal", "common-pile/caselaw_access_project", "3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90", "cap_00000.jsonl.gz", "public domain (CAP via Common Pile)"),
    ("finance", "eloukas/edgar-corpus", "7e90f0f342569b35213445f809cfaf3b91f9964f", "2018/train.jsonl", "apache-2.0"),
    ("science", "common-pile/arxiv_abstracts", "828e35d1000f94579da8850f5f640c138279bdb5", "00000_arxiv-abstracts.jsonl.gz", "per-record licence"),
    ("encyclopedic", "wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en/train-00000-of-00041.parquet", "cc-by-sa-3.0"),
]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, repo, rev, name, licence in SOURCES:
        dest = out / domain
        path = Path(hf_hub_download(repo, name, revision=rev, repo_type="dataset", local_dir=dest))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.chmod(0o444)
        rows.append({"domain": domain, "repo": repo, "revision": rev, "file": name, "licence": licence,
                     "path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": digest})
        print(json.dumps(rows[-1]), flush=True)
    manifest = out / "manifest.json"
    body = json.dumps({"schema": "text-pilot-acquisition-v1", "sources": rows}, indent=1, sort_keys=True)
    if manifest.exists():
        assert manifest.read_text() == body, "MNLM_TEXT_PILOT_DRIFT: acquired bytes differ from the sealed manifest"
    else:
        manifest.write_text(body)
    print("sealed", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
