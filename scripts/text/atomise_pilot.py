"""LEGACY filtered diagnostic atoms; use constellation_corpus.py for text sequences.

Per domain, documents are read in the pinned file's order (no sampling randomness). Each document is
segmented by the deterministic Stage 2 segmenter (canon/segment.py, identity SEGMENTER_ID). Sentence
atoms whose stripped text is 24..600 characters and has at least four words are kept in document order
(rejected atoms such as headings are skipped), at most MAX_PER_DOC per document from the start. In the
hard-wrapped domains (legal, finance) single newlines are joined first; blank lines stay paragraph breaks. A domain stops when it reaches TARGET atoms.
Cost: O(characters read) per domain; one streaming pass, no all-pairs work.
Output: atoms.parquet (atom_id, domain, doc, pos, text) and manifest.json beside it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_SENTENCE, _self_id, segment
from senses.trajectory_population import publish

MAX_PER_DOC, MIN_CHARS, MAX_CHARS = 48, 24, 600
WORDS = re.compile(r"\w{2,}")
FINANCE_SECTIONS = ("section_1", "section_1A", "section_7")


def records(path: Path):
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                yield r
    else:
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", encoding="utf-8") as h:
            for line in h:
                yield json.loads(line)


HARD_WRAPPED = ("legal", "finance")  # court opinions and 10-K text are hard-wrapped: a single newline is not a boundary
WRAP = re.compile(r"(?<!\n)[ \t]*\n(?!\n)[ \t]*")


def text_of(domain: str, r: dict) -> str:
    t = "\n\n".join(r[k] for k in FINANCE_SECTIONS if r.get(k)) if domain == "finance" else (r.get("text") or "")
    if domain in HARD_WRAPPED:
        t = WRAP.sub(" ", t)  # join wrapped lines; blank lines stay paragraph breaks
    return t


def doc_id(domain: str, r: dict, n: int) -> str:
    for k in ("id", "filename", "url"):
        if r.get(k):
            return str(r[k])
    return f"{domain}:{n}"


def atoms_of(text: str):
    out = []
    for s, e, role in segment(text):
        if role != ROLE_SENTENCE:
            continue
        t = text[s:e].strip()
        if MIN_CHARS <= len(t) <= MAX_CHARS and len(WORDS.findall(t)) >= 4:
            out.append(t)  # rejected atoms (headings, fragments) are skipped; kept atoms stay in document order
        if len(out) == MAX_PER_DOC:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="raw/text-pilot-v1 (sealed manifest.json)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=int, default=160_000, help="atoms per domain")
    p.add_argument("--legacy-filtered-diagnostic", action="store_true",
                   help="explicitly reproduce old filtered/capped atoms, never contiguous training")
    a = p.parse_args()
    if not a.legacy_filtered_diagnostic:
        raise SystemExit("MNLM_PILOT_FILTERED_SEQUENCE: use scripts/text/constellation_corpus.py "
                         "for lossless source-order text; legacy reproduction requires "
                         "--legacy-filtered-diagnostic")
    manifest = json.loads((a.raw / "manifest.json").read_bytes())
    a.output.mkdir(parents=True, exist_ok=False)
    cols = {k: [] for k in ("atom_id", "domain", "doc", "pos", "text")}
    per = {}
    for src in manifest["sources"]:
        path = a.raw / src["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != src["sha256"]:
            raise SystemExit(f"MNLM_TEXT_PILOT_HASH: {src['path']} differs from its sealed digest")
        dom, kept, docs = src["domain"], 0, 0
        for n, r in enumerate(records(path)):
            run = atoms_of(text_of(dom, r))
            if len(run) < 3:
                continue
            d = doc_id(dom, r, n); docs += 1
            for i, t in enumerate(run):
                cols["atom_id"].append(hashlib.sha256(f"{dom}|{d}|{i}|{t}".encode()).hexdigest()[:24])
                cols["domain"].append(dom); cols["doc"].append(d); cols["pos"].append(i); cols["text"].append(t)
            kept += len(run)
            if kept >= a.target:
                break
        per[dom] = {"atoms": kept, "documents": docs, "source_sha256": src["sha256"]}
        print(dom, per[dom], flush=True)
    pq.write_table(pa.table(cols), a.output / "atoms.parquet", compression="zstd")
    publish(a.output / "manifest.json", {"schema": "text-pilot-atoms-v1", "segmenter": _self_id(), "per_domain": per,
        "atoms": len(cols["text"]), "rules": {"max_per_doc": MAX_PER_DOC, "chars": [MIN_CHARS, MAX_CHARS], "min_words": 4, "contiguous": False, "unwrapped_domains": list(HARD_WRAPPED)},
        "raw_manifest_sha256": hashlib.sha256((a.raw / "manifest.json").read_bytes()).hexdigest(),
        "atoms_sha256": hashlib.sha256((a.output / "atoms.parquet").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"atoms": len(cols["text"])}))


if __name__ == "__main__":
    main()
