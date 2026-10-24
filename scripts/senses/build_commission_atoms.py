"""Atoms of one commissioning corpus (commission-v1): the training data of one commissioned sense.

Same atomiser as the pilot (text/atomise_pilot.py: segment() sentences, 24-600 chars, >= 4 words; legal and
finance unwrapped) with a training cap of MAX_PER_DOC atoms per document and MAX_ATOMS per sense. Documents are
split by sha256(doc) % 100: 0-97 train, 98-99 validation. Output atoms.parquet (split, doc, pos, text) + manifest.
Cost: one streaming pass over the corpus files, O(bytes); one process per sense.
"""
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_SENTENCE, segment
from senses.trajectory_population import publish
from text.atomise_pilot import MAX_CHARS, MIN_CHARS, WORDS, records, text_of

MAX_PER_DOC, MAX_ATOMS = 256, 8_000_000
DOMAIN_RULES = {"legal": "legal", "finance": "finance"}  # text_of: unwrap hard-wrapped legal text; 10-K sections


def main():
    raw, sense, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    man = json.loads((raw / "manifest.json").read_bytes())
    files = [r for r in man["sources"] if r["sense"] == sense]
    if not files:
        raise SystemExit(f"MNLM_COMMISSION_SENSE: {sense} not in {raw}/manifest.json")
    out.mkdir(parents=True, exist_ok=False)
    cols = {"split": [], "doc": [], "pos": [], "text": []}
    docs = 0
    for f in files:
        for r in records(raw / f["path"]):
            t = text_of(DOMAIN_RULES.get(sense, "general"), r)
            if not t:
                continue
            key = str(r.get("id") or r.get("filename") or r.get("url") or f"{f['path']}:{docs}")
            split = "val" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100 >= 98 else "train"
            pos = 0
            for s, e, role in segment(t):
                if role != ROLE_SENTENCE:
                    continue
                a = t[s:e].strip()
                if MIN_CHARS <= len(a) <= MAX_CHARS and len(WORDS.findall(a)) >= 4:
                    cols["split"].append(split); cols["doc"].append(docs); cols["pos"].append(pos); cols["text"].append(a)
                    pos += 1
                    if pos == MAX_PER_DOC:
                        break
            docs += 1
            if len(cols["text"]) >= MAX_ATOMS:
                break
        if len(cols["text"]) >= MAX_ATOMS:
            break
    if not cols["text"]:
        raise SystemExit(f"MNLM_COMMISSION_EMPTY: {sense} produced no atoms")
    pq.write_table(pa.table({"split": cols["split"], "doc": pa.array(cols["doc"], pa.int32()), "pos": pa.array(cols["pos"], pa.int16()),
                             "text": cols["text"]}), out / "atoms.parquet")
    publish(out / "manifest.json", {"schema": "commission-atoms-v1", "sense": sense, "files": files, "documents": docs,
        "atoms": len(cols["text"]), "val_atoms": cols["split"].count("val"), "max_per_doc": MAX_PER_DOC, "max_atoms": MAX_ATOMS,
        "atoms_sha256": hashlib.sha256((out / "atoms.parquet").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"sense": sense, "documents": docs, "atoms": len(cols["text"])}))


if __name__ == "__main__":
    main()
