"""A2 record anchors per domain family (a2-domain-record-v2, #12/#25): where domain knowledge should matter.

Rules (predeclared before any reading; recorded in each summary):
  families    biomedical, finance, legal, science, code; each a fixed list of Wikidata P31 classes (CLASSES);
  entities    per family up to PER_FAMILY entities, balanced over its classes in hash order (seed 25), that have
              a description text; an entity in several families' classes goes to the first family listed;
  text        the whole description up to MAX_CHARS characters, cut at the last sentence end before the cap
              (embedders still truncate at their own window; truncation is recorded per lens by measure_pilot);
  properties  per family, its N_PROPS most frequent properties among its sampled entities, P31 excluded
              (the sampling class would otherwise be the label);
  labels      per property, the N_VALUES most frequent values plus OTHER and ABSENT (explicit), 33-way.
Output per family: rows.jsonl, atoms.parquet (text column for measure_pilot), summary.json.
Cost: two streaming passes over the triples and one over the texts, O(statements + texts).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

CLASSES = {"biomedical": ["Q12136", "Q11173", "Q7187", "Q12140"],
           "finance": ["Q4830453", "Q783794", "Q891723"],
           "legal": ["Q2334719", "Q1137809", "Q131569"],
           "science": ["Q5633421", "Q11862829"],
           "code": ["Q7397", "Q9143"]}
PER_FAMILY, N_PROPS, N_VALUES, SEED, MAX_CHARS = 3000, 20, 31, 25, 1500
SHARDS = ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test")
END = re.compile(r"[.!?](?=\s|$)")


def lines(path):
    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        yield from io.TextIOWrapper(zr, encoding="utf-8")


def order(s):
    return hashlib.sha256(f"{SEED}|a2-domain|{s}".encode()).hexdigest()


def cut(t):
    t = t.strip()
    if len(t) <= MAX_CHARS:
        return t
    ends = [m.end() for m in END.finditer(t, 0, MAX_CHARS)]
    return t[: ends[-1]] if ends else t[:MAX_CHARS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canon", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-family", type=int, default=PER_FAMILY, help="sample size per family (power, not a gate)")
    a = p.parse_args()
    per_family = a.per_family
    a.output.mkdir(parents=True, exist_ok=False)
    family_of = {}
    for fam, cs in CLASSES.items():
        for c in cs:
            family_of.setdefault(c, fam)
    members = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for sh in SHARDS:
        for line in lines(a.canon / f"{sh}.jsonl.zst"):
            if '"p": "P31"' not in line:
                continue
            f = json.loads(line)["f"]
            if f["p"] == "P31" and f["o"] in family_of and f["s"] not in seen:
                seen.add(f["s"]); members[family_of[f["o"]]][f["o"]].append(f["s"])
    cand = {}
    for fam, by in members.items():
        per = per_family // len(CLASSES[fam])
        for c in CLASSES[fam]:
            for s in sorted(by.get(c, []), key=order)[: per * 2]:
                cand[s] = (fam, c)
    texts = {}
    for line in lines(a.canon / "wikidata5m_text.jsonl.zst"):
        rec = json.loads(line)
        q = rec["f"].get("entity_id")
        if q in cand and q not in texts and rec["text"].strip():
            texts[q] = cut(rec["text"])
    chosen = collections.defaultdict(list)
    taken = collections.Counter()
    for q in sorted(cand, key=order):
        fam, c = cand[q]
        if q in texts and taken[(fam, c)] < per_family // len(CLASSES[fam]):
            chosen[fam].append(q); taken[(fam, c)] += 1
    keep = {q for v in chosen.values() for q in v}
    facts = collections.defaultdict(dict)
    for sh in SHARDS:
        for line in lines(a.canon / f"{sh}.jsonl.zst"):
            f = json.loads(line)["f"]
            if f["s"] in keep and f["p"] != "P31":
                facts[f["s"]].setdefault(f["p"], f["o"])
    for fam, ents in chosen.items():
        out = a.output / fam; out.mkdir()
        pc = collections.Counter(pp for e in ents for pp in facts[e])
        props = [pp for pp, _ in pc.most_common(N_PROPS)]
        values = {pp: [v for v, _ in collections.Counter(facts[e][pp] for e in ents if pp in facts[e]).most_common(N_VALUES)] for pp in props}
        rows = []
        for e in ents:
            y = {pp: (N_VALUES + 1 if pp not in facts[e] else (values[pp].index(facts[e][pp]) if facts[e][pp] in values[pp] else N_VALUES)) for pp in props}
            rows.append({"entity": e, "class": cand[e][1], "text": texts[e], "y": y})
        with (out / "rows.jsonl").open("x", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        pq.write_table(pa.table({"atom_id": [r["entity"] for r in rows], "text": [r["text"] for r in rows]}), out / "atoms.parquet")
        publish(out / "summary.json", {"schema": "a2-wikidata-record-v1", "family": fam, "anchor_version": "a2-domain-record-v2",
            "rules": {"n_props": N_PROPS, "n_values": N_VALUES, "per_family": per_family, "seed": SEED, "max_chars": MAX_CHARS,
                      "classes": CLASSES[fam], "p31_excluded": True, "labels": "top values + OTHER + ABSENT"},
            "properties": props, "present_per_property": {pp: pc[pp] for pp in props}, "values": values,
            "classes_taken": {c: taken[(fam, c)] for c in CLASSES[fam]}, "rows": len(rows),
            "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print(fam, len(rows), props[:8], flush=True)


if __name__ == "__main__":
    main()
