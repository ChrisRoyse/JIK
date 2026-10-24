"""Anchor family A2, record form (a2-wikidata-record-v1, #12/#25): an entity's structured Wikidata record.

Rules (predeclared before any reading; recorded in the summary):
  properties  the 30 most frequent properties in Wikidata5m (statement count), excluding none;
  labels      per property, the 31 most frequent values among sampled entities, plus OTHER and ABSENT
              (absence is an explicit class), so each property is a 33-way label;
  entities    the 60 most frequent P31 classes, up to 150 entities each in hash order (seed 25) that have
              a text record; the atom is the first sentence of the description (>= 25 chars, else two);
  clusters    the entity (one row per entity).
The anchor is a vector of 30 typed labels; its bits are the sum of per-property held-out bits under a
factorised probe, a usable-information lower bound of the record's information. Reads only canonical
Wikidata5m (triples + texts); no embedder sees a label.
Cost: two streaming passes over the triples, one over the texts; O(statements + texts).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import sys

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_SENTENCE, segment
from senses.trajectory_population import publish

N_PROPS, N_VALUES, N_CLASSES, PER_CLASS, SEED = 30, 31, 60, 150, 25
SHARDS = ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test")


def lines(path):
    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        yield from io.TextIOWrapper(zr, encoding="utf-8")


def order(s):
    return hashlib.sha256(f"{SEED}|a2-record|{s}".encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canon", type=Path, required=True, help="canon/p0micro-v1/wikidata5m")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    pcount, p31 = collections.Counter(), {}
    for sh in SHARDS:
        for line in lines(a.canon / f"{sh}.jsonl.zst"):
            f = json.loads(line)["f"]
            pcount[f["p"]] += 1
            if f["p"] == "P31":
                p31.setdefault(f["s"], f["o"])
    props = [q for q, _ in pcount.most_common(N_PROPS)]
    classes = [c for c, _ in collections.Counter(p31.values()).most_common(N_CLASSES)]
    by_class = collections.defaultdict(list)
    for s, o in p31.items():
        if o in classes:
            by_class[o].append(s)
    cand = {}
    for o in classes:
        for s in sorted(by_class[o], key=order)[: PER_CLASS * 3]:
            cand[s] = o
    texts = {}
    for line in lines(a.canon / "wikidata5m_text.jsonl.zst"):
        rec = json.loads(line)
        q = rec["f"].get("entity_id")
        if q in cand and q not in texts:
            t = rec["text"]
            sent = [(s, e) for s, e, r in segment(t) if r == ROLE_SENTENCE]
            if not sent:
                continue
            s, e = sent[0]
            if e - s < 25 and len(sent) > 1:
                e = sent[1][1]
            if t[s:e].strip():
                texts[q] = t[s:e].strip()
    chosen, taken = [], collections.Counter()
    for q in sorted(cand, key=order):
        if q in texts and taken[cand[q]] < PER_CLASS:
            chosen.append(q); taken[cand[q]] += 1
    keep = set(chosen)
    facts = collections.defaultdict(dict)
    for sh in SHARDS:
        for line in lines(a.canon / f"{sh}.jsonl.zst"):
            f = json.loads(line)["f"]
            if f["s"] in keep and f["p"] in props:
                facts[f["s"]].setdefault(f["p"], f["o"])  # first statement per property (canonical order)
    values = {}
    for q_ in props:
        c = collections.Counter(facts[e][q_] for e in chosen if q_ in facts[e])
        values[q_] = [v for v, _ in c.most_common(N_VALUES)]
    rows = []
    for e in chosen:
        y = {}
        for q_ in props:
            v = facts[e].get(q_)
            y[q_] = N_VALUES + 1 if v is None else (values[q_].index(v) if v in values[q_] else N_VALUES)
        rows.append({"entity": e, "class": cand[e], "text": texts[e], "y": y})
    with (a.output / "rows.jsonl").open("x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    present = {q_: sum(1 for r in rows if r["y"][q_] != N_VALUES + 1) for q_ in props}
    publish(a.output / "summary.json", {"schema": "a2-wikidata-record-v1", "rules": {"n_props": N_PROPS, "n_values": N_VALUES,
        "classes": N_CLASSES, "per_class": PER_CLASS, "seed": SEED, "labels": "top values + OTHER + ABSENT"},
        "properties": props, "property_statements": {q_: pcount[q_] for q_ in props}, "present_per_property": present,
        "classes": [(c, taken[c]) for c in classes], "values": values, "rows": len(rows),
        "rows_sha256": hashlib.sha256((a.output / "rows.jsonl").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"rows": len(rows), "properties": props[:10], "present": dict(list(present.items())[:10])}))


if __name__ == "__main__":
    main()
