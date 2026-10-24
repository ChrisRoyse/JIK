"""Chunked categorical anchors (a2-domain-chunked-v1, #12): reality pulled in as chunks, as Calyx does.

Operator 2026-10-19: senses must read reality in chunks and build constellations of the chunks; the whole-text
pooled reading used by record_assay.py is not how the reference systems measure (Leapable roster v3.1, lawsearch
51: pooling 4-8 segments loses ~0.2 bits per lane). Rules (predeclared, recorded in the summary):
  anchors   one categorical anchor per Wikidata property of the family (the v3 family's 20 properties): an
            entity carries it iff the property is present; classes = the property's most frequent values that
            each hold >= target_per_class entities, at most MAX_CLASSES; entities with another value are not
            in that anchor's population. target_per_class = max(ceil(192 / C), 32), as calyx bits_categorical.
  sample    per anchor, entities in a fixed hash order (seed 25), the first target_per_class of each class
            (deterministic, label-balanced, the calyx sampler).
  chunks    every sampled entity's description split by the corpus atomiser (segment(): sentences, 24-600
            chars, >= 4 words); each chunk is its own atom, in order. Output chunks/atoms.parquet
            (entity, row, chunk, text) for the measurement scripts; anchors.json; summary.json.
"""
import collections
import hashlib
import json
from math import ceil
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import ROLE_SENTENCE, segment
from senses.trajectory_population import publish
from text.atomise_pilot import MAX_CHARS, MIN_CHARS, WORDS

SAMPLE_TARGET, MIN_CLASS_SAMPLE, MAX_CLASSES, SEED = 192, 32, 6, 25


def chunks_of(text):
    out = []
    for s, e, role in segment(text):
        if role == ROLE_SENTENCE:
            t = text[s:e].strip()
            if MIN_CHARS <= len(t) <= MAX_CHARS and len(WORDS.findall(t)) >= 4:
                out.append(t)
    return out or [text.strip()[:MAX_CHARS]]  # a description with no kept sentence is one chunk (recorded)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    summ = json.loads((src / "summary.json").read_bytes())
    rows = [json.loads(l) for l in (src / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{SEED}|chunked|{rows[i]['entity']}".encode()).hexdigest())
    absent, other = summ["rules"]["n_values"] + 1, summ["rules"]["n_values"]
    anchors = {}
    for p in summ["properties"]:
        counts = collections.Counter(rows[i]["y"][p] for i in order if rows[i]["y"][p] not in (absent, other))
        for c in range(min(MAX_CLASSES, len(counts)), 1, -1):
            top = [v for v, _ in counts.most_common(c)]
            per = max(ceil(SAMPLE_TARGET / c), MIN_CLASS_SAMPLE)
            if all(counts[v] >= per for v in top):
                break
        else:
            continue
        taken, sample = collections.Counter(), []
        for i in order:
            v = rows[i]["y"][p]
            if v in top and taken[v] < per:
                taken[v] += 1; sample.append(i)
        anchors[p] = {"classes": [summ["values"][p][v] for v in top], "per_class": per, "rows": sample,
                      "labels": [top.index(rows[i]["y"][p]) for i in sample]}
    if not anchors:
        raise SystemExit(f"MNLM_CHUNK_ANCHORS: no property of {src} has two classes of >= {MIN_CLASS_SAMPLE} entities")
    used = sorted({i for a in anchors.values() for i in a["rows"]})
    out.mkdir(parents=True, exist_ok=False); (out / "chunks").mkdir()
    cols = {"entity": [], "row": [], "chunk": [], "text": []}
    single = 0
    for i in used:
        cs = chunks_of(rows[i]["text"]); single += len(cs) == 1
        for j, t in enumerate(cs):
            cols["entity"].append(rows[i]["entity"]); cols["row"].append(i); cols["chunk"].append(j); cols["text"].append(t)
    pq.write_table(pa.table({"entity": cols["entity"], "row": pa.array(cols["row"], pa.int32()), "chunk": pa.array(cols["chunk"], pa.int16()),
                             "text": cols["text"]}), out / "chunks" / "atoms.parquet")
    (out / "anchors.json").write_text(json.dumps(anchors, sort_keys=True))
    publish(out / "summary.json", {"schema": "a2-domain-chunked-v1", "family": summ.get("family"), "parent_rows_sha256": summ["rows_sha256"],
        "rules": {"sample_target": SAMPLE_TARGET, "min_class_sample": MIN_CLASS_SAMPLE, "max_classes": MAX_CLASSES, "seed": SEED,
                  "chunker": "canon.segment sentences 24-600 chars >= 4 words"},
        "anchors": {p: {"classes": a["classes"], "per_class": a["per_class"], "n": len(a["rows"])} for p, a in anchors.items()},
        "entities": len(used), "chunks": len(cols["text"]), "single_chunk_entities": single,
        "anchors_sha256": hashlib.sha256((out / "anchors.json").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"family": summ.get("family"), "anchors": len(anchors), "entities": len(used), "chunks": len(cols["text"]), "single_chunk": single}))


if __name__ == "__main__":
    main()
