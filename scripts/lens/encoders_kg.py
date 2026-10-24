"""Label-free Wikidata5m graph lenses for the A2 statement questions (issue #12, readback finding F5). The Stage 3 context
lenses `kg-relation-signature` / `kg-neighbour-classes` (encoders_ctx.py) exclude the atom's own P31 statements, which is
right for the P31 questions and wrong for a question about P17 or P106: there the entity's own P17 or P106 edge is the
label and its object would enter the neighbour-class histogram. These two lenses read the same triple store but exclude
the entity's own statements of the property the question asks about (P31, P17 or P106, from the family name), so the
same lens never reads its label on any A2 question. They are a separate module so that the frozen identities of the
first context lenses do not move; the catalogue restricts the first pair to the P31 questions and this pair to the
P17 and P106 questions, so no question is measured by both. Cost: one streamed pass over the triples (cached under
<lens_root>/ctx/ per key set), then O(rows × edges of the entity)."""
from __future__ import annotations

import collections
import math
import re
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot, _hash_index  # noqa: E402
from encoders_ctx import CLASS_DIMS, KG_SOURCE, REL_DIMS, _cached, _cal_rows_of  # noqa: E402
from lenscommon import Paths  # noqa: E402

ENCODERS_KG_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
_TRIPLE = re.compile(r'"o": "(Q\d+)", "p": "(P\d+)", "s": "(Q\d+)"')
LABEL_PROPERTY = {"a2-wikidata-p31": "P31", "a2-wikidata-p31-rare": "P31", "a2-wikidata-p17": "P17", "a2-wikidata-p106": "P106"}
SHARDS = ["wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test"]


def _kg_ctx_by_property(paths: Paths) -> dict:
    """Per calibration entity: out-edges grouped by property (counts and the objects' P31 classes) and in-edges grouped by
    property (counts and the subjects' P31 classes); nothing excluded here, the exclusion is applied per question."""
    ents = sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE)})

    def build() -> dict:
        import io

        import zstandard

        want = set(ents)
        out_edges, in_edges = collections.defaultdict(list), collections.defaultdict(list)
        p31: dict[str, str] = {}
        triples = 0
        for sh in SHARDS:
            with (paths.canon(KG_SOURCE) / f"{sh}.jsonl.zst").open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
                for line in io.TextIOWrapper(zr, encoding="utf-8"):
                    m = _TRIPLE.search(line)
                    if not m:
                        continue
                    o, p, s = m.groups()
                    triples += 1
                    if p == "P31":
                        p31.setdefault(s, o)
                    if s in want:
                        out_edges[s].append((p, o))
                    if o in want:
                        in_edges[o].append((p, s))
        per = {}
        for e in ents:
            out = collections.defaultdict(lambda: {"n": 0, "classes": collections.Counter()})
            for p, o in out_edges[e]:
                out[p]["n"] += 1
                if o in p31:
                    out[p]["classes"][p31[o]] += 1
            inn = collections.defaultdict(lambda: {"n": 0, "classes": collections.Counter()})
            for p, s in in_edges[e]:
                inn[p]["n"] += 1
                if s in p31:
                    inn[p]["classes"][p31[s]] += 1
            per[e] = {"out": {p: {"n": v["n"], "classes": dict(v["classes"])} for p, v in out.items()}, "in": {p: {"n": v["n"], "classes": dict(v["classes"])} for p, v in inn.items()}}
        return {"source": KG_SOURCE, "shards": SHARDS, "triples": triples, "entities_with_p31": len(p31), "entities": per}

    return _cached(paths, "kgq", ["shards", *SHARDS, "by-property", "entities", *ents], build)


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    relation = name == "kg-relation-signature-q"
    dims = 2 * REL_DIMS + 4 if relation else CLASS_DIMS
    X, absent = np.zeros((len(rows), dims), np.float32), np.ones(len(rows), bool)
    if not any(r["source"] == KG_SOURCE and r.get("family") in LABEL_PROPERTY for r in rows):
        return Slot(X, absent, {"absent": "no A2 statement question in this family"}, "cosine")
    ents = _kg_ctx_by_property(paths)["entities"]
    for i, r in enumerate(rows):
        label = LABEL_PROPERTY.get(r.get("family", ""))
        e = ents.get(r["cluster"]) if (r["source"] == KG_SOURCE and label) else None
        if not e:
            continue
        out = {p: v for p, v in e["out"].items() if p != label}  # the entity's own statements of the asked property are the label
        inn = e["in"]
        out_deg, in_deg = sum(v["n"] for v in out.values()), sum(v["n"] for v in inn.values())
        if out_deg + in_deg == 0:
            continue
        if relation:
            for p, v in out.items():
                j, _ = _hash_index("out", p, dims=REL_DIMS)
                X[i, j] += v["n"]
            for p, v in inn.items():
                j, _ = _hash_index("in", p, dims=REL_DIMS)
                X[i, REL_DIMS + j] += v["n"]
            X[i, :2 * REL_DIMS] = np.log1p(X[i, :2 * REL_DIMS])
            X[i, 2 * REL_DIMS:] = [math.log1p(out_deg), math.log1p(in_deg), math.log1p(len(out)), math.log1p(len(inn))]
        else:
            for v in list(out.values()) + list(inn.values()):
                for c, n in v["classes"].items():
                    j, _ = _hash_index("class", c, dims=CLASS_DIMS)
                    X[i, j] += n
            if X[i].sum() == 0:
                continue
            X[i] /= X[i].sum()
        absent[i] = False
    fit = {"hash_dims": REL_DIMS if relation else CLASS_DIMS, "excluded": "the entity's own statements of the property the question asks about (per family)",
           "layout": "log1p out-relation histogram | log1p in-relation histogram | log1p out/in degree, distinct out/in relations" if relation else "P31 class histogram of out- and in-neighbours, normalised"}
    return Slot(X, absent, fit, "cosine")


SOURCE_OF = {"kg-relation-signature-q": KG_SOURCE, "kg-neighbour-classes-q": KG_SOURCE}
