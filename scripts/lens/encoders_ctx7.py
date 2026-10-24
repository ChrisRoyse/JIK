"""Round-4 senses (issue #12, agent decisions D-2/D-7 of 2026-10-16). `scripture-crossrefs` reads the text of the verses the
Stage 2 cross-reference edges (openbible, `edges/crossrefs.npz`) link the row's verse to, keeping only links into *other*
books: a link chosen by content, not by the label's unit (A4's genre is a function of the book, so same-book links would be
the label's siblings, decision D-2). `kg-entity-names` reads the A2 entity's own alias-table entries beyond its label (other
names, transliterations, abbreviations), with every alias of every class the question declares deleted (decision D-1).
Contexts are cached under <lens_root>/ctx/ as in encoders_ctx.py; earlier encoder modules are unchanged. Cost: one load of
the cross-reference arrays and the verse index, one fetch per edition shard touched; one streamed pass over the entity alias
table, one regex substitution per row."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot  # noqa: E402
from encoders_ctx import KG_SOURCE, _cached, _cal_rows_of  # noqa: E402
from encoders_ctx2 import SCRIPTURE_VERSION, TRIGRAM_DIMS, _absent, _is_verse, _scripture_rows, _sparse_rows, _spaths, _trigram_hist  # noqa: E402
from encoders_ctx6 import _is_a2, _masks  # noqa: E402
from lenscommon import LensError, Paths, fetch_records, iter_canon_docs, population_shards, read_jsonl  # noqa: E402

ENCODERS_CTX7_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
CROSSREFS_PER_VERSE = 10


# ------------------------------------------------------------------ A4: verses linked by cross-reference, other books only
def _crossref_ctx(paths: Paths) -> dict:
    want = sorted({(r["source"], r["shard"], int(r["cluster"][4:])) for r in _scripture_rows(paths) if _is_verse(r)})

    def build() -> dict:
        sp = _spaths(paths)
        adir = sp.out / "atoms" / SCRIPTURE_VERSION
        p = adir / "edges" / "crossrefs.npz"
        if not p.exists():
            raise LensError("MNLM_LENS_CONTEXT_MISSING", f"{p} absent: the Stage 2 cross-reference edges are required")
        e = np.load(p)
        book = {k["key_id"]: k["book"] for k in read_jsonl(adir / "align" / "verse_keys.jsonl")}
        keys = {k for _, _, k in want}
        links: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
        for a, b, v in zip(e["from_key"].tolist(), e["to_key"].tolist(), e["votes"].tolist()):
            if a < 0 or b < 0 or a == b:
                continue
            for x, y in ((a, b), (b, a)):
                if x in keys and book.get(y) != book.get(x):
                    links[x].add((-v, y))
        others = {k: [y for _, y in sorted(links[k])[:CROSSREFS_PER_VERSE]] for k in keys}
        vi = np.load(adir / "align" / "verse_index.npz")
        need = {y for ys in others.values() for y in ys}
        shards = population_shards(sp, SCRIPTURE_VERSION)
        shard_idx = {(s["source"], s["shard"]): i for i, s in enumerate(shards)}
        loc: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for k, s, r in zip(vi["key_id"].tolist(), vi["shard"].tolist(), vi["row"].tolist()):
            if k in need:
                loc[k].append((s, r))
        plan: dict[str, list[tuple[int, int]]] = {}
        fetch: dict[int, set[int]] = collections.defaultdict(set)
        for src, sh, k in want:
            own = shard_idx[(src, sh)]
            picks = []
            for y in others[k]:
                cands = sorted(loc.get(y, []))
                if cands:  # the row's own edition when it carries the linked verse, otherwise the first edition in shard order
                    s, r = next(((s, r) for s, r in cands if s == own), cands[0])
                    picks.append((s, r))
                    fetch[s].add(r)
            plan[f"{src}:{sh}:{k}"] = picks
        texts: dict[tuple[int, int], str] = {}
        for s, rows in sorted(fetch.items()):
            for r, rec in fetch_records(sp.canon(shards[s]["source"]), shards[s]["shard"], rows).items():
                texts[(s, r)] = rec["text"]
        entries = {k: [texts[sr] for sr in v if sr in texts] for k, v in plan.items()}
        return {"edges": str(p), "per_verse": CROSSREFS_PER_VERSE, "read": "verses linked by openbible cross-references into other books, highest votes first", "entries": entries,
                "verses_with_links": sum(1 for v in entries.values() if v)}

    return _cached(paths, "crossrefs", ["other-books", str(CROSSREFS_PER_VERSE), *[f"{s}:{sh}:{k}" for s, sh, k in want]], build)


def _crossref_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(_is_verse(r) for r in rows):
        return _absent(rows, TRIGRAM_DIMS + 1, "cosine", "no scripture atom in this family", sparse=True)
    entries = _crossref_ctx(paths)["entries"]
    counts = []
    for r in rows:
        e = entries.get(f"{r['source']}:{r['shard']}:{r['cluster'][4:]}") if _is_verse(r) else None
        if e is None:
            counts.append({})
            continue
        tf = _trigram_hist(" ".join(e), TRIGRAM_DIMS) if e else {}
        tf[TRIGRAM_DIMS] = 1.0 if e else -1.0  # a verse with no cross-reference into another book is a reading, not an absence
        counts.append(tf)
    m, empty = _sparse_rows(counts, TRIGRAM_DIMS + 1)
    return Slot(m, empty, {"dims": TRIGRAM_DIMS, "features": "signed hashed character trigrams of cross-referenced verses in other books, L2-normalised; last dim +1 links, -1 none",
                           "per_verse": CROSSREFS_PER_VERSE}, "cosine", sparse=True)


# ------------------------------------------------------------------ A2: the entity's own names, the answer masked
def _names_ctx(paths: Paths) -> dict:
    ents = sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE) if _is_a2(r)})

    def build() -> dict:
        want, out = set(ents), {}
        for rec in iter_canon_docs(paths.canon(KG_SOURCE), "wikidata5m_entity"):
            e = rec["f"].get("id")
            if e in want:
                out[e] = rec["text"].split("\t")[2:]  # every alias beyond the label (the first alias), in table order
        return {"source": KG_SOURCE, "read": "alias-table entries beyond the label", "names": out}

    return _cached(paths, "entnames", ["entities", *ents], build)


def _names_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(_is_a2(r) for r in rows):
        return _absent(rows, TRIGRAM_DIMS + 1, "cosine", "no A2 question in this family", sparse=True)
    names, masks = _names_ctx(paths)["names"], _masks(paths, rows)
    counts = []
    for r in rows:
        if not _is_a2(r):
            counts.append({})
            continue
        t = masks[r["family"]].sub(" ", " | ".join(names.get(r["cluster"]) or []))
        has = bool(t.strip(" |"))
        tf = _trigram_hist(t, TRIGRAM_DIMS) if has else {}
        tf[TRIGRAM_DIMS] = 1.0 if has else -1.0
        counts.append(tf)
    m, empty = _sparse_rows(counts, TRIGRAM_DIMS + 1)
    return Slot(m, empty, {"dims": TRIGRAM_DIMS, "features": "signed hashed character trigrams of the entity's aliases beyond its label with the question's class aliases deleted, L2-normalised; last dim +1 names, -1 none"},
                "cosine", sparse=True)


LENSES = {"scripture-crossrefs": lambda _n, rows, paths: _crossref_lens(rows, paths), "kg-entity-names": lambda _n, rows, paths: _names_lens(rows, paths)}
SOURCE_OF = {"scripture-crossrefs": "crosswire-sword-pd", "kg-entity-names": KG_SOURCE}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
