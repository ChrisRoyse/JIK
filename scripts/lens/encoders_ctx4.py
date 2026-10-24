"""Cross-source perspectives (issue #12; operator doctrine 2026-10-16 "five lenses means five senses"; the paper's A2 anchor is
the text-to-structure join and its A4 anchor is cross-source agreement). Each lens here senses the same referent through
a different source than the atom's own: `wiki-article-body` reads the body of the entity's Wikipedia article beyond its
lead (the Wikidata5m description is the lead, so the body is what the description does not contain), joined by alias to
title; `wiki-mentions` reads windows of text from *other* Wikipedia articles that name the entity (what the rest of the
corpus says about it); `scripture-other-editions` reads the same verse key in other editions (other languages and
centuries) than the row's. Joins use the entity alias table (wikidata5m_entity) and the alignment keys; nothing reads a
label. Contexts are cached under <lens_root>/ctx/ as in encoders_ctx.py; earlier encoder modules are byte-identical. Cost:
one streamed pass over the Wikipedia shards for the title join (process pool), one ripgrep pass per shard for mentions
with a per-alias cap, one streamed fetch per shard touched, the alignment index in memory for editions."""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot  # noqa: E402
from encoders_ctx import KG_SOURCE, _cached, _cal_rows_of  # noqa: E402
from encoders_ctx2 import TRIGRAM_DIMS, WORD_DIMS, SCRIPTURE_VERSION, _absent, _is_verse, _scripture_rows, _sparse_rows, _spaths, _trigram_hist, _word_hist  # noqa: E402
from lenscommon import Paths, fetch_records, iter_canon_docs, population_shards, read_jsonl  # noqa: E402

ENCODERS_CTX4_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
WIKI_SOURCE = "wikipedia-en-20231101"
BODY_SKIP, BODY_CHARS = 200, 8000  # the body window starts this far past the lead (the abstract) and reads this many characters
MENTION_DOCS, MENTION_WINDOW, MENTIONS_PER_DOC = 40, 220, 3
MIN_ALIAS_CHARS = 6
OTHER_EDITIONS = 6
_WORDISH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,'\-()&]+$")


def _a2_entities(paths: Paths) -> list[str]:
    return sorted({r["cluster"] for r in _cal_rows_of(paths, KG_SOURCE) if str(r.get("family", "")).startswith("a2-")})


def _aliases(paths: Paths, ents: list[str]) -> dict[str, list[str]]:
    """Entity -> aliases as the alias table lists them (first alias is the label); only aliases a title join can use."""
    want = set(ents)
    out: dict[str, list[str]] = {}
    for rec in iter_canon_docs(paths.canon(KG_SOURCE), "wikidata5m_entity"):
        e = rec["f"].get("id")
        if e in want:
            parts = rec["text"].split("\t")[1:]
            out[e] = [a for a in parts if len(a) >= MIN_ALIAS_CHARS and _WORDISH.match(a)]
    return out


def _wiki_shards(paths: Paths) -> list[str]:
    return sorted(s["shard"] for s in population_shards(paths, paths.version) if s["source"] == WIKI_SOURCE)


# ------------------------------------------------------------------ article body by alias -> title
def _title_scan(job: dict) -> dict:
    want = set(job["titles"])
    found = {}
    for rec in iter_canon_docs(Path(job["canon_dir"]), job["shard"]):
        t = rec["f"].get("title")
        if t in want and t not in found:
            found[t] = rec["row"]
    return {"shard": job["shard"], "found": found}


def _article_ctx(paths: Paths) -> dict:
    ents = _a2_entities(paths)

    def build() -> dict:
        aliases = _aliases(paths, ents)
        title_to_ents: dict[str, list[str]] = collections.defaultdict(list)
        for e, al in aliases.items():
            for a in al:
                title_to_ents[a].append(e)
        shards = _wiki_shards(paths)
        jobs = [{"shard": sh, "canon_dir": str(paths.canon(WIKI_SOURCE)), "titles": sorted(title_to_ents)} for sh in shards]
        with ProcessPoolExecutor(max_workers=min(28, os.cpu_count() or 1)) as ex:
            scans = list(ex.map(_title_scan, jobs))
        loc: dict[str, tuple[str, int]] = {}
        for sc in scans:
            for t, row in sc["found"].items():
                loc.setdefault(t, (sc["shard"], row))
        chosen: dict[str, tuple[str, int, str]] = {}
        for e, al in aliases.items():
            for a in al:  # the first alias that is a title and names only this entity
                if a in loc and len(title_to_ents[a]) == 1:
                    chosen[e] = (*loc[a], a)
                    break
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for sh, row, _ in chosen.values():
            by_shard[sh].add(row)
        texts: dict[tuple[str, int], str] = {}
        for sh, rows in sorted(by_shard.items()):
            for row, rec in fetch_records(paths.canon(WIKI_SOURCE), sh, rows).items():
                texts[(sh, row)] = rec["text"]
        bodies = {}
        for e, (sh, row, title) in chosen.items():
            t = texts.get((sh, row), "")
            lead = t.find("\n\n")  # the lead ends at the first paragraph break; the Wikidata5m description is drawn from the lead
            start = (lead if lead > 0 else 0) + BODY_SKIP
            bodies[e] = {"title": title, "shard": sh, "row": row, "body": t[start:start + BODY_CHARS]}
        return {"source": WIKI_SOURCE, "join": "first alias that is a title naming exactly one calibration entity", "entities_joined": len(chosen), "entities": len(ents), "bodies": bodies}

    return _cached(paths, "wikiart", ["entities", *ents, "body", str(BODY_SKIP), str(BODY_CHARS)], build)


def _article_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == KG_SOURCE and str(r.get("family", "")).startswith("a2-") for r in rows):
        return _absent(rows, WORD_DIMS + 1, "cosine", "no A2 question in this family", sparse=True)
    bodies = _article_ctx(paths)["bodies"]
    counts = []
    for r in rows:
        b = bodies.get(r["cluster"]) if r["source"] == KG_SOURCE else None
        if not b:
            counts.append({})
            continue
        tf = _word_hist(b["body"], WORD_DIMS)
        tf[WORD_DIMS] = 1.0 if b["body"].strip() else -1.0
        counts.append(tf)
    m, empty = _sparse_rows(counts, WORD_DIMS + 1)
    return Slot(m, empty, {"dims": WORD_DIMS, "features": "signed hashed word unigrams and bigrams of the article body beyond the lead, L2-normalised; last dim +1 body present, -1 article is lead only", "join": "alias -> title"}, "cosine", sparse=True)


# ------------------------------------------------------------------ mentions by other articles (ripgrep over the shards)
def _mention_scan(job: dict) -> dict:
    """Per shard: for each alias, the first MENTION_DOCS line numbers (rows) that contain it, via ripgrep fixed strings."""
    rg = shutil.which("rg") or "/usr/bin/rg"
    pat = Path(job["pattern_file"])
    zst = Path(job["canon_dir"]) / f"{job['shard']}.jsonl.zst"
    cap: dict[str, list[int]] = collections.defaultdict(list)
    p1 = subprocess.Popen(["zstdcat", str(zst)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen([rg, "-F", "-n", "-o", "-f", str(pat)], stdin=p1.stdout, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    p1.stdout.close()
    for line in p2.stdout:
        ln, _, alias = line.rstrip("\n").partition(":")
        if not alias:
            continue
        hits = cap[alias]
        if len(hits) < MENTION_DOCS and (not hits or hits[-1] != int(ln) - 1):
            hits.append(int(ln) - 1)
    p2.wait()
    p1.wait()
    return {"shard": job["shard"], "hits": dict(cap)}


def _mention_ctx(paths: Paths) -> dict:
    ents = _a2_entities(paths)

    def build() -> dict:
        aliases = _aliases(paths, ents)
        own = {e: b["title"] for e, b in _article_ctx(paths)["bodies"].items()}
        ownloc = {(b["shard"], b["row"]) for b in _article_ctx(paths)["bodies"].values()}
        alias_to_ents: dict[str, list[str]] = collections.defaultdict(list)
        for e, al in aliases.items():
            for a in al[:3]:
                alias_to_ents[a].append(e)
        patterns = sorted(a for a, es in alias_to_ents.items() if len(es) == 1)
        pf = paths.out / "lenses" / paths.version / "ctx" / "wikimen-patterns.txt"
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("\n".join(patterns) + "\n", encoding="utf-8")
        shards = _wiki_shards(paths)
        jobs = [{"shard": sh, "canon_dir": str(paths.canon(WIKI_SOURCE)), "pattern_file": str(pf)} for sh in shards]
        with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as ex:
            scans = list(ex.map(_mention_scan, jobs))
        per_ent_rows: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
        for sc in scans:
            for alias, rows in sc["hits"].items():
                e = alias_to_ents[alias][0]
                for row in rows:
                    if (sc["shard"], row) not in ownloc and len(per_ent_rows[e]) < MENTION_DOCS:
                        per_ent_rows[e].append((sc["shard"], row, alias))
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for lst in per_ent_rows.values():
            for sh, row, _ in lst:
                by_shard[sh].add(row)
        texts: dict[tuple[str, int], str] = {}
        for sh, rows in sorted(by_shard.items()):
            for row, rec in fetch_records(paths.canon(WIKI_SOURCE), sh, rows).items():
                texts[(sh, row)] = rec["text"]
        windows: dict[str, list[str]] = {}
        for e, lst in per_ent_rows.items():
            w = []
            for sh, row, alias in lst:
                t = texts.get((sh, row), "")
                i, k = t.find(alias), 0
                while i >= 0 and k < MENTIONS_PER_DOC:
                    w.append(t[max(0, i - MENTION_WINDOW): i + len(alias) + MENTION_WINDOW])
                    i, k = t.find(alias, i + len(alias)), k + 1
            windows[e] = w
        return {"source": WIKI_SOURCE, "patterns": len(patterns), "docs_cap": MENTION_DOCS, "window_chars": MENTION_WINDOW, "entities_with_mentions": sum(1 for w in windows.values() if w),
                "excluded": "the entity's own article; aliases naming more than one calibration entity", "windows": windows}

    return _cached(paths, "wikimen", ["entities", *ents, "cap", str(MENTION_DOCS), str(MENTION_WINDOW), str(MENTIONS_PER_DOC)], build)


def _mention_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(r["source"] == KG_SOURCE and str(r.get("family", "")).startswith("a2-") for r in rows):
        return _absent(rows, WORD_DIMS + 1, "cosine", "no A2 question in this family", sparse=True)
    windows = _mention_ctx(paths)["windows"]
    counts = []
    for r in rows:
        if r["source"] != KG_SOURCE or not str(r.get("family", "")).startswith("a2-"):
            counts.append({})
            continue
        w = windows.get(r["cluster"]) or []  # an entity no other article names is a reading (-1), not an absence
        tf = _word_hist(" ".join(w), WORD_DIMS)
        tf[WORD_DIMS] = 1.0 if w else -1.0
        counts.append(tf)
    m, empty = _sparse_rows(counts, WORD_DIMS + 1)
    return Slot(m, empty, {"dims": WORD_DIMS, "features": "signed hashed word unigrams and bigrams of mention windows in other articles, L2-normalised; last dim +1 mentions found, -1 none", "cap": [MENTION_DOCS, MENTIONS_PER_DOC]}, "cosine", sparse=True)


# ------------------------------------------------------------------ the same verse in other editions
def _editions_ctx(paths: Paths) -> dict:
    want = sorted({(r["source"], r["shard"], int(r["cluster"][4:])) for r in _scripture_rows(paths) if _is_verse(r)})

    def build() -> dict:
        spaths = _spaths(paths)
        align = spaths.out / "atoms" / SCRIPTURE_VERSION / "align"
        vi = np.load(align / "verse_index.npz")
        shards = population_shards(spaths, SCRIPTURE_VERSION)
        shard_idx = {(s["source"], s["shard"]): i for i, s in enumerate(shards)}
        loc: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        keys_wanted = {k for _, _, k in want}
        for k, s, r in zip(vi["key_id"].tolist(), vi["shard"].tolist(), vi["row"].tolist()):
            if k in keys_wanted:
                loc[k].append((s, r))
        need: dict[int, set[int]] = collections.defaultdict(set)
        plan = {}
        for src, sh, kid in want:
            si = shard_idx[(src, sh)]
            others = sorted((s, r) for s, r in loc[kid] if s != si)[:OTHER_EDITIONS]
            plan[f"{src}:{sh}:{kid}"] = others
            for s, r in others:
                need[s].add(r)
        texts: dict[tuple[int, int], str] = {}
        for s, rows in sorted(need.items()):
            src, shn = shards[s]["source"], shards[s]["shard"]
            for r, rec in fetch_records(spaths.canon(src), shn, rows).items():
                texts[(s, r)] = rec["text"]
        entries = {k: {"editions": [shards[s]["shard"] for s, _ in v], "texts": [texts[(s, r)] for s, r in v if (s, r) in texts]} for k, v in plan.items()}
        return {"version": SCRIPTURE_VERSION, "other_editions": OTHER_EDITIONS, "read": "the same alignment key in other edition shards, in shard order; the row's own edition excluded", "entries": entries}

    return _cached(paths, "editions", ["other", str(OTHER_EDITIONS), *[f"{s}:{sh}:{k}" for s, sh, k in want]], build)


def _editions_lens(rows: list[dict], paths: Paths) -> Slot:
    if not any(_is_verse(r) for r in rows):
        return _absent(rows, TRIGRAM_DIMS, "cosine", "no scripture atom in this family", sparse=True)
    entries = _editions_ctx(paths)["entries"]
    counts = []
    for r in rows:
        e = entries.get(f"{r['source']}:{r['shard']}:{r['cluster'][4:]}") if _is_verse(r) else None
        counts.append(_trigram_hist(" ".join(e["texts"]), TRIGRAM_DIMS) if e and e["texts"] else {})
    m, empty = _sparse_rows(counts, TRIGRAM_DIMS)
    return Slot(m, empty, {"dims": TRIGRAM_DIMS, "features": "signed hashed character trigrams of the verse in other editions, L2-normalised", "other_editions": OTHER_EDITIONS}, "cosine", sparse=True)


LENSES = {"wiki-article-body": lambda _n, rows, paths: _article_lens(rows, paths), "wiki-mentions": lambda _n, rows, paths: _mention_lens(rows, paths),
          "scripture-other-editions": lambda _n, rows, paths: _editions_lens(rows, paths)}
SOURCE_OF = {"wiki-article-body": KG_SOURCE, "wiki-mentions": KG_SOURCE, "scripture-other-editions": "crosswire-sword-pd"}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
