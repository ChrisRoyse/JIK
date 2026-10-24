"""A2 context senses with the answer masked (issue #12, readback J1 of run 10). The readback found that `wiki-article-body`
names the row's own class in 83 % of P54 rows (the atom sentence it does not read: 24 %), and a probe over all sixteen A2
questions found the same for the abstract outside the atom span and for mention windows (13-96 % per question). Wikidata's
statements are drawn from that prose, so a context that names the class re-reads the oracle's source rather than sensing
the entity. These lenses read the same contexts with every alias of every class the question declares (`top_classes` in
the question's calibration summary, fixed before any row is labelled) deleted, the same mask for every row of a question,
so what remains is what the context says besides the answer. The atom's own sentence is not masked: the A2 anchor is the
join from that sentence to the structured statement. The unmasked lenses are retired in the catalogue. Contexts reuse the
caches of encoders_ctx3/ctx4 (article bodies, mention windows); the abstract rest is rebuilt here for all A2 questions
(encoders_ctx3's cache covers the first ten). Cost: one streamed pass over the entity alias table for the class aliases,
one fetch per shard holding a calibration record, one regex substitution per row."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import Slot  # noqa: E402
from encoders_ctx import KG_SOURCE, _cached, _cal_rows_of  # noqa: E402
from encoders_ctx2 import WORD_DIMS, _absent, _sparse_rows, _word_hist  # noqa: E402
from encoders_ctx4 import _article_ctx, _mention_ctx  # noqa: E402
from lenscommon import LensError, Paths, fetch_records, iter_canon_docs, lens_root  # noqa: E402

ENCODERS_CTX6_B3 = blake3.blake3(Path(__file__).read_bytes()).hexdigest()
MIN_MASK_ALIAS = 3


def _is_a2(r: dict) -> bool:
    return r["source"] == KG_SOURCE and str(r.get("family", "")).startswith("a2-")


def _question_classes(paths: Paths, fam: str) -> list[str]:
    p = lens_root(paths) / "cal" / f"{fam}.summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    if not s.get("top_classes"):
        raise LensError("MNLM_LENS_MASK_NO_CLASSES", f"{fam}: calibration summary declares no top_classes; the mask cannot be built")
    return [c for c, _ in s["top_classes"]]


def _class_aliases(paths: Paths, fams: list[str]) -> dict:
    classes = sorted({c for f in fams for c in _question_classes(paths, f)})

    def build() -> dict:
        want, out = set(classes), {}
        for rec in iter_canon_docs(paths.canon(KG_SOURCE), "wikidata5m_entity"):
            e = rec["f"].get("id")
            if e in want:
                out[e] = sorted({a for a in rec["text"].split("\t")[1:] if len(a) >= MIN_MASK_ALIAS})
        missing = sorted(want - set(out))
        if missing:
            raise LensError("MNLM_LENS_MASK_ALIAS_MISSING", f"{len(missing)} declared classes have no alias row: {missing[:5]}")
        return {"aliases": out, "min_alias_chars": MIN_MASK_ALIAS}

    return _cached(paths, "classalias", ["classes", *classes, str(MIN_MASK_ALIAS)], build)


def _masks(paths: Paths, rows: list[dict]) -> dict[str, re.Pattern]:
    fams = sorted({r["family"] for r in rows if _is_a2(r)})
    al = _class_aliases(paths, fams)["aliases"]
    out = {}
    for f in fams:
        words = sorted({a for c in _question_classes(paths, f) for a in al[c]}, key=lambda a: (-len(a), a))
        out[f] = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.IGNORECASE)
    return out


def _abstract_all(paths: Paths) -> dict:
    rows = [r for r in _cal_rows_of(paths, KG_SOURCE) if _is_a2(r)]
    want = {(r["shard"], int(r["row"])): (int(r["span"][0]), int(r["span"][1])) for r in rows}
    keys = sorted(f"{s}:{r}:{a}:{b}" for (s, r), (a, b) in want.items())

    def build() -> dict:
        by_shard: dict[str, set[int]] = collections.defaultdict(set)
        for s, r in want:
            by_shard[s].add(r)
        per = {}
        for s, rws in sorted(by_shard.items()):
            for r, rec in fetch_records(paths.canon(KG_SOURCE), s, rws).items():
                a, b = want[(s, r)]
                per[f"{s}:{r}"] = (rec["text"][:a] + " " + rec["text"][b:]).strip()
        return {"source": KG_SOURCE, "read": "record abstract outside the row's atom span, every A2 question", "rest": per}

    return _cached(paths, "abstractall", ["rest", *keys], build)


def _masked_lens(rows: list[dict], paths: Paths, text_of, what: str) -> Slot:
    if not any(_is_a2(r) for r in rows):
        return _absent(rows, WORD_DIMS + 1, "cosine", "no A2 question in this family", sparse=True)
    masks = _masks(paths, rows)
    counts = []
    for r in rows:
        t = text_of(r) if _is_a2(r) else None
        if t is None:
            counts.append({})
            continue
        m = masks[r["family"]].sub(" ", t)
        tf = _word_hist(m, WORD_DIMS)
        tf[WORD_DIMS] = 1.0 if m.strip() else -1.0
        counts.append(tf)
    sl, empty = _sparse_rows(counts, WORD_DIMS + 1)
    return Slot(sl, empty, {"dims": WORD_DIMS, "features": f"signed hashed word unigrams and bigrams of {what} with every alias of every class the question declares deleted, L2-normalised; last dim +1 text remains, -1 none",
                            "mask": "aliases of the question's top_classes (calibration summary), case-insensitive, word-bounded"}, "cosine", sparse=True)


def _article_masked(rows, paths):
    bodies = _article_ctx(paths)["bodies"]
    return _masked_lens(rows, paths, lambda r: (bodies.get(r["cluster"]) or {}).get("body"), "the Wikipedia article body beyond the lead")


def _mentions_masked(rows, paths):
    windows = _mention_ctx(paths)["windows"]
    return _masked_lens(rows, paths, lambda r: " ".join(windows.get(r["cluster"]) or []), "mention windows in other articles")


def _abstract_masked(rows, paths):
    rest = _abstract_all(paths)["rest"]
    return _masked_lens(rows, paths, lambda r: rest.get(f"{r['shard']}:{r['row']}"), "the record abstract outside the atom span")


LENSES = {"wiki-article-body-masked": lambda _n, rows, paths: _article_masked(rows, paths), "wiki-mentions-masked": lambda _n, rows, paths: _mentions_masked(rows, paths),
          "a2-abstract-rest-masked": lambda _n, rows, paths: _abstract_masked(rows, paths)}
SOURCE_OF = {k: KG_SOURCE for k in LENSES}


def encode(name: str, rows: list[dict], paths: Paths) -> Slot:
    return LENSES[name](name, rows, paths)
