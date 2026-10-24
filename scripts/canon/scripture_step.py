"""Contract `scripture-v1` reader for STEPBible-Data (issue #9; CC BY 4.0, Tyndale House). Tab-separated UTF-8 text files:
TAHOT/TAGNT word-per-line tagged texts become one verse record per reference (original-language words joined; the
per-word glosses, Strong's numbers, morphology and edition sets are explicit fields); TBESH/TBESG/TFLSJ lexicon rows
become one entry record each (definition text latent, lemma/Strong's/gloss explicit); TVTMS, TIPNR, TEGMC and TEHMC
become row records (the line is the record, its `$` section explicit). "Older Formats/" and "Tagged-Bibles/" are not read
(superseded duplicates; the Arabic SVD XML and the CC BY-NC ESV tags are stated, not taken). Cost: O(bytes), one pass.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterator
from pathlib import Path

from scripture import PROSE, ROW, ScriptureError

DATASETS = {  # path prefix (under the repo) -> (dataset id, record kind, language)
    "Translators Amalgamated OT+NT/TAHOT ": ("TAHOT", PROSE, "hbo"), "Translators Amalgamated OT+NT/TAGNT ": ("TAGNT", PROSE, "grc"),
    "Lexicons/TBESH ": ("TBESH", PROSE, "hbo"), "Lexicons/TBESG ": ("TBESG", PROSE, "grc"), "Lexicons/TFLSJ ": ("TFLSJ", PROSE, "grc"),
    "Versification/TVTMS ": ("TVTMS", ROW, "eng"), "Proper Nouns/TIPNR ": ("TIPNR", ROW, "eng"),
    "Morphology codes/TEGMC ": ("TEGMC", ROW, "grc"), "Morphology codes/TEHMC ": ("TEHMC", ROW, "hbo"),
}
_WORDLINE = re.compile(r"^([1-3]?[A-Za-z]{2,3})\.(\d+)\.(\d+)([a-z]?)#(\d+)=(\S*)\t")
_LEXLINE = re.compile(r"^([GH]\d{4}[A-Za-z]?)\t")
_BR = re.compile(r"<br\s*/?>", re.I)
_RULE = re.compile(r"[\t '=\-‖_.,]+")
_TAG = re.compile(r"<[^>]+>")


def dataset_of(rel: str) -> tuple[str, str, str] | None:
    for prefix, ds in DATASETS.items():
        if rel.startswith(prefix):
            return ds
    return None


def shard_name(rel: str) -> str:
    ds = dataset_of(rel)
    tag = Path(rel).name.split(" - ")[0].replace(ds[0], "").strip().replace(" ", "")
    return ds[0] + (f"-{tag}" if tag else "")


def _lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for ln in f:
            yield ln.rstrip("\r\n")


def _hebrew(w: str) -> str:
    return w.replace("/", "").replace("\\", "")  # morpheme separators and marker escapes are tagging, not text


def _greek(w: str) -> tuple[str, str]:
    word, _, tr = w.partition(" (")
    return word.replace("¶", "").strip(), tr.rstrip(")").strip()


def _join(words: list[str], lang: str) -> str:
    if lang != "hbo":
        return " ".join(words)
    out = ""
    for w in words:
        out += w if (not out or out.endswith("־")) else " " + w
    return out


def tagged_docs(path: Path, ds: str, lang: str) -> Iterator[tuple[int, str | None, dict, str]]:
    row, cur, words, f = 0, None, [], {}

    def flush():
        nonlocal row, cur, words, f
        if cur is not None and words:
            yield row, _join(words, lang), {**f, "words": len(words)}, PROSE
            row += 1
        cur, words, f = None, [], {}

    for ln in _lines(path):
        m = _WORDLINE.match(ln)
        if not m:
            continue
        book, ch, vs, sub, _, typ = m.groups()
        ref = f"{book}.{ch}.{vs}{sub}"
        cols = ln.split("\t")
        if ref != cur:
            yield from flush()
            cur = ref
            f = {"edition": ds, "lang": lang, "book": book, "chapter": int(ch), "verse": vs + sub, "role": "verse", "ref": ref,
                 "types": [], "gloss": [], "strongs": [], "morph": []}
            if ds == "TAGNT":
                f["editions"] = []
        f["types"].append(typ)
        if ds == "TAHOT":
            words.append(_hebrew(cols[1]))
            f["gloss"].append(cols[3])
            f["strongs"].append(cols[4])
            f["morph"].append(cols[5])
        else:
            word, _ = _greek(cols[1])
            words.append(word)
            f["gloss"].append(cols[2])
            ds_gr = cols[3].split("=", 1)
            f["strongs"].append(ds_gr[0])
            f["morph"].append(ds_gr[1] if len(ds_gr) > 1 else "")
            f["editions"].append(cols[5])
    yield from flush()
    if row == 0:
        raise ScriptureError("MNLM_SCRIPTURE_STEP_EMPTY", f"{path.name}: no tagged word line")


def lexicon_docs(path: Path, ds: str, lang: str) -> Iterator[tuple[int, str | None, dict, str]]:
    row = 0
    for ln in _lines(path):
        if not _LEXLINE.match(ln):
            continue
        cols = ln.split("\t")
        if len(cols) < 8:
            continue
        meaning = html.unescape(_TAG.sub("", _BR.sub("\n", cols[7])))
        text = "\n".join(x for x in (" ".join(p.split()) for p in meaning.split("\n")) if x)
        if not text:
            continue
        yield row, text, {"edition": ds, "lang": lang, "role": "lexicon", "estrong": cols[0], "dstrong": cols[1].split()[0] if cols[1].split() else cols[1],
                          "relation": " ".join(cols[1].split()[1:]), "ustrong": cols[2].strip(), "lemma": cols[3], "translit": cols[4], "morph": cols[5],
                          "gloss": cols[6], "ref": f"{ds}:{cols[0]}:{row}"}, PROSE
        row += 1
    if row == 0:
        raise ScriptureError("MNLM_SCRIPTURE_STEP_EMPTY", f"{path.name}: no lexicon line")


def row_docs(path: Path, ds: str, lang: str) -> Iterator[tuple[int, str | None, dict, str]]:
    """Every non-blank line from the first `$` line on is a row; the current `$` header is its section."""
    row, section, started = 0, "", False
    for n, ln in enumerate(_lines(path)):
        if ln.startswith("$"):
            started, section = True, " ".join(ln[1:].split("\t")[0].split())
            continue
        if not started or not ln.strip() or not _RULE.sub("", ln).strip():
            continue  # blank lines and rule lines ('====, ‖====) are layout, not rows
        yield row, ln, {"edition": ds, "lang": lang, "role": "row", "section": section, "line": n, "ref": f"{ds}:{n}"}, ROW
        row += 1
    if row == 0:
        raise ScriptureError("MNLM_SCRIPTURE_STEP_EMPTY", f"{path.name}: no row after a $ section line")


def step_docs(path: Path, rel: str) -> Iterator[tuple[int, str | None, dict, str]]:
    ds = dataset_of(rel)
    if ds is None:
        raise ScriptureError("MNLM_SCRIPTURE_STEP_UNPLANNED", rel)
    name, kind, lang = ds
    if name in ("TAHOT", "TAGNT"):
        return tagged_docs(path, name, lang)
    if kind == PROSE:
        return lexicon_docs(path, name, lang)
    return row_docs(path, name, lang)
