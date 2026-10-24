"""Contract `scripture-v1` readers for the non-USFM formats (issue #9): UXLC XML, SBLGNT text, Byzantine csv, Nestle 1904
word rows, OpenBible cross-reference rows, Paratext .vrs rows and Sefaria version JSON (SWORD modules: scripture_sword.py).
The licence gate (`gate`, `licence_token`) and the record contract live in scripture.py; every reader here yields
(row, text, fields, kind) in file order and refuses with a typed ScriptureError. Cost: O(bytes) per shard, one pass.
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

from scripture import LICENCE_NC, PROSE, ROW, ScriptureError, gate, licence_token


# ----------------------------------------------------------------------------------------------------------- other formats
def uxlc_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    """UXLC Books/*.xml: <c n><v n><w> (qere <q> read, ketiv <k> kept in fields, notes <x> dropped)."""
    import xml.etree.ElementTree as ET

    row = 0
    with zipfile.ZipFile(path) as z:
        # the five Books/*.DH.xml files mark the Documentary Hypothesis on the same text with filler and split verses:
        # not a second edition (readback finding F-4, issue #9); only the plain books are read
        for name in sorted(n for n in z.namelist() if n.startswith("Books/") and n.endswith(".xml") and not n.endswith(".DH.xml")):
            root = ET.fromstring(z.read(name))
            book = root.findtext("./tanach/book/names/abbrev") or Path(name).stem
            for c in root.iter("c"):
                for v in c.findall("v"):
                    words, ketiv = [], []
                    for el in v:
                        if el.tag in ("w", "q"):
                            words.append(_wtext(el))
                        elif el.tag == "k":
                            ketiv.append(_wtext(el))
                    f = {"edition": "UXLC", "lang": "hbo", "book": book, "chapter": int(c.get("n")), "verse": v.get("n"), "role": "verse",
                         "ref": f"{book}.{c.get('n')}.{v.get('n')}", "file": name}
                    if ketiv:
                        f["ketiv"] = ketiv
                    yield row, _join_hebrew(words), f, PROSE
                    row += 1


def _wtext(el) -> str:
    """A UXLC word: its text without the <x> note letters (readback finding F-2: `<w>לְ<x>t</x>ךָ</w>` is `לְךָ`)."""
    parts = [el.text or ""]
    for ch in el:
        if ch.tag != "x":
            parts.append("".join(ch.itertext()))
        parts.append(ch.tail or "")
    return "".join(parts).strip()


def _join_hebrew(words: list[str]) -> str:
    """Words joined by a space, except after a maqaf (U+05BE), which binds the next word."""
    out = ""
    for w in words:
        if not w:
            continue
        out += w if (not out or out.endswith("־")) else " " + w
    return out


_SIGLA = re.compile(r"[\u2e00-\u2e03\u2e06-\u2e0d\u2e2e\u2e3a\u2e3b⸀⸂⸃⸄⸅⸆⸇⸈⸉⸊⸋⸌⸍]")


def sblgnt_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    row = 0
    for ln in path.read_text(encoding="utf-8-sig").split("\n"):
        ln = ln.rstrip("\r")
        if not ln.strip():
            continue
        if "\t" not in ln:
            yield row, ln.strip(), {"edition": "SBLGNT", "lang": "grc", "role": "title", "ref": f"{path.stem}.title"}, PROSE
            row += 1
            continue
        ref, t = ln.split("\t", 1)
        book, cv = ref.rsplit(" ", 1)
        ch, vs = cv.split(":")
        yield row, _SIGLA.sub("", t).strip(), {"edition": "SBLGNT", "lang": "grc", "book": book, "chapter": int(ch), "verse": vs, "role": "verse",
                                                "ref": f"{book}.{ch}.{vs}", "sigla": len(_SIGLA.findall(t))}, PROSE
        row += 1


def byz_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    book = path.stem
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row, rec in enumerate(csv.DictReader(f)):
            t = rec["text"]
            para = t.startswith("¶")
            yield row, t.lstrip("¶").strip(), {"edition": "RP2018", "lang": "grc", "book": book, "chapter": int(rec["chapter"]), "verse": rec["verse"],
                                              "role": "verse", "ref": f"{book}.{rec['chapter']}.{rec['verse']}", "para": para}, PROSE


def nestle_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    row, cur, words = 0, None, []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            bcv = rec["BCV"]
            if bcv != cur and cur is not None:
                book, cv = cur.rsplit(" ", 1)
                ch, vs = cv.split(":")
                yield row, " ".join(words), {"edition": "Nestle1904", "lang": "grc", "book": book, "chapter": int(ch), "verse": vs, "role": "verse",
                                              "ref": f"{book}.{ch}.{vs}", "words": len(words)}, PROSE
                row, words = row + 1, []
            cur = bcv
            words.append(rec["text"])
    if cur is not None:
        book, cv = cur.rsplit(" ", 1)
        ch, vs = cv.split(":")
        yield row, " ".join(words), {"edition": "Nestle1904", "lang": "grc", "book": book, "chapter": int(ch), "verse": vs, "role": "verse",
                                      "ref": f"{book}.{ch}.{vs}", "words": len(words)}, PROSE


def crossrefs_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        text = z.read(name).decode("utf-8")
    row, token = 0, None
    for ln in text.split("\n"):
        ln = ln.rstrip("\r")
        if ln.startswith("From Verse"):  # header: `From Verse\tTo Verse\tVotes\t#www.openbible.info CC-BY 2026-10-12`
            tail = ln.rsplit("\t", 1)[-1]
            token = gate(next((t for t in map(licence_token, [tail] + tail.split()) if t), None), f"{path.name} header {tail!r}")
            continue
        if not ln:
            continue
        if token is None:
            raise ScriptureError("MNLM_SCRIPTURE_LICENCE_REFUSED", f"{path.name}: no licence header before the first row")
        a, b, votes = ln.split("\t")[:3]
        yield row, ln, {"from_ref": a, "to_ref": b, "votes": int(votes), "licence": token}, ROW
        row += 1


def vrs_docs(path: Path) -> Iterator[tuple[int, str | None, dict, str]]:
    row = 0
    for ln in path.read_text(encoding="utf-8-sig").split("\n"):
        ln = ln.rstrip("\r")
        if not ln.strip() or ln.startswith("#"):
            continue
        kind = "mapping" if "=" in ln else ("excluded" if ln.startswith("-") else "maxverses")
        yield row, ln, {"scheme": path.stem, "kind": kind}, ROW
        row += 1


# ------------------------------------------------------------------------------------------------------------------- Sefaria
_HTML = re.compile(r"<[^>]+>")
_HTML_BREAK = re.compile(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6]|blockquote)\b[^>]*>", re.I)
_FOOTNOTE = re.compile(r"<sup class=\"footnote-marker\">.*?</sup>|<i class=\"footnote\">.*?</i>", re.S)


def _daf(i: int) -> str:
    return f"{i // 2 + 1}{'ab'[i % 2]}"


def sefaria_docs(path: Path, allow_nc: bool, schema_dir: Path | None) -> Iterator[tuple[int, str | None, dict, str]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    token = gate(licence_token(d.get("license", "")), f"{path.name} license={d.get('license')!r}", allow_nc=allow_nc)
    if allow_nc and token not in LICENCE_NC:
        raise ScriptureError("MNLM_SCRIPTURE_LICENCE_REFUSED", f"{path.name}: {token} belongs to the open partition, not the NC alias")
    title, vt = d.get("title"), d.get("versionTitle")
    lang = d.get("actualLanguage") or d.get("language")
    addr = None
    if schema_dir is not None:
        sp = schema_dir / (re.sub(r"[ ,]", "_", title or "").replace("__", "_") + ".json")
        if sp.is_file():
            addr = (json.loads(sp.read_text(encoding="utf-8")).get("schema") or {}).get("addressTypes")
    if addr is None and d.get("sectionNames"):
        addr = ["Talmud" if s == "Daf" else "Integer" for s in d["sectionNames"]]
    base = {"title": title, "versionTitle": vt, "lang": lang, "licence": token, "categories": d.get("categories"), "versionSource": d.get("versionSource")}
    row = 0

    def walk(node, keys: list, depth: int):
        nonlocal row
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, keys + [k], depth)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                a = addr[depth] if addr and depth < len(addr) else "Integer"
                yield from walk(v, keys + [_daf(i) if a == "Talmud" else str(i + 1)], depth + 1)
        elif isinstance(node, str):
            # an inline tag (<i>, <b>, <big>, <span>) carries no space: `<i>vav</i>,` is `vav,` (readback finding F-1);
            # a block or break tag separates words; a footnote carries no text
            t = _HTML.sub("", _HTML_BREAK.sub(" ", _FOOTNOTE.sub("", node)))
            t = " ".join(t.replace("&nbsp;", " ").split())
            if t:
                yield row, t, {**base, "ref": f"{title} {':'.join(keys)}", "path": keys, "role": "segment"}, PROSE
                row += 1

    yield from walk(d.get("text"), [], 0)
    if row == 0:  # 33 versions ship an empty text array: a typed refusal, never a 0-record seal (readback finding, issue #9)
        raise ScriptureError("MNLM_SCRIPTURE_EMPTY_VERSION", f"{path.name}: version {vt!r} of {title!r} carries no text")
