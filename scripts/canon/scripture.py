"""Contract `scripture-v1` (issue #9): readers for the scripture partition. One canonical record per verse (or per
Sefaria segment / structured row) with the coordinate explicit in `f` and the text as the latent field; `canon-v1`
text normalisation applies to the text. Every edition is gated by the licence read from the distributed bytes
(eBible copr.htm, Sefaria per-version `license`, SWORD .conf DistributionLicense, a repository LICENSE blob); an
edition whose licence is not in LICENCE_OPEN is refused with MNLM_SCRIPTURE_LICENCE_REFUSED (CC-BY-NC texts only through
the `sefaria-export-nc` alias). Unknown USFM markers refuse the shard (fail closed). Cost: O(bytes) per shard, one pass.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

CONTRACT_ID = "scripture-v1"
PROSE, ROW = "prose", "row"
LICENCE_OPEN = {"PD", "CC0", "CC-BY", "CC-BY-SA", "MIT", "Unlicense"}
LICENCE_NC = {"CC-BY-NC", "CC-BY-NC-SA"}
RAW_DIR_ALIAS = {"sefaria-export-nc": "sefaria-export"}
SOURCES = {"ebible-english", "berean-bsb", "open-english-bible", "unfoldingword-uhb", "unfoldingword-ugnt", "unfoldingword-ult",
           "unfoldingword-ust", "tanach-us-uxlc", "sblgnt", "byztxt-rp2018", "nestle1904", "ubsicap-versification", "crosswire-sword-pd",
           "openbible-crossrefs", "sefaria-export", "sefaria-export-nc", "stepbible-data"}
USFM_GIT = {"unfoldingword-uhb": "hbo", "unfoldingword-ugnt": "grc", "unfoldingword-ult": "eng", "unfoldingword-ust": "eng"}
LICENCE_BLOB = {  # git sources: the licence file in the tree must carry the phrase; the whole source refuses otherwise
    "unfoldingword-uhb": ("LICENSE.md", "Attribution-ShareAlike 4.0", "CC-BY-SA"), "unfoldingword-ugnt": ("LICENSE.md", "Attribution-ShareAlike 4.0", "CC-BY-SA"),
    "unfoldingword-ult": ("LICENSE.md", "Attribution-ShareAlike 4.0", "CC-BY-SA"), "unfoldingword-ust": ("LICENSE.md", "Attribution-ShareAlike 4.0", "CC-BY-SA"),
    "sblgnt": ("LICENSE", "Attribution 4.0 International", "CC-BY"), "byztxt-rp2018": ("LICENSE.txt", "released into the public domain", "Unlicense"),
    "nestle1904": ("morph/README.md", "CC0", "CC0"), "ubsicap-versification": ("LICENSE", "MIT License", "MIT"),
    "stepbible-data": ("README.md", "CC BY 4.0", "CC-BY"),
}


class ScriptureError(Exception):
    def __init__(self, code: str, message: str, **detail):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.detail = code, message, detail


# ------------------------------------------------------------------------------------------------------------------ licences
def licence_token(text: str) -> str | None:
    """Normalise a licence statement (any of the providers' forms) to one token, or None when it is not recognised."""
    t = " ".join((text or "").split()).lower()
    if not t:
        return None
    nd = bool(re.search(r"\bnd\b|no ?-?derivative", t))  # NoDerivatives: admitted by no gate, so the token keeps it
    if "noncommercial" in t or "non-commercial" in t or "by-nc" in t or "nc-sa" in t:
        return "CC-BY-NC-ND" if nd else ("CC-BY-NC-SA" if ("sa" in t.split("nc")[-1][:6]) else "CC-BY-NC")
    if nd:
        return "CC-BY-ND"
    if t in ("public domain", "pd") or "public domain" in t:
        return "PD"
    if t == "cc0" or "cc0" in t or "creative commons: cc0" in t:
        return "CC0"
    if t in ("cc-by-sa", "cc by-sa") or "share-alike" in t or "sharealike" in t or "share alike" in t or "by-sa" in t:
        return "CC-BY-SA"
    if t in ("cc-by", "cc by") or "attribution license" in t or "attribution 4.0" in t or "creative commons: by" in t or "attribution licence" in t:
        return "CC-BY"
    if "mit license" in t:
        return "MIT"
    if "unlicense" in t:
        return "Unlicense"
    return None


def gate(token: str | None, who: str, allow_nc: bool = False) -> str:
    if token in LICENCE_OPEN or (allow_nc and token in LICENCE_NC):
        return token
    raise ScriptureError("MNLM_SCRIPTURE_LICENCE_REFUSED", f"{who}: licence {token!r} is not admitted")


def _accepted(receipts: dict) -> dict:
    return {k: v for k, v in receipts.items() if v.get("verified") is True
            or (v.get("verified") is None and v.get("attestation") == "content-length" and v.get("size_match") is True)}


# ---------------------------------------------------------------------------------------------------------------------- USFM
USFM_SKIP = {"id", "ide", "usfm", "h", "toc1", "toc2", "toc3", "toca1", "toca2", "toca3", "mt", "mt1", "mt2", "mt3", "mt4", "imt", "imt1", "imt2", "imt3", "imte",
             "ip", "is", "is1", "is2", "io1", "io2", "iot", "ipi", "im", "ie", "rem", "sts", "cl", "cp", "ca", "ca*", "va", "va*", "vp", "vp*",
             "periph", "restore", "mte", "mte1", "imi", "ili", "ili1", "ili2", "iq", "iq1", "iq2", "ib", "ipq", "ipr", "iex", "imq", "ior", "ior*"}
USFM_HEADING = {"s", "s1", "s2", "s3", "s4", "ms", "ms1", "ms2", "mr", "r", "sp", "qa", "sr", "d", "sd", "sd1", "sd2"}
USFM_PARA = {"p", "m", "pi", "pi1", "pi2", "pi3", "pc", "pr", "pmo", "pm", "pmc", "pmr", "mi", "nb", "cls", "li", "li1", "li2", "li3", "li4", "lh", "lf",
             "lim", "lim1", "lim2", "q", "q1", "q2", "q3", "q4", "qr", "qc", "qm", "qm1", "qm2", "qm3", "qd", "b", "po", "ph", "ph1", "ph2", "tr", "th1",
             "th2", "th3", "th4", "thr1", "thr2", "tc1", "tc2", "tc3", "tc4", "tcr1", "tcr2", "tcr3", "tcr4", "ts", "pb", "lit"}
USFM_CHAR = {"add", "nd", "wj", "bd", "it", "bdit", "tl", "sc", "qs", "bk", "k", "pn", "em", "sup", "no", "sls", "ord", "png", "addpn", "dc", "ndx", "rq",
             "wh", "wg", "wa", "qt", "sig", "lik", "liv", "liv1", "liv2", "litl", "rb", "va", "vp", "jmp", "fv", "xt", "xo", "xq", "xk", "xta", "w", "ft", "qac", "ref",
             "fr", "fq", "fqa", "fk", "fl", "fw", "fp", "fdc", "fm", "ca", "cat"}
_NOTE = re.compile(r"\\(f|fe|x|fig|ef|ex)\s.*?\\\1\*", re.S)
_MILESTONE = re.compile(r"\\[a-z0-9]+-[se](\s[^\\]*?)?\\\*")
_WORD = re.compile(r"\\\+?(w|ref)\s(.*?)\\\+?\1\*", re.S)
_CHAR_OPEN = re.compile(r"\\\+?([a-z0-9]+)\s")
_CHAR_CLOSE = re.compile(r"\\\+?([a-z0-9]+)\*")
_WS = re.compile(r"[ \t]+")
_NL_WS = re.compile(r" *\n *")  # a marker-backed newline carries no spaces (milestone removal left `"\n "`: readback finding, issue #9)
_INLINE_VC = re.compile(r"(?=\\(?:[vc]|d|s[1-4]?|r|q[1-4]?|p|m|b)\s)")  # markers that may follow another one mid-line


def usfm_clean(s: str, where: str) -> str:
    """Verse-level character content: notes dropped, milestones dropped, \\w word|attrs\\w* -> word, known character styles
    unwrapped; an unknown marker refuses."""
    s = _NOTE.sub("", s)  # a note or a milestone carries no text and no space: "earth\zaln-e\*." is "earth." (the sealed uW
    s = _MILESTONE.sub("", s)  # English carried a space before 99% of its punctuation when these were replaced by " ")
    s = _WORD.sub(lambda m: m.group(2).split("|", 1)[0], s)
    for m in _CHAR_CLOSE.finditer(s):
        if m.group(1) not in USFM_CHAR:
            raise ScriptureError("MNLM_SCRIPTURE_USFM_MARKER", f"{where}: unknown character marker \\{m.group(1)}*")
    s = _CHAR_CLOSE.sub("", s)
    for m in _CHAR_OPEN.finditer(s):
        if m.group(1) not in USFM_CHAR:
            raise ScriptureError("MNLM_SCRIPTURE_USFM_MARKER", f"{where}: unknown character marker \\{m.group(1)}")
    s = _CHAR_OPEN.sub("", s)
    if "\\" in s:
        raise ScriptureError("MNLM_SCRIPTURE_USFM_MARKER", f"{where}: unconsumed backslash in {s[:80]!r}")
    return _NL_WS.sub("\n", _WS.sub(" ", s)).strip()


def usfm_docs(text: str, edition: str, lang: str, where: str, start_row: int = 0) -> Iterator[tuple[int, str | None, dict, str]]:
    """One record per verse (role verse) and per heading (role heading/superscription), in file order. A record's text
    runs from its marker line to the next marker line, so notes that span lines are closed before cleaning."""
    lines = text.lstrip("\ufeff").split("\n")
    if not lines or not lines[0].startswith("\\id "):
        raise ScriptureError("MNLM_SCRIPTURE_USFM_NOID", f"{where}: first line is not \\id")
    book = lines[0].split()[1]
    row, chapter, verse, buf = start_row, 0, None, []
    cur: dict | None = None  # the open record: {"role", "marker", "verse"}
    base = {"edition": edition, "lang": lang, "book": book}

    def flush():
        nonlocal row, buf, cur
        if cur is not None:
            t = usfm_clean(" ".join(buf), f"{where} {book} {chapter}:{cur['verse']} \\{cur['marker']}")
            if t:
                ref = f"{book}.{chapter}.{cur['verse']}" if cur["role"] == "verse" else f"{book}.{chapter}.{cur['marker']}@{cur['verse']}"
                f = {**base, "chapter": chapter, "verse": cur["verse"], "role": cur["role"], "ref": ref}
                if cur["role"] != "verse":
                    f["marker"] = cur["marker"]
                yield row, t, f, PROSE
                row += 1
            elif cur["role"] == "verse":  # a verse with no text (a bare marker or a footnote only) is counted as empty, never dropped silently
                yield row, None, {**base, "chapter": chapter, "verse": cur["verse"], "role": "verse", "ref": f"{book}.{chapter}.{cur['verse']}"}, PROSE
        cur, buf = None, []

    for raw_ln in lines[1:]:
        for ln in _INLINE_VC.split(raw_ln.rstrip("\r")):  # `\d ... \v 1 ...` and `\q1 \v 2 ...` carry a verse mid-line
            if not ln:
                continue
            if not ln.startswith("\\"):
                if cur is not None:
                    buf.append(ln)
                continue
            mk, _, rest = ln[1:].partition(" ")
            if mk.endswith("\\*"):  # a standalone milestone such as \ts\* (uW translation section)
                continue
            if mk in ("f", "fe", "x", "fig"):  # a note opened at line start belongs to the open record, or to nothing
                if cur is not None:
                    buf.append(ln)
                continue
            if mk.startswith("+"):  # nested character marker (\+w) at line start
                mk = mk[1:]
            if mk == "c":
                yield from flush()
                verse, chapter = None, int(rest.split()[0])
            elif mk == "v":
                yield from flush()
                vtok, _, vtext = rest.partition(" ")
                verse, cur, buf = vtok, {"role": "verse", "marker": "v", "verse": vtok}, [vtext]
            elif mk in USFM_HEADING:
                yield from flush()
                cur, buf = {"role": "superscription" if mk == "d" else "heading", "marker": mk, "verse": verse}, [rest]
            elif mk in USFM_PARA:
                if cur is not None and cur["role"] == "verse":
                    buf.append("\n" + rest if rest else "\n")
                elif cur is not None:
                    yield from flush()
            elif mk in USFM_SKIP:
                if cur is not None and cur["role"] != "verse":
                    yield from flush()
            elif mk.rstrip("*") in USFM_CHAR or mk.endswith("-s") or mk.endswith("-e"):
                if cur is not None:
                    buf.append(ln)
            else:
                raise ScriptureError("MNLM_SCRIPTURE_USFM_MARKER", f"{where} {book}: unknown marker \\{mk}")
    yield from flush()


def _usfm_zip_docs(path: Path, edition: str, lang: str) -> Iterator[tuple[int, str | None, dict, str]]:
    row = 0
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".usfm") and "__MACOSX/" not in n and not Path(n).name.startswith("._")):
            text = z.read(name).decode("utf-8")
            if text.lstrip("\ufeff").startswith("\\id FRT") or text.lstrip("\ufeff").startswith("\\id BAK") or text.lstrip("\ufeff").startswith("\\id GLO"):
                continue
            for r in usfm_docs(text, edition, lang, f"{path.name}/{name}", row):
                yield r
                if r[1] is not None:  # rows number written records only; an empty verse is counted, not numbered
                    row = r[0] + 1


def ebible_licence(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as z:
        html = z.read("copr.htm").decode("utf-8", "replace")
    plain = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    m = re.search(r"Language:\s*([A-Za-z ]+?)\s+(Dialect|Translation|Letters|This|$)", plain)
    lang = (m.group(1).strip() if m else "unknown")
    return gate(licence_token(plain), f"{path.name} copr.htm"), lang


# ---------------------------------------------------------------------------------------------------------------- plan + open
def _shard_name(rel: str) -> str:
    """Sanitised, bounded and unique: five Sefaria paths collide after sanitisation and one is 278 chars long."""
    import hashlib

    return re.sub(r"[^A-Za-z0-9._-]+", "_", rel)[:120] + "~" + hashlib.blake2b(rel.encode("utf-8"), digest_size=6).hexdigest()


def _check_licence_blob(source_id: str, raw_dir: Path) -> str:
    rel, phrase, token = LICENCE_BLOB[source_id]
    p = raw_dir / rel
    if not p.is_file() or phrase not in p.read_text(encoding="utf-8", errors="replace"):
        raise ScriptureError("MNLM_SCRIPTURE_LICENCE_REFUSED", f"{source_id}: {rel} does not carry {phrase!r}")
    return token


def list_shards(source_id: str, raw_dir: Path, receipts: dict) -> list[dict]:
    raw_dir = raw_dir.parent / RAW_DIR_ALIAS.get(source_id, source_id)
    acc = _accepted(receipts)
    files = {Path(k).name if k.startswith("http") else k: v for k, v in acc.items()}
    plan: list[dict] = []

    def add(shard: str, rel: str, kind: str = PROSE, **extra):
        r = files.get(rel) or files.get(rel.split("?")[0])
        if r is None:
            raise ScriptureError("MNLM_SCRIPTURE_RECEIPT_MISSING", f"{source_id}: no accepted receipt for {rel}")
        stored = r.get("stored")  # acquire may store an over-long basename under a digest-suffixed name
        plan.append({"shard": shard, "files": [Path(stored) if stored else raw_dir / rel], "sha256": [r["sha256"]], "kind": kind, **extra})

    if source_id == "ebible-english":
        for rel in sorted(k for k in files if k.endswith("_usfm.zip")):
            add(rel[: -len("_usfm.zip")], rel)
    elif source_id == "berean-bsb":
        add("bsb", "bsb_usfm.zip")
    elif source_id == "open-english-bible":
        for rel in sorted(k for k in files if k.endswith(".usfm.zip")):
            add(rel.replace(".usfm.zip", ""), rel)
    elif source_id in USFM_GIT:
        for rel in sorted(k for k in files if k.endswith(".usfm")):
            r = files[rel]
            with open(Path(r["stored"]) if r.get("stored") else raw_dir / rel, "r", encoding="utf-8") as fh:
                head = fh.readline().lstrip("﻿")
            if head.split()[:2] in (["\\id", "FRT"], ["\\id", "BAK"], ["\\id", "GLO"]):
                continue  # front/back matter and glossaries are skipped by content, as in the zip reader
            add(Path(rel).stem, rel, lang=USFM_GIT[source_id])
    elif source_id == "tanach-us-uxlc":
        add("uxlc", "Tanach.xml.zip")
    elif source_id == "sblgnt":
        for rel in sorted(k for k in files if k.startswith("data/sblgnt/text/") and k.endswith(".txt")):
            add(Path(rel).stem, rel)
    elif source_id == "byztxt-rp2018":
        for rel in sorted(k for k in files if k.startswith("csv-unicode/ccat/no-variants/") and k.endswith(".csv")):
            add(Path(rel).stem, rel)
    elif source_id == "nestle1904":
        add("nestle1904", "morph/Nestle1904.csv")
    elif source_id == "ubsicap-versification":
        for rel in sorted(k for k in files if k.endswith(".vrs")):
            add(Path(rel).stem, rel, ROW)
    elif source_id == "crosswire-sword-pd":
        for rel in sorted(k for k in files if k.endswith(".zip")):
            add(Path(rel).stem, rel)
    elif source_id == "openbible-crossrefs":
        add("cross-references", "cross-references.zip", ROW)
    elif source_id == "stepbible-data":
        from scripture_step import DATASETS, dataset_of, shard_name

        for rel in sorted(k for k in files if dataset_of(k)):
            add(shard_name(rel), rel, dataset_of(rel)[1], step_rel=rel)
        if not plan:
            raise ScriptureError("MNLM_SCRIPTURE_RECEIPT_MISSING", f"{source_id}: none of {len(DATASETS)} datasets has an accepted receipt")
    elif source_id in ("sefaria-export", "sefaria-export-nc"):
        for rel in sorted(k for k in files if k.startswith("json/") and k.endswith(".json")):
            if source_id.endswith("-nc"):  # the NC alias plans only NC-licensed versions; the open partition records every refusal
                try:
                    lic = licence_token(json.loads(Path(files[rel].get("stored") or raw_dir / rel).read_text(encoding="utf-8")).get("license", ""))
                except (OSError, ValueError):
                    lic = None
                if lic not in LICENCE_NC:
                    continue
            add(_shard_name(rel[5:-5]), rel)
    else:
        raise ScriptureError("MNLM_SCRIPTURE_UNKNOWN_SOURCE", source_id)
    if source_id in LICENCE_BLOB:
        tok = _check_licence_blob(source_id, raw_dir)
        for p in plan:
            p["licence"] = tok
    return plan


def open_shard(source_id: str, shard: dict) -> Iterator[tuple[str, str, Iterator]]:
    from scripture_formats import byz_docs, crossrefs_docs, nestle_docs, sblgnt_docs, sefaria_docs, uxlc_docs, vrs_docs
    from scripture_step import step_docs
    from scripture_sword import sword_docs

    path: Path = shard["files"][0]
    name = shard["shard"]
    if source_id == "ebible-english":
        tok, lang = ebible_licence(path)
        yield name, PROSE, _tag(_usfm_zip_docs(path, name, lang), licence=tok)
    elif source_id == "berean-bsb":
        with zipfile.ZipFile(path) as z:
            has_copr = "copr.htm" in z.namelist()
        tok, lang = ebible_licence(path) if has_copr else ("PD", "English")
        yield name, PROSE, _tag(_usfm_zip_docs(path, "BSB", lang), licence=tok)
    elif source_id == "open-english-bible":
        yield name, PROSE, _tag(_usfm_zip_docs(path, name, "English"), licence="CC0")
    elif source_id in USFM_GIT:
        yield name, PROSE, _tag(usfm_docs(path.read_text(encoding="utf-8"), source_id.split("-")[1].upper(), shard["lang"], path.name), licence=shard["licence"])
    elif source_id == "tanach-us-uxlc":
        yield name, PROSE, _tag(uxlc_docs(path), licence="PD")
    elif source_id == "sblgnt":
        yield name, PROSE, _tag(sblgnt_docs(path), licence=shard["licence"])
    elif source_id == "byztxt-rp2018":
        yield name, PROSE, _tag(byz_docs(path), licence=shard["licence"])
    elif source_id == "nestle1904":
        yield name, PROSE, _tag(nestle_docs(path), licence=shard["licence"])
    elif source_id == "ubsicap-versification":
        yield name, ROW, _tag(vrs_docs(path), licence=shard["licence"])
    elif source_id == "crosswire-sword-pd":
        yield name, PROSE, sword_docs(path, path.parent)
    elif source_id == "openbible-crossrefs":
        yield name, ROW, crossrefs_docs(path)
    elif source_id == "stepbible-data":
        yield name, shard["kind"], _tag(step_docs(path, shard["step_rel"]), licence=shard["licence"])
    elif source_id in ("sefaria-export", "sefaria-export-nc"):
        yield name, PROSE, sefaria_docs(path, source_id.endswith("-nc"), _schema_dir(path))
    else:
        raise ScriptureError("MNLM_SCRIPTURE_UNKNOWN_SOURCE", source_id)


def _schema_dir(path: Path) -> Path | None:
    for parent in path.parents:
        if (parent / "schemas").is_dir() and (parent / "json").is_dir():
            return parent / "schemas"
    return None


def _tag(docs: Iterator, **extra) -> Iterator[tuple[int, str | None, dict, str]]:
    for row, text, f, kind in docs:
        yield row, text, {**f, **extra}, kind
