"""Stage 1 FSV B1 for the scripture partition (issue #9): re-extract one sealed record from the RAW bytes by its explicit
coordinate (edition, ref, path) through parsers written separately from scripture*.py (a token walk over USFM instead of a
line walk; direct JSON/XML/csv/zip/SWORD access), and return the candidate texts. The caller normalises each candidate
under its own canon-v1 restatement and compares bytes; a record whose format has no parser here returns "SKIP" and is
counted as such, never as agreement. Cost: O(raw shard bytes) per sampled record.
"""
from __future__ import annotations

import csv
import io
import json
import re
import struct
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

USFM_SOURCES = {"ebible-english", "berean-bsb", "open-english-bible", "unfoldingword-ult", "unfoldingword-ust", "unfoldingword-uhb", "unfoldingword-ugnt"}
HEADINGS = {"s", "s1", "s2", "s3", "s4", "ms", "ms1", "ms2", "mr", "r", "sp", "qa", "sr", "d", "sd", "sd1", "sd2"}
PARAS = {"p", "m", "pi", "pi1", "pi2", "pi3", "pc", "pr", "pmo", "pm", "pmc", "pmr", "mi", "nb", "cls", "li", "li1", "li2", "li3", "li4", "lh", "lf", "lim", "lim1",
         "lim2", "q", "q1", "q2", "q3", "q4", "qr", "qc", "qm", "qm1", "qm2", "qm3", "qd", "b", "po", "ph", "ph1", "ph2", "tr", "th1", "th2", "th3", "th4", "thr1",
         "thr2", "tc1", "tc2", "tc3", "tc4", "tcr1", "tcr2", "tcr3", "tcr4", "ts", "pb", "lit"}
BOOKLEVEL = {"id", "ide", "usfm", "h", "toc1", "toc2", "toc3", "toca1", "toca2", "toca3", "mt", "mt1", "mt2", "mt3", "mt4", "imt", "imt1", "imt2", "imt3", "imte", "ip",
             "is", "is1", "is2", "io1", "io2", "iot", "ipi", "im", "ie", "rem", "sts", "cl", "cp", "ca", "va", "vp", "periph", "restore", "mte", "mte1", "imi", "ili",
             "ili1", "ili2", "iq", "iq1", "iq2", "ib", "ipq", "ipr", "iex", "imq", "ior"}
NOTE_RE = re.compile(r"\\(f|fe|x|fig|ef|ex)\s.*?\\\1\*", re.S)
MILESTONE_RE = re.compile(r"\\[a-z0-9]+-[se](\s[^\\]*?)?\\\*|\\[a-z0-9]+\\\*")
WORD_RE = re.compile(r"\\\+?(w|ref)\s(.*?)\\\+?\1\*", re.S)
TOKEN_RE = re.compile(r"(\\\+?[a-z0-9]+\*?)")
FOOT_RE = re.compile(r"<sup class=\"footnote-marker\">.*?</sup>|<i class=\"footnote\">.*?</i>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def usfm_records(text: str) -> list[tuple]:
    """Token walk (not line walk): (role, marker, chapter, verse, text); a paragraph marker inside a verse is a newline."""
    text = WORD_RE.sub(lambda m: m.group(2).split("|", 1)[0], MILESTONE_RE.sub("", NOTE_RE.sub("", text.lstrip("\ufeff"))))
    toks = TOKEN_RE.split(text)
    out, chapter, verse, cur, buf = [], 0, None, None, []

    def flush():
        nonlocal cur, buf
        if cur is not None:
            t = re.sub(r"[ \t]+", " ", "".join(buf))
            t = "\n".join(ln.strip() for ln in t.split("\n")).strip("\n").strip()
            if t:
                out.append((cur[0], cur[1], chapter, cur[2], t))
        cur, buf = None, []

    i = 0
    while i < len(toks):
        tk = toks[i]
        if i % 2 == 0:
            if cur is not None and tk:
                buf.append(tk.replace("\r", "").replace("\n", " "))
            i += 1
            continue
        mk = tk[1:]
        nested = mk.startswith("+")
        mk = mk.lstrip("+")
        following = toks[i + 1] if i + 1 < len(toks) else ""
        if mk.endswith("*"):
            pass
        elif mk == "c":
            flush()
            m = re.match(r"\s*(\d+)", following)
            chapter, verse = (int(m.group(1)) if m else chapter), None
            toks[i + 1] = ""
        elif mk == "v":
            flush()
            m = re.match(r"\s*(\S+)\s?", following)
            verse = m.group(1) if m else "?"
            cur, toks[i + 1] = ("verse", "v", verse), (following[m.end():] if m else "")
        elif mk in HEADINGS:
            flush()
            cur, toks[i + 1] = ("superscription" if mk == "d" else "heading", mk, verse), following.lstrip(" ")
        elif mk in PARAS:
            if cur is not None and cur[0] == "verse":
                buf.append("\n")
                toks[i + 1] = following.lstrip(" ")
            else:
                flush()
                toks[i + 1] = ""
        elif mk in BOOKLEVEL and not nested:
            if cur is not None and cur[0] != "verse":
                flush()
            toks[i + 1] = ""
        i += 1
    flush()
    return out


def _usfm_text(zpath_or_file: Path, book: str) -> str | None:
    if zpath_or_file.suffix.lower() == ".zip":
        with zipfile.ZipFile(zpath_or_file) as z:
            for name in sorted(n for n in z.namelist() if n.lower().endswith(".usfm") and "__MACOSX/" not in n and not n.rsplit("/", 1)[-1].startswith("._")):
                data = z.read(name).decode("utf-8")
                if data.lstrip("\ufeff").startswith("\\id " + book):
                    return data
        return None
    return zpath_or_file.read_text(encoding="utf-8")


def _usfm(raw: Path, f: dict) -> list[str]:
    data = _usfm_text(raw, f["book"])
    if data is None:
        return []
    recs = usfm_records(data)
    if f.get("role") == "verse":
        return [r[4] for r in recs if r[0] == "verse" and r[2] == f.get("chapter") and r[3] == f.get("verse")]
    return [r[4] for r in recs if r[0] != "verse" and r[2] == f.get("chapter") and r[1] == f.get("marker") and r[3] == f.get("verse")]


def _sefaria(raw: Path, f: dict) -> list[str]:
    node = json.loads(raw.read_text(encoding="utf-8"))["text"]
    for key in f["path"]:
        if isinstance(node, dict):
            node = node[key]
        elif key.isdigit():
            node = node[int(key) - 1]
        else:
            node = node[(int(key[:-1]) - 1) * 2 + "ab".index(key[-1])]
    if not isinstance(node, str):
        return []
    t = re.sub(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6]|blockquote)\b[^>]*>", " ", FOOT_RE.sub("", node), flags=re.I)
    return [" ".join(TAG_RE.sub("", t).replace("&nbsp;", " ").split())]


def _uxlc(raw: Path, f: dict) -> list[str]:
    with zipfile.ZipFile(raw) as z:
        root = ET.fromstring(z.read(f["file"]))
    for c in root.iter("c"):
        if c.get("n") != str(f["chapter"]):
            continue
        for v in c.findall("v"):
            if v.get("n") == str(f["verse"]):
                out = ""

                def wtext(el):  # the word without its <x> note letters
                    return "".join([el.text or ""] + [("" if ch.tag == "x" else "".join(ch.itertext())) + (ch.tail or "") for ch in el]).strip()

                for w in (wtext(el) for el in v if el.tag in ("w", "q")):
                    if w:
                        out += w if (not out or out.endswith("\u05be")) else " " + w
                return [out]
    return []


def _sblgnt(raw: Path, f: dict) -> list[str]:
    lines = raw.read_text(encoding="utf-8-sig").split("\n")
    if f.get("role") == "title":
        return [ln.strip() for ln in lines if ln.strip() and "\t" not in ln][:1]
    want = f"{f['book']} {f['chapter']}:{f['verse']}"
    for ln in lines:
        if "\t" in ln and ln.split("\t", 1)[0] == want:
            t = ln.split("\t", 1)[1].rstrip("\r")
            return ["".join(ch for ch in t if not (0x2E00 <= ord(ch) <= 0x2E3B)).strip()]
    return []


def _byz(raw: Path, f: dict) -> list[str]:
    with raw.open("r", encoding="utf-8-sig", newline="") as fh:
        return [r["text"].lstrip("¶").strip() for r in csv.DictReader(fh) if r["chapter"] == str(f["chapter"]) and r["verse"] == str(f["verse"])]


def _nestle(raw: Path, f: dict) -> list[str]:
    want = f"{f['book']} {f['chapter']}:{f['verse']}"
    with raw.open("r", encoding="utf-8-sig", newline="") as fh:
        words = [r["text"] for r in csv.DictReader(fh, delimiter="\t") if r["BCV"] == want]
    return [" ".join(words)] if words else []


def _line_member(raw: Path, text: str, member: str | None = None) -> list[str]:
    """Row kinds: the record is a raw line; it agrees when that exact line exists in the raw file."""
    if raw.suffix.lower() == ".zip":
        with zipfile.ZipFile(raw) as z:
            name = member or [n for n in z.namelist() if n.endswith(".txt")][0]
            data = z.read(name).decode("utf-8")
    else:
        data = raw.read_text(encoding="utf-8-sig")
    want = unicodedata.normalize("NFC", text.rstrip())
    return [ln.rstrip() for ln in data.split("\n") if unicodedata.normalize("NFC", ln.rstrip()) == want][:1]


def _step(raw: Path, f: dict, text: str) -> list[str]:
    ds = f["edition"]
    if f.get("role") == "row":
        return _line_member(raw, text)
    lines = raw.read_text(encoding="utf-8-sig").split("\n")
    if f.get("role") == "lexicon":
        for ln in lines:
            cols = ln.split("\t")
            if len(cols) >= 8 and cols[0] == f["estrong"] and cols[6] == f["gloss"] and cols[1].split()[:1] == [f["dstrong"]]:
                import html

                t = html.unescape(TAG_RE.sub("", re.sub(r"<br\s*/?>", "\n", cols[7], flags=re.I)))
                return ["\n".join(x for x in (" ".join(p.split()) for p in t.split("\n")) if x)]
        return []
    prefix = f["ref"] + "#"
    words = []
    for ln in lines:
        if ln.startswith(prefix):
            w = ln.split("\t")[1]
            words.append(w.replace("/", "").replace("\\", "") if ds == "TAHOT" else w.split(" (")[0].replace("¶", "").strip())
    if ds != "TAHOT":
        return [" ".join(words)] if words else []
    out = ""
    for w in words:
        out += w if (not out or out.endswith("\u05be")) else " " + w
    return [out] if out else []


def _sword(raw: Path, f: dict) -> list[str]:
    """zText/zCom through pysword's verse index (LZSS decoded here); RawGenBook by locating the node named by `ref`."""
    from pysword.bible import SwordBible
    from pysword.modules import SwordModules

    from scripture_sword import lzss_decompress, markup_text

    with tempfile.TemporaryDirectory(prefix="mnlm-fsv-sword-", ignore_cleanup_errors=True) as tmp:
        with zipfile.ZipFile(raw) as z:
            z.extractall(tmp)
        name, conf = next(iter(SwordModules(tmp).parse_modules().items()))
        if f.get("driver") == "RawGenBook" or conf.get("moddrv") == "RawGenBook":
            dp = str(Path(tmp) / conf["datapath"])
            idx, dat, bdt = Path(dp + ".idx").read_bytes(), Path(dp + ".dat").read_bytes(), Path(dp + ".bdt").read_bytes()
            names = {}
            for i in range(len(idx) // 4):
                off = struct.unpack_from("<I", idx, 4 * i)[0]
                par = struct.unpack_from("<i", dat, off)[0]
                e = dat.index(b"\0", off + 12)
                ud = struct.unpack_from("<H", dat, e + 1)[0]
                names[4 * i] = (par, dat[off + 12:e].decode("utf-8", "replace"), dat[e + 3:e + 3 + ud])
            for k, (par, nm, u) in names.items():
                path, p = [nm], par
                while p >= 0:
                    p, pn, _ = names[p]
                    if pn:
                        path.append(pn)
                if "/".join(reversed(path)) == f["ref"] and len(u) == 8:
                    o, s = struct.unpack("<II", u)
                    return [markup_text(bdt[o:o + s].decode("utf-8", "replace"))]
            return []

        compress = conf.get("compresstype", "ZIP").upper()
        b = SwordBible(str(Path(tmp) / conf["datapath"]), "ztext", conf.get("versification", "kjv").lower(), conf.get("encoding", "utf-8").lower(),
                       conf.get("sourcetype"), conf.get("blocktype", "BOOK"), compress, None)
        if compress == "LZSS":
            b._decompress = lzss_decompress
        for books in b.get_structure().get_books().values():
            for bk in books:
                if bk.osis_name == f["book"]:
                    return [markup_text(b.get(books=[bk.name], chapters=[int(f["chapter"])], verses=[int(f["verse"])], clean=False) or "")]
    return []


def reparse(source: str, raw: Path, f: dict, text: str) -> list[str] | str:
    """Candidate raw texts for one sealed scripture record, or "SKIP" when no parser exists for the source."""
    if source in USFM_SOURCES:
        return _usfm(raw, f)
    if source in ("sefaria-export", "sefaria-export-nc"):
        return _sefaria(raw, f)
    if source == "tanach-us-uxlc":
        return _uxlc(raw, f)
    if source == "sblgnt":
        return _sblgnt(raw, f)
    if source == "byztxt-rp2018":
        return _byz(raw, f)
    if source == "nestle1904":
        return _nestle(raw, f)
    if source in ("ubsicap-versification", "openbible-crossrefs"):
        return _line_member(raw, text)
    if source == "stepbible-data":
        return _step(raw, f, text)
    if source == "crosswire-sword-pd":
        return _sword(raw, f)
    return "SKIP"
