"""Contract `scripture-v1` readers for CrossWire SWORD modules (issue #9): zText Bibles through pysword (with the LZSS block
compression pysword lacks), zCom commentaries through the same block/verse index, and RawGenBook tree modules walked from
their .idx/.dat/.bdt bytes. Licence from the module .conf; a module that yields no record refuses (never a 0-record seal).
Cost: O(module bytes), one pass; the tree walk is O(nodes).
"""
from __future__ import annotations

import html
import re
import struct
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

from scripture import PROSE, ScriptureError, gate, licence_token

_NOTE = re.compile(r"<note\b.*?</note>|<RF\b[^>]*>.*?<Rf>", re.S | re.I)  # OSIS/ThML notes and GBF footnotes (Murdock, readback F-3)
_BREAK = re.compile(r"</(?:p|title|l|lg|div|item|head|verse)\s*>|<(?:lb|br|p|milestone)\b[^>]*/?>", re.I)
_SCRIPREF = re.compile(r"<scripRef\b[^>]*>(.*?)</scripRef>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def markup_text(s: str) -> str:
    """OSIS/ThML fragment -> plain text: notes dropped, block ends become newlines, other tags removed, entities decoded,
    whitespace collapsed per line, blank lines dropped."""
    s = _NOTE.sub(" ", s)
    s = _SCRIPREF.sub(lambda m: m.group(1), s)
    s = _BREAK.sub("\n", s)
    s = html.unescape(_TAG.sub("", s))
    return "\n".join(ln for ln in (" ".join(x.split()) for x in s.split("\n")) if ln)


def lzss_decompress(data: bytes) -> bytes:
    """SWORD's LZSS block codec (Okumura: ring 4096, lookahead 18, threshold 2, ring primed with spaces, no header)."""
    n, f, th = 4096, 18, 2
    ring = bytearray(b" " * (n + f - 1))
    r, i, flags, out = n - f, 0, 0, bytearray()
    while True:
        flags >>= 1
        if not flags & 0x100:
            if i >= len(data):
                break
            flags = data[i] | 0xFF00
            i += 1
        if flags & 1:
            if i >= len(data):
                break
            c = data[i]
            i += 1
            out.append(c)
            ring[r] = c
            r = (r + 1) & (n - 1)
        else:
            if i + 1 >= len(data):
                break
            c1, c2 = data[i], data[i + 1]
            i += 2
            pos, ln = c1 | ((c2 & 0xF0) << 4), (c2 & 0x0F) + th
            for k in range(ln + 1):
                c = ring[(pos + k) & (n - 1)]
                out.append(c)
                ring[r] = c
                r = (r + 1) & (n - 1)
    return bytes(out)


def _bible(root: Path, conf: dict):
    """A pysword SwordBible over the module's data directory, reading `compresstype` from the conf (pysword looks for a
    `compress_type` key that no .conf carries, so every module was decoded as ZIP) and decoding LZSS itself."""
    from pysword.bible import SwordBible

    compress = conf.get("compresstype", "ZIP").upper()
    b = SwordBible(str(root / conf["datapath"]), "ztext", conf.get("versification", "kjv").lower(), conf.get("encoding", "utf-8").lower(),
                   conf.get("sourcetype"), conf.get("blocktype", "BOOK"), compress, None)  # __new__ picks the driver class: patch the instance
    if compress == "LZSS":
        b._decompress = lzss_decompress
    return b


def _verses(bible, base: dict, role: str, clean) -> Iterator[tuple[int, str | None, dict, str]]:
    row, failed = 0, {}
    for testament, books in bible.get_structure().get_books().items():
        for b in books:
            for ch, n in enumerate(b.chapter_lengths, start=1):
                for v in range(1, n + 1):
                    try:
                        t = bible.get(books=[b.name], chapters=[ch], verses=[v], clean=False)
                    except Exception as exc:  # noqa: BLE001 a verse absent from the module (counted, reported if nothing is read)
                        failed[type(exc).__name__] = failed.get(type(exc).__name__, 0) + 1
                        continue
                    t = clean(t or "")
                    if not t:
                        continue
                    yield row, t, {**base, "book": b.osis_name, "chapter": ch, "verse": str(v), "role": role, "ref": f"{b.osis_name}.{ch}.{v}", "testament": testament}, PROSE
                    row += 1
    if row == 0:
        raise ScriptureError("MNLM_SCRIPTURE_SWORD_EMPTY", f"{base['edition']}: no verse decoded", failed=failed)


def genbook_docs(root: Path, conf: dict, base: dict) -> Iterator[tuple[int, str | None, dict, str]]:
    """RawGenBook: .idx = uint32 offsets into .dat; a .dat node = parent, next, first-child (int32 byte offsets into .idx),
    name NUL, uint16 user-data size, user data (uint32 offset + uint32 size into .bdt when 8 bytes). One record per node
    that carries text, in .idx order (document order); `ref` is the node path."""
    dp = str(root / conf["datapath"])
    idx, dat, bdt = Path(dp + ".idx").read_bytes(), Path(dp + ".dat").read_bytes(), Path(dp + ".bdt").read_bytes()
    nodes = []
    for i in range(len(idx) // 4):
        off = struct.unpack_from("<I", idx, 4 * i)[0]
        par = struct.unpack_from("<i", dat, off)[0]
        e = dat.index(b"\0", off + 12)
        ud = struct.unpack_from("<H", dat, e + 1)[0]
        nodes.append((par, dat[off + 12:e].decode("utf-8", "replace"), dat[e + 3:e + 3 + ud]))
    row = 0
    for i, (par, name, u) in enumerate(nodes):
        path, p = [name], par
        while p >= 0:
            pn, pname, _ = nodes[p // 4]
            if pname:
                path.append(pname)
            p = pn
        path.reverse()
        if len(u) != 8:
            continue
        o, s = struct.unpack("<II", u)
        t = markup_text(bdt[o:o + s].decode("utf-8", "replace"))
        if not t:
            continue
        yield row, t, {**base, "ref": "/".join(path), "role": "section", "node": i, "depth": len(path)}, PROSE
        row += 1
    if row == 0:
        raise ScriptureError("MNLM_SCRIPTURE_SWORD_EMPTY", f"{base['edition']}: no node carries text", nodes=len(nodes))


def sword_docs(path: Path, conf_dir: Path) -> Iterator[tuple[str, str | None, dict, str]]:
    """A SWORD rawzip module: licence gated from its .conf; zText and zCom through the block index, RawGenBook through
    the tree; any other driver refuses MNLM_SCRIPTURE_SWORD_UNSUPPORTED."""
    from pysword.modules import SwordModules

    with tempfile.TemporaryDirectory(prefix="mnlm-sword-", ignore_cleanup_errors=True) as tmp:  # pysword keeps handles open
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        found = SwordModules(tmp).parse_modules()
        if len(found) != 1:
            raise ScriptureError("MNLM_SCRIPTURE_SWORD", f"{path.name}: {len(found)} modules in zip")
        name, conf = next(iter(found.items()))
        token = gate(licence_token(conf.get("distributionlicense", "")), f"{path.name} {name}.conf")
        drv = conf.get("moddrv", "")
        base = {"edition": name, "lang": conf.get("lang"), "licence": token, "module_version": conf.get("version"), "versification": conf.get("versification"), "driver": drv}
        if drv == "zText":
            yield from _verses(_bible(Path(tmp), conf), base, "verse", markup_text)
        elif drv == "zCom":
            yield from _verses(_bible(Path(tmp), conf), base, "commentary", markup_text)
        elif drv == "RawGenBook":
            yield from genbook_docs(Path(tmp), conf, base)
        else:
            raise ScriptureError("MNLM_SCRIPTURE_SWORD_UNSUPPORTED", f"{path.name}: driver {drv!r} has no reader")
