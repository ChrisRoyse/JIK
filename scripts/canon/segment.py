"""Stage 2 (issue #11, spec §18): the deterministic prose segmenter. Atoms are character spans that tile the text
exactly (whitespace between atoms attaches to the preceding atom), so conservation is a property of the
representation. Rule set: a newline is a hard boundary; a boundary candidate is a terminator run plus optional
closing quotes, followed by whitespace and an opener (upper-case Latin/Greek/Cyrillic, any Hebrew or Arabic-script
letter, a digit, an opening quote or bracket) or the end of the paragraph; an abbreviation table and single-letter
initials suppress it; Hebrew sof pasuq and the Greek question mark are terminators. The segmenter is identified by
the BLAKE3 of this file (SEGMENTER_ID) so a rule change mints a new identity; no learned component.
Cost: O(chars) per document, one regex pass per paragraph.
"""
from __future__ import annotations

import re
from pathlib import Path

ROLE_SENTENCE, ROLE_WS, ROLE_ROW, ROLE_CUT, ROLE_VERSE = 1, 2, 3, 4, 5
ROLE_NAMES = {ROLE_SENTENCE: "sentence", ROLE_WS: "ws", ROLE_ROW: "row", ROLE_CUT: "cut", ROLE_VERSE: "verse"}

ABBREV = frozenset("""mr mrs ms dr prof sr jr st vs etc e.g i.e cf fig no vol pp ch sec u.s u.k inc ltd co corp bros
jan feb mar apr jun jul aug sep sept oct nov dec mon tue wed thu fri sat sun
gen ex exod lev num deut josh judg ruth sam kgs chr chron ezra neh esth ps pss prov eccl isa jer lam ezek dan hos
joel amos obad jon mic nah hab zeph hag zech mal mt matt mk lk jn rom cor gal eph phil col thess tim tit philem heb
jas pet rev a.m p.m al ed eds trans rev op cit ibid id viz approx dept est max min""".split())

_TERM = "[.!?…。！？׃;;]"
_CLOSE = "[\"'”’)\\]»״]"
_OPEN = "[\"'“‘(\\[«״]"
_UPPER = "[A-ZΑ-ΩЀ-Яא-ת؀-ۿ0-9]"
_CAND = re.compile(f"{_TERM}++{_CLOSE}*+(?=\\s+(?:{_OPEN}*{_UPPER})|\\s*$)")  # possessive runs: no backtracking over a run of terminators
_PARA = re.compile(r"[^\n]+(?:\n+|$)")
_LASTWORD = re.compile(r"(\S+)\s*$")
_STRIP_PUNCT = ".!?…\"'”’)]»״׃;;"

SEGMENTER_ID = None  # filled at import from this file's bytes


def _self_id() -> str:
    import blake3

    return blake3.blake3(Path(__file__).read_bytes()).hexdigest()[:32]


def segment(text: str) -> list[tuple[int, int, int]]:
    """(start, end, role) char spans covering `text` exactly: spans start at 0, tile without gaps, end at len(text)."""
    out: list[tuple[int, int, int]] = []
    pos = 0
    for para in _PARA.finditer(text):
        p0, p1 = para.start(), para.end()
        if p0 > pos:  # leading newlines (only possible at the very start)
            out.append((pos, p0, ROLE_WS))
        body = text[p0:p1]
        start = 0
        for m in _CAND.finditer(body):
            w = _LASTWORD.search(body, max(0, m.end() - 64), m.end())  # bounded window: an abbreviation is short; an unbounded search is quadratic
            token = w.group(1).rstrip(_STRIP_PUNCT).lower() if w else ""
            if token in ABBREV or (len(token) == 1 and token.isalpha()) or "." in token and token.replace(".", "").isalpha() and len(token) <= 5:
                continue  # abbreviation, initial, or dotted acronym such as u.s.a
            e = m.end()
            while e < len(body) and body[e].isspace():
                e += 1
            if e - start < 2:
                continue
            out.append((p0 + start, p0 + e, ROLE_SENTENCE))
            start = e
        if start < len(body):
            out.append((p0 + start, p1, ROLE_SENTENCE))
        pos = p1
    if pos < len(text):
        out.append((pos, len(text), ROLE_WS))
    return out


def conserved(text: str, spans: list[tuple[int, int, int]]) -> bool:
    """True iff the spans tile the text exactly (the Stage 2 boundary 1 predicate, also used by the FSV)."""
    if not text:
        return not spans
    if not spans or spans[0][0] != 0 or spans[-1][1] != len(text):
        return False
    return all(spans[i][1] == spans[i + 1][0] and spans[i][0] < spans[i][1] for i in range(len(spans) - 1)) and spans[-1][0] < spans[-1][1]


SEGMENTER_ID = _self_id()
