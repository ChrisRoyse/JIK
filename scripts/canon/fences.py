"""Declared fences with exact character offsets; never normalize source bytes.

One linear scan, O(text bytes + fence count). Only LF, CRLF and CR are line
terminators: Unicode separators inside code strings are not source newlines.
"""
import hashlib
from pathlib import Path
import re

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LINE_END = re.compile(r"\r\n|\r|\n")
OPEN = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*([^\r\n`]*)")


def lines(text):
    """Yield (start, body_end, line_end), including an unterminated last line."""
    pos = 0
    for end in LINE_END.finditer(text):
        yield pos, end.start(), end.end()
        pos = end.end()
    if pos < len(text):
        yield pos, len(text), len(text)


def fences(text):
    """(open_start, body_start, body_end, close_end, language, closed)."""
    opened = None
    for start, body_end, line_end in lines(text):
        line = text[start:body_end]
        if opened is None:
            match = OPEN.fullmatch(line)
            if match is not None:
                fence, info = match.groups()
                language = info.split()[0].lower() if info.strip() else "unspecified"
                close = re.compile(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*")
                opened = (start, line_end, language, close)
        elif opened[3].fullmatch(line) is not None:
            yield opened[0], opened[1], start, line_end, opened[2], True
            opened = None
    if opened is not None:
        yield opened[0], opened[1], len(text), len(text), opened[2], False
