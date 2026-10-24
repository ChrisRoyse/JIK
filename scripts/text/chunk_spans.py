"""Lossless, bounded text spans for sense measurements (#12/#25).

Use the existing sentence segmenter, subdividing oversized sentences at whitespace
when possible. No minimum length/word filter and no head-only fallback. Offsets
are Python character offsets; exact concatenation recovers the original text.
Cost O(characters); a new contract requires new chunks and new slot measurements.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import segment, conserved

CONTRACT = "lossless-sentence-spans-v1"
MAX_CHARS = 600


def chunk_spans(text, max_chars=MAX_CHARS):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("MNLM_CHUNK_EMPTY: a nonblank text observation is required")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("MNLM_CHUNK_LIMIT: positive character limit required")
    sentences = segment(text)
    if not conserved(text, sentences):
        raise ValueError("MNLM_CHUNK_SEGMENT: source segmenter did not conserve text")
    out = []
    for begin, end, role in sentences:
        start = begin
        while start < end:
            stop = min(start + max_chars, end)
            forced = stop < end
            if forced:
                # One backwards scan per bounded chunk; do not rescan the tail.
                boundary = next((j + 1 for j in range(stop - 1, start, -1)
                                 if text[j].isspace()), None)
                if boundary is not None:
                    stop = boundary
                    forced = False
            out.append({"start": start, "end": stop, "role": role,
                        "forced_cut": forced, "text": text[start:stop]})
            start = stop
    if (out[0]["start"] != 0 or out[-1]["end"] != len(text)
            or any(a["end"] != b["start"] for a, b in zip(out, out[1:]))
            or any(not 0 < len(s["text"]) <= max_chars for s in out)):
        raise ValueError("MNLM_CHUNK_CONSERVATION: invalid bounded source tiling")
    return out
