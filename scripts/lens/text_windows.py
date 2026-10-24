"""Exact tokenizer-window provenance, separate from native measured slots (#16).

Offsets are Python character indices in prefix+surface. Source spans clip away
the declared prefix. Unmapped source spans expose tokenizer normalization gaps;
they are not silently described as text the model read. Cost O(chars+tokens).
"""
from __future__ import annotations

import hashlib
import json
from array import array

from lenscommon import LensError

SCHEMA = "text-window-provenance-v1"


def probe_digest(windows, segments):
    """Freeze the exact window state before the next embed overwrites it."""
    raw = json.dumps({"windows": windows, "segments": segments}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_probe(before, after, name):
    if before != after:
        raise LensError("MNLM_LENS_NONDETERMINISTIC", f"{name}: tokenizer window provenance changed")


def require(ok, message):
    if not ok:
        raise LensError("MNLM_LENS_WINDOW_PROVENANCE", message)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def empty(text, prefix):
    return {"schema": SCHEMA, "offset_unit": "unicode-codepoint", "surface_sha256": sha(text) if text is not None else None,
            "surface_chars": len(text) if text is not None else 0,
            "surface_utf8_bytes": len(text.encode("utf-8")) if text is not None else 0,
            "prefix": prefix, "input_sha256": sha(prefix + text) if text is not None else None,
            "windows": [], "unmapped_source_spans": [[0, len(text)]] if text else []}


def source_span(start, end, prefix_chars, text_chars):
    return [min(text_chars, max(0, start - prefix_chars)), min(text_chars, max(0, end - prefix_chars))]


def gaps(spans, length):
    changes = array("q", [0]) * (length + 1)
    for start, end in spans:
        changes[start] += 1
        changes[end] -= 1
    depth, opening, result = 0, None, []
    for i in range(length):
        depth += changes[i]
        if depth == 0 and opening is None:
            opening = i
        elif depth and opening is not None:
            result.append([opening, i])
            opening = None
    if opening is not None:
        result.append([opening, length])
    return result


def build(texts, prefix, enc, mapping, kind):
    """Capture CPU tokenizer tensors before any metadata is removed for forward."""
    required = {"input_ids", "attention_mask", "offset_mapping", "special_tokens_mask"}
    require(required <= set(enc), "tokenizer must expose IDs, attention, special markers and character offsets")
    arrays = {k: enc[k].tolist() if hasattr(enc[k], "tolist") else enc[k]
              for k in required | ({"token_type_ids"} if "token_type_ids" in enc else set())}
    require(all(len(a) == len(mapping) for a in arrays.values()), "tokenizer window/mapping lengths disagree")
    traces = [empty(t, prefix) for t in texts]
    spans = [[] for _ in texts]
    native = [0] * len(texts)
    for w, sample in enumerate(mapping):
        require(type(sample) is int and 0 <= sample < len(texts), "overflow points outside the input batch")
        trace, text = traces[sample], texts[sample]
        window = {k: arrays[k][w] for k in arrays}
        window["ordinal"] = len(trace["windows"])
        active = sum(window["attention_mask"])
        if kind == "multi":
            window["native_token_span"] = [native[sample], native[sample] + active]
            native[sample] += active
        starts = []
        for (start, end), attended, special in zip(window["offset_mapping"], window["attention_mask"], window["special_tokens_mask"]):
            if attended and not special and start < end:
                a, b = source_span(start, end, len(prefix), len(text))
                if a < b:
                    spans[sample].append([a, b])
                    starts.append([a, b])
        window["source_char_span"] = [min(s[0] for s in starts), max(s[1] for s in starts)] if starts else None
        trace["windows"].append(window)
    for t, seen, text in zip(traces, spans, texts):
        t["unmapped_source_spans"] = gaps(seen, len(text))
        validate(t, kind, len(t["windows"]), text=text)
        require(bool(t["windows"]), "tokenizer omitted an input instead of returning a window")
    return traces


def validate(trace, kind, count, *, native_rows=None, text=None):
    require(isinstance(trace, dict) and trace.get("schema") == SCHEMA, "historical count-only record; remeasure in a new generation")
    require(trace.get("offset_unit") == "unicode-codepoint", "offset unit is not declared as source characters")
    require(type(trace.get("surface_chars")) is int and trace["surface_chars"] >= 0 and
            isinstance(trace.get("prefix"), str) and isinstance(trace.get("windows"), list), "invalid source extent or window list")
    length, prefix, windows = trace["surface_chars"], trace["prefix"], trace["windows"]
    require(type(trace.get("surface_utf8_bytes")) is int and trace["surface_utf8_bytes"] >= length, "invalid UTF-8 byte extent")
    for key in ("surface_sha256", "input_sha256"):
        value = trace.get(key)
        require((value is None and length == 0 and not windows and trace["surface_utf8_bytes"] == 0) or
                (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)),
                "missing or invalid retained source/input hash")
    require(len(windows) == count, "native window count differs from provenance")
    if text is not None:
        require(trace["surface_sha256"] == sha(text) and length == len(text) and
                trace["surface_utf8_bytes"] == len(text.encode("utf-8")) and
                trace["input_sha256"] == sha(prefix + text), "window provenance binds different surface bytes")
    covered, cursor = [], 0
    for ordinal, window in enumerate(windows):
        require(window.get("ordinal") == ordinal, "window sequence has gaps or is reordered")
        keys = ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping")
        require(all(isinstance(window.get(k), list) for k in keys), "missing tokenizer inputs")
        n = len(window["input_ids"])
        require(all(len(window[k]) == n for k in keys) and
                ("token_type_ids" not in window or len(window["token_type_ids"]) == n), "token alignment lengths disagree")
        require(all(type(i) is int and i >= 0 for i in window["input_ids"]) and
                all(type(i) is int and i in (0, 1) for k in ("attention_mask", "special_tokens_mask") for i in window[k]),
                "invalid ID or attention/special mask")
        local = []
        for span, attended, special in zip(window["offset_mapping"], window["attention_mask"], window["special_tokens_mask"]):
            require(isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span) and
                    0 <= span[0] <= span[1] <= len(prefix) + length, "offset outside bound prefix+surface")
            require(not special or span == [0, 0], "special-token marker carries an unexplained source span")
            require(attended or span == [0, 0], "padding token falsely claims source characters")
            if attended and not special and span[0] < span[1]:
                a, b = source_span(*span, len(prefix), length)
                if a < b:
                    covered.append([a, b])
                    local.append([a, b])
        expected = [min(s[0] for s in local), max(s[1] for s in local)] if local else None
        require(window.get("source_char_span") == expected, "window source extent differs from token offsets")
        if kind == "multi":
            end = cursor + sum(window["attention_mask"])
            require(window.get("native_token_span") == [cursor, end], "Multi token rows lost window alignment")
            cursor = end
        else:
            require("native_token_span" not in window, "pooled output falsely claims native token rows")
    require(trace.get("unmapped_source_spans") == gaps(covered, length), "unmapped source spans were hidden or altered")
    if kind == "multi" and native_rows is not None:
        require(cursor == native_rows, "Multi output token count differs from recorded forward inputs")
    return True


def validate_slot(path, contract, views):
    """Structural native/metadata alignment at publication and checkpoint reads."""
    if contract.get("modality") != "text" or contract.get("repo") is None:
        return
    require(contract.get("segmentation") == SCHEMA, "unattested learned text segmentation contract")
    from slotio import npz_member
    windows = npz_member(path, "windows")
    absent = npz_member(path, "absent")
    offsets = npz_member(path, "offsets") if contract["kind"] == "multi" else None
    try:
        details = views.get("details")
        require(isinstance(details, list) and len(details) == len(windows), "missing per-row window provenance")
        for i, detail in enumerate(details):
            require(isinstance(detail, dict) and isinstance(detail.get("text_windows"), dict), "missing typed text window detail")
            require(detail["text_windows"].get("prefix") == contract.get("prefix", ""), "window prefix differs from frozen instrument")
            require(windows[i] > 0 or absent[i], "present native value has no measured window")
            validate(detail.get("text_windows"), contract["kind"], int(windows[i]),
                     native_rows=int(offsets[i + 1] - offsets[i]) if offsets is not None else None)
    finally:
        for a in (windows, offsets, absent):
            if getattr(a, "_mmap", None) is not None:
                a._mmap.close()
