"""Frozen, source-agnostic channel instruments. No per-source fit or label access.

Each typed text field retains its ordered lexical tokens in a Multi slot. The
original fields stay exact in UCR; these are deterministic measurement features,
not a replacement for source bytes. All surface embeddings count as `surface`.
"""
from __future__ import annotations

import hashlib
import math
import re
import platform
from pathlib import Path

from .contract import CHANNELS, canonical, digest, require, time_value, validate_record

DIM = 128
VERSION = "universal-instruments-v2"


def contracts():
    import numpy as np
    base = Path(__file__).resolve().parent
    code = {n: hashlib.sha256((base / n).read_bytes()).hexdigest() for n in ("contract.py", "instruments.py")}
    runtime = {"device": "algorithmic-cpu", "python": platform.python_version(), "numpy": np.__version__,
               "system": platform.system(), "machine": platform.machine()}
    return [{"name": "ucr-" + c + "-v1", "channel": c, "perspective": c, "inputs": "ucr",
             "kind": "encoder", "role": "universal_instrument", "dtype": "f32", "shape": [None, DIM],
             "runtime": runtime, "cpu_fallback": False, "weights_sha256": digest(code),
             "contract": VERSION, "norm": "none", "sim": "maxsim", "corpus_hash": "none",
             "lens_id": digest([VERSION, c, DIM, code, runtime])} for c in CHANNELS]


def roster():
    c = contracts()
    return {"schema": VERSION, "instruments": c, "panel_digest": digest(c),
            "admitted": [], "served": False, "status": "MNLM_SHANNON_CONDITIONAL_UNMEASURED"}


def _features(item, channel):
    if channel == "surface":
        return [(t.casefold(), 1.0) for t in re.findall(r"\w+|[^\w\s]", item["text"])]
    if channel in ("fields", "provenance", "structure"):
        k, v, name = item["type"], item["value"], item["name"]
        if k in ("reference", "text"):
            # References are lineage. Latent text is measured token by token
            # below, never hashed as one categorical identity or pooled here.
            return []  # identities are lineage, not predictive numeric features
        key = f"{name}|{k}|{item['unit']}|{item['clock_role']}"
        if k in ("integer", "number"):
            magnitude = math.log(abs(v) + 1) if type(v) is int else math.log1p(abs(v))
            return [(key, -magnitude if v < 0 else magnitude), (key + "|observed", 1.0)]
        if k == "clock":
            epoch = time_value(v)
            return [(key + "|sin_day", math.sin(epoch * 2 * math.pi / 86400)),
                    (key + "|cos_day", math.cos(epoch * 2 * math.pi / 86400)),
                    (key + "|epoch_days", epoch / 86400)]
        return [(key + "|" + str(v), 1.0)]
    if channel == "relations":
        out = [("relation|" + item["rel_type"] + "|" + item["direction"] + "|" + item["polarity"], 1.0)]
        return out + [f for q in item["qualifiers"] for f in _features(q, "fields")]
    if channel in ("history", "co_reference"):
        # Links alone do not manufacture information from other sources. The view
        # resolver supplies their measured fields/text through separate items.
        return []
    if channel == "behaviour":
        return [("event|" + item["mode"] + "|" + item["event"], 1.0)] + [f for q in item["measurements"] for f in _features(q, "fields")]
    if channel == "perception":
        # Only observed sensor measurements, never a file hash masquerading as perception.
        return [f for q in item["measurements"] for f in _features(q, "fields")]
    require(False, "MNLM_UCR_CHANNEL", channel)


def text_fields(item, channel):
    """Typed text only; numeric/category/reference values never enter a lexer."""
    if channel in ("fields", "provenance", "structure"):
        fields = [item]
    elif channel == "relations":
        fields = item["qualifiers"]
    elif channel in ("behaviour", "perception"):
        fields = item["measurements"]
    else:
        fields = ()
    return (f for f in fields if f["type"] == "text")


def groups(item, channel):
    features = _features(item, channel)
    if channel == "surface":
        yield from ([f] for f in features)
        return
    if features:
        yield features
    for field in text_fields(item, channel):
        # Preserve field identity and units without creating a whole-text key.
        # Token repetition/order survive as rows; changing one word changes
        # that row, not every token in the observation.
        key = f"{field['name']}|text|{field['unit']}|{field['clock_role']}"
        for match in re.finditer(r"\w+|[^\w\s]", field["value"]):
            yield [(key, 1.0), (key + "|token|" + match.group().casefold(), 1.0)]


def tokens(items, channel):
    import numpy as np
    rows = []
    for item in items:
        for group in groups(item, channel):
            row = np.zeros(DIM, dtype=np.float32)
            for name, value in group:
                h = hashlib.sha256(name.encode()).digest()
                row[int.from_bytes(h[:4], "little") % DIM] += (1 if h[4] & 1 else -1) * value
            require(bool(np.isfinite(row).all()), "MNLM_UCR_NONFINITE", "instrument output overflow")
            rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape(-1, DIM)


def measure(record, channel, context=None):
    validate_record(record)
    require(channel in CHANNELS, "MNLM_UCR_CHANNEL", str(channel))
    ch = record["channels"][channel]
    import numpy as np
    if channel in ("history", "co_reference"):
        resolved = context or []
        arrays = [tokens(r["channels"][c]["items"], c) for r in resolved for c in ("surface", "fields", "relations")]
        values = np.concatenate(arrays, axis=0) if arrays else np.empty((0, DIM), np.float32)
        reason = "no_eligible_context" if not len(values) else None
    else:
        values = tokens(ch["items"], channel)
        reason = ch["reason"] or "no_measurable_items" if not len(values) else None
    return {"kind": "Multi" if len(values) else "Absent", "channel": channel, "reason": reason,
            "tokens": values, "shape": list(values.shape),
            "dependencies": [r["atom_id"] for r in (context or [])]}


def slot_digest(slot):
    h = hashlib.sha256(canonical({k: v for k, v in slot.items() if k != "tokens"}))
    h.update(slot["tokens"].astype("<f4").tobytes())
    return h.hexdigest()


def coverage(slots, admitted=()):
    names = {c["name"]: c["channel"] for c in contracts()}
    require(all(n in names and s["channel"] == names[n] for n, s in slots.items()), "MNLM_UCR_UNKNOWN_LENS", "slot/channel identity mismatch")
    require(set(admitted) <= names.keys(), "MNLM_UCR_UNKNOWN_LENS", "unknown admission identity")
    available = sorted({s["channel"] for s in slots.values() if s["kind"] != "Absent"})
    # Current diagnostic estimator cannot certify D2. Populated != admitted.
    require(not admitted, "MNLM_SHANNON_CONDITIONAL_UNMEASURED", "this instrument roster has no certified admissions")
    return {"available_channels": available, "available_count": len(available), "admitted_channels": [],
            "admitted_lenses": 0, "d4": False, "sufficient": False,
            "status": "MNLM_SHANNON_CONDITIONAL_UNMEASURED"}
