"""Exact event intervals plus derived positional context, never a new sense.

Contextual encoder rows retain the complete input window's support. A row's
position inside an encoder tensor does not establish an instantaneous reading.
PC-07/35: derive the small interval catalogue once from authenticated metadata.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import json
import math

import torch

from model.native import NativeInput
from senses.contract import canonical, digest, require

FEATURES = 8
SCHEMA = "native-event-time-v1"


def q(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) is int for x in value) and value[1] > 0,
            "MNLM_TEMPORAL_RATIONAL", "retain exact source clock fractions")
    return Fraction(*value)


def pair(value):
    return [value.numerator, value.denominator]


def descriptor(stream):
    meta = json.loads(stream.metadata)
    event, entry = meta["moment"], meta["entry"]
    row = entry["input"]
    require(row["clock_id"] == event["source"]["clock_id"]
            and row["source"] == event["source"]["bundle"]
            and row["segment"] == event["source"]["segment"],
            "MNLM_TEMPORAL_SOURCE", "reading and event must share an attested source")
    windows = []
    medium = stream.instrument.split("_", 1)[0]
    if medium == "image":
        frame, tb = row["frame"], q(row["time_base"])
        interval = [pair(frame["pts"]*tb), pair((frame["pts"]+frame["duration"])*tb)]
        windows.append(dict(rows=[0, entry["shape"][0]], interval=interval,
                            components=[interval], basis="decoded_frame_presentation"))
    elif medium == "audio":
        audio = row["source_audio"]
        origin = Fraction(audio["start_sample"], audio["rate"])
        # Keep the entire waveform dependency: resampling and preprocessing can
        # consult samples outside a nominal encoder chunk.
        for window in entry["native"]:
            rate = window.get("target_rate", 16000)
            lo = origin + Fraction(window["resampled_start"], rate)
            hi = min(origin + Fraction(window["resampled_end"], rate),
                     Fraction(audio["stop_sample"], audio["rate"]))
            interval = [pair(lo), pair(hi)]
            windows.append(dict(rows=[window["token_start"], window["token_end"]],
                interval=interval, components=[interval],
                basis="complete_contextual_audio_window"))
    else:
        require(medium == "text", "MNLM_TEMPORAL_MEDIUM", "explicit native modality required")
        words, cursor = [], 0
        for word in row["text"]["words"]:
            if word["tag"] != "w":
                continue
            surface = word["text"] or ""
            words.append((cursor, cursor+len(surface), word))
            cursor += len(surface)+1
        require(" ".join(w["text"] or "" for _, _, w in words) == row["text"]["surface"],
                "MNLM_TEMPORAL_TEXT", "restore the source word-to-character map")
        for i, window in enumerate(entry["native"]["windows"]):
            lo, hi = window["source_char_span"]
            selected = [w for start, end, w in words if start < hi and end > lo]
            require(selected, "MNLM_TEMPORAL_TEXT", "window has no attested timed words")
            spans = [w["span"] for w in selected]
            interval = [pair(min(q(s[0]) for s in spans)), pair(max(q(s[1]) for s in spans))]
            windows.append(dict(rows=[i, i+1], interval=interval, components=spans,
                word_ids=[w["attributes"]["{http://nite.sourceforge.net/}id"] for w in selected],
                basis="contextual_text_window_forced_alignment"))
    value = dict(schema=SCHEMA, row=event["id"], instrument=stream.instrument,
        measurement=stream.identity, reading_sha256=entry["sha256"],
        bank=meta["bank"], source=event["source"], at=event["content"]["at"],
        context=row["support"], available_after=row["available_after"],
        through=meta["scope"]["through"], timing=event["content"]["timing"],
        physical_sync_calibrated=False, windows=windows,
        dependency="audio" if medium == "text" else "recording",
        active_language=event["content"]["language"])
    validate(value, event["id"], entry["shape"][0])
    return canonical(value)


def validate(value, row, count):
    require(value["schema"] == SCHEMA and value["row"] == row
            and value["physical_sync_calibrated"] is False
            and value["timing"]["measured_error"] is None,
            "MNLM_TEMPORAL_SCOPE", "retain unresolved physical synchronization")
    at, through, available = (q(value[k]) for k in ("at", "through", "available_after"))
    lo, hi = map(q, value["context"])
    require(lo <= at < hi and available >= hi and through >= available,
            "MNLM_TEMPORAL_CUTOFF", "retrospective readings are unavailable before their cutoff")
    cursor = 0
    for window in value["windows"]:
        start, end = window["rows"]
        a, b = map(q, window["interval"])
        require(type(start) is int and type(end) is int and start == cursor < end <= count
                and a < b and b <= available, "MNLM_TEMPORAL_COVERAGE",
                "cover every native row once with its attested window")
        for component in window["components"]:
            x, y = map(q, component)
            require(a <= x <= y <= b, "MNLM_TEMPORAL_COMPONENT", "preserve window containment")
        cursor = end
    require(cursor == count, "MNLM_TEMPORAL_COVERAGE", "incomplete native temporal coverage")


def features(records, counts, width, device):
    """Exact subtraction precedes fp32 conversion; persisted fractions stay exact."""
    result = torch.zeros(len(records), width, FEATURES, dtype=torch.float32, device=device)
    for i, (raw, count) in enumerate(zip(records, counts, strict=True)):
        value = json.loads(raw)
        validate(value, value["row"], count)
        at, available = q(value["at"]), q(value["available_after"])
        context_lo, context_hi = map(q, value["context"])
        for window in value["windows"]:
            lo, hi = map(q, window["interval"])
            offsets = (lo-at, hi-at, hi-lo, available-at, context_lo-at, context_hi-at, at)
            coordinates = [math.copysign(math.log1p(abs(float(x))), float(x)) for x in offsets]
            coordinates.append(1.)  # Unknown synchronization error, never assumed zero.
            start, end = window["rows"]
            result[i, start:end] = torch.tensor(coordinates, dtype=torch.float32, device=device)
    return result


@dataclass(frozen=True)
class TimedNativeInput(NativeInput):
    temporal: tuple[bytes, ...] = ()

    def to(self, device):
        base = super().to(device)
        return replace(self, values=base.values, padding=base.padding, absent=base.absent,
                       indices=base.indices)

    def positions(self):
        counts = tuple((~self.padding).sum(1).tolist())
        require(len(self.temporal) == len(self.rows), "MNLM_TEMPORAL_MISSING",
                "every observation needs retained temporal support")
        for row, count, raw in zip(self.rows, counts, self.temporal, strict=True):
            validate(json.loads(raw), row, count)
        return features(self.temporal, counts, self.values.shape[1], self.values.device)


def check_catalogue(native, catalogue):
    """A model consumes only the temporal bindings pinned by its source adapter."""
    first = None
    for lane, x in native.items():
        require(isinstance(x, TimedNativeInput) and len(x.temporal) == len(x.rows),
                "MNLM_TEMPORAL_MISSING", "timed source observations cannot lose their clock")
        common = []
        for row, raw in zip(x.rows, x.temporal, strict=True):
            require(row in catalogue and lane in catalogue[row]
                    and digest(json.loads(raw)) == catalogue[row][lane],
                    "MNLM_TEMPORAL_BINDING", "re-resolve changed time/source support through memory")
            value = json.loads(raw)
            require(value["instrument"] == lane and value["row"] == row,
                    "MNLM_TEMPORAL_BINDING", "timing belongs to another instrument or event")
            common.append((value["source"], value["at"], value["available_after"], value["through"]))
        require(first is None or first == common, "MNLM_TEMPORAL_ALIGNMENT",
                "all branches must refer to the same event and source clock")
        first = common
