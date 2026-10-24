"""Immutable native evidence joined by exact source time, never by pooled vectors.

PC-07/28/35: check source/prefix/wrapper/value bytes once at opening; index native
blocks once. Moment lookup is logarithmic in blocks plus its selected references.
The checker certifies recording contents, not an event's truth or sensory admission.
"""
from bisect import bisect_right
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import soundfile as sf

from claims.episodic_memory import canonical, digest
from senses import audio_bank, visual_bank
from senses.native_observation import public_views
from senses.perception_corpus import pinned
from senses.perception_native import NativePrefix

SCOPE = "retained_native_recording_measurements"


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_MEMORY_"+code)


def rational(value):
    need(type(value) == list and len(value) == 2
         and all(type(x) == int for x in value) and value[0] >= 0 and value[1] > 0,
         "TIME")
    result = Fraction(*value)
    need(value == pair(result), "CANONICAL_TIME")
    return result


def pair(value):
    return [value.numerator, value.denominator]


def episode(kind, content, source, time, evidence=()):
    body = dict(claim_kind=kind, content=content, source=source,
                event_time=time, evidence=list(evidence))
    return dict(id=digest(body), **body)


class NativeEvidence:
    def __init__(self, bank, report_pin, inputs, corpus):
        bank, inputs, corpus = map(Path, (bank, inputs, corpus))
        report = pinned(bank/"report.json", report_pin)
        manifest = pinned(inputs/"report.json", report["input_pin"])
        need(report["author_checked"] and not report["scientific_admission"]
             and not report["training_ready"] and not manifest["physical_sync_calibrated"],
             "UNADMITTED_FLAGS")
        self.report_pin = report_pin
        self._sources, self._readings, self._measurements, self._indexes = {}, {}, {}, {}
        modules = dict(image=visual_bank, audio=audio_bank)
        views = {m: public_views(m, mod.input_views(inputs/m, manifest["input_pins"][m]))
                 for m, mod in modules.items()}
        need(set(report["banks"]) == set(modules) and
             all({r["clip"] for r, _, _ in v} ==
                 {s["clip"] for s in manifest["sources"]} for v in views.values()),
             "SOURCE_ROSTER")
        for declared in manifest["sources"]:
            clip = declared["clip"]
            need(clip not in self._sources and Path(clip).name == clip, "CLIP")
            self._sources[clip] = canonical(declared)
            for medium in modules:
                d = declared["media"][medium]
                prefix = NativePrefix(inputs/d["root"], expected=d["pin"],
                    episodes=corpus/"clips"/clip, episode_pin=declared["episode_pin"],
                    media=(medium,))
                decoded = list(prefix.consume(**d["request"]))
                need(prefix.value["source"] == declared["source"], "SOURCE_BINDING")
                rows = [v for v in views[medium] if v[0]["clip"] == clip]
                self._bind(prefix, decoded, rows, medium)
                nodes = [x["node"] for x in decoded]
                starts = [rational(n["span"][0]) for n in nodes]
                need(starts == sorted(set(starts)) and
                     all(rational(a["span"][1]) <= rational(b["span"][0])
                         for a, b in zip(nodes, nodes[1:])), "BLOCK_ORDER")
                offsets, total = [], 0
                for node in nodes:
                    offsets.append(total)
                    total += node["sample_count"] or 0
                self._indexes[clip, medium] = (
                    starts, tuple(canonical(n) for n in nodes), offsets,
                    [nodes[0]["span"][0], nodes[-1]["span"][1]],
                    pair(max(rational(n["available_after"]) for n in nodes)))
        for medium, mod in modules.items():
            proof = report["banks"][medium]
            cls = visual_bank.VisualBank if medium == "image" else audio_bank.AudioBank
            reader = cls(bank/medium, proof["bank_pin"], inputs/medium,
                         manifest["input_pins"][medium], proof["instruments"])
            entries = {(name, digest(e["input"])): e
                       for name in mod.NAMES for e in reader.value["readings"][name]}
            selected = {clip: [] for clip in self._sources}
            for name, row, values in reader.consume():
                e = entries[name, digest(row)]
                raw = values.tobytes()
                need(hashlib.sha256(raw).hexdigest() == e["sha256"], "VALUE_PIN")
                ref = digest(dict(medium=medium, name=name, entry=e))
                self._readings[ref] = (canonical(dict(medium=medium, name=name, entry=e)), raw)
                selected[row["clip"]].append(ref)
            for clip in self._sources:
                self._measurements[clip, medium] = canonical(
                    self._measurement(clip, medium, proof["bank_pin"], selected[clip]))
        self.pin = digest(dict(report=report_pin, inputs=report["input_pin"],
                               episodes=[digest(v.decode()) for v in self._measurements.values()]))

    @staticmethod
    def _bind(prefix, decoded, rows, medium):
        v = prefix.value
        if medium == "image":
            need(len(decoded) == 1, "INITIAL_FRAME")
            rgb = decoded[0]["values"]
            h, w, _ = rgb.shape
            rectangles = [[x, y, min(512, w-x), min(512, h-y)]
                          for y in range(0, h, 512) for x in range(0, w, 512)]
            need([r["descriptor"]["rectangle"] for r, _, _ in rows] == rectangles,
                 "COMPLETE_TILES")
            for row, _, blob in rows:
                x, y, tw, th = row["descriptor"]["rectangle"]
                with Image.open(io.BytesIO(blob.raw)) as im:
                    tile = np.asarray(im)
                need(tile.shape == (th, tw, 3) and tile.dtype == np.uint8
                     and tile.tobytes() == rgb[y:y+th, x:x+tw].tobytes()
                     and row["descriptor"]["node"] == decoded[0]["node"]
                     and row["descriptor"]["source_payload"] ==
                         v["records"]["image"][0]["sha256"], "IMAGE_SOURCE")
        else:
            pcm = np.concatenate([x["values"] for x in decoded])
            need([span["channel"] for _, span, _ in rows] == list(range(pcm.shape[1])),
                 "COMPLETE_CHANNELS")
            for row, span, blob in rows:
                wave, rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
                need(wave.tobytes() == pcm.tobytes() and rate == v["formats"]["audio"]["sample_rate"]
                     and span["start"] == 0 and span["end"] == len(pcm)
                     and row["descriptor"]["nodes"] == [x["node"] for x in decoded]
                     and row["descriptor"]["source_payloads"] ==
                         [r["sha256"] for r in v["records"]["audio"]]
                     and row["source"] == v["source"] and row["clock"] == v["clock"]
                     and row["split_group"] == v["split_group"]
                     and row["through"] == v["through"], "AUDIO_SOURCE")
        prefix_pin = (prefix.root/"COMMITTED").read_text()
        need(all(r["prefix_pin"] == prefix_pin for r, _, _ in rows), "PREFIX_BINDING")

    def _source(self, clip):
        need(clip in self._sources, "UNKNOWN_CLIP")
        s = json.loads(self._sources[clip])
        request = s["media"]["image"]["request"]
        return dict(clip=clip, source=s["source"], clock_id=request["clock_id"],
                    split_group=request["split_group"], role=s["role"],
                    bank_report=self.report_pin)

    def _measurement(self, clip, medium, bank_pin, refs):
        s = json.loads(self._sources[clip])
        index = self._indexes[clip, medium]
        return episode("measurement", dict(what="retained_native_candidate_readings",
            who=None, where=None, why=None, medium=medium, readings=refs,
            bank_pin=bank_pin, prefix_pin=s["media"][medium]["pin"],
            scientific_admission=False, training_ready=False, meaning_asserted=False),
            self._source(clip), dict(clock_id=self._source(clip)["clock_id"],
                support=index[3], available_after=index[4]))

    def measurement(self, clip, medium):
        need((clip, medium) in self._measurements, "MISSING_MEDIUM")
        return json.loads(self._measurements[clip, medium])

    def moment(self, clip, at, required_media=("image", "audio")):
        need(tuple(required_media) == ("image", "audio"), "MISSING_TIMED_TEXT_OR_MEDIUM")
        t, members, evidence, availability = rational(at), [], [], []
        for medium in required_media:
            need((clip, medium) in self._indexes, "UNKNOWN_CLIP")
            starts, raw, offsets, _, _ = self._indexes[clip, medium]
            i = bisect_right(starts, t)-1
            need(i >= 0, "NO_TEMPORAL_OVERLAP")
            node = json.loads(raw[i])
            need(t < rational(node["span"][1]), "NO_TEMPORAL_OVERLAP")
            measurement = self.measurement(clip, medium)
            coordinate = dict(node=node, temporal_resolution=pair(
                rational(node["span"][1])-rational(node["span"][0])))
            if medium == "audio":
                rate = Fraction(node["sample_count"], 1)/(
                    rational(node["span"][1])-rational(node["span"][0]))
                need(rate.denominator == 1, "SAMPLE_RATE")
                offset = int((t-starts[i])*rate)
                coordinate.update(sample_in_block=offset,
                    sample_in_prefix=offsets[i]+offset,
                    temporal_resolution=pair(1/rate),
                    sample_time=pair(starts[i]+offset/rate))
            members.append(dict(medium=medium, measurement=measurement["id"],
                                coordinate=coordinate, context=measurement["event_time"]))
            evidence.append(measurement["id"])
            availability.append(rational(measurement["event_time"]["available_after"]))
        return episode("derived_relation", dict(what="source_relative_temporal_constellation",
            who=None, where=None, why=None, at=at, members=members,
            text="absent_no_attested_timed_text", timestamp_representation="exact_rational",
            physical_sync_calibrated=False, measured_alignment_error=None,
            scientific_admission=False, meaning_asserted=False, training_ready=False),
            self._source(clip), dict(clock_id=self._source(clip)["clock_id"], at=at,
                                    available_after=pair(max(availability))), evidence)

    def unresolved(self, moment):
        return episode("interpretation", dict(what="why_unresolved", explanation=None),
                       moment["source"], moment["event_time"], [moment["id"]])

    def validate_episode(self, ep):
        need(type(ep) == dict and set(ep) ==
             {"id", "claim_kind", "content", "source", "event_time", "evidence"}, "EPISODE_SCHEMA")
        clip = ep["source"]["clip"]
        if ep["claim_kind"] == "measurement":
            expected = self.measurement(clip, ep["content"]["medium"])
        elif ep["claim_kind"] == "derived_relation":
            expected = self.moment(clip, ep["content"]["at"])
        elif ep["claim_kind"] == "interpretation":
            expected = self.unresolved(self.moment(clip, ep["event_time"]["at"]))
        else:
            need(False, "UNSUPPORTED_CLAIM")
        need(ep == expected, "EPISODE_BINDING")

    def assess(self, ep):
        self.validate_episode(ep)
        need(ep["claim_kind"] != "interpretation", "WHY_UNRESOLVED")
        body = dict(checker="native-recording-evidence-v1", evidence_pin=self.pin,
            episode=ep["id"], claim_kind=ep["claim_kind"], scope=SCOPE,
            world_validity="unassessed", physical_sync_calibrated=False,
            scientific_admission=False, training_ready=False)
        return dict(id=digest(body), **body)

    def readings(self, moment):
        self.validate_episode(moment)
        need(moment["claim_kind"] == "derived_relation", "MOMENT_REQUIRED")
        result = []
        for member in moment["content"]["members"]:
            measurement = self.measurement(moment["source"]["clip"], member["medium"])
            for ref in measurement["content"]["readings"]:
                meta, raw = self._readings[ref]
                value = json.loads(meta)
                result.append(dict(reference=ref, **value,
                    values=np.frombuffer(raw, dtype="<f4").reshape(value["entry"]["shape"])))
        return tuple(result)
