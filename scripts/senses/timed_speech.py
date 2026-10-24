"""Source-published speech intervals with separate native video, audio and text.

AMI is an adapter, never a new sense. Annotation time is retrospective and its
forced alignment is not a measurement of physical synchronization. PC-04/07/11:
bind sources/indexes once, seek PCM, decode one bounded range per utterance.
"""
from bisect import bisect_left, bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import wave
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from claims.episodic_memory import canonical, digest
from senses.perception_corpus import sha
from senses.perception_decode import runtime

NITE = "{http://nite.sourceforge.net/}"
FLAGS = dict(scientific_admission=False, training_ready=False,
             learned_understanding=False, physical_sync_calibrated=False,
             online_feedback_eligible=False)


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_TIMED_SPEECH_"+code)


def pair(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def rational(value):
    need(type(value) == list and len(value) == 2
         and all(type(v) == int for v in value) and value[1] > 0, "TIME")
    return Fraction(*value)


def save(path, value):
    with Path(path).open("xb") as f:
        f.write(canonical(value))


class AmiSource:
    """Read the recorded selection, without exposing other annotation layers."""

    def __init__(self, root, pin):
        self.root, self.pin = Path(root), pin
        need((self.root/"manifest.json").is_file()
             and sha(self.root/"manifest.json") == pin, "SOURCE_MANIFEST")
        self.manifest = json.loads((self.root/"manifest.json").read_bytes())
        m = self.manifest
        need(m["schema"] == "ami-source-v1" and m["meeting"] == "ES2002a"
             and m["agents"] == ["A", "B"]
             and m["selection"] == "first-two-provider-segments-per-agent"
             and m["license"] == "CC-BY-4.0"
             and m["physical_sync_calibrated"] is False, "SOURCE_SCOPE")
        for name, row in m["files"].items():
            need(Path(name).name == name, "SOURCE_PATH")
            p = self.root/name
            need(p.is_file() and p.stat().st_size == row["bytes"]
                 and sha(p) == row["sha256"], "SOURCE_BYTES:"+name)
        self._events, self.counts = {}, {}
        self.quality = json.loads((self.root/"native-census.json").read_bytes())
        archive = self.root/"ami_public_manual_1.6.2.zip"
        with zipfile.ZipFile(archive) as z:
            meetings = ET.fromstring(z.read("corpusResources/meetings.xml"))
            matches = [e for e in meetings if e.get("observation") == m["meeting"]]
            need(len(matches) == 1, "MEETING")
            meeting = matches[0]
            self.mapping_pin = hashlib.sha256(z.read("corpusResources/meetings.xml")).hexdigest()
            for agent in m["agents"]:
                people = [e for e in meeting if e.get("nxt_agent") == agent]
                need(len(people) == 1, "SPEAKER")
                person = dict(people[0].attrib)
                word_name, seg_name = (f"{folder}/ES2002a.{agent}.{suffix}.xml"
                    for folder, suffix in (("words", "words"), ("segments", "segments")))
                word_raw, seg_raw = z.read(word_name), z.read(seg_name)
                words, segments = list(ET.fromstring(word_raw)), list(ET.fromstring(seg_raw))
                positions = {e.get(NITE+"id"): i for i, e in enumerate(words)}
                need(None not in positions and len(positions) == len(words), "WORD_IDS")
                self.counts[agent] = dict(nodes=len(words), segments=len(segments))
                video = f"ES2002a.{person['camera']}.avi"
                audio = f"ES2002a.Headset-{person['channel']}.wav"
                need(video in m["files"] and audio in m["files"], "CHANNEL_MAPPING")
                probe = json.loads((self.root/(video+".frames.json")).read_bytes())
                need(len(probe["streams"]) == 1, "VIDEO_STREAM")
                stream = probe["streams"][0]
                tb = Fraction(stream["time_base"])
                frames = probe["frames"]
                need(stream["codec_type"] == "video" and tb > 0 and frames
                     and all(type(f.get("pts")) == int and type(f.get("duration")) == int
                             and f["duration"] > 0 for f in frames)
                     and all(a["pts"]+a["duration"] <= b["pts"]
                             for a, b in zip(frames, frames[1:])), "VIDEO_TIMELINE")
                starts = [f["pts"]*tb for f in frames]
                ends = [(f["pts"]+f["duration"])*tb for f in frames]
                with wave.open(str(self.root/audio)) as w:
                    fmt = dict(rate=w.getframerate(), channels=w.getnchannels(),
                               width=w.getsampwidth(), samples=w.getnframes())
                    need(w.getcomptype() == "NONE" and fmt["width"] == 2
                         and fmt["channels"] == 1, "PCM_FORMAT")
                for segment in segments[:2]:
                    need(segment.get("channel") == person["channel"]
                         and len(segment) == 1, "SEGMENT_CHANNEL")
                    href = segment[0].get("href")
                    pattern = re.escape(Path(word_name).name)+r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?"
                    match = re.fullmatch(pattern, href or "")
                    need(match is not None, "SEGMENT_REFERENCE")
                    first, last = match.group(1), match.group(2) or match.group(1)
                    need(first in positions and last in positions
                         and positions[first] <= positions[last], "SEGMENT_WORDS")
                    chosen = words[positions[first]:positions[last]+1]
                    rows = []
                    for e in chosen:
                        lo, hi = Fraction(e.get("starttime")), Fraction(e.get("endtime"))
                        need(0 <= lo <= hi, "WORD_TIME")
                        rows.append(dict(tag=e.tag, attributes=dict(e.attrib),
                                         text=e.text, span=[pair(lo), pair(hi)]))
                    segment_span = [Fraction(segment.get(k))
                                    for k in ("transcriber_start", "transcriber_end")]
                    lo = min([segment_span[0]]+[rational(r["span"][0]) for r in rows])
                    hi = max([segment_span[1]]+[rational(r["span"][1]) for r in rows])
                    need(0 <= lo < hi, "SEGMENT_TIME")
                    begin, end = bisect_right(ends, lo), bisect_left(starts, hi)
                    selected = [dict(ordinal=i, **frames[i]) for i in range(begin, end)]
                    need(selected and selected[0]["pts"]*tb <= lo
                         and (selected[-1]["pts"]+selected[-1]["duration"])*tb >= hi
                         and all(a["pts"]+a["duration"] == b["pts"]
                                 for a, b in zip(selected, selected[1:])), "VIDEO_COVERAGE")
                    start = (lo*fmt["rate"]).__floor__()
                    stop = (hi*fmt["rate"]).__ceil__()
                    need(0 <= start < stop <= fmt["samples"], "AUDIO_COVERAGE")
                    clock = digest(["ami-published-common-timeline", pin, m["meeting"]])
                    body = dict(meeting=m["meeting"], agent=agent, speaker=person,
                        segment=segment.attrib, source=pin, clock_id=clock,
                        split_group=digest(["ami-recording-session", pin, "ES2002"]),
                        transcriber_span=[pair(t) for t in segment_span],
                        support=[pair(lo), pair(hi)],
                        # Conservative source-relative bound; not an online transcript.
                        available_after=pair(max(Fraction(fmt["samples"], fmt["rate"]),
                            (frames[-1]["pts"]+frames[-1]["duration"])*tb)),
                        timing=dict(authority="AMI published common timeline",
                            word_method="provider_forced_alignment", measured_error=None,
                            availability="retrospective_full_recording_annotation"),
                        text=dict(words=rows, reference=href, words_file=word_name,
                            segments_file=seg_name, words_pin=hashlib.sha256(word_raw).hexdigest(),
                            segments_pin=hashlib.sha256(seg_raw).hexdigest(),
                            surface=" ".join(e.text or "" for e in chosen if e.tag == "w"),
                            surface_method="space_joined_orthographic_words",
                            independent_audio_witness=False),
                        image=dict(file=video, pin=m["files"][video]["sha256"],
                            frames=selected, time_base=pair(tb),
                            width=stream["width"], height=stream["height"]),
                        audio=dict(file=audio, pin=m["files"][audio]["sha256"],
                            start_sample=start, stop_sample=stop, **fmt), flags=FLAGS,
                        quality=dict(native_census=m["files"]["native-census.json"]["sha256"],
                            provider_problems=m["files"]["problems.html"]["sha256"],
                            complete_media_quality_certified=False))
                    key = segment.get(NITE+"id")
                    need(key not in self._events, "DUPLICATE_SEGMENT")
                    self._events[key] = canonical(dict(id=digest(body), **body))

    def events(self):
        return tuple(json.loads(raw) for raw in self._events.values())

    def event(self, segment):
        need(segment in self._events, "UNKNOWN_SEGMENT")
        return json.loads(self._events[segment])


def decode(source, event, output):
    """Persist every selected pixel/sample; strict prefix decode, no resampling."""
    need(event == source.event(event["segment"][NITE+"id"]), "EVENT_BINDING")
    output = Path(output)
    output.mkdir(exist_ok=False)
    image, audio = event["image"], event["audio"]
    first, last = image["frames"][0]["ordinal"], image["frames"][-1]["ordinal"]
    decoder = runtime()
    cmd = [decoder["executable"], "-nostdin", "-hide_banner", "-nostats",
           "-loglevel", "info", "-xerror", "-err_detect", "explode", "-threads", "0",
           "-copyts", "-noautorotate", "-i", str(source.root/image["file"]),
           "-map", "0:0", "-vf", f"select=between(n\\,{first}\\,{last}),format=rgb24,showinfo",
           "-frames:v", str(len(image["frames"])), "-fps_mode", "passthrough",
           "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "rawvideo", "pipe:1"]
    with (output/"frames.rgb").open("xb") as out, (output/"decode.log").open("xb") as log:
        r = subprocess.run(cmd, stdout=out, stderr=log)
    need(r.returncode == 0, "VIDEO_DECODE")
    log = (output/"decode.log").read_text()
    actual = [int(v) for v in re.findall(r"Parsed_showinfo_[^\n]*\bn:\s*\d+\s+pts:\s*(-?\d+)", log)]
    need(actual == [f["pts"] for f in image["frames"]], "DECODE_COORDINATES")
    expected = len(actual)*image["width"]*image["height"]*3
    need((output/"frames.rgb").stat().st_size == expected, "PIXEL_COUNT")
    with wave.open(str(source.root/audio["file"])) as w:
        w.setpos(audio["start_sample"])
        raw = w.readframes(audio["stop_sample"]-audio["start_sample"])
    need(len(raw) == (audio["stop_sample"]-audio["start_sample"])*2, "SAMPLE_COUNT")
    (output/"audio.pcm").write_bytes(raw)
    save(output/"text.json", event["text"])
    manifest = dict(event=event, source=source.pin, runtime=decoder, command=cmd,
        code=sha(Path(__file__)), files={n: dict(bytes=(output/n).stat().st_size,
            sha256=sha(output/n)) for n in ("frames.rgb", "audio.pcm", "text.json", "decode.log")},
        flags=FLAGS)
    save(output/"manifest.json", manifest)
    return dict(directory=output.name, pin=sha(output/"manifest.json"))


class SpeechViews:
    def __init__(self, source, root, pin):
        self.source, self.root, self.pin = source, Path(root), pin
        need((self.root/"manifest.json").is_file()
             and sha(self.root/"manifest.json") == pin, "VIEW_MANIFEST")
        m = json.loads((self.root/"manifest.json").read_bytes())
        need(type(m) == dict and set(m) ==
             {"event", "source", "runtime", "command", "code", "files", "flags"}, "VIEW_SCHEMA")
        need(m["source"] == source.pin and m["code"] == sha(Path(__file__))
             and m["flags"] == FLAGS and m["event"] ==
             source.event(m["event"]["segment"][NITE+"id"]), "VIEW_BINDING")
        for name, row in m["files"].items():
            need(name in ("frames.rgb", "audio.pcm", "text.json", "decode.log")
                 and (self.root/name).is_file()
                 and (self.root/name).stat().st_size == row["bytes"]
                 and sha(self.root/name) == row["sha256"], "VIEW_BYTES")
        need(set(m["files"]) == {"frames.rgb", "audio.pcm", "text.json", "decode.log"},
             "VIEW_FILES")
        need(json.loads((self.root/"text.json").read_bytes()) == m["event"]["text"], "TEXT_BYTES")
        self._manifest = canonical(m)

    @property
    def manifest(self):
        return json.loads(self._manifest)

    def read(self):
        """Consumer owns immutable arrays; later source withdrawal invalidates use."""
        m, event = self.manifest, self.manifest["event"]
        payloads = {}
        for medium, name, dtype, shape in (
            ("image", "frames.rgb", "u1",
             (len(event["image"]["frames"]), event["image"]["height"], event["image"]["width"], 3)),
            ("audio", "audio.pcm", "<i2",
             (event["audio"]["stop_sample"]-event["audio"]["start_sample"], 1))):
            raw = (self.root/name).read_bytes()
            need(hashlib.sha256(raw).hexdigest() == m["files"][name]["sha256"], "CONSUMER_BYTES")
            payloads[medium] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        raw = (self.root/"text.json").read_bytes()
        need(hashlib.sha256(raw).hexdigest() == m["files"]["text.json"]["sha256"], "CONSUMER_TEXT")
        return dict(image=payloads["image"], audio=payloads["audio"], text=json.loads(raw),
                    provenance=json.loads(canonical(event)), flags=dict(FLAGS))
