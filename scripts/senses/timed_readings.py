"""Timed speech observations through existing, separately typed frozen instruments.

PC-07/35: prepare native wrappers once per source view, load a lens once per bank.
This is candidate measurement infrastructure; it cannot authorize training.
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from claims.timed_speech_memory import SpeechEvidence
from senses import audio_bank, visual_bank
from senses.contract import CHANNELS, absent, canonical, digest, field, present, seal_record
from senses.perception_corpus import pinned, sha
from senses.perception_instruments import Blob, audio_input
from senses.timed_speech import AmiSource, FLAGS, SpeechViews

TEXT = ("all-mpnet-base-v2", "codebert-mean", "e5-base-v2",
        "finbert-mean", "pubmedbert-embeddings", "specter2-base")
ROSTER = tuple("image_"+n for n in visual_bank.NAMES) + tuple(
    "audio_"+n for n in audio_bank.NAMES) + tuple("text_"+n for n in TEXT)
SCHEMA = "timed-speech-candidate-readings-v1"


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_TIMED_READINGS_"+code)


class Inputs:
    def __init__(self, root, pin, source):
        self.root, self.report_pin = Path(root), pin
        self.report = report = pinned(self.root/"report.json", pin)
        need(report["flags"] == FLAGS, "SOURCE_FLAGS")
        self.source = AmiSource(source, report["source_pin"])
        self.views = [SpeechViews(self.source, self.root/r["directory"], r["pin"])
                      for r in report["views"]]
        self.evidence = SpeechEvidence(self.source, self.views)
        self.moments = report["moments"]
        need(len(self.moments) == len(self.views)
             and len({m["source"]["segment"] for m in self.moments}) == len(self.views),
             "MOMENT_ROSTER")
        for moment in self.moments:
            self.evidence.validate_episode(moment)
        self.items = dict(image=[], audio=[], text=[])
        for view in self.views:
            payload = view.read()
            ev = payload["provenance"]
            key = ev["segment"]["{http://nite.sourceforge.net/}id"]
            common = dict(segment=key, view=view.pin, source=self.source.pin,
                          clock_id=ev["clock_id"], split_group=ev["split_group"],
                          support=ev["support"], available_after=ev["available_after"])
            for frame, rgb in zip(ev["image"]["frames"], payload["image"], strict=True):
                height, width = rgb.shape[:2]
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format="PNG")
                blob = Blob(stream.getvalue(), "image/png")
                with Image.open(io.BytesIO(blob.raw)) as image:
                    need(np.asarray(image).tobytes() == rgb.tobytes(), "PNG_LOSS")
                row = dict(**common, frame=frame, time_base=ev["image"]["time_base"],
                    descriptor=dict(rectangle=[0, 0, width, height]),
                    blob_sha256=blob.pin, pixels_sha256=Blob(rgb.tobytes(), "RGB").pin)
                item = dict(modality="image", sha256=blob.pin, media_type=blob.media,
                    measurements=[field("width", width, "integer"), field("height", height, "integer"),
                                  field("channels", 3, "integer"), field("layout", "RGB", "category")])
                record = dict(schema="ucr-v1", unit=ev["split_group"], referents=[],
                    locator=dict(source="ami-native", shard=key, row=frame["ordinal"],
                        span=[0, width*height], record_digest=digest(row),
                        manifest_digest=self.source.pin),
                    channels={n: absent("not_observed") for n in CHANNELS})
                record["channels"]["perception"] = present([item])
                self.items["image"].append((row, seal_record(record), blob))
            # PCM16 -> float32 / 32768 is exact. No channel mixing or resampling.
            pcm = payload["audio"].astype(np.float32)/np.float32(32768)
            need(np.array_equal((pcm*32768).astype("<i2"), payload["audio"]), "PCM_LOSS")
            span, blob = audio_input(pcm, dict(samples=len(pcm), channels=1,
                                               rate=ev["audio"]["rate"], channel=0))
            row = dict(**common, span=span, blob_sha256=blob.pin,
                source_audio=ev["audio"], sample_units="PCM16 / 32768",
                original_sample_mapping="source_audio.start_sample + wrapper_sample")
            self.items["audio"].append((row, span, blob))
            row = dict(**common, text=payload["text"],
                       audio_dependency=self.evidence.parents(key)[1]["id"])
            self.items["text"].append((row, payload["text"]["surface"], None))
        self.rows = {m: [r for r, _, _ in items] for m, items in self.items.items()}
        self.pin = digest(dict(source=self.source.pin, report=pin, rows=self.rows,
                               code=sha(Path(__file__))))
        self.by_key = {digest(row): (row, value, blob)
                       for items in self.items.values() for row, value, blob in items}

    def selected(self, medium):
        if medium != "image":
            return self.items[medium]
        frames = {(m["source"]["segment"], m["content"]["image"]["frame"]["ordinal"])
                  for m in self.moments}
        return [item for item in self.items["image"]
                if (item[0]["segment"], item[0]["frame"]["ordinal"]) in frames]


def text_contract(lens):
    from text import window_readings as runtime
    import torch
    return {**lens.contract(), "native_runtime": runtime.RUNTIME, "frozen": True,
        "window_reduction": "one vector per complete tokenizer window; no window pooling",
        "execution_unit": "one full source utterance per forward",
        "shape": f"MultiWindows({lens.dim})", "kind": "dense-window-multi",
        "native_code": sha(Path(runtime.__file__)), "binding_code": sha(Path(__file__)),
        "torch": torch.__version__, "transformers": runtime.transformers.__version__,
        "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0),
        "scientific_admission": False}


def text_store(root, identity, row, output):
    values, offsets, traces = output
    need(list(offsets) == [0, len(values)] and len(traces) == 1, "TEXT_OFFSETS")
    need(values.dtype == np.float32 and bool(np.isfinite(values).all()), "TEXT_VALUES")
    raw = values.astype("<f4").tobytes()
    path = Path(root)/(digest(dict(instrument=identity, input=row))+".bin")
    with path.open("xb") as f:
        f.write(raw)
    need(path.read_bytes() == raw, "PERSISTENCE")
    return dict(instrument_id=identity, input=row, file=path.name, sha256=sha(path),
                shape=list(values.shape), native=traces[0])


class Bank:
    def __init__(self, root, pin, inputs, identities):
        self.root, self.inputs = Path(root), inputs
        v = pinned(self.root/"bank.json", pin)
        need(v["schema"] == SCHEMA and v["flags"] == FLAGS, "FLAGS")
        need(v["input_pin"] == inputs.pin and v["binding_code"] == sha(Path(__file__)),
             "SOURCE")
        need(set(v["instruments"]) == set(v["readings"]) == set(identities) == set(ROSTER),
             "ROSTER")
        need({n: digest(c) for n, c in v["instruments"].items()} == identities, "IDENTITIES")
        self.entries = {}
        for lane in ROSTER:
            medium, name = lane.split("_", 1)
            entries = v["readings"][lane]
            need([e["input"] for e in entries] == inputs.rows[medium], "INPUT_ROSTER")
            for entry in entries:
                need(set(entry) == {"instrument_id", "input", "file", "sha256", "shape", "native"},
                     "ENTRY_SCHEMA")
                need(entry["instrument_id"] == identities[lane], "INSTRUMENT")
                need(entry["file"] == digest(dict(instrument=identities[lane],
                                                   input=entry["input"]))+".bin", "FILE_KEY")
                if medium == "image":
                    need(entry["shape"] == visual_bank.expected(name, entry["input"])
                         and entry["native"] == visual_bank.native_window(name, entry["input"]),
                         "NATIVE_LAYOUT")
                elif medium == "audio":
                    need(entry["shape"] == audio_bank.shape(name, entry["input"]["span"])
                         and entry["native"] == audio_bank.native_windows(name, entry["input"]["span"]),
                         "NATIVE_LAYOUT")
                else:
                    import text_windows
                    text_windows.validate(entry["native"], "dense", entry["shape"][0],
                                          text=entry["input"]["text"]["surface"])
                    need(entry["shape"][1] == 768 and 0 < entry["shape"][0] <= 8, "TEXT_LAYOUT")
                self.entries[(lane, digest(entry["input"]))] = canonical(entry)
        self._manifest, self.pin = canonical(v), pin

    @property
    def manifest(self):
        return json.loads(self._manifest)

    def read(self, lane, row):
        need((lane, digest(row)) in self.entries, "UNKNOWN_READING")
        entry = json.loads(self.entries[(lane, digest(row))])
        path = self.root/lane/entry["file"]
        need(path.is_file(), "READING_BYTES")
        raw = path.read_bytes()
        need(len(raw) == int(np.prod(entry["shape"]))*4
             and Blob(raw, "native/f32").pin == entry["sha256"], "READING_BYTES")
        values = np.frombuffer(raw, dtype="<f4").reshape(entry["shape"])
        need(bool(np.isfinite(values).all()), "NONFINITE")
        return entry, values

    def streams(self, memory, moment, *, purpose="engineering_boundary", **scope):
        from model.native_stream_input import ReadingStream
        need(purpose == "engineering_boundary", "ADMISSION_REQUIRED")
        need(memory.evidence.pin == self.inputs.evidence.pin, "MEMORY_BINDING")
        retained = memory.consume(moment, purpose=purpose, **scope)
        ep = retained["moment"]
        result = []
        for lane in ROSTER:
            medium = lane.split("_", 1)[0]
            selected = [r for r in self.inputs.rows[medium]
                        if r["segment"] == ep["source"]["segment"]
                        and (medium != "image" or r["frame"] == ep["content"]["image"]["frame"])]
            need(len(selected) == 1, "MOMENT_READING")
            entry, values = self.read(lane, selected[0])
            metadata = canonical(dict(bank=self.pin, moment=ep, entry=entry,
                scope=scope, memory_head=retained["head"], flags=FLAGS))
            result.append(ReadingStream(digest([self.pin, lane, entry["file"]]),
                lane, entry["instrument_id"], metadata, values, memory, retained["head"]))
        need(memory.retained()["head"] == retained["head"], "STALE_CONSUMER")
        return tuple(result)
