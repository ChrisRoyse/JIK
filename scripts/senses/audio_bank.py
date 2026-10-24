"""Native audio candidate bank; separate instruments, channels and complete windows."""
import json
from pathlib import Path

import numpy as np

from native_audio import SPECS, geometry
from senses.contract import digest
from senses.perception_instruments import Blob, sha

NAMES = ("whisper", "wavlm", "hubert", "ast", "clap")
FLAGS = dict(scientific_admission=False, meaning_asserted=False, training_ready=False,
             candidate_readings_only=True, no_flatten=True, physical_sync_calibrated=False)


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_AUDIO_BANK_"+code)


def input_views(root, pin):
    root = Path(root)
    raw = (root/"manifest.json").read_bytes()
    need(sha(raw) == pin, "INPUT_PIN")
    manifest = json.loads(raw)
    need(manifest["schema"] == "native-audio-witness-input-v1", "INPUT_SCHEMA")
    blobs, views = {}, []
    for row in manifest["rows"]:
        name = row["blob"]
        need(Path(name).name == name, "INPUT_PATH")
        if name not in blobs:
            blobs[name] = Blob((root/name).read_bytes(), "audio/wav")
        blob = blobs[name]
        need(blob.pin == row["blob_sha256"] == row["span"]["sha256"], "INPUT_BYTES")
        views.append((row, row["span"], blob))
    need(len({(x[0]["clip"], x[1]["channel"]) for x in views}) == len(views), "DUPLICATE_CHANNEL")
    return views


def native_windows(name, span):
    rate, seconds = (16000, 30) if name == "whisper" else (SPECS[name]["rate"], 10)
    n = ((span["end"]-span["start"])*rate+span["rate"]-1)//span["rate"]
    windows, offset = [], 0
    for start in range(0, n, rate*seconds):
        end = min(n, start+rate*seconds)
        if name == "whisper":
            count = (end-start+319)//320
            window = dict(**span, resampled_start=start, resampled_end=end,
                token_start=offset, token_end=offset+count,
                original_coordinate_mapping="span.start + resampled_index * rate / 16000")
        else:
            g = geometry(name, end-start)
            count = g["shape"][0]
            window = dict(**span, **g, target_rate=rate, resampled_start=start, resampled_end=end,
                token_start=offset, token_end=offset+count,
                original_coordinate_mapping="span.start + resampled_index * source_rate / target_rate")
        windows.append(window)
        offset += count
    return windows


def shape(name, span):
    return [native_windows(name, span)[-1]["token_end"], 384 if name == "whisper" else 768]


def store(root, name, identity, row, output):
    values, windows = output
    need(name in NAMES and list(values.shape) == shape(name, row["span"])
         and values.dtype == np.float32 and bool(np.isfinite(values).all()), "NATIVE_SHAPE")
    need(windows == native_windows(name, row["span"]), "NATIVE_WINDOWS")
    raw = values.astype("<f4", copy=False).tobytes()
    filename = digest(dict(instrument=identity, input=row))+".bin"
    path = Path(root)/filename
    with path.open("xb") as f:
        f.write(raw)
    need(path.read_bytes() == raw, "PERSISTENCE")
    return dict(instrument_id=identity, input=row, file=filename, shape=list(values.shape),
                sha256=sha(raw), native=windows)


class AudioBank:
    def __init__(self, root, pin, inputs, input_pin, identities):
        self.root = Path(root)
        raw = (self.root/"bank.json").read_bytes()
        need(sha(raw) == pin, "PIN")
        self.value = v = json.loads(raw)
        need(v["schema"] == "native-audio-bank-v1" and v["flags"] == FLAGS, "FLAGS")
        need(v["input_pin"] == input_pin, "SOURCE")
        need(set(v["instruments"]) == set(v["readings"]) == set(identities) == set(NAMES), "ROSTER")
        need({n: digest(x) for n, x in v["instruments"].items()} == identities, "IDENTITIES")
        views = [x[0] for x in input_views(inputs, input_pin)]
        for name in NAMES:
            need([x["input"] for x in v["readings"][name]] == views, "CHANNEL_ROSTER")
            for e in v["readings"][name]:
                need(e["instrument_id"] == identities[name], "SWAPPED_INSTRUMENT")
                need(e["shape"] == shape(name, e["input"]["span"]), "NATIVE_SHAPE")
                need(e["file"] == digest(dict(instrument=identities[name], input=e["input"]))+".bin",
                     "FILE_KEY")
                need(e["native"] == native_windows(name, e["input"]["span"]), "NATIVE_WINDOWS")

    def consume(self, purpose="engineering_boundary"):
        need(purpose == "engineering_boundary", "NATIVE_ADMISSION_REQUIRED")
        for name in NAMES:
            for e in self.value["readings"][name]:
                raw = (self.root/e["file"]).read_bytes()
                need(len(raw) == np.prod(e["shape"])*4 and sha(raw) == e["sha256"], "READING_BYTES")
                values = np.frombuffer(raw, dtype="<f4").reshape(e["shape"])
                need(bool(np.isfinite(values).all()), "NONFINITE")
                yield name, e["input"], values
