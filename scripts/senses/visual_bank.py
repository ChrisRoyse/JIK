"""Typed native visual candidate bank: separate identities, windows and dimensions."""
import json
from pathlib import Path

import numpy as np

from native_vision import geometry
from senses.contract import canonical, digest, validate_record
from senses.perception_instruments import Blob, sha

NAMES = ("dinov2", "siglip2", "mae", "sam", "convnext")
FLAGS = dict(scientific_admission=False, meaning_asserted=False, training_ready=False,
             candidate_readings_only=True, no_flatten=True)


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_VISUAL_BANK_"+code)


def input_views(root, pin):
    root = Path(root)
    raw = (root/"manifest.json").read_bytes()
    need(sha(raw) == pin, "INPUT_PIN")
    manifest = json.loads(raw)
    need(manifest["schema"] == "native-vision-witness-input-v1", "INPUT_SCHEMA")
    views = []
    for row in manifest["rows"]:
        need(all(Path(row[k]).name == row[k] for k in ("record", "blob")), "INPUT_PATH")
        data, image = (root/row["record"]).read_bytes(), (root/row["blob"]).read_bytes()
        need(sha(data) == row["record_sha256"] and sha(image) == row["blob_sha256"], "INPUT_BYTES")
        record, blob = json.loads(data), Blob(image, "image/png")
        validate_record(record)
        need(record["channels"]["perception"]["items"][0]["sha256"] == blob.pin, "INPUT_BINDING")
        views.append((row, record, blob))
    need(len({(x[0]["clip"], x[0]["ordinal"]) for x in views}) == len(views), "DUPLICATE_INPUT")
    return views


def expected(name, row):
    _, _, width, height = row["descriptor"]["rectangle"]
    if name == "dinov2":
        return [1+((height+13)//14)*((width+13)//14), 384]
    return geometry(name, height, width)["shape"]


def native_window(name, row):
    _, _, width, height = row["descriptor"]["rectangle"]
    if name == "dinov2":
        ph, pw = (height+13)//14*14, (width+13)//14*14
        g = dict(height=height, width=width, padded_height=ph, padded_width=pw,
                 grid=[ph//14, pw//14], patch=14, order="CLS then row-major spatial patches",
                 padding="right/bottom zero RGB before normalization")
    else:
        g = geometry(name, height, width)
    return [dict(**g, sha256=row["blob_sha256"], token_start=0, token_end=expected(name, row)[0])]


def store(root, name, identity, view, output):
    root = Path(root)
    need(name in NAMES, "INSTRUMENT")
    values = output["tokens"]
    need(output["kind"] == "Multi" and list(values.shape) == expected(name, view)
         and values.dtype == np.float32 and bool(np.isfinite(values).all()), "NATIVE_SHAPE")
    need(output["regions"] == native_window(name, view), "NATIVE_WINDOW")
    key = digest(dict(instrument=identity, input=view))
    path = root/(key+".bin")
    raw = values.astype("<f4", copy=False).tobytes()
    with path.open("xb") as f:
        f.write(raw)
    need(path.read_bytes() == raw, "PERSISTENCE")
    return dict(instrument_id=identity, input=view, file=path.name,
                shape=list(values.shape), sha256=sha(raw), native=output["regions"])


class VisualBank:
    def __init__(self, root, pin, inputs, input_pin, identities):
        self.root = Path(root)
        raw = (self.root/"bank.json").read_bytes()
        need(sha(raw) == pin, "PIN")
        self.value = json.loads(raw)
        v = self.value
        need(v["schema"] == "native-visual-bank-v1" and v["flags"] == FLAGS, "FLAGS")
        need(v["input_pin"] == input_pin, "SOURCE")
        need(set(v["instruments"]) == set(v["readings"]) == set(identities) == set(NAMES), "ROSTER")
        need({n: digest(x) for n, x in v["instruments"].items()} == identities, "IDENTITIES")
        views = [x[0] for x in input_views(inputs, input_pin)]
        for name in NAMES:
            need([x["input"] for x in v["readings"][name]] == views, "WINDOW_ROSTER")
            for entry in v["readings"][name]:
                need(entry["instrument_id"] == identities[name], "SWAPPED_INSTRUMENT")
                need(entry["shape"] == expected(name, entry["input"]), "NATIVE_SHAPE")
                need(Path(entry["file"]).name == entry["file"]
                     and entry["file"] == digest(dict(instrument=identities[name], input=entry["input"]))+".bin",
                     "FILE_KEY")
                need(entry["native"] == native_window(name, entry["input"]), "NATIVE_WINDOW")

    def consume(self, purpose="engineering_boundary"):
        need(purpose == "engineering_boundary", "NATIVE_ADMISSION_REQUIRED")
        for name in NAMES:
            for entry in self.value["readings"][name]:
                raw = (self.root/entry["file"]).read_bytes()
                need(len(raw) == np.prod(entry["shape"])*4 and sha(raw) == entry["sha256"], "READING_BYTES")
                values = np.frombuffer(raw, dtype="<f4").reshape(entry["shape"])
                need(bool(np.isfinite(values).all()), "NONFINITE")
                yield name, entry["input"], values
