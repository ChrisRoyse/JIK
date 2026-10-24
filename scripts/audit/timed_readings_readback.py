"""Independent persisted-byte audit of speech sources, wrappers and native slots."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import wave
import zipfile
from xml.dom import minidom

STARTED = time.perf_counter()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from PIL import Image
from senses.contract import digest
from senses.perception_corpus import pinned, sha
from senses.timed_speech import FLAGS, save


def need(ok, label):
    if not ok:
        raise ValueError("MNLM_TIMED_READBACK_"+label)


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    bank = pinned(a.bank/"bank.json", a.bank_pin)
    report = pinned(a.inputs/"report.json", a.input_pin)
    source = pinned(a.source/"manifest.json", report["source_pin"])
    source_bytes = 0
    for name, row in source["files"].items():
        need(Path(name).name == name and sha(a.source/name) == row["sha256"]
             and (a.source/name).stat().st_size == row["bytes"], "ORIGINAL_BYTES")
        source_bytes += row["bytes"]
    views = {}
    for row in report["views"]:
        root = a.inputs/row["directory"]
        manifest = pinned(root/"manifest.json", row["pin"])
        for name, entry in manifest["files"].items():
            need(Path(name).name == name and sha(root/name) == entry["sha256"], "VIEW_BYTES")
        views[row["directory"]] = (root, manifest, row["pin"])
    phases = dict(original_and_view_bytes=time.perf_counter()-STARTED)
    tick = time.perf_counter()
    input_rows = json.loads((a.bank/"inputs.json").read_bytes())
    need(input_rows["input_pin"] == bank["input_pin"] and bank["flags"] == FLAGS, "INPUT_PIN")
    need(digest(dict(source=report["source_pin"], report=a.input_pin,
        rows=input_rows["rows"], code=bank["binding_code"])) == bank["input_pin"], "INPUT_BINDING")
    with zipfile.ZipFile(a.source/"ami_public_manual_1.6.2.zip") as archive:
        word_documents = {}
        for medium, rows in input_rows["rows"].items():
            for row in rows:
                root, manifest, pin = views[row["segment"]]
                event = manifest["event"]
                need(row["view"] == pin and row["source"] == report["source_pin"]
                     and row["clock_id"] == event["clock_id"]
                     and row["split_group"] == event["split_group"]
                     and row["support"] == event["support"]
                     and row["available_after"] == event["available_after"], "CLOCK_LINEAGE")
                if medium == "image":
                    frame = row["frame"]
                    index = event["image"]["frames"].index(frame)
                    height, width = event["image"]["height"], event["image"]["width"]
                    count = height*width*3
                    with (root/"frames.rgb").open("rb") as f:
                        f.seek(index*count)
                        raw = f.read(count)
                    need(len(raw) == count and hash_bytes(raw) == row["pixels_sha256"]
                         and row["time_base"] == event["image"]["time_base"], "PIXEL_COORDINATE")
                    stream = io.BytesIO()
                    Image.fromarray(np.frombuffer(raw, "u1").reshape(height, width, 3)).save(
                        stream, format="PNG")
                    need(hash_bytes(stream.getvalue()) == row["blob_sha256"], "LOSSLESS_PNG")
                elif medium == "audio":
                    spec = event["audio"]
                    with wave.open(str(a.source/spec["file"])) as w:
                        w.setpos(spec["start_sample"])
                        raw = w.readframes(spec["stop_sample"]-spec["start_sample"])
                    need(raw == (root/"audio.pcm").read_bytes()
                         and row["source_audio"] == spec, "ORIGINAL_PCM")
                    values = np.frombuffer(raw, "<i2").astype("<f4")/np.float32(32768)
                    pcm = values.tobytes()
                    wrapper = (b"RIFF"+struct.pack("<I", 36+len(pcm))+b"WAVEfmt "+
                        struct.pack("<IHHIIHH", 16, 3, 1, spec["rate"], spec["rate"]*4, 4, 32)
                        +b"data"+struct.pack("<I", len(pcm))+pcm)
                    need(hash_bytes(wrapper) == row["blob_sha256"] == row["span"]["sha256"]
                         and row["span"]["start"] == 0 and row["span"]["end"] == len(values),
                         "EXACT_FLOAT_WRAPPER")
                else:
                    text = row["text"]
                    path = text["words_file"]
                    if path not in word_documents:
                        doc = minidom.parseString(archive.read(path))
                        word_documents[path] = {
                            e.getAttributeNS("http://nite.sourceforge.net/", "id"): e
                            for e in doc.documentElement.childNodes if e.nodeType == e.ELEMENT_NODE}
                    strings = []
                    for word in text["words"]:
                        node = word_documents[path][word["attributes"]["{http://nite.sourceforge.net/}id"]]
                        observed = "".join(c.data for c in node.childNodes if c.nodeType == c.TEXT_NODE)
                        need((observed or None) == word["text"], "ORIGINAL_WORD")
                        if node.tagName == "w":
                            strings.append(observed)
                    need(" ".join(strings) == text["surface"] == event["text"]["surface"]
                         and text["independent_audio_witness"] is False, "DEPENDENT_TRANSCRIPT")
    phases["independent_wrappers_and_words"] = time.perf_counter()-tick
    tick = time.perf_counter()
    counts, total = {}, 0
    for lane, entries in bank["readings"].items():
        medium, name = lane.split("_", 1)
        need([e["input"] for e in entries] == input_rows["rows"][medium], "NATIVE_ROSTER")
        identity = digest(bank["instruments"][lane])
        count = 0
        for entry in entries:
            row = entry["input"]
            need(entry["instrument_id"] == identity
                 and entry["file"] == digest(dict(instrument=identity, input=row))+".bin", "SLOT_KEY")
            raw = (a.bank/lane/entry["file"]).read_bytes()
            shape = entry["shape"]
            need(hash_bytes(raw) == entry["sha256"] and len(raw) == shape[0]*shape[1]*4,
                 "NATIVE_BYTES")
            if medium == "image":
                _, _, w, h = row["descriptor"]["rectangle"]
                stride = dict(dinov2=14, siglip2=16, mae=16, sam=16, convnext=32)[name]
                ph, pw = (1024, 1024) if name == "sam" else (
                    ((h+stride-1)//stride)*stride, ((w+stride-1)//stride)*stride)
                expected = [(ph//stride)*(pw//stride)+(name in ("dinov2", "mae")),
                            dict(dinov2=384, siglip2=768, mae=768, sam=256, convnext=768)[name]]
                need(shape == expected, "SPATIAL_ROWS")
            elif medium == "audio":
                samples = row["span"]["end"]
                n = samples
                if name in ("wavlm", "hubert"):
                    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
                        n = (n-kernel)//stride+1
                else:
                    n = {"whisper": (samples+319)//320, "ast": 1214, "clap": 64}[name]
                need(shape == [n, 384 if name == "whisper" else 768], "TEMPORAL_ROWS")
            else:
                import text_windows
                text_windows.validate(entry["native"], "dense", shape[0], text=row["text"]["surface"])
                need(shape == [len(entry["native"]["windows"]), 768], "TEXT_WINDOWS")
            values = np.frombuffer(raw, "<f4")
            need(bool(np.isfinite(values).all()), "NONFINITE")
            count += len(raw)
        counts[lane] = dict(readings=len(entries), bytes=count)
        total += count
    phases["all_native_value_bytes"] = time.perf_counter()-tick
    result = dict(bank_pin=a.bank_pin, original_source_bytes=source_bytes, lanes=counts,
        value_bytes=total, flags=FLAGS, author_checked=True,
        source_frames=len(input_rows["rows"]["image"]), independent_wrapper_readback=True,
        phases=phases, seconds=time.perf_counter()-STARTED, code=sha(Path(__file__)))
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    save(a.output/"report.json", result)
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    print(json.dumps(dict(report=sha(a.output/"report.json"), value_bytes=total,
        seconds=result["seconds"], phase_coverage=result["phase_coverage"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "inputs", "source", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("bank-pin", "input-pin"):
        p.add_argument("--"+name, required=True)
    main(p.parse_args())
