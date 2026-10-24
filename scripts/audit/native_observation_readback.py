"""Cold source-to-wrapper evidence for a fixed native observation publication."""
import argparse
import copy
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from PIL import Image
import soundfile as sf
from senses.perception_corpus import pinned, save, sha
from senses.perception_targets import Corpus
from senses.perception_native import NativePrefix
from senses.visual_bank import input_views as images
from senses.audio_bank import input_views as sounds
from senses.native_observation import public_views


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_OBSERVATION_READBACK_"+code)


def readback(a):
    tick = time.perf_counter()
    report = pinned(a.inputs/"report.json", a.input_pin)
    need(report["schema"] == "native-observation-inputs-v1"
         and not any(report[n] for n in ("scientific_admission", "training_ready",
             "physical_sync_calibrated", "audio_label_correspondence_established")), "FLAGS")
    corpus = Corpus(a.corpus, report["corpus_pin"], a.plan)
    image_views = images(a.inputs/"image", report["input_pins"]["image"])
    audio_views = sounds(a.inputs/"audio", report["input_pins"]["audio"])
    public_views("image", image_views)
    public_views("audio", audio_views)
    need(set(r["clip"] for r, _, _ in image_views) ==
         set(r["clip"] for r, _, _ in audio_views) ==
         {r["clip"] for r in report["sources"]}, "SOURCE_ROSTER")
    a.output.mkdir(parents=True, exist_ok=False)
    proofs, controls, phases = [], [], dict(setup=time.perf_counter()-tick)
    for medium, original in (("image", image_views), ("audio", audio_views)):
        row, value, blob = original[0]
        altered = copy.deepcopy(row)
        altered["provider_answer"] = "private target must never be an instrument input"
        try:
            public_views(medium, [(altered, value, blob)])
        except ValueError as exc:
            need(str(exc) == "MNLM_NATIVE_OBSERVATION_PUBLIC_INPUT_FIELDS", "LABEL_CONTROL")
            controls.append(dict(medium=medium, control="private-label-field", error=str(exc)))
        else:
            raise ValueError("MNLM_OBSERVATION_READBACK_LABEL_ACCEPTED")
    row, value, blob = image_views[0]
    altered = copy.deepcopy(value)
    altered["referents"] = ["private target"]
    try:
        public_views("image", [(row, altered, blob)])
    except ValueError as exc:
        need(str(exc) == "MNLM_NATIVE_OBSERVATION_PRIVATE_LABEL_INPUT", "LABEL_CONTROL")
        controls.append(dict(medium="image", control="private-label-referent", error=str(exc)))
    else:
        raise ValueError("MNLM_OBSERVATION_READBACK_LABEL_ACCEPTED")
    for declared in report["sources"]:
        start = time.perf_counter()
        clip = declared["clip"]
        graph, streams, source = corpus.read(clip)
        need(source == declared["source"]
             and corpus.rows[clip]["role"] == declared["role"]
             and corpus.rows[clip]["report_pin"] == declared["episode_pin"], "SOURCE")
        episode = corpus.path.parent/"clips"/clip
        by_medium = {}
        for medium in ("image", "audio"):
            d = declared["media"][medium]
            prefix = NativePrefix(a.inputs/d["root"], expected=d["pin"],
                episodes=episode, episode_pin=declared["episode_pin"], media=(medium,))
            request = d["request"]
            decoded = list(prefix.consume(**request))
            need(request["required_media"] == [medium], "MEDIA")
            need(all(Fraction(*x["node"]["available_after"]) <= Fraction(*request["through"])
                     for x in decoded), "FUTURE")
            by_medium[medium] = prefix, decoded
            for label, change, code in (
                ("future", dict(through=[999, 1]), "UNDECODED_FUTURE"),
                ("clock", dict(clock_id="foreign"), "CLOCK"),
                ("split", dict(split_group="foreign"), "SPLIT"),
                ("unobserved-medium", dict(required_media=["audio" if medium == "image" else "image"]),
                 "MISSING_MODALITY"),
            ):
                try:
                    list(prefix.consume(**{**request, **change}))
                except ValueError as exc:
                    need(str(exc) == "MNLM_NATIVE_PREFIX_"+code, "CONTROL_REASON")
                    controls.append(dict(clip=clip, medium=medium, control=label, error=str(exc)))
                else:
                    raise ValueError("MNLM_OBSERVATION_READBACK_CONTROL_ACCEPTED:"+label)
        prefix, decoded = by_medium["image"]
        need(len(decoded) == 1 and decoded[0]["node"]["ordinal"] == 0, "FIRST_IMAGE")
        original = decoded[0]["values"]
        h, w, _ = original.shape
        selected = [(r, record, blob) for r, record, blob in image_views if r["clip"] == clip]
        rectangles = [[x,y,min(512,w-x),min(512,h-y)]
                      for y in range(0,h,512) for x in range(0,w,512)]
        need([r["descriptor"]["rectangle"] for r,_,_ in selected] == rectangles
             and [r["ordinal"] for r,_,_ in selected] == list(range(len(rectangles))),
             "COMPLETE_TILE_ROSTER")
        reconstructed = np.empty_like(original)
        for row, record, blob in selected:
            x,y,tw,th = row["descriptor"]["rectangle"]
            need(row["prefix_pin"] == declared["media"]["image"]["pin"]
                 and row["descriptor"]["node"] == decoded[0]["node"]
                 and row["descriptor"]["source_payload"] ==
                     prefix.value["records"]["image"][0]["sha256"]
                 and record["referents"] == [], "IMAGE_BINDING")
            with Image.open(io.BytesIO(blob.raw)) as im:
                tile = np.asarray(im)
            need(tile.dtype == np.uint8 and tile.shape == (th,tw,3), "TILE_SHAPE")
            reconstructed[y:y+th,x:x+tw] = tile
        need(reconstructed.tobytes() == original.tobytes(), "ALL_SOURCE_PIXELS")
        # Separate command construction and raw file readback from the source decoder.
        ffmpeg = prefix.value["runtime"]["executable"]
        image_file = a.output/(clip+"-independent.rgb")
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-i",
               str(episode/"media"/(clip+".mp4")), "-map", "0:"+str(streams["video"]["index"]),
               "-frames:v", "1", "-c:v", "rawvideo", "-pix_fmt", "rgb24",
               "-f", "rawvideo", str(image_file)]
        subprocess.run(cmd, check=True, capture_output=True)
        need(image_file.read_bytes() == original.tobytes(), "INDEPENDENT_SOURCE_RGB")
        prefix, decoded = by_medium["audio"]
        selected = [(r, span, blob) for r,span,blob in audio_views if r["clip"] == clip]
        pcm = np.concatenate([x["values"] for x in decoded])
        need([span["channel"] for _,span,_ in selected] == list(range(pcm.shape[1])),
             "COMPLETE_CHANNEL_ROSTER")
        for row, span, blob in selected:
            wave, rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
            need(wave.tobytes() == pcm.tobytes()
                 and rate == int(streams["audio"]["sample_rate"])
                 and span["start"] == 0 and span["end"] == len(pcm)
                 and row["descriptor"]["nodes"] == [x["node"] for x in decoded]
                 and row["descriptor"]["source_payloads"] ==
                     [r["sha256"] for r in prefix.value["records"]["audio"]]
                 and row["prefix_pin"] == declared["media"]["audio"]["pin"]
                 and row["source"] == source
                 and row["clock"] == graph["clock"]
                 and row["split_group"] == graph["split_group"]
                 and row["through"] == [2,1], "ALL_SOURCE_AUDIO")
        expected_bytes = sum(n["sample_count"] for n in graph["nodes"]
                             if n["modality"] == "audio")*pcm.shape[1]*4
        need(expected_bytes <= 64*1024*1024, "INDEPENDENT_PCM_BUDGET")
        audio_file = a.output/(clip+"-independent-full.pcm")
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(episode/"media"/(clip+".mp4")),
               "-map", "0:"+str(streams["audio"]["index"]), "-vn", "-c:a", "pcm_f32le",
               "-f", "f32le", str(audio_file)]
        subprocess.run(cmd, check=True, capture_output=True)
        need(audio_file.stat().st_size == expected_bytes, "INDEPENDENT_PCM_SIZE")
        with audio_file.open("rb") as f:
            need(f.read(pcm.nbytes) == pcm.tobytes(), "INDEPENDENT_SOURCE_PCM")
        proofs.append(dict(clip=clip, source=source, tiles=len(rectangles),
            rgb_bytes=original.nbytes, rgb_sha256=sha(image_file),
            audio_blocks=len(decoded), samples=len(pcm), channels=pcm.shape[1],
            pcm_prefix_bytes=pcm.nbytes, independent_full_pcm_sha256=sha(audio_file)))
        phases[clip] = time.perf_counter()-start
    value = dict(author_checked=True, input_pin=a.input_pin, source_proofs=proofs,
        controls=controls, phases=phases, scientific_admission=False, training_ready=False,
        code_pin=sha(Path(__file__)), seconds=time.perf_counter()-tick)
    value["phase_coverage"] = sum(phases.values())/value["seconds"]
    pin = save(a.output/"report.json", value)
    print(json.dumps(dict(report_pin=pin, sources=len(proofs), controls=len(controls),
        seconds=value["seconds"], phase_coverage=value["phase_coverage"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "corpus", "plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--input-pin", required=True)
    readback(p.parse_args())
