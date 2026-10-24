"""Fixed native first-frame/audio observations for candidate instrument execution.

This is an engineering observation boundary, not sensory admission. It never
reads private labels. A single frame is not a representation of the whole video.
PC-04/07/28: stream clips; bind the frozen census once; preserve every selected
pixel, audio block and channel. Source reads at decoder/consumer boundaries are
intentional physical checks, not per-tile work.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses.perception_corpus import pinned, save, sha
from senses.perception_decode import build
from senses.perception_instruments import image_input, audio_input
from senses.perception_native import NativePrefix
from senses.perception_targets import Corpus


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_OBSERVATION_"+code)


def census(path, pin):
    report = pinned(path/"report.json", pin)
    rows = pinned(path/"public.json", report["public_pin"])
    need(report["observations"] == len(rows) == 512 and not report["rejected"],
         "CENSUS_ROSTER")
    need(not report["scientific_admission"] and not report["training_ready"],
         "CENSUS_FLAGS")
    need(len({r["clip"] for r in rows}) == len(rows), "DUPLICATE_CLIP")
    return report, {r["clip"]: r for r in rows}


def public_views(medium, views):
    """Reject private annotations before any candidate instrument is invoked."""
    need(medium in ("image", "audio"), "MEDIA")
    expected = ({"clip", "ordinal", "descriptor", "prefix_pin", "record", "record_sha256",
                 "blob", "blob_sha256", "blob_bytes"} if medium == "image" else
                {"clip", "descriptor", "source", "clock", "split_group", "through",
                 "prefix_pin", "span", "blob", "blob_sha256", "blob_bytes"})
    for row, value, _ in views:
        need(set(row) == expected, "PUBLIC_INPUT_FIELDS")
        if medium == "image":
            need(value["referents"] == [] and
                 all(v["state"] == "absent" and not v["items"]
                     for k,v in value["channels"].items() if k != "perception"),
                 "PRIVATE_LABEL_INPUT")
            need(len(value["channels"]["perception"]["items"]) == 1
                 and value["channels"]["perception"]["items"][0]["modality"] == "image",
                 "PUBLIC_IMAGE")
        else:
            need(set(value) == {"sha256","media_type","rate","channel","start","end"},
                 "PRIVATE_LABEL_INPUT")
    return views


def prepare(a):
    started = time.perf_counter()
    report, population = census(a.census, a.census_pin)
    corpus = Corpus(a.corpus, a.corpus_pin, a.plan)
    need(report["corpus_pin"] == a.corpus_pin, "CORPUS_BINDING")
    need(a.clips and len(set(a.clips)) == len(a.clips)
         and set(a.clips) <= set(population), "CLIPS")
    a.output.mkdir(parents=True, exist_ok=False)
    for medium in ("image", "audio"):
        (a.output/medium).mkdir()
    phases = dict(setup=time.perf_counter()-started)
    rows = dict(image=[], audio=[])
    sources, projected = [], {}
    for clip in a.clips:
        tick = time.perf_counter()
        selected = population[clip]
        graph, streams, bound = corpus.read(clip)
        need(bound == selected["source"] and corpus.rows[clip]["role"] == selected["role"],
             "SOURCE_BINDING")
        need(Fraction(*selected["qa_limit"]) >= 2, "QA_CUTOFF")
        expected_image = [n for n in graph["nodes"] if n["modality"] == "image"][0]
        expected_audio = [n for n in graph["nodes"] if n["modality"] == "audio"
                          and Fraction(*n["available_after"]) <= 2]
        need(expected_image == selected["image"]["node"]
             and expected_audio == selected["audio"]["nodes"], "CENSUS_NODES")
        episode = corpus.path.parent/"clips"/clip
        episode_pin = corpus.rows[clip]["report_pin"]
        source_row = dict(clip=clip, role=selected["role"], source=bound,
                          episode_pin=episode_pin, media={})
        for medium, through in (("image", expected_image["available_after"]),
                                ("audio", [2, 1])):
            prefix_root = a.output/("prefix-"+clip+"-"+medium)
            receipt = build(episode, episode_pin, clip, through, prefix_root, media=(medium,))
            prefix = NativePrefix(prefix_root, expected=receipt["pin"],
                                  episodes=episode, episode_pin=episode_pin, media=(medium,))
            request = dict(clock_id=graph["clock"]["id"], split_group=graph["split_group"],
                           through=through, required_media=[medium])
            decoded = list(prefix.consume(**request))
            source_row["media"][medium] = dict(root=prefix_root.name, pin=receipt["pin"],
                request=request, bytes=receipt["bytes"], phases=receipt["phases"])
            if medium == "image":
                need(len(decoded) == 1 and decoded[0]["node"] == expected_image, "FIRST_FRAME")
                rgb = decoded[0]["values"]
                h, w, _ = rgb.shape
                need([h, w] == [selected["image"]["height"], selected["image"]["width"]],
                     "FRAME_GEOMETRY")
                count = 0
                for y in range(0, h, 512):
                    for x in range(0, w, 512):
                        th, tw = min(512, h-y), min(512, w-x)
                        descriptor = dict(modality="image", node=expected_image,
                            source_payload=prefix.value["records"]["image"][0]["sha256"],
                            rectangle=[x, y, tw, th], frame_shape=[h, w, 3],
                            shape=[1+((th+13)//14)*((tw+13)//14), 384])
                        record, blob = image_input(rgb, descriptor, prefix)
                        stem = clip+"-"+str(count)
                        save(a.output/"image"/(stem+".json"), record)
                        with (a.output/"image"/(stem+".png")).open("xb") as f:
                            f.write(blob.raw)
                        rows["image"].append(dict(clip=clip, ordinal=count,
                            descriptor=descriptor, prefix_pin=receipt["pin"],
                            record=stem+".json", record_sha256=sha(a.output/"image"/(stem+".json")),
                            blob=stem+".png", blob_sha256=blob.pin, blob_bytes=len(blob.raw)))
                        count += 1
                need(count == selected["image"]["tiles"], "TILE_COVERAGE")
            else:
                need([r["node"] for r in decoded] == expected_audio, "AUDIO_BLOCKS")
                pcm = np.concatenate([r["values"] for r in decoded])
                rate = prefix.value["formats"]["audio"]["sample_rate"]
                channels = pcm.shape[1]
                need(pcm.shape == (selected["audio"]["samples"], selected["audio"]["channels"])
                     and rate == selected["audio"]["rate"], "AUDIO_GEOMETRY")
                need(all(x["span"][1] == y["span"][0]
                         for x,y in zip(expected_audio, expected_audio[1:])), "AUDIO_GAP")
                for channel in range(channels):
                    descriptor = dict(modality="audio", nodes=expected_audio,
                        source_payloads=[r["sha256"] for r in prefix.value["records"]["audio"]],
                        channel=channel, channels=channels, rate=rate, samples=len(pcm),
                        span=[expected_audio[0]["span"][0], expected_audio[-1]["span"][1]],
                        shape=[(((len(pcm)*16000+rate-1)//rate)+319)//320, 384])
                    span, blob = audio_input(pcm, descriptor)
                    name = clip+".wav"
                    if channel == 0:
                        with (a.output/"audio"/name).open("xb") as f:
                            f.write(blob.raw)
                    need(sha(a.output/"audio"/name) == blob.pin, "SHARED_WAVEFORM")
                    rows["audio"].append(dict(clip=clip, descriptor=descriptor,
                        source=prefix.value["source"], clock=prefix.value["clock"],
                        split_group=prefix.value["split_group"], through=through,
                        prefix_pin=receipt["pin"], span=span, blob=name,
                        blob_sha256=blob.pin, blob_bytes=len(blob.raw)))
        sources.append(source_row)
        for name, size in selected["projected_slot_bytes"].items():
            projected[name] = projected.get(name, 0)+size
        phases[clip] = time.perf_counter()-tick
    pins = {}
    for medium, schema in (("image", "native-vision-witness-input-v1"),
                           ("audio", "native-audio-witness-input-v1")):
        pins[medium] = save(a.output/medium/"manifest.json", dict(schema=schema,
            purpose="complete declared native observations; candidate readings only",
            rows=rows[medium]))
    value = dict(schema="native-observation-inputs-v1", corpus_pin=a.corpus_pin,
        census_pin=a.census_pin, code_pin=sha(Path(__file__)), sources=sources,
        input_pins=pins, projected_slot_bytes=projected, phases=phases,
        scientific_admission=False, training_ready=False, physical_sync_calibrated=False,
        audio_label_correspondence_established=False,
        image_scope="one complete initial frame; later video unobserved",
        seconds=time.perf_counter()-started)
    value["phase_coverage"] = sum(phases.values())/value["seconds"]
    pin = save(a.output/"report.json", value)
    print(json.dumps(dict(report_pin=pin, input_pins=pins,
        windows={k:len(v) for k,v in rows.items()}, projected_slot_bytes=projected,
        seconds=value["seconds"], phase_coverage=value["phase_coverage"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("census", "corpus", "plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("census-pin", "corpus-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--clips", nargs="+", required=True)
    prepare(p.parse_args())
