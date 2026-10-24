"""Real native prefix decode, consumption, cold readback and fault controls."""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.perception_decode import build
from senses.perception_native import NativePrefix
from audit.perception_native_readback import read, alternate, need, sha


CLIPS = ("video_10993", "video_874")


def run(a):
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    phases, receipts, controls = {}, [], []
    for clip in CLIPS:
        t = time.perf_counter()
        root = a.prefixes/(a.host+"-prefix-"+clip)
        if root.exists():
            pin = (root/"COMMITTED").read_text()
            build_result = dict(reused_pinned_prefix=pin)
        else:
            build_result = build(a.episodes, a.episode_pin, clip, [2, 1], root, a.ffmpeg)
            pin = build_result["pin"]
        manifest, independent = read(root, a.episodes, pin, a.episode_pin)
        consumer = NativePrefix(root, expected=pin, episodes=a.episodes, episode_pin=a.episode_pin)
        phases[clip+"_bind_build_read"] = time.perf_counter()-t
        views = []
        for through in ([1, 1], [2, 1]):
            t = time.perf_counter()
            request = dict(clock_id=manifest["clock"]["id"], split_group=manifest["split_group"],
                           through=through, required_media=["image", "audio"])
            observed = []
            for item in consumer.consume(**request):
                values, node = item["values"], item["node"]
                need(not values.flags.writeable and Fraction(*node["available_after"]) <= Fraction(*through),
                     "CONSUMER_FUTURE")
                observed.append(dict(id=node["id"], modality=item["modality"], shape=list(values.shape),
                                     sha256=hashlib.sha256(values.tobytes()).hexdigest()))
            expected = sorted((r for rows in manifest["records"].values() for r in rows
                               if Fraction(*r["node"]["available_after"]) <= Fraction(*through)),
                              key=lambda r: (Fraction(*r["node"]["span"][0]),
                                             r["node"]["stream"], r["node"]["ordinal"]))
            need([v["id"] for v in observed] == [r["node"]["id"] for r in expected]
                 and [v["sha256"] for v in observed] == [r["sha256"] for r in expected], "CONSUMER_BYTES")
            independent_decode = alternate(root, a.episodes, manifest, through, a.output/f"{clip}-{through[0]}")
            views.append(dict(through=through, consumed=observed, alternate=independent_decode))
            phases[clip+f"_view_{through[0]}"] = time.perf_counter()-t
        receipts.append(dict(clip=clip, pin=pin, build=build_result, independent=independent, views=views))
        if clip == CLIPS[0]:
            t = time.perf_counter()
            def refuse(name, code, call):
                try:
                    call()
                except ValueError as exc:
                    need(str(exc) == "MNLM_NATIVE_PREFIX_"+code, "CONTROL_REASON")
                    controls.append(dict(name=name, code=str(exc)))
                else:
                    raise ValueError("MNLM_NATIVE_READ_CONTROL_ACCEPTED")
            request = dict(clock_id=manifest["clock"]["id"], split_group=manifest["split_group"],
                           through=[1, 1], required_media=["image", "audio"])
            for name, change, code in (
                ("wrong_clock", dict(clock_id="foreign"), "CLOCK"),
                ("wrong_split", dict(split_group="foreign"), "SPLIT"),
                ("future", dict(through=[3, 1]), "UNDECODED_FUTURE"),
                ("missing_text", dict(required_media=["text"]), "MISSING_MODALITY"),
                ("unavailable", dict(through=[0, 1]), "UNAVAILABLE_MODALITY")):
                refuse(name, code, lambda change=change: list(consumer.consume(**{**request, **change})))
            shadow = a.output/"owned-shadow"
            shutil.copytree(root, shadow, copy_function=os.link)
            # Replace only owned metadata directory entries; linked native files stay immutable.
            for name in ("manifest.json", "COMMITTED"):
                (shadow/name).unlink()
            original = copy.deepcopy(manifest)
            def corrupt(name, edit, code):
                changed = copy.deepcopy(original); edit(changed)
                raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
                (shadow/"manifest.json").write_bytes(raw)
                altered_pin = hashlib.sha256(raw).hexdigest()
                (shadow/"COMMITTED").write_text(altered_pin)
                refuse(name, code, lambda: NativePrefix(shadow, expected=altered_pin,
                    episodes=a.episodes, episode_pin=a.episode_pin))
            corrupt("shifted_native_pts", lambda v: v["records"]["image"][0]["node"].update(pts=1), "NATIVE_COORDINATES")
            corrupt("wrong_source", lambda v: v["source"].update(media="0"*64), "SOURCE_IDENTITY")
            corrupt("omitted_frame", lambda v: v["records"]["image"].pop(), "RECORD_ROSTER")
            corrupt("extra_block", lambda v: v["records"]["audio"].append(v["records"]["audio"][-1]), "RECORD_ROSTER")
            corrupt("resampled_format", lambda v: v["formats"]["audio"].update(sample_rate=16000), "FORMAT")
            corrupt("invented_transcript", lambda v: v.update(transcript="private annotation"), "MANIFEST_SCHEMA")
            corrupt("meaning_asserted", lambda v: v["flags"].update(meaning_asserted=True), "FLAGS")
            (shadow/"manifest.json").write_bytes((root/"manifest.json").read_bytes())
            (shadow/"COMMITTED").write_text(pin)
            for medium in ("image", "audio"):
                entry = manifest["records"][medium][0]
                path = shadow/entry["file"]
                raw = path.read_bytes()
                path.unlink()  # Remove the owned link before creating a corrupt owned file.
                path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
                refuse("corrupt_"+medium, "PAYLOAD_BYTES", lambda: NativePrefix(shadow, expected=pin,
                    episodes=a.episodes, episode_pin=a.episode_pin))
                path.write_bytes(raw)
            # Mutation after binding must also fail at the actual consumption boundary.
            reopened = NativePrefix(shadow, expected=pin, episodes=a.episodes, episode_pin=a.episode_pin)
            path = shadow/manifest["records"]["audio"][0]["file"]
            raw = path.read_bytes(); path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
            refuse("changed_after_bind", "CONSUMPTION_BYTES", lambda: list(reopened.consume(**request)))
            path.write_bytes(raw)
            for fault in ("fps", "audio_rate"):
                proof = alternate(root, a.episodes, manifest, [1, 1], a.output/fault, fault=fault)
                controls.append(dict(name="actual_"+fault, rejected_mismatch=True, receipt=proof))
            phases["controls"] = time.perf_counter()-t
        t = time.perf_counter()
        cold, cold_receipt = read(root, a.episodes, pin, a.episode_pin)
        need(cold == manifest and cold_receipt == independent, "COLD_REPLAY")
        phases[clip+"_cold"] = time.perf_counter()-t
    wall = time.perf_counter()-start
    report = dict(complete=True, acceptance="author-checked", episode_pin=a.episode_pin,
                  receipts=receipts, controls=controls, phases=phases, wall_seconds=wall,
                  timing_coverage=sum(phases.values())/wall, meaning_asserted=False,
                  scientific_admission=False, model_learning=False)
    (a.output/"report.json").write_text(json.dumps(report, sort_keys=True), encoding="utf8")
    print(json.dumps(dict(complete=True, clips=len(receipts), controls=len(controls),
                         wall_seconds=wall, timing_coverage=report["timing_coverage"],
                         report_sha256=sha(a.output/"report.json"))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("episodes", "prefixes", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--episode-pin", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    run(p.parse_args())
