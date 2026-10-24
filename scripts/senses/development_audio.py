"""Audit original WAV/MP4 correspondence for the fixed development partition.

PC-07/11/28/32/37: bind the directory once, stream one clip, retain unresolved
recordings and intervals. This publishes source evidence, never sensory admission.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"census")]
import numpy as np
from range_zip import DirectoryReader
from range_member import header, compressed, decompress
from senses.audio_correspondence import Alignment, POLICY, need, profile, summary
from senses.audio_target_binding import SoundCorrespondence
from senses.contract import digest
from senses.perception_corpus import PLAN_PIN, pinned, save, sha
from senses.perception_targets import Corpus, Targets

DIRECTORY_PIN = "4ee125660811d9702d4357d6916aad5e1eff6a2a95f11f463e412e982971f068"
ACQUIRED_PIN = "a217b1a31a23d3b6a69f5c6df5a2663085be07f92868bac0663fba78467ec54f"
CORPUS_PIN = "3dce43b6bf0357c26d11bf38ac80c51193cf9817d2dbd7ab933a85a81fa55a11"
TARGET_PIN = "0f1f50117e7030c7965ec939a4e607b2093ddb95b173b8ff959339fa31c22a79"
FLAGS = dict(scientific_admission=False, training_ready=False,
             physical_sync_calibrated=False, annotation_truth_established=False,
             online_feedback_eligible=False)


def directory(root):
    d = pinned(root/"directory-report.json", DIRECTORY_PIN)
    reader = DirectoryReader(root/"directory", d["identity"], offline_pin=d["directory_pin"])
    with zipfile.ZipFile(reader) as z:
        rows = {Path(e.filename).stem: dict(name=e.filename, crc32=e.CRC,
            bytes=e.file_size, compressed_bytes=e.compress_size,
            header_offset=e.header_offset, method=e.compress_type, flags=e.flag_bits)
            for e in z.infolist()}
    need(len(rows) == d["entry_count"] == 2184, "DIRECTORY_ROSTER")
    return d, rows


def acquire(root, member, identity, reused=None):
    if reused:
        path, pin = reused
        receipt = pinned(path/"receipt.json", pin)
        need(receipt["member"] == member and receipt["identity"] == identity
             and sha(path/"source.wav") == receipt["media"]["sha256"], "REUSED_SOURCE")
        return path, pin
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reader = DirectoryReader(root/"header", identity)
    offset = header(reader, member)
    header_pin = reader.seal()
    phases = dict(header=time.perf_counter()-started)
    tick = time.perf_counter()
    packed, receipt = compressed(identity, member, offset, root)
    phases["network"] = time.perf_counter()-tick
    tick = time.perf_counter()
    media = decompress(packed, member, root/"source.wav")
    need(sha(root/"source.wav") == media["sha256"], "ACQUIRED_BYTES")
    phases["decompress_and_readback"] = time.perf_counter()-tick
    pin = save(root/"receipt.json", dict(identity=identity, member=member,
        data_start=offset, header_pin=header_pin, compressed=receipt, media=media,
        phases=phases, seconds=time.perf_counter()-started, **FLAGS))
    return root, pin


def decode(path, channels):
    raw = subprocess.check_output(["/usr/bin/ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path), "-vn", "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", "-"])
    need(len(raw) > 0 and len(raw) % (4*channels) == 0, "DECODE_LAYOUT")
    return raw, np.frombuffer(raw, dtype="<i4").reshape(-1, channels).astype(np.float64)/2**31


def observe(clip, root, source, receipt_pin, corpus):
    started = time.perf_counter()
    graph, streams, binding = corpus.read(clip)
    receipt = pinned(source/"receipt.json", receipt_pin)
    paths = dict(wav=source/"source.wav",
                 mp4=corpus.path.parent/"clips"/clip/"media"/(clip+".mp4"))
    probe = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error",
        "-select_streams", "a", "-show_streams", "-of", "json", str(paths["wav"])]))
    need(len(probe["streams"]) == 1, "WAV_STREAMS")
    wav, mp4 = probe["streams"][0], streams["audio"]
    rate, channels = int(mp4["sample_rate"]), mp4["channels"]
    need(rate % 2 == 0 and int(wav["sample_rate"]) == rate
         and wav["channels"] == channels, "NATIVE_LAYOUT")
    records, values = {}, {}
    for kind, path in paths.items():
        raw, values[kind] = decode(path, channels)
        with (root/(kind+"-s32le.bin")).open("xb") as f:
            f.write(raw)
        records[kind] = dict(source_file=str(path), source_sha256=(binding["media"] if
            kind == "mp4" else receipt["media"]["sha256"]), pcm_sha256=sha(root/(kind+"-s32le.bin")),
            samples=len(values[kind]), rate=rate, channels=channels)
    need(sum(n["sample_count"] for n in graph["nodes"] if n["modality"] == "audio")
         == records["mp4"]["samples"], "NATIVE_SAMPLE_COUNT")
    phases = dict(source_and_decode=time.perf_counter()-started)
    tick = time.perf_counter()
    (root/"profiles").mkdir()
    common, width = min(len(v) for v in values.values()), rate//2
    windows = []
    for channel in range(channels):
        for index, first in enumerate(range(0, common-width+1, width)):
            scores, row = profile(values["mp4"][:, channel], values["wav"][:, channel],
                                  first, first+width, rate)
            name = f"profiles/{channel}-{index:03d}.bin"
            with (root/name).open("xb") as f:
                f.write(scores.astype("<f8").tobytes())
            windows.append(dict(index=index, channel=channel, **row,
                                profile=name, profile_sha256=sha(root/name)))
    phases["profiles"] = time.perf_counter()-tick
    record = dict(clip=clip, rate=rate, channels=channels, sources=records,
        graph_file=str(corpus.path.parent/"clips"/clip/"public"/(clip+".json")),
        graph_sha256=binding["graph"], native_clock_pin=digest(graph["clock"]), windows=windows,
        summary=summary(windows, channels),
        untested_tail_mp4=[common//width*width, records["mp4"]["samples"]],
        untested_tail_wav=[common//width*width, records["wav"]["samples"]])
    pin = save(root/"report.json", dict(schema="source-audio-alignment-v1", policy=POLICY,
        clips=[record], source=str(source), receipt_pin=receipt_pin, phases=phases,
        seconds=time.perf_counter()-started, **FLAGS))
    return pin, record


def attach(clip, root, pin, targets):
    reader = Alignment(root, pin)
    bound = SoundCorrespondence(targets, reader, [clip])
    _, clock, events = bound.clips[clip]
    rows = [bound.bind(clip, key, target_pin=targets.entries[clip]["pin"]) for key in events]
    record = reader.records[clip]
    graph = clock.graph
    request = dict(wav_sha256=record["sources"]["wav"]["source_sha256"],
        mp4_sha256=graph["blob_sha256"], clock_id=graph["clock"]["id"],
        split_group=graph["split_group"], required_channels=list(range(record["channels"])))
    projections, unresolved = [], []
    for window in record["windows"]:
        if window["channel"] != 0:
            continue
        try:
            projections.append(reader.project(clip, window["wav_samples"], **request))
        except ValueError as exc:
            need(str(exc) in ("MNLM_AUDIO_ALIGNMENT_CORRESPONDENCE_UNRESOLVED",
                 "MNLM_AUDIO_ALIGNMENT_UNCOVERED_INTERVAL"), "PROJECTION_ERROR")
            unresolved.append(dict(wav_samples=window["wav_samples"], code=str(exc)))
    return save(root/"attachments.json", dict(alignment_pin=pin, targets_pin=TARGET_PIN,
        target_pin=targets.entries[clip]["pin"], rows=rows, projections=projections,
        unresolved_projections=unresolved, **FLAGS)), rows


def main(a):
    started = time.perf_counter()
    d, members = directory(a.witness)
    corpus = Corpus(a.corpus, CORPUS_PIN, a.plan)
    targets = Targets(a.targets, expected=TARGET_PIN, corpus=corpus)
    chosen = [r for r in corpus.plan["splits"] if r["role"] != "held_out"]
    need(Counter(r["role"] for r in chosen) == dict(increment_one=256, increment_two=128),
         "DEVELOPMENT_ROSTER")
    old = pinned(a.witness/"acquisition-report.json", ACQUIRED_PIN)
    reused = {r["clip"]: (a.witness/"members"/r["clip"], r["receipt_sha256"]) for r in old["results"]}
    a.output.mkdir(parents=True, exist_ok=False)
    code = {n: sha(Path(__file__).parents[1]/n) for n in (
        "senses/development_audio.py", "senses/audio_correspondence.py",
        "senses/audio_target_binding.py", "census/range_member.py", "census/range_zip.py")}
    binding = dict(directory_pin=DIRECTORY_PIN, plan_pin=PLAN_PIN, corpus_pin=CORPUS_PIN,
        targets_pin=TARGET_PIN, corpus=str(a.corpus), targets=str(a.targets), plan=str(a.plan),
        witness=str(a.witness), code=code, policy=POLICY,
        runtime={k: sha(Path("/usr/bin")/k) for k in ("ffmpeg", "ffprobe")},
        clips=[r["clip"] for r in chosen], **FLAGS)
    save(a.output/"binding.json", binding)
    phases = dict(setup=time.perf_counter()-started)
    results, errors = [], []
    for row in chosen:
        tick = time.perf_counter()
        clip = row["clip"]
        root = a.output/"clips"/clip
        root.mkdir(parents=True)
        try:
            member = members[clip]
            need(member["name"] == "audios/"+clip+".wav", "AUDIO_MEMBER")
            source, receipt_pin = acquire(root/"source", member, d["identity"], reused.get(clip))
            acquired = time.perf_counter()
            pin, record = observe(clip, root, source, receipt_pin, corpus)
            observed = time.perf_counter()
            attached, rows = attach(clip, root, pin, targets)
            result = dict(clip=clip, role=row["role"], source=str(source), receipt_pin=receipt_pin,
                report_pin=pin, attachments_pin=attached, summary=record["summary"],
                targets=len(rows), covered=sum(r["status"] == "covered" for r in rows),
                windows=len(record["windows"]), reused=clip in reused,
                phases=dict(acquire=acquired-tick, observe=observed-acquired,
                            attach=time.perf_counter()-observed))
            results.append(result)
        except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            result = dict(clip=clip, role=row["role"], error=type(exc).__name__+": "+str(exc))
            errors.append(result)
        save(root/"receipt.json", result)
        phases[clip] = time.perf_counter()-tick
        with (a.output/"journal.jsonl").open("ab") as f:
            f.write(json.dumps(dict(clip=clip, ok="error" not in result,
                done=len(results)+len(errors), seconds=phases[clip])).encode()+b"\n")
            f.flush()
        print(json.dumps(dict(done=len(results)+len(errors), errors=len(errors), clip=clip)), flush=True)
    seconds = time.perf_counter()-started
    report = dict(binding_pin=sha(a.output/"binding.json"), results=results, errors=errors,
        complete=len(results) == 384 and not errors, phases=phases, seconds=seconds,
        phase_coverage=sum(phases.values())/seconds, **FLAGS)
    print(json.dumps(dict(report_pin=save(a.output/"report.json", report),
                          complete=report["complete"], seconds=seconds)), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("witness", "corpus", "plan", "targets", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
