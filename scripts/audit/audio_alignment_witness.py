"""Pinned two-source audio correspondence and native sample-clock projection."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from senses.audio_correspondence import Alignment, POLICY, need, profile, summary
from senses.contract import digest
from senses.perception_corpus import pinned, save, sha


def main(a):
    start = time.perf_counter()
    diagnostic = pinned(a.source/"diagnostic.json", a.diagnostic_pin)
    acquisition = pinned(a.source/"acquisition-report.json", diagnostic["acquisition_sha256"])
    need([r["clip"] for r in diagnostic["results"]] ==
         [r["clip"] for r in acquisition["results"]], "SOURCE_ROSTER")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"profiles").mkdir()
    phases, clips, controls, arrays = {}, [], [], {}
    phases["setup"] = time.perf_counter()-start

    def refusal(name, operation, code):
        try:
            operation()
        except ValueError as exc:
            need(str(exc) == code, "CONTROL_REASON:"+name+":"+str(exc))
            controls.append(dict(name=name, code=str(exc)))
        else:
            raise ValueError("MNLM_AUDIO_ALIGNMENT_CONTROL_ACCEPTED:"+name)

    for row in diagnostic["results"]:
        tick = time.perf_counter()
        clip = row["clip"]
        need(Path(clip).name == clip, "CLIP_PATH")
        root = a.source/"members"/clip
        rec = row["records"]
        rate, channels = rec["mp4"]["rate"], rec["mp4"]["channels"]
        need(rate % 2 == 0 and rate == rec["wav"]["rate"]
             and channels == rec["wav"]["channels"], "NATIVE_LAYOUT")
        values = {}
        for kind in ("mp4", "wav"):
            original = (a.corpus/"clips"/clip/"media"/(clip+".mp4") if kind == "mp4"
                        else root/"source.wav")
            need(sha(original) == rec[kind]["source_sha256"], "SOURCE_PIN")
            path = root/(kind+"-s32le.bin")
            need(sha(path) == rec[kind]["pcm_sha256"]
                 and path.stat().st_size == rec[kind]["samples"]*channels*4, "PCM_PIN")
            values[kind] = np.frombuffer(path.read_bytes(), dtype="<i4").reshape(-1, channels)
        arrays[clip] = {k: v.astype(np.float64)/(2**31) for k, v in values.items()}
        graph_path = a.corpus/"clips"/clip/"public"/(clip+".json")
        graph = json.loads(graph_path.read_bytes())
        episode = json.loads((a.corpus/"clips"/clip/"report.json").read_bytes())
        need(sha(graph_path) == episode["files"]["public/"+clip+".json"]["sha256"]
             and graph["blob_sha256"] == rec["mp4"]["source_sha256"], "GRAPH_PIN")
        phases[clip+":sources"] = time.perf_counter()-tick
        tick = time.perf_counter()
        common, width = min(len(v) for v in values.values()), rate//2
        windows = []
        for channel in range(channels):
            mp4, wav = (arrays[clip][k][:, channel] for k in ("mp4", "wav"))
            for index, first in enumerate(range(0, common-width+1, width)):
                scores, measured = profile(mp4, wav, first, first+width, rate)
                filename = f"profiles/{clip}-{channel}-{index:03d}.bin"
                raw = scores.astype("<f8").tobytes()
                with (a.output/filename).open("xb") as f:
                    f.write(raw)
                windows.append(dict(index=index, channel=channel, **measured,
                    profile=filename, profile_sha256=hashlib.sha256(raw).hexdigest()))
        result = dict(clip=clip, rate=rate, channels=channels, sources=rec,
            graph_file=str(graph_path), graph_sha256=sha(graph_path),
            native_clock_pin=digest(graph["clock"]), windows=windows,
            summary=summary(windows, channels),
            untested_tail_mp4=[common//width*width, rec["mp4"]["samples"]],
            untested_tail_wav=[common//width*width, rec["wav"]["samples"]])
        clips.append(result)
        phases[clip+":profiles"] = time.perf_counter()-tick

    tick = time.perf_counter()
    for record in clips:
        clip, rate = record["clip"], record["rate"]
        mp4, wav = arrays[clip]["mp4"][:, 0], arrays[clip]["wav"][:, 0]
        same_channel = [r for r in record["windows"] if r["channel"] == 0]
        chosen = same_channel[len(same_channel)//2]
        lo, hi = chosen["mp4_samples"]
        shift = rate//100
        _, shifted = profile(mp4, wav[shift:], lo, hi, rate)
        need(shifted["lag_samples"] == chosen["lag_samples"]-shift, "SHIFT_SIGN_REPLAY")
        controls.append(dict(name=clip+":known-cropped-origin", shift=shift,
            before=chosen["lag_samples"], after=shifted["lag_samples"],
            lineage="provider waveform viewed after removing its first shift samples"))
        _, reversed_time = profile(mp4, wav[::-1], lo, hi, rate)
        need(not reversed_time["window_qualified"], "REVERSED_TIME_FALSE_POSITIVE")
        controls.append(dict(name=clip+":reversed-time", qualified=False,
            peak=reversed_time["peak"], margin=reversed_time["margin"],
            lineage="original provider waveform viewed in reverse sample order; control only"))
    phases["signal_controls"] = time.perf_counter()-tick
    report = dict(schema="source-audio-alignment-v1", policy=POLICY, author_checked=True,
        source=str(a.source), diagnostic_pin=a.diagnostic_pin, corpus=str(a.corpus),
        clips=clips, controls=controls, phases=phases, scientific_admission=False,
        training_ready=False, physical_sync_calibrated=False,
        method_reference="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html",
        runtime=dict(numpy=np.__version__, scipy=scipy.__version__),
        code={n: sha(Path(__file__).resolve().parents[1]/n) for n in
              ("senses/audio_correspondence.py", "audit/audio_alignment_witness.py")},
        seconds=time.perf_counter()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report["phase_coverage"] = sum(phases.values())/report["seconds"]
    need(report["phase_coverage"] >= .9, "PHASE_COVERAGE")
    report_pin = save(a.output/"report.json", report)
    tick = time.perf_counter()
    reader = Alignment(a.output, report_pin)
    projections, unresolved_projections = [], []
    for record in clips:
        clip = record["clip"]
        graph = json.loads(Path(record["graph_file"]).read_bytes())
        request = dict(wav_sha256=record["sources"]["wav"]["source_sha256"],
            mp4_sha256=record["sources"]["mp4"]["source_sha256"], clock_id=graph["clock"]["id"],
            split_group=graph["split_group"], required_channels=list(range(record["channels"])))
        if not record["summary"]["correspondence_supported"]:
            refusal(clip+":unsupported-mapping", lambda: reader.project(clip, [0, 1], **request),
                    "MNLM_AUDIO_ALIGNMENT_CORRESPONDENCE_UNRESOLVED")
            continue
        # Exercise every half-second interval jointly supported by all original channels.
        candidates = [r for r in record["windows"] if r["channel"] == 0 and r["window_qualified"]]
        for row in candidates:
            span = row["wav_samples"]
            try:
                projections.append(reader.project(clip, span, **request))
            except ValueError as exc:
                need(str(exc) == "MNLM_AUDIO_ALIGNMENT_UNCOVERED_INTERVAL", "UNEXPECTED_PROJECTION")
                unresolved_projections.append(dict(clip=clip, wav_samples=span, code=str(exc)))
        first = next((p for p in projections if p["clip"] == clip), None)
        if first is None:
            continue
        span = first["wav_samples"]
        for label, change, code in (
            ("wrong-wav", dict(wav_sha256="0"*64), "WAV_SOURCE"),
            ("wrong-mp4", dict(mp4_sha256="0"*64), "NATIVE_SOURCE_CLOCK"),
            ("wrong-clock", dict(clock_id="foreign"), "NATIVE_SOURCE_CLOCK"),
            ("wrong-split", dict(split_group="foreign"), "NATIVE_SOURCE_CLOCK"),
            ("missing-channel", dict(required_channels=[]), "MISSING_CHANNEL"),
            ("untyped-channel", dict(required_channels=[
                bool(c) for c in range(record["channels"])]), "MISSING_CHANNEL"),
            ("training", dict(purpose="training"), "ADMISSION_REQUIRED"),
        ):
            refusal(clip+":"+label, lambda change=change: reader.project(
                clip, span, **{**request, **change}), "MNLM_AUDIO_ALIGNMENT_"+code)
        refusal(clip+":beyond-source", lambda: reader.project(clip,
            [record["sources"]["wav"]["samples"], record["sources"]["wav"]["samples"]+1],
            **request), "MNLM_AUDIO_ALIGNMENT_SAMPLE_INTERVAL")
        tail = record["untested_tail_wav"]
        if tail[1] > tail[0]:
            refusal(clip+":unmeasured-tail", lambda: reader.project(clip, tail, **request),
                    "MNLM_AUDIO_ALIGNMENT_UNCOVERED_INTERVAL")
    corrupt = a.output/"corrupt-profile"
    (corrupt/"profiles").mkdir(parents=True)
    (corrupt/"report.json").symlink_to(a.output/"report.json")
    for path in (a.output/"profiles").iterdir():
        (corrupt/"profiles"/path.name).symlink_to(path)
    damaged = corrupt/clips[0]["windows"][0]["profile"]
    damaged.unlink()
    damaged.write_bytes(b"changed-profile")
    refusal("corrupt-profile", lambda: Alignment(corrupt, report_pin),
            "MNLM_AUDIO_ALIGNMENT_PROFILE_BYTES")
    need(sha(a.output/clips[0]["windows"][0]["profile"]) ==
         clips[0]["windows"][0]["profile_sha256"], "ORIGINAL_PROFILE_MUTATED")
    end = time.perf_counter()
    consumer = dict(report_pin=report_pin, projections=projections,
        unresolved_projections=unresolved_projections, controls=controls,
        seconds=end-tick, physical_sync_calibrated=False, training_ready=False,
        authoritative_sources_unchanged=all(
            sha(a.source/"members"/r["clip"]/(kind+"-s32le.bin")) == r["sources"][kind]["pcm_sha256"]
            for r in clips for kind in ("mp4", "wav")))
    need(consumer["authoritative_sources_unchanged"], "SOURCE_MUTATION")
    consumer_pin = save(a.output/"consumer.json", consumer)
    print(json.dumps(dict(report_pin=report_pin, consumer_pin=consumer_pin,
        clips=[dict(clip=r["clip"], **r["summary"]) for r in clips],
        controls=len(controls), projections=len(projections), seconds=report["seconds"],
        phase_coverage=report["phase_coverage"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--diagnostic-pin", required=True)
    main(p.parse_args())
