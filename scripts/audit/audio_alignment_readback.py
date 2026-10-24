"""Independent decode, FFT/direct-dot, native-coordinate and target coverage audit."""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from senses.perception_corpus import pinned, save, sha


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_ALIGNMENT_READBACK_"+code)


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def rat(x):
    return [x.numerator, x.denominator]


def independent_profile(x, y):
    """Linear convolution with reversed template, explicit zero pad; no scipy correlate."""
    width = len(x)
    template = x-x.mean()
    size = 1 << (len(y)+width-2).bit_length()
    convolution = np.fft.irfft(np.fft.rfft(y, size)*np.fft.rfft(template[::-1], size), size)
    numerator = convolution[width-1:len(y)]
    c, q = np.r_[0., np.cumsum(y)], np.r_[0., np.cumsum(y*y)]
    centered_energy = np.maximum(q[width:]-q[:-width]-(c[width:]-c[:-width])**2/width, 0)
    denominator = np.sqrt(np.sum(template*template)*centered_energy)
    values = np.full(len(numerator), -1.)
    np.divide(numerator, denominator, out=values, where=denominator > 0)
    return np.clip(values, -1, 1)


def main(a):
    started = time.perf_counter()
    r = pinned(a.root/"report.json", a.report_pin)
    consumer = pinned(a.root/"consumer.json", a.consumer_pin)
    bindings = pinned(a.root/"target-bindings.json", a.targets_pin)
    need(consumer["report_pin"] == bindings["alignment_pin"] == a.report_pin, "PUBLICATION_JOIN")
    policy = r["policy"]
    need(policy["minimum_correlation"] == .99 and policy["minimum_rms"] == .0001
         and policy["minimum_margin"] == .05 and policy["near_peak_delta"] == .01
         and policy["maximum_lag_disagreement_samples"] == 1
         and policy["minimum_windows_per_channel"] == 3
         and policy["window_seconds"] == [1, 2] and policy["search_seconds"] == [1, 4]
         and policy["alternative_exclusion_seconds"] == [1, 1000], "FIXED_POLICY")
    for name, expected in r["code"].items():
        need(sha(Path(__file__).resolve().parents[1]/name) == expected, "CODE")
    need(sha(Path(__file__).resolve().parents[1]/"senses/audio_target_binding.py") ==
         bindings["code_pin"], "BINDING_CODE")
    source = Path(r["source"])
    diagnostic = pinned(source/"diagnostic.json", r["diagnostic_pin"])
    need(sha(Path("/usr/bin/ffmpeg")) == diagnostic["ffmpeg_sha256"], "DECODER")
    records, graphs, usable, measured, controls = {}, {}, {}, [], []
    phases = dict(setup=time.perf_counter()-started)
    for clip in r["clips"]:
        tick = time.perf_counter()
        key, rate, channels = clip["clip"], clip["rate"], clip["channels"]
        graph = pinned(Path(clip["graph_file"]), clip["graph_sha256"])
        graphs[key], records[key] = graph, clip
        wave = {}
        for kind in ("wav", "mp4"):
            path = (source/"members"/key/"source.wav" if kind == "wav" else
                    Path(r["corpus"])/"clips"/key/"media"/(key+".mp4"))
            need(sha(path) == clip["sources"][kind]["source_sha256"], "SOURCE_HASH")
            decoded = subprocess.check_output(["/usr/bin/ffmpeg", "-nostdin", "-v", "error",
                "-i", str(path), "-vn", "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", "-"])
            need(hashlib.sha256(decoded).hexdigest() == clip["sources"][kind]["pcm_sha256"]
                 and len(decoded) == clip["sources"][kind]["samples"]*channels*4, "PCM_REPLAY")
            wave[kind] = np.frombuffer(decoded, dtype="<i4").reshape(-1, channels).astype(np.float64)/2**31
        phases[key+":decode"] = time.perf_counter()-tick
        tick = time.perf_counter()
        maximum_error, direct_checks = 0., 0
        counts = {c: 0 for c in range(channels)}
        peaks, offset_values = [], []
        for row in clip["windows"]:
            lo, hi = row["mp4_samples"]
            channel = row["channel"]
            x = wave["mp4"][lo:hi, channel]
            left, right = max(0, lo-rate//4), min(len(wave["wav"]), hi+rate//4)
            y = wave["wav"][left:right, channel]
            path = a.root/row["profile"]
            need(sha(path) == row["profile_sha256"], "PROFILE_HASH")
            stored = np.frombuffer(path.read_bytes(), dtype="<f8")
            cold = independent_profile(x, y)
            need(stored.shape == cold.shape and np.isfinite(stored).all(), "PROFILE_SHAPE")
            error = float(np.max(abs(cold-stored)))
            need(error < 1e-8, "PROFILE_REPLAY")
            maximum_error = max(maximum_error, error)
            offsets = np.arange(len(cold))+left-lo
            index = int(np.argmax(cold))
            lag = int(offsets[index])
            outside = abs(offsets-lag)*1000 > rate
            competitor = np.flatnonzero(outside)[np.argmax(cold[outside])]
            rms = float(np.sqrt(np.mean((x-x.mean())**2)))
            margin = float(cold[index]-cold[competitor])
            qualifies = rms >= .0001 and cold[index] >= .99 and margin >= .05
            near = offsets[cold >= cold[index]-.01]
            need(lag == row["lag_samples"] and bool(qualifies) == row["window_qualified"]
                 and [int(near.min()), int(near.max())] == row["near_peak_lag_range"]
                 and abs(float(cold[index])-row["peak"]) < 1e-8
                 and abs(margin-row["margin"]) < 1e-8, "DECISION_REPLAY")
            for test in set([0, len(cold)-1, index, int(competitor),
                             max(0, min(len(cold)-1, lo-left))]):
                candidate = y[test:test+len(x)]
                cx, cy = x-x.mean(), candidate-candidate.mean()
                denominator = np.sqrt(np.sum(cx*cx)*np.sum(cy*cy))
                exact = float(np.sum(cx*cy)/denominator) if denominator > 0 else -1.
                need(abs(exact-stored[test]) < 1e-8, "DIRECT_DOT_REPLAY")
                direct_checks += 1
            usable[key, channel, row["index"]] = dict(row, window_qualified=bool(qualifies))
            peaks.append(float(cold[index]))
            if qualifies:
                counts[channel] += 1
                offset_values.append(lag)
        offset_values.sort()
        median = offset_values[(len(offset_values)-1)//2] if offset_values else None
        supported = all(n >= 3 for n in counts.values()) and bool(offset_values) and all(
            abs(n-median) <= 1 for n in offset_values)
        need(bool(supported) == clip["summary"]["correspondence_supported"]
             and median == clip["summary"]["offset_samples"], "CONSENSUS")
        measured.append(dict(clip=key, windows=len(clip["windows"]), qualified=sum(counts.values()),
            profile_max_absolute_error=maximum_error, direct_dot_checks=direct_checks,
            offset_samples=median, minimum_peak=min(peaks), per_channel=counts))
        phases[key+":profiles"] = time.perf_counter()-tick

    def project(p):
        clip = p["clip"]
        graph, record = graphs[clip], records[clip]
        need(p["wav_sha256"] == record["sources"]["wav"]["source_sha256"]
             and p["mp4_sha256"] == graph["blob_sha256"] and p["clock"] == graph["clock"]
             and p["split_group"] == graph["split_group"] and not p["physical_sync_calibrated"],
             "PROJECTION_SOURCE")
        span, pieces = p["wav_samples"], []
        for channel in range(record["channels"]):
            support = [s for s in p["support"] if s["channel"] == channel]
            cursor = span[0]
            for s in support:
                row = usable[clip, channel, s["window"]]
                need(row["window_qualified"] and s["wav_samples"][0] == cursor
                     and row["wav_samples"][0] <= cursor < s["wav_samples"][1] <= row["wav_samples"][1]
                     and s["mp4_samples"] == [v-row["lag_samples"] for v in s["wav_samples"]]
                     and s["empirical_lag_range"] == row["near_peak_lag_range"], "COVERAGE")
                cursor = s["wav_samples"][1]
                pieces.append(tuple(s["mp4_samples"]))
            need(cursor == span[1], "COVERAGE")
        expected = []
        for lo, hi in sorted(set(pieces)):
            cursor = 0
            for n in (n for n in graph["nodes"] if n["modality"] == "audio"):
                first, last = max(lo, cursor), min(hi, cursor+n["sample_count"])
                if last > first:
                    expected.append(dict(node=n["id"], sample_in_block=[first-cursor, last-cursor],
                        span=[rat(Fraction(*n["span"][0])+Fraction(first-cursor, record["rate"])),
                              rat(Fraction(*n["span"][0])+Fraction(last-cursor, record["rate"]))],
                        available_after=n["available_after"]))
                cursor += n["sample_count"]
        need(p["native"] == expected, "NATIVE_PROJECTION")

    tick = time.perf_counter()
    for p in consumer["projections"]:
        project(p)
    target_root = Path(bindings["targets"])
    target_report = pinned(target_root/"report.json", bindings["targets_pin"])
    entries = {v["clip"]: v for v in target_report["results"] if v["clip"] in records}
    targets = {c: pinned(target_root/"private"/(c+".json"), e["pin"]) for c, e in entries.items()}

    def target(row):
        clip = row["clip"]
        original = next(e for e in targets[clip]["targets"]["sound_localisation"]
                        if e["provider"]["id"] == row["annotation_id"])
        need(row["original_target_digest"] == digest(original)
             and row["target_sha256"] == entries[clip]["pin"]
             and row["native_interval"] == original["native"], "TARGET_IDENTITY")
        start, end = (Fraction(*t) for t in original["native"]["span"])
        nodes = [n for n in graphs[clip]["nodes"] if n["modality"] == "audio"]
        outside = start < Fraction(*nodes[0]["span"][0]) or end > Fraction(*nodes[-1]["span"][1])
        bounds, cursor = [], 0
        for n in nodes:
            left, right = Fraction(*n["span"][0]), Fraction(*n["span"][1])
            if left < end and start < right:
                a0 = max(start, left)-left
                b0 = min(end, right)-left
                a0, b0 = a0*records[clip]["rate"], b0*records[clip]["rate"]
                bounds.extend([cursor+a0.numerator//a0.denominator,
                               cursor-(-b0.numerator//b0.denominator)])
            cursor += n["sample_count"]
        covered = bool(bounds) and not outside
        if covered:
            lo, hi = min(bounds), max(bounds)
            for channel in range(records[clip]["channels"]):
                position = lo
                for w in records[clip]["windows"]:
                    if w["channel"] == channel and w["window_qualified"]:
                        a0, b0 = w["mp4_samples"]
                        if a0 <= position < b0:
                            position = min(hi, b0)
                covered = covered and position == hi
        need((row["status"] == "covered") == bool(covered), "TARGET_COVERAGE")
        if covered:
            need(row["correspondence"]["wav_samples"] == [min(bounds), max(bounds)], "TARGET_SAMPLES")
            project(row["correspondence"])
        else:
            need(row["code"] == "MNLM_AUDIO_ALIGNMENT_"+(
                "TARGET_OUTSIDE_AUDIO" if outside else "UNCOVERED_INTERVAL"), "TARGET_REFUSAL")
        need(not row["online_feedback_eligible"] and not row["annotation_truth_established"]
             and not row["scientific_admission"], "TARGET_SCOPE")

    for row in bindings["rows"]:
        target(row)
    need(len(bindings["rows"]) == sum(len(t["targets"]["sound_localisation"]) for t in targets.values()),
         "TARGET_ROSTER")
    phases["native_and_private_targets"] = time.perf_counter()-tick
    tick = time.perf_counter()
    def refuse(name, call):
        try:
            call()
        except ValueError as exc:
            controls.append(dict(name=name, code=str(exc)))
        else:
            need(False, "CONTROL_ACCEPTED:"+name)
    for label, change in (
        ("changed-source", lambda p: p.update(wav_sha256="0"*64)),
        ("missing-channel-support", lambda p: p.update(support=[])),
        ("false-synchronization", lambda p: p.update(physical_sync_calibrated=True)),
        ("shifted-native-time", lambda p: p["native"][0].update(span=[[0, 1], [1, 1]])),
    ):
        altered = copy.deepcopy(consumer["projections"][0])
        change(altered)
        refuse(label, lambda p=altered: project(p))
    altered = copy.deepcopy(next(r for r in bindings["rows"] if r["status"] == "unresolved"))
    altered["status"] = "covered"
    refuse("promoted-uncovered-target", lambda: target(altered))
    altered2 = copy.deepcopy(bindings["rows"][0])
    altered2["original_target_digest"] = "0"*64
    refuse("changed-target", lambda: target(altered2))
    phases["negative_controls"] = time.perf_counter()-tick
    result = dict(author_checked=True, report_pin=a.report_pin, consumer_pin=a.consumer_pin,
        bindings_pin=a.targets_pin, measured=measured, projections=len(consumer["projections"]),
        targets=len(bindings["rows"]), covered=bindings["covered"], unresolved=bindings["unresolved"],
        controls=controls, phases=phases, seconds=time.perf_counter()-started,
        physical_sync_calibrated=False, scientific_admission=False, training_ready=False,
        code_pin=sha(Path(__file__)))
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    result_pin = save(a.output, result)
    print(json.dumps(dict(pin=result_pin, measured=measured, covered=result["covered"],
        unresolved=result["unresolved"], controls=len(controls), seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    for key in ("report-pin", "consumer-pin", "targets-pin"):
        p.add_argument("--"+key, required=True)
    main(p.parse_args())
