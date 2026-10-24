"""Cold original-source, waveform-profile and private-target correspondence audit.

Does not import the producer or correspondence implementation. One recording is
held in memory; source bytes, native graph and target publication are pinned.
"""
import argparse
from bisect import bisect_right
import bz2
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import time
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"census"))
import numpy as np
from range_zip import DirectoryReader
from audit.audio_alignment_readback import independent_profile
from senses.perception_corpus import pinned, save, sha


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_DEVELOPMENT_AUDIO_READBACK_"+code)


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def rat(value):
    return [value.numerator, value.denominator]


def project(p, record, graph):
    need(p["wav_sha256"] == record["sources"]["wav"]["source_sha256"]
         and p["mp4_sha256"] == graph["blob_sha256"] and p["clock"] == graph["clock"]
         and p["split_group"] == graph["split_group"] and not p["physical_sync_calibrated"]
         and not p["scientific_admission"] and not p["annotation_truth_established"], "PROJECTION_SOURCE")
    windows = {(w["channel"], w["index"]): w for w in record["windows"]}
    mapped = []
    for channel in range(record["channels"]):
        cursor = p["wav_samples"][0]
        for s in [s for s in p["support"] if s["channel"] == channel]:
            w = windows[channel, s["window"]]
            need(w["window_qualified"] and s["wav_samples"][0] == cursor
                 and w["wav_samples"][0] <= cursor < s["wav_samples"][1] <= w["wav_samples"][1]
                 and s["mp4_samples"] == [v-w["lag_samples"] for v in s["wav_samples"]]
                 and s["empirical_lag_range"] == w["near_peak_lag_range"], "PROJECTION_COVERAGE")
            cursor = s["wav_samples"][1]
            mapped.append(tuple(s["mp4_samples"]))
        need(cursor == p["wav_samples"][1], "PROJECTION_COVERAGE")
    nodes = [n for n in graph["nodes"] if n["modality"] == "audio"]
    offsets = np.r_[0, np.cumsum([n["sample_count"] for n in nodes])].tolist()
    expected = []
    for start, end in sorted(set(mapped)):
        i, cursor = bisect_right(offsets, start)-1, start
        while i < len(nodes) and cursor < end:
            n, base = nodes[i], offsets[i]
            stop = min(end, offsets[i+1])
            need(base <= cursor < stop, "NATIVE_GAP")
            expected.append(dict(node=n["id"], sample_in_block=[cursor-base, stop-base],
                span=[rat(Fraction(*n["span"][0])+Fraction(cursor-base, record["rate"])),
                      rat(Fraction(*n["span"][0])+Fraction(stop-base, record["rate"]))],
                available_after=n["available_after"]))
            cursor, i = stop, i+1
        need(cursor == end, "NATIVE_GAP")
    need(p["native"] == expected, "NATIVE_PROJECTION")


def target(row, original, target_pin, record, graph):
    need(row["original_target_digest"] == digest(original) and row["target_sha256"] == target_pin
         and row["native_interval"] == original["native"], "TARGET_IDENTITY")
    for key in ("online_feedback_eligible", "annotation_truth_established", "scientific_admission",
                "training_ready", "physical_sync_calibrated"):
        need(row[key] is False, "TARGET_SCOPE")
    native, code, bounds = original["native"], None, []
    if native["status"] != "bound":
        code = "TARGET_UNRESOLVED"
    elif not record["summary"]["correspondence_supported"]:
        code = "CORRESPONDENCE_UNRESOLVED"
    elif any(w["lag_samples"] != 0 for w in record["windows"] if w["window_qualified"]):
        code = "NONZERO_MAPPING_REQUIRES_PROVIDER_CLOCK"
    else:
        start, end = (Fraction(*v) for v in native["span"])
        nodes = [n for n in graph["nodes"] if n["modality"] == "audio"]
        if start < Fraction(*nodes[0]["span"][0]) or end > Fraction(*nodes[-1]["span"][1]):
            code = "TARGET_OUTSIDE_AUDIO"
        else:
            cursor = 0
            for n in nodes:
                lo, hi = (Fraction(*v) for v in n["span"])
                if lo < end and start < hi:
                    a, b = (max(start, lo)-lo)*record["rate"], (min(end, hi)-lo)*record["rate"]
                    bounds.extend([cursor+a.numerator//a.denominator, cursor-(-b.numerator//b.denominator)])
                cursor += n["sample_count"]
            need(bounds, "TARGET_NATIVE_OVERLAP")
            if max(bounds) > record["sources"]["wav"]["samples"]:
                code = "SAMPLE_INTERVAL"
            else:
                for channel in range(record["channels"]):
                    cursor = min(bounds)
                    for w in record["windows"]:
                        if w["channel"] == channel and w["window_qualified"]:
                            if w["mp4_samples"][0] <= cursor < w["mp4_samples"][1]:
                                cursor = min(max(bounds), w["mp4_samples"][1])
                    if cursor != max(bounds):
                        code = "UNCOVERED_INTERVAL"
    need((row["status"] == "covered") == (code is None), "TARGET_DECISION")
    if code:
        need(row["code"] == "MNLM_AUDIO_ALIGNMENT_"+code, "TARGET_REFUSAL")
    else:
        need(row["correspondence"]["wav_samples"] == [min(bounds), max(bounds)], "TARGET_SAMPLES")
        project(row["correspondence"], record, graph)


def audit_clip(root, entry, binding, native_entry, target_entry, directory, member):
    started = time.perf_counter()
    r = pinned(root/"report.json", entry["report_pin"])
    need(len(r["clips"]) == 1 and r["policy"] == binding["policy"], "ALIGNMENT_POLICY")
    record = r["clips"][0]
    clip = entry["clip"]
    need(record["clip"] == clip and record["summary"] == entry["summary"], "CLIP")
    acquisition = pinned(Path(entry["source"])/"receipt.json", entry["receipt_pin"])
    need(acquisition["member"] == member and acquisition["identity"] == directory["identity"]
         and member["name"] == "audios/"+clip+".wav", "MEMBER")
    hr = DirectoryReader(Path(entry["source"])/"header", directory["identity"],
                         offline_pin=acquisition["header_pin"])
    hr.seek(member["header_offset"])
    raw_header = hr.read(256)
    signature, _, flags, method, _, _, crc, packed_bytes, size, nname, nextra = struct.unpack("<I5H3I2H", raw_header[:30])
    need(signature == 0x04034b50 and flags == member["flags"] == 0
         and method == member["method"] == 12 and crc == member["crc32"]
         and packed_bytes == member["compressed_bytes"] and size == member["bytes"]
         and raw_header[30:30+nname].decode() == member["name"]
         and acquisition["data_start"] == member["header_offset"]+30+nname+nextra, "HEADER")
    packed = Path(entry["source"])/"compressed.bz2"
    need(sha(packed) == acquisition["compressed"]["sha256"]
         and packed.stat().st_size == member["compressed_bytes"], "COMPRESSED_BYTES")
    cold = bz2.decompress(packed.read_bytes())
    need(len(cold) == member["bytes"] and zlib.crc32(cold) == member["crc32"]
         and hashlib.sha256(cold).hexdigest() == acquisition["media"]["sha256"]
         and cold == (Path(entry["source"])/"source.wav").read_bytes(), "SOURCE_REPLAY")
    episode = Path(binding["corpus"]).parent/"clips"/clip
    ep = pinned(episode/"report.json", native_entry["report_pin"])
    graph = pinned(episode/"public"/(clip+".json"), ep["files"]["public/"+clip+".json"]["sha256"])
    need(graph["blob_sha256"] == record["sources"]["mp4"]["source_sha256"]
         and graph["clip"] == clip and record["graph_sha256"] == sha(episode/"public"/(clip+".json")),
         "SOURCE_GRAPH")
    channels, rate = record["channels"], record["rate"]
    wave = {}
    for kind in ("wav", "mp4"):
        path = Path(record["sources"][kind]["source_file"])
        need(path == (Path(entry["source"])/"source.wav" if kind == "wav" else
                      episode/"media"/(clip+".mp4")), "SOURCE_PATH")
        need(sha(path) == record["sources"][kind]["source_sha256"], "SOURCE_HASH")
        metadata = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error",
            "-select_streams", "a", "-show_streams", "-of", "json", str(path)]))["streams"]
        need(len(metadata) == 1 and metadata[0]["channels"] == channels
             and int(metadata[0]["sample_rate"]) == rate, "NATIVE_LAYOUT")
        decoded = subprocess.check_output(["/usr/bin/ffmpeg", "-nostdin", "-v", "error",
            "-i", str(path), "-vn", "-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "s32le", "-"])
        need(hashlib.sha256(decoded).hexdigest() == record["sources"][kind]["pcm_sha256"]
             == sha(root/(kind+"-s32le.bin"))
             and len(decoded) == record["sources"][kind]["samples"]*channels*4, "PCM_REPLAY")
        wave[kind] = np.frombuffer(decoded, dtype="<i4").reshape(-1, channels).astype(np.float64)/2**31
    phases = dict(source_and_decode=time.perf_counter()-started)
    tick = time.perf_counter()
    maximum, dots, lags, counts = 0., 0, [], Counter()
    expected = [(c, i) for c in range(channels) for i in
                range(min(len(v) for v in wave.values())//(rate//2))]
    need([(w["channel"], w["index"]) for w in record["windows"]] == expected, "WINDOW_ROSTER")
    for w in record["windows"]:
        lo, hi = w["mp4_samples"]
        need([lo, hi] == [w["index"]*(rate//2), (w["index"]+1)*(rate//2)], "WINDOW_RANGE")
        x = wave["mp4"][lo:hi, w["channel"]]
        left, right = max(0, lo-rate//4), min(len(wave["wav"]), hi+rate//4)
        y = wave["wav"][left:right, w["channel"]]
        path = root/w["profile"]
        need(sha(path) == w["profile_sha256"], "PROFILE_BYTES")
        saved, cold = np.frombuffer(path.read_bytes(), dtype="<f8"), independent_profile(x, y)
        need(saved.shape == cold.shape and np.isfinite(saved).all(), "PROFILE_LAYOUT")
        error = float(np.max(abs(saved-cold)))
        need(error < 1e-8, "PROFILE_REPLAY")
        maximum = max(maximum, error)
        offsets = np.arange(len(saved))+left-lo
        k = int(np.argmax(saved))
        other = np.flatnonzero(abs(offsets-offsets[k])*1000 > rate)
        rival = int(other[np.argmax(saved[other])])
        margin, rms = float(saved[k]-saved[rival]), float(np.sqrt(np.mean((x-x.mean())**2)))
        qualified = rms >= .0001 and saved[k] >= .99 and margin >= .05
        near = offsets[saved >= saved[k]-.01]
        need(w["lag_samples"] == int(offsets[k]) and w["window_qualified"] == bool(qualified)
             and abs(w["peak"]-saved[k]) < 1e-12 and abs(w["margin"]-margin) < 1e-12
             and w["near_peak_lag_range"] == [int(near.min()), int(near.max())], "WINDOW_DECISION")
        if qualified:
            need(int(offsets[int(np.argmax(cold))]) == w["lag_samples"], "COLD_QUALIFIED_LAG")
            lags.append(w["lag_samples"])
            counts[w["channel"]] += 1
        for index in {0, len(saved)-1, k, rival}:
            cx, cy = x-x.mean(), y[index:index+len(x)]-y[index:index+len(x)].mean()
            denominator = np.sqrt(np.sum(cx*cx)*np.sum(cy*cy))
            value = float(np.sum(cx*cy)/denominator) if denominator > 0 else -1.
            need(abs(value-saved[index]) < 1e-8, "DIRECT_DOT")
            dots += 1
    lags.sort()
    median = lags[(len(lags)-1)//2] if lags else None
    enough = all(counts[c] >= 3 for c in range(channels))
    consistent = bool(lags) and all(abs(v-median) <= 1 for v in lags)
    need(record["summary"] == dict(offset_samples=median, original_channels=channels,
        qualified_windows=len(lags), unresolved_windows=len(record["windows"])-len(lags),
        sufficient_anchors=enough, consistent_offset=consistent,
        correspondence_supported=enough and consistent,
        physical_sync_calibrated=False, source_waveforms_identical=False), "CONSENSUS")
    phases["profiles"] = time.perf_counter()-tick
    tick = time.perf_counter()
    attachments = pinned(root/"attachments.json", entry["attachments_pin"])
    need(attachments["alignment_pin"] == entry["report_pin"]
         and attachments["targets_pin"] == binding["targets_pin"]
         and attachments["target_pin"] == target_entry["pin"], "ATTACHMENT_BINDING")
    for obj in (r, attachments):
        for key in ("scientific_admission", "training_ready", "physical_sync_calibrated",
                    "annotation_truth_established", "online_feedback_eligible"):
            need(obj[key] is False, "SCOPE")
    original = pinned(Path(binding["targets"])/"private"/(clip+".json"), target_entry["pin"])
    originals = {r["provider"]["id"]: r for r in original["targets"]["sound_localisation"]}
    need(len(originals) == len(original["targets"]["sound_localisation"]), "DUPLICATE_TARGET")
    need([r["annotation_id"] for r in attachments["rows"]] == list(originals), "TARGET_ROSTER")
    attempts = [p["wav_samples"] for p in attachments["projections"]+attachments["unresolved_projections"]]
    need(sorted(attempts) == sorted(w["wav_samples"] for w in record["windows"] if w["channel"] == 0),
         "PROJECTION_ROSTER")
    for p in attachments["projections"]:
        need(p["clip"] == clip and p["alignment_report"] == entry["report_pin"]
             and record["summary"]["correspondence_supported"], "PROJECTION_BINDING")
        project(p, record, graph)
    for p in attachments["unresolved_projections"]:
        if not record["summary"]["correspondence_supported"]:
            expected_code = "CORRESPONDENCE_UNRESOLVED"
        else:
            complete = True
            for channel in range(channels):
                cursor = p["wav_samples"][0]
                for w in sorted((w for w in record["windows"] if w["channel"] == channel
                                 and w["window_qualified"]), key=lambda w:w["wav_samples"][0]):
                    if w["wav_samples"][0] <= cursor < w["wav_samples"][1]:
                        cursor = min(p["wav_samples"][1], w["wav_samples"][1])
                complete = complete and cursor == p["wav_samples"][1]
            need(not complete, "FALSE_UNCOVERED_INTERVAL")
            expected_code = "UNCOVERED_INTERVAL"
        need(p["code"] == "MNLM_AUDIO_ALIGNMENT_"+expected_code, "PROJECTION_REFUSAL")
    for row in attachments["rows"]:
        target(row, originals[row["annotation_id"]], target_entry["pin"], record, graph)
    need(entry["covered"] == sum(r["status"] == "covered" for r in attachments["rows"])
         and entry["targets"] == len(attachments["rows"]), "TARGET_COUNTS")
    phases["native_and_targets"] = time.perf_counter()-tick
    return dict(clip=clip, windows=len(record["windows"]), qualified=len(lags),
        maximum_profile_error=maximum, direct_dots=dots, targets=len(originals),
        covered=entry["covered"], phases=phases, seconds=time.perf_counter()-started), (record, graph, attachments, originals)


def main(a):
    started = time.perf_counter()
    report = pinned(a.root/"report.json", a.report_pin)
    binding = pinned(a.root/"binding.json", report["binding_pin"])
    need(binding["plan_pin"] == "fce7f893a44b569356114bf46c09a4cdadd52bd43d944106af32f11d2013e64b"
         and binding["corpus_pin"] == "3dce43b6bf0357c26d11bf38ac80c51193cf9817d2dbd7ab933a85a81fa55a11"
         and binding["targets_pin"] == "0f1f50117e7030c7965ec939a4e607b2093ddb95b173b8ff959339fa31c22a79"
         and binding["directory_pin"] == "4ee125660811d9702d4357d6916aad5e1eff6a2a95f11f463e412e982971f068",
         "PRECOMMITTED_SOURCE")
    plan = pinned(Path(binding["plan"]), binding["plan_pin"])
    chosen = [r for r in plan["splits"] if r["role"] != "held_out"]
    need([r["clip"] for r in chosen] == binding["clips"] and len(chosen) == 384, "ROSTER")
    need(binding["policy"]["minimum_correlation"] == .99
         and binding["policy"]["minimum_rms"] == .0001 and binding["policy"]["minimum_margin"] == .05
         and binding["policy"]["minimum_windows_per_channel"] == 3
         and binding["policy"]["window_seconds"] == [1, 2]
         and binding["policy"]["search_seconds"] == [1, 4]
         and binding["policy"]["near_peak_delta"] == .01
         and binding["policy"]["alternative_exclusion_seconds"] == [1, 1000]
         and binding["policy"]["maximum_lag_disagreement_samples"] == 1, "FIXED_POLICY")
    directory = pinned(Path(binding["witness"])/"directory-report.json", binding["directory_pin"])
    dr = DirectoryReader(Path(binding["witness"])/"directory", directory["identity"],
                         offline_pin=directory["directory_pin"])
    with zipfile.ZipFile(dr) as z:
        members = {Path(e.filename).stem: dict(name=e.filename, crc32=e.CRC,
            bytes=e.file_size, compressed_bytes=e.compress_size, header_offset=e.header_offset,
            method=e.compress_type, flags=e.flag_bits) for e in z.infolist()}
    for k, pin in binding["runtime"].items():
        need(sha(Path("/usr/bin")/k) == pin, "RUNTIME")
    for k, pin in binding["code"].items():
        need(sha(Path(__file__).parents[1]/k) == pin, "CODE")
    native = pinned(Path(binding["corpus"]), binding["corpus_pin"])
    targets = pinned(Path(binding["targets"])/"report.json", binding["targets_pin"])
    native = {r["clip"]: r for r in native["results"]}
    targets = {r["clip"]: r for r in targets["results"]}
    need(report["complete"] == (len(report["results"]) == 384 and not report["errors"])
         and report["phase_coverage"] >= .9, "PRODUCER_COMPLETION")
    phases, results, controls = dict(setup=time.perf_counter()-started), [], []
    for entry in report["results"]:
        tick = time.perf_counter()
        clip = entry["clip"]
        need(clip in binding["clips"] and entry["role"] == native[clip]["role"], "DEVELOPMENT_ONLY")
        value, (record, graph, attachments, originals) = audit_clip(a.root/"clips"/clip, entry, binding, native[clip], targets[clip], directory, members[clip])
        results.append(value)
        phases[clip] = time.perf_counter()-tick
        print(json.dumps(dict(done=len(results), clip=clip)), flush=True)
        # Same persisted real source; scratch mutations never replace authoritative bytes.
        if not controls and attachments["projections"]:
            for name, mutate in (
                ("wrong-source", lambda p: p.update(wav_sha256="0"*64)),
                ("wrong-clock", lambda p: p["clock"].update(id="foreign")),
                ("missing-channel", lambda p: p.update(support=[])),
                ("false-physical-sync", lambda p: p.update(physical_sync_calibrated=True)),
                ("shifted-native-time", lambda p: p["native"][0].update(span=[[0, 1], [1, 1]])),
            ):
                changed = copy.deepcopy(attachments["projections"][0])
                mutate(changed)
                try:
                    project(changed, record, graph)
                except ValueError as exc:
                    controls.append(dict(name=name, code=str(exc)))
                else:
                    need(False, "CONTROL_ACCEPTED")
            for name, predicate in (
                ("changed-target", lambda r: r.update(original_target_digest="0"*64)),
                ("promoted-uncovered-target", lambda r: r.update(status="covered")),
            ):
                original_row = (next(r for r in attachments["rows"] if r["status"] == "unresolved")
                                if name == "promoted-uncovered-target" else attachments["rows"][0])
                changed = copy.deepcopy(original_row)
                predicate(changed)
                try:
                    target(changed, originals[changed["annotation_id"]], targets[clip]["pin"], record, graph)
                except ValueError as exc:
                    controls.append(dict(name=name, code=str(exc)))
                else:
                    need(False, "CONTROL_ACCEPTED")
            original_file = a.root/"clips"/clip/record["windows"][0]["profile"]
            before = sha(original_file)
            scratch = a.output.with_suffix(".changed-profile.bin")
            scratch.write_bytes(b"changed profile")
            try:
                need(sha(scratch) == before, "PROFILE_BYTES")
            except ValueError as exc:
                controls.append(dict(name="changed-profile-bytes", code=str(exc)))
            else:
                need(False, "CONTROL_ACCEPTED")
            need(sha(original_file) == before, "CONTROL_MUTATED_SOURCE")
    # Processing errors are preserved, and their roster must account for every omitted clip.
    need({r["clip"] for r in report["results"]}.isdisjoint({r["clip"] for r in report["errors"]})
         and {r["clip"] for r in report["results"]+report["errors"]} == set(binding["clips"]), "ERROR_ROSTER")
    need(len(results)+len(report["errors"]) == 384 and len({r["clip"] for r in results}) == len(results), "RESULT_ROSTER")
    seconds = time.perf_counter()-started
    result = dict(author_checked=True, report_pin=a.report_pin, results=results, controls=controls,
        producer_errors=report["errors"], phases=phases, seconds=seconds,
        phase_coverage=sum(phases.values())/seconds, code_pin=sha(Path(__file__)),
        scientific_admission=False, training_ready=False, physical_sync_calibrated=False)
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    print(json.dumps(dict(pin=save(a.output, result), clips=len(results), seconds=seconds)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-pin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
