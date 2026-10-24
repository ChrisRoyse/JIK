"""Pinned original clips -> resumable native clock graphs for the cutoff decoder.

PC-07/11/28/32/37: one probe per new clip, no label reads, bounded per-clip
publication. Reuse requires a caller-pinned prior run, never an untrusted cache.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.perception_episodes import command, timeline, FLAGS

PLAN_PIN = "fce7f893a44b569356114bf46c09a4cdadd52bd43d944106af32f11d2013e64b"
ACQUISITION_PIN = "b588b9683e7f7fa639d20b250ff8bcf5039124becb38602f0ba97f425f1a97f4"
COLD_PIN = "bf9fd8354f36bb207d0c2366dc63679bd43f06970f205dbd93861e7c334ce66c"
SCHEMA = "perception-native-corpus-v1"
EXPOSURE = "native_timing_only"


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_CORPUS_"+code)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pinned(path, pin):
    need(sha(path) == pin, "PIN:"+str(path))
    return json.loads(Path(path).read_bytes())


def save(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    need(sha(path) == hashlib.sha256(raw).hexdigest(), "PERSISTENCE")
    return hashlib.sha256(raw).hexdigest()


def sources(args):
    plan = pinned(args.plan, PLAN_PIN)
    acquisition = pinned(args.acquisition_report, ACQUISITION_PIN)
    cold = pinned(args.source_readback, COLD_PIN)
    need(acquisition["complete"] and not acquisition["errors"]
         and acquisition["plan_pin"] == cold["plan_pin"] == PLAN_PIN
         and cold["acquisition_report_pin"] == ACQUISITION_PIN, "SOURCE_COMPLETION")
    rows, results, checked = plan["splits"], acquisition["results"], cold["checked"]
    need(len(rows) == len(results) == len(checked) == 512, "SOURCE_ROSTER")
    need(Counter(x["role"] for x in rows) ==
         dict(increment_one=256, increment_two=128, held_out=128), "SPLIT_COUNTS")
    need(len({x["clip"] for x in rows}) == 512, "SOURCE_DUPLICATE")
    for row, result, observed in zip(rows, results, checked):
        need(row["clip"] == result["clip"] == observed["clip"]
             and row["role"] == result["role"] == observed["role"], "SOURCE_JOIN")
        need(result["media"] == {k: observed[k] for k in ("bytes", "crc32", "sha256")}
             and result["media"]["bytes"] == row["archive"]["bytes"], "SOURCE_BYTES")
    return plan, {r["clip"]: (r, a) for r, a in zip(rows, results)}


def runtime(executable):
    exe = shutil.which(executable)
    need(exe is not None, "FFPROBE_MISSING")
    p = subprocess.run([exe, "-version"], capture_output=True, check=True)
    return dict(executable=exe, executable_sha256=sha(exe), version=p.stdout.decode())


def binding(args, decoder):
    parent = Path(__file__).resolve().parents[1]
    return dict(schema=SCHEMA, plan_pin=PLAN_PIN, acquisition_pin=ACQUISITION_PIN,
        source_readback_pin=COLD_PIN, runtime=decoder, exposure=EXPOSURE,
        code_sha256={n: sha(parent/n) for n in
                     ("senses/perception_corpus.py", "senses/perception_episodes.py")},
        flags=FLAGS, contributor_disjointness_established=False)


def episode_counts(graph):
    return dict(clip=graph["clip"], blob_sha256=graph["blob_sha256"], provider_split="train",
        image_frames=sum(n["modality"] == "image" for n in graph["nodes"]),
        audio_blocks=sum(n["modality"] == "audio" for n in graph["nodes"]),
        audio_samples=sum(n["sample_count"] or 0 for n in graph["nodes"]),
        contexts=len(graph["contexts"]))


def graph_for(clip, blob, probe):
    graph = timeline(clip, blob, probe, "train")
    graph["exposure"] = EXPOSURE
    return graph


def read_item(root, pin, row, result, bound, *, source_root):
    clip = row["clip"]
    v = pinned(root/"report.json", pin)
    need(v["schema"] == "perception-native-clip-v1" and v["complete"]
         and not v["errors"] and v["binding"] == bound and v["role"] == row["role"],
         "CLIP_BINDING")
    names = {f"media/{clip}.mp4", f"probes/{clip}.json", f"public/{clip}.json"}
    need(set(v["files"]) == names, "CLIP_FILES")
    for name in names:
        path = root/name
        need(path.stat().st_size == v["files"][name]["bytes"]
             and sha(path) == v["files"][name]["sha256"], "CLIP_ARTIFACT")
    media = v["files"][f"media/{clip}.mp4"]
    need(media == {k: result["media"][k] for k in ("bytes", "sha256")}
         and sha(source_root/clip/"media.mp4") == media["sha256"], "ORIGINAL")
    probe = json.loads((root/f"probes/{clip}.json").read_bytes())
    graph = json.loads((root/f"public/{clip}.json").read_bytes())
    need(graph == graph_for(clip, media["sha256"], probe),
         "NATIVE_GRAPH")
    need(v["episodes"] == [episode_counts(graph)] and v["flags"] == FLAGS, "CLIP_COUNTS")
    return v


def build_item(args, row, result, bound):
    clip = row["clip"]
    root = args.output/"clips"/clip
    need(not root.exists(), "UNACCEPTED_PARTIAL:"+clip)
    root.mkdir(parents=True)
    phases = {}
    t = time.perf_counter()
    member_root = args.acquisition_report.parent/"members"/clip
    receipt = pinned(member_root/"receipt.json", result["receipt_sha256"])
    need(receipt["media"] == result["media"] and receipt["member"] == row["archive"],
         "MEMBER_RECEIPT")
    original = member_root/"media.mp4"
    need(original.stat().st_size == result["media"]["bytes"]
         and sha(original) == result["media"]["sha256"], "ORIGINAL")
    for name in ("media", "probes", "public"):
        (root/name).mkdir()
    path = root/"media"/(clip+".mp4")
    os.link(original, path)
    need(os.path.samefile(original, path), "SOURCE_LINK")
    phases["source_bind_link"] = time.perf_counter()-t
    t = time.perf_counter()
    cmd = command(bound["runtime"]["executable"], path)
    p = subprocess.run(cmd, capture_output=True, check=True)
    need(not p.stderr.strip(), "PROBE_STDERR")
    (root/"probes"/(clip+".json")).write_bytes(p.stdout)
    phases["native_probe"] = time.perf_counter()-t
    t = time.perf_counter()
    probe = json.loads(p.stdout)
    graph = graph_for(clip, result["media"]["sha256"], probe)
    save(root/"public"/(clip+".json"), graph)
    phases["graph"] = time.perf_counter()-t
    t = time.perf_counter()
    files = {}
    for name in (f"media/{clip}.mp4", f"probes/{clip}.json", f"public/{clip}.json"):
        item = root/name
        files[name] = dict(bytes=item.stat().st_size, sha256=sha(item))
    need(files[f"media/{clip}.mp4"]["sha256"] == result["media"]["sha256"], "SOURCE_DRIFT")
    phases["artifact_hashes"] = time.perf_counter()-t
    report = dict(schema="perception-native-clip-v1", complete=True, errors=[],
        binding=bound, role=row["role"], episodes=[episode_counts(graph)], files=files,
        flags=FLAGS, command=cmd, phases=phases)
    pin = save(root/"report.json", report)
    return report, pin


def run(args):
    started = time.perf_counter()
    plan, roster = sources(args)
    decoder = runtime(args.ffprobe)
    bound = binding(args, decoder)
    wanted = args.clips or [x["clip"] for x in plan["splits"]]
    need(len(set(wanted)) == len(wanted) and set(wanted) <= set(roster), "REQUEST_ROSTER")
    wanted = [x["clip"] for x in plan["splits"] if x["clip"] in set(wanted)]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"binding.json"
    if path.exists():
        need(path.read_bytes() == encoded(bound), "ROOT_BINDING")
    else:
        save(path, bound)
    need(bool(args.resume_report) == bool(args.resume_pin), "RESUME_ARGS")
    previous = {}
    if args.resume_report:
        old = pinned(args.resume_report, args.resume_pin)
        need(old["binding"] == bound, "RESUME_BINDING")
        previous = {x["clip"]: x for x in old["results"]}
        need(len(previous) == len(old["results"]) and set(previous) <= set(roster), "RESUME_ROSTER")
    lock = args.output/".writer"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, encoded(dict(pid=os.getpid(), requested=wanted)))
    os.close(fd)
    results, errors = [], []
    phases = dict(setup=time.perf_counter()-started)
    try:
        for clip in wanted:
            t = time.perf_counter()
            row, result = roster[clip]
            try:
                if clip in previous:
                    pin = previous[clip]["report_pin"]
                    report = read_item(args.output/"clips"/clip, pin, row, result, bound,
                                       source_root=args.acquisition_report.parent/"members")
                else:
                    report, pin = build_item(args, row, result, bound)
                results.append(dict(clip=clip, role=row["role"], report_pin=pin,
                    reused=clip in previous, **{k: v for k, v in report["episodes"][0].items()
                                               if k not in ("clip", "provider_split")}))
                event = dict(clip=clip, ok=True, reused=clip in previous)
            except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
                errors.append(dict(clip=clip, error=str(exc), type=type(exc).__name__))
                event = dict(clip=clip, ok=False, error=str(exc))
            phases[clip] = time.perf_counter()-t
            print(json.dumps({**event, "seconds": phases[clip]}), flush=True)
        seconds = time.perf_counter()-started
        report = dict(schema=SCHEMA, binding=bound, requested=wanted,
            complete=not errors and len(results) == len(wanted),
            corpus_complete=not errors and len(results) == len(roster),
            results=results, errors=errors, phases=phases, seconds=seconds,
            phase_coverage=sum(phases.values())/seconds, flags=FLAGS)
        name = "run-"+uuid.uuid4().hex+".json"
        pin = save(args.output/name, report)
        print(json.dumps(dict(report=name, pin=pin, clips=len(results), errors=len(errors),
                              corpus_complete=report["corpus_complete"])), flush=True)
        return 0 if report["complete"] else 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "acquisition-report", "source-readback", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--clips", nargs="+")
    ap.add_argument("--resume-report", type=Path)
    ap.add_argument("--resume-pin")
    raise SystemExit(run(ap.parse_args()))
