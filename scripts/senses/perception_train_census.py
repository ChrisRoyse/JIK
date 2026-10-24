"""Provider-train annotation and compressed-media inventory; no sensory/label mixing.

O(annotation bytes + directory bytes + annotations). The public media roster
contains only archive membership and source geometry. Human answer sheets and
class frequencies remain explicitly private calibration information.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"census")]
from range_zip import DirectoryReader, entries, encode, sha, require

ANNOTATION = dict(bytes=26247728, sha256="9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78",
                  member="all_train.json", unpacked_bytes=132417530, crc32=2693650891)
TASKS = ("object_tracking", "point_tracking", "action_localisation", "sound_localisation",
         "mc_question", "grounded_question")
FLAGS = dict(meaning_asserted=False, scientific_admission=False, training_ready=False,
             occurrence_established=False, native_timing_attested=False, sealed_evaluation=False,
             calibration_metadata_only=True)


def load_annotations(path):
    path = Path(path)
    require(path.stat().st_size == ANNOTATION["bytes"], "ANNOTATION_SIZE")
    with path.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == ANNOTATION["sha256"], "ANNOTATION_PIN")
    with zipfile.ZipFile(path) as z:
        require(z.namelist() == [ANNOTATION["member"]], "ANNOTATION_MEMBERS")
        member = z.getinfo(ANNOTATION["member"])
        require(member.file_size == ANNOTATION["unpacked_bytes"] and member.CRC == ANNOTATION["crc32"],
                "ANNOTATION_LAYOUT")
        raw = z.read(member)
    value = json.loads(raw)
    require(type(value) is dict and len(value) == 2184, "TRAIN_ROSTER")
    return value, sha(raw)


def inventory(annotations, members, sample_ids):
    archive = {Path(e["name"]).stem: e for e in members}
    require(set(annotations) == set(archive), "MEDIA_ANNOTATION_ROSTER")
    public, private, errors = [], [], []
    totals, coverage = Counter(), Counter()
    labels = {k: defaultdict(set) for k in ("action_localisation", "sound_localisation")}
    segments = {k: Counter() for k in labels}
    label_names = {k: defaultdict(set) for k in labels}
    group_limits = Counter()
    for clip in sorted(annotations):
        x = annotations[clip]
        try:
            require(set(x) == {"metadata", *TASKS}, "TASK_SCHEMA")
            m = x["metadata"]
            require(m["split"] == "train" and m["video_id"] == clip, "PROVIDER_TRAIN_ONLY")
            require(type(m["num_frames"]) is int and m["num_frames"] > 0
                    and isinstance(m["resolution"], list) and len(m["resolution"]) == 2
                    and all(type(n) is int and n > 0 for n in m["resolution"])
                    and type(m["audio_samples"]) is int and m["audio_samples"] > 0
                    and m["frame_rate"] > 0 and m["audio_sample_rate"] > 0, "PROVIDER_GEOMETRY")
            for key in TASKS:
                require(isinstance(x[key], list), "TASK_LIST")
            # Facts about the file are public. Dataset/task flags and answer texts are not.
            public.append(dict(clip=clip, provider_split="train", archive=archive[clip],
                reported_geometry={k: m[k] for k in ("frame_rate", "num_frames", "resolution",
                                                     "audio_samples", "audio_sample_rate")},
                exposed_sample=clip in sample_ids, source_group=clip,
                group_scope="source_clip_only; contributor/session identities unavailable"))
            row = dict(clip=clip, annotation_sha256=sha(encode(x)),
                       task_counts={k: len(x[k]) for k in TASKS}, issues=[])
            for task in TASKS:
                totals[task] += len(x[task])
                coverage[task] += bool(x[task])
            for task in labels:
                for event in x[task]:
                    require(type(event["label_id"]) is int and isinstance(event["label"], str), "EVENT_LABEL")
                    label = str(event["label_id"])
                    labels[task][label].add(clip)
                    label_names[task][label].add(event["label"])
                    segments[task][label] += 1
                    for field in ("timestamps", "frame_ids"):
                        span = event[field]
                        if not (isinstance(span, list) and len(span) == 2
                                and all(type(n) is int for n in span) and 0 <= span[0] < span[1]):
                            row["issues"].append(dict(task=task, id=event["id"], code="nonpositive_"+field))
                    if isinstance(event["frame_ids"], list) and len(event["frame_ids"]) == 2:
                        if event["frame_ids"][1] > m["num_frames"]:
                            row["issues"].append(dict(task=task, id=event["id"], code="frame_end_beyond_reported_count"))
            row["qa_reasoning"] = dict(Counter(q.get("reasoning", "unspecified")
                                               for k in ("mc_question", "grounded_question") for q in x[k]))
            private.append(row)
            group_limits["source_clip_only"] += 1
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(dict(clip=clip, code=str(exc)))
    frequencies = {task: [dict(label_id=int(label), names=sorted(label_names[task][label]),
                               clips=len(clips), segments=segments[task][label])
                          for label, clips in sorted(table.items(), key=lambda kv: (-len(kv[1]), int(kv[0])))]
                   for task, table in labels.items()}
    summary = dict(clips=len(public), task_annotations=dict(totals), task_clip_coverage=dict(coverage),
        class_frequencies=frequencies, sample_overlap=sum(x["exposed_sample"] for x in public),
        media_bytes=sum(x["archive"]["bytes"] for x in public),
        compressed_media_bytes=sum(x["archive"]["compressed_bytes"] for x in public),
        group_limits=dict(group_limits), issue_clips=sum(bool(x["issues"]) for x in private),
        issue_counts=dict(Counter(e["code"] for x in private for e in x["issues"])), errors=errors)
    return public, private, summary


def run(args):
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    phases = {}
    t = time.perf_counter()
    annotations, annotation_member_pin = load_annotations(args.source/"train_annotations.zip")
    phases["annotation_read"] = time.perf_counter()-t
    t = time.perf_counter()
    acquired = json.loads(args.directory_receipt.read_bytes())
    reader = DirectoryReader(args.source/"video-directory", acquired["identity"], offline_pin=acquired["pin"])
    members = entries(reader)
    sample_raw = args.sample_report.read_bytes()
    require(sha(sample_raw) == args.sample_report_pin, "SAMPLE_REPORT_PIN")
    sample = json.loads(sample_raw)
    require(sample["complete"] and not sample["errors"], "SAMPLE_REPORT")
    sample_ids = {e["clip"] for e in sample["episodes"]}
    phases["directory_and_exposure"] = time.perf_counter()-t
    t = time.perf_counter()
    public, private, summary = inventory(annotations, members, sample_ids)
    phases["census"] = time.perf_counter()-t
    t = time.perf_counter()
    artifacts = {"media.jsonl": b"".join(encode(r)+b"\n" for r in public),
                 "private-census.jsonl": b"".join(encode(r)+b"\n" for r in private),
                 "private-summary.json": encode(summary)}
    for name, raw in artifacts.items():
        (out/name).write_bytes(raw)
        require((out/name).read_bytes() == raw, "PUBLICATION_BYTES")
    phases["write_and_readback"] = time.perf_counter()-t
    wall = time.perf_counter()-start
    report = dict(schema="perception-train-census-v1", complete=not summary["errors"],
        annotation={**ANNOTATION, "member_sha256": annotation_member_pin},
        media_archive=acquired["identity"], directory_pin=acquired["pin"],
        sample_report_pin=args.sample_report_pin, flags=FLAGS, errors=summary["errors"],
        files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in artifacts.items()},
        phases=phases, seconds=wall, phase_coverage=sum(phases.values())/wall,
        code_sha256={str(Path(__file__).name): sha(Path(__file__).read_bytes()),
                     "range_zip.py": sha((Path(__file__).parents[1]/"census/range_zip.py").read_bytes())})
    (out/"report.json").write_bytes(encode(report))
    print(json.dumps(dict(pin=sha(encode(report)), complete=report["complete"], clips=summary["clips"],
                          issue_clips=summary["issue_clips"], seconds=wall)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("source", "directory-receipt", "sample-report", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--sample-report-pin", required=True)
    run(ap.parse_args())
