"""Cold source/selection/member readback, independent of the acquisition writer.

O(annotation bytes + selected compressed/original bytes), bounded member buffers.
No encoder, label-to-sense join, native clock assertion, or training promotion.
"""
import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"census"))
from range_zip import DirectoryReader


def need(ok, message):
    if not ok:
        raise ValueError("MNLM_TRAIN_READBACK_"+message)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def pinned(path, pin):
    need(digest(path) == pin, "PIN:"+str(path))
    return json.loads(path.read_bytes())


def census(source, root, pin, plan_path, plan_pin):
    report = pinned(root/"report.json", pin)
    need(report["complete"] and not report["errors"], "CENSUS_INCOMPLETE")
    for name, entry in report["files"].items():
        need(Path(name).name == name, "CENSUS_PATH")
        need((root/name).stat().st_size == entry["bytes"]
             and digest(root/name) == entry["sha256"], "CENSUS_FILE:"+name)
    annotation = source/"train_annotations.zip"
    need(digest(annotation) == report["annotation"]["sha256"], "ANNOTATION_PIN")
    with zipfile.ZipFile(annotation) as z:
        need(z.namelist() == ["all_train.json"], "ANNOTATION_MEMBER")
        raw = z.read("all_train.json")
        need(hashlib.sha256(raw).hexdigest() == report["annotation"]["member_sha256"],
             "ANNOTATION_CONTENT")
        annotations = json.loads(raw)
    reader = DirectoryReader(source/"video-directory", report["media_archive"],
                             offline_pin=report["directory_pin"])
    with zipfile.ZipFile(reader) as z:
        members = {Path(x.filename).stem: dict(name=x.filename, crc32=x.CRC,
                    bytes=x.file_size, compressed_bytes=x.compress_size,
                    method=x.compress_type, flags=x.flag_bits, header_offset=x.header_offset)
                   for x in z.infolist()}
    public = [json.loads(line) for line in (root/"media.jsonl").read_bytes().splitlines()]
    private = [json.loads(line) for line in (root/"private-census.jsonl").read_bytes().splitlines()]
    need(len(public) == len(private) == len(members) == len(annotations) == 2184,
         "SOURCE_CARDINALITY")
    need({x["clip"] for x in public} == set(annotations) == set(members), "SOURCE_JOIN")
    private_by_id = {x["clip"]: x for x in private}
    totals = Counter()
    for row in public:
        need(set(row) == {"clip", "provider_split", "archive", "reported_geometry",
                         "exposed_sample", "source_group", "group_scope"}, "PUBLIC_LABEL_LEAK")
        clip = row["clip"]
        a = annotations[clip]
        need(a["metadata"]["split"] == row["provider_split"] == "train"
             and a["metadata"]["video_id"] == row["source_group"] == clip, "TRAIN_SCOPE")
        need(row["archive"] == members[clip], "ARCHIVE_JOIN")
        need(row["reported_geometry"] == {k: a["metadata"][k] for k in
             ("frame_rate", "num_frames", "resolution", "audio_samples", "audio_sample_rate")},
             "GEOMETRY_JOIN")
        task_counts = {k: len(v) for k, v in a.items() if k != "metadata"}
        totals.update(task_counts)
        need(private_by_id[clip]["task_counts"] == task_counts
             and private_by_id[clip]["annotation_sha256"] ==
             hashlib.sha256(canonical(a)).hexdigest(), "PRIVATE_JOIN")
    summary = json.loads((root/"private-summary.json").read_bytes())
    need(summary["task_annotations"] == dict(totals), "TASK_COUNTS")
    plan = pinned(plan_path, plan_pin)
    need(plan["identity"] == report["media_archive"]
         and plan["directory_pin"] == report["directory_pin"], "PLAN_SOURCE")
    need(plan["flags"]["scientific_admission"] is False
         and plan["flags"]["training_ready"] is False, "PLAN_PROMOTION")
    selection = plan["selection"]
    need(selection["target"] == 512 and selection["seed"] == 25
         and selection["namespace"] == "mnlm-native-foundation-v1", "SELECTION_RULE")
    excluded = set(selection["excluded_clips"])
    need(excluded == {"video_10993", "video_2564", "video_3501", "video_874", "video_9431"},
         "EXPOSURE_RULE")
    order = sorted(set(members)-excluded, key=lambda c:
                   hashlib.sha256(("mnlm-native-foundation-v1|25|"+c).encode()).digest())[:512]
    roles = ["increment_one"]*256+["increment_two"]*128+["held_out"]*128
    expected = [dict(clip=c, role=r, archive=members[c]) for c, r in zip(order, roles)]
    need(plan["splits"] == expected, "FROZEN_SELECTION")
    return plan, dict(clips=len(public), selected=len(expected), roles=dict(Counter(roles)),
        task_counts=dict(totals), compressed_bytes=sum(x["archive"]["compressed_bytes"] for x in expected),
        media_bytes=sum(x["archive"]["bytes"] for x in expected))


def original_member(root, identity, member, receipt_pin):
    r = pinned(root/"receipt.json", receipt_pin)
    need(r["identity"] == identity and r["member"] == member, "RECEIPT_SOURCE")
    need(r["scientific_admission"] is False and r["training_ready"] is False, "MEDIA_PROMOTION")
    binding = json.loads((root/"range.json").read_bytes())
    need(binding == dict(identity=identity, member=member, data_start=r["data_start"]), "RANGE_BINDING")
    hr = pinned(root/"header/ranges.json", r["header_pin"])
    need(hr["identity"] == identity, "HEADER_IDENTITY")
    need((root/"header.pin").read_text() == r["header_pin"], "HEADER_PIN")
    data = {}
    for e in hr["ranges"]:
        path = root/"header"/f"{e['start']}-{e['end']}.bin"
        need(digest(path) == e["sha256"], "HEADER_BYTES")
        b = path.read_bytes()
        need(len(b) == e["end"]-e["start"]+1, "HEADER_LENGTH")
        data.update(enumerate(b, e["start"]))
    def span(offset, length):
        need(all(i in data for i in range(offset, offset+length)), "HEADER_MISSING")
        return bytes(data[i] for i in range(offset, offset+length))
    offset = member["header_offset"]
    h = struct.unpack("<I5H3I2H", span(offset, 30))
    need(h[0] == 0x04034B50 and h[2] == member["flags"] == 0
         and h[3] == member["method"] == 12 and h[6] == member["crc32"]
         and h[7] == member["compressed_bytes"] and h[8] == member["bytes"], "LOCAL_HEADER")
    need(span(offset+30, h[9]).decode("utf8") == member["name"]
         and offset+30+h[9]+h[10] == r["data_start"], "LOCAL_NAME_OFFSET")
    packed = root/"compressed.bz2"
    need(packed.stat().st_size == member["compressed_bytes"]
         and digest(packed) == r["compressed"]["sha256"], "PACKED_CONTENT")
    # Independent stdlib stream interface, rather than the writer's low-level decoder.
    total, crc, sha = 0, 0, hashlib.sha256()
    with bz2.open(packed, "rb") as f:
        while raw := f.read(1024*1024):
            total += len(raw)
            need(total <= member["bytes"], "UNPACKED_BUDGET")
            crc = zlib.crc32(raw, crc)
            sha.update(raw)
    observed = dict(bytes=total, crc32=crc, sha256=sha.hexdigest())
    need(observed == r["media"] and total == member["bytes"] and crc == member["crc32"],
         "UNPACKED_CONTENT")
    need((root/"media.mp4").stat().st_size == total
         and digest(root/"media.mp4") == sha.hexdigest(), "PERSISTED_MEDIA")
    need(not (root/"compressed.part").exists() and not (root/"decoded.partial").exists(),
         "UNACCEPTED_PARTIAL")
    return observed


def run(args):
    started = time.perf_counter()
    plan, counts = census(args.source, args.census, args.census_pin, args.plan, args.plan_pin)
    phases = dict(census_and_selection=time.perf_counter()-started)
    checked, duplicates = [], []
    if args.acquisition_report:
        t = time.perf_counter()
        acquisition = pinned(args.acquisition_report, args.acquisition_report_pin)
        need(acquisition["complete"] and not acquisition["errors"]
             and acquisition["plan_pin"] == args.plan_pin, "ACQUISITION_INCOMPLETE")
        need(len(acquisition["results"]) == 512, "ACQUISITION_COUNT")
        hashes = {}
        for row, result in zip(plan["splits"], acquisition["results"]):
            need((row["clip"], row["role"]) == (result["clip"], result["role"]), "RESULT_ORDER")
            media = original_member(args.acquisition_report.parent/"members"/row["clip"],
                                    plan["identity"], row["archive"], result["receipt_sha256"])
            need(media == result["media"], "RESULT_CONTENT")
            checked.append(dict(clip=row["clip"], role=row["role"], **media))
            hashes.setdefault(media["sha256"], []).append(row["clip"])
        duplicates = [v for v in hashes.values() if len(v) > 1]
        need(sum(x["bytes"] for x in checked) == counts["media_bytes"], "TOTAL_BYTES")
        phases["member_readback"] = time.perf_counter()-t
    result = dict(schema="native-train-source-readback-v1", author_checked=True,
        census_pin=args.census_pin, plan_pin=args.plan_pin,
        acquisition_report_pin=args.acquisition_report_pin, counts=counts, checked=checked,
        exact_duplicate_clips=duplicates, phase_seconds=phases, seconds=time.perf_counter()-started,
        flags=dict(meaning_asserted=False, scientific_admission=False, training_ready=False,
                   native_timing_attested=False, contributor_disjointness_established=False),
        code_sha256=digest(Path(__file__)))
    with args.output.open("xb") as f:
        f.write(canonical(result))
    print(json.dumps(dict(output=str(args.output), pin=digest(args.output),
                          clips=len(checked), duplicate_groups=len(duplicates))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("source", "census", "plan", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    for name in ("census-pin", "plan-pin"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--acquisition-report", type=Path)
    ap.add_argument("--acquisition-report-pin")
    args = ap.parse_args()
    need(bool(args.acquisition_report) == bool(args.acquisition_report_pin), "ACQUISITION_ARGS")
    run(args)
