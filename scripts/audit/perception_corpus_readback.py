"""Cold native corpus reader: independent graph reconstruction and source binding."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.perception_episodes_readback import graph_read


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_CORPUS_READBACK_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pinned(path, pin):
    need(sha(path) == pin, "PIN")
    return json.loads(Path(path).read_bytes())


def inspect_clip(root, entry, row, result, bound, originals, *, replay=False):
    clip = row["clip"]
    receipt = pinned(root/"report.json", entry["report_pin"])
    need(set(receipt) == {"schema", "complete", "errors", "binding", "role", "episodes",
                         "files", "flags", "command", "phases"}, "CLIP_SCHEMA")
    need(receipt["schema"] == "perception-native-clip-v1" and receipt["complete"] is True
         and receipt["errors"] == [] and receipt["binding"] == bound
         and receipt["role"] == entry["role"] == row["role"], "CLIP_IDENTITY")
    need(receipt["flags"] == bound["flags"] and not any(receipt["flags"].values()), "FLAGS")
    names = {f"{d}/{clip}.{ext}" for d, ext in
             (("media", "mp4"), ("probes", "json"), ("public", "json"))}
    need(set(receipt["files"]) == names, "PUBLIC_LABEL_LEAK")
    for name in names:
        file = root/name
        need(dict(bytes=file.stat().st_size, sha256=sha(file)) == receipt["files"][name],
             "ARTIFACT")
    media = receipt["files"][f"media/{clip}.mp4"]
    need(media == {k: result["media"][k] for k in ("bytes", "sha256")}
         and sha(originals/clip/"media.mp4") == media["sha256"], "ORIGINAL_SOURCE")
    probe = json.loads((root/f"probes/{clip}.json").read_bytes())
    graph = json.loads((root/f"public/{clip}.json").read_bytes())
    counts = graph_read(graph, probe, clip, media["sha256"], "train", exposure="native_timing_only")
    need(receipt["episodes"] == [counts], "NATIVE_COUNTS")
    need({k: v for k, v in entry.items() if k not in ("role", "report_pin", "reused")}
         == {k: v for k, v in counts.items() if k != "provider_split"}, "CORPUS_COUNTS")
    if replay:
        exe = bound["runtime"]["executable"]
        need(sha(exe) == bound["runtime"]["executable_sha256"], "RUNTIME")
        # Independent invocation over the acquisition path, not the producer's link/helper.
        argv = [exe, "-v", "error", "-threads", "0", "-show_streams", "-show_frames",
            "-show_entries", "stream=index,codec_type,codec_name,time_base,sample_rate,channels,"
            "width,height,start_pts,duration_ts,nb_frames:frame=media_type,stream_index,pts,"
            "duration,pkt_duration,nb_samples", "-of", "json", str(originals/clip/"media.mp4")]
        observed = subprocess.run(argv, capture_output=True, check=True)
        need(not observed.stderr.strip() and json.loads(observed.stdout) == probe, "NATIVE_REPLAY")
    return counts


def read(args):
    started = time.perf_counter()
    report = pinned(args.report, args.report_pin)
    bound = report["binding"]
    need(report["schema"] == bound["schema"] == "perception-native-corpus-v1", "SCHEMA")
    need(report["complete"] is True and not report["errors"], "INCOMPLETE")
    need(bound["exposure"] == "native_timing_only"
         and bound["contributor_disjointness_established"] is False, "EXPOSURE")
    plan = pinned(args.plan, bound["plan_pin"])
    acquisition = pinned(args.acquisition_report, bound["acquisition_pin"])
    need(acquisition["complete"] is True and acquisition["errors"] == []
         and acquisition["plan_pin"] == bound["plan_pin"], "ACQUISITION")
    need(report["requested"] == [r["clip"] for r in plan["splits"]
                                if r["clip"] in set(report["requested"])], "SELECTION_ORDER")
    need([r["clip"] for r in report["results"]] == report["requested"], "COMPLETE_ROSTER")
    need(report["corpus_complete"] == (len(report["requested"]) == 512), "COMPLETE_SCOPE")
    need(not args.require_complete or report["corpus_complete"], "CORPUS_INCOMPLETE")
    need(set(args.replay or []) <= set(report["requested"]), "REPLAY_SCOPE")
    need(0.9 <= report["phase_coverage"] <= 1
         and abs(sum(report["phases"].values())/report["seconds"]-report["phase_coverage"]) < 1e-12,
         "COST")
    root = args.report.parent
    need((root/"binding.json").read_bytes() ==
         json.dumps(bound, sort_keys=True, separators=(",", ":")).encode(), "ROOT_BINDING")
    code = Path(__file__).resolve().parents[1]
    need(all(sha(code/n) == h for n, h in bound["code_sha256"].items()), "CODE")
    roster = {p["clip"]: (p, a) for p, a in zip(plan["splits"], acquisition["results"])}
    phases, counts = dict(setup=time.perf_counter()-started), []
    for entry in report["results"]:
        t = time.perf_counter()
        row, result = roster[entry["clip"]]
        counts.append(inspect_clip(root/"clips"/entry["clip"], entry, row, result, bound,
            args.acquisition_report.parent/"members", replay=entry["clip"] in (args.replay or [])))
        phases[entry["clip"]] = time.perf_counter()-t
        print(json.dumps(dict(clip=entry["clip"], readback=True, seconds=phases[entry["clip"]])),
              flush=True)
    result = dict(author_checked=True, report_pin=args.report_pin, clips=len(counts),
        corpus_complete=report["corpus_complete"], roles=dict(Counter(x["role"] for x in report["results"])),
        totals={k: sum(x[k] for x in counts) for k in
                ("image_frames", "audio_blocks", "audio_samples", "contexts")},
        native_replays=args.replay or [], phases=phases, seconds=time.perf_counter()-started,
        flags=bound["flags"], code_sha256=sha(__file__))
    with args.output.open("xb") as f:
        f.write(json.dumps(result, sort_keys=True, separators=(",", ":")).encode())
    print(json.dumps(dict(cold_pin=sha(args.output), totals=result["totals"])), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "plan", "acquisition-report", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--report-pin", required=True)
    ap.add_argument("--replay", nargs="+")
    ap.add_argument("--require-complete", action="store_true")
    read(ap.parse_args())
