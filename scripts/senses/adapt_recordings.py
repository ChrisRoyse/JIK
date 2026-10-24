"""Adapt the explicitly approved acquisition into native, provisional UCR records."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.acoustics import runtime
from senses.contract import SenseError, require
from senses.pipeline import file_hash, write_records
from senses.recorded import audio_records, trajectory_records, web_chapters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acquisition", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--canon", type=Path, required=True)
    ap.add_argument("--text-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trajectory-scope", type=Path, help="optional pinned one-attempt-per-task selection")
    ap.add_argument("--trajectory-scope-sha256")
    args = ap.parse_args()
    require(file_hash(args.acquisition) == args.sha256, "MNLM_UCR_INPUT_DIGEST", str(args.acquisition))
    acquired = json.loads(args.acquisition.read_bytes())
    files = acquired["files"]
    trajectories = [f for f in files if f["source"] == "swe-agent-trajectories"]
    audio = [f for f in files if f["source"] == "web-audio-mpj"]
    require(len(files) == 3 and len(trajectories) == 1 and len(audio) == 2
            and {Path(f["stored"]).name for f in audio} == {"Psalm001.ogg", "Psalm002.ogg"},
            "MNLM_RECORDINGS_SCOPE", "expected one approved trajectory shard and two approved chapters")
    require(sum(f["bytes"] for f in trajectories) <= 256 * 1024 ** 2 and
            sum(f["bytes"] for f in audio) <= 64 * 1024 ** 2,
            "MNLM_RECORDINGS_SCOPE", "approved source size caps exceeded")
    binding = {"acquisition_sha256": args.sha256, "selection_sha256": acquired["selection_sha256"],
               "approval": acquired["approval"], "status": "provisional; no scientific admission"}
    trajectory_binding = {**binding, "trajectory_scope": "first 128 complete trajectories; evaluation labels excluded"}
    selection = None
    require(bool(args.trajectory_scope) == bool(args.trajectory_scope_sha256), "MNLM_TRAJECTORY_SCOPE", "selection path and digest are required together")
    if args.trajectory_scope:
        require(file_hash(args.trajectory_scope) == args.trajectory_scope_sha256, "MNLM_TRAJECTORY_SCOPE", "selection digest differs")
        selection = json.loads(args.trajectory_scope.read_bytes())
        trajectory_binding.update({"trajectory_scope": "one attempt per distinct task; outcome-blind hash order",
                                   "trajectory_scope_sha256": args.trajectory_scope_sha256})
    args.output.mkdir(parents=True, exist_ok=False)
    source = trajectories[0]
    traj = write_records(args.output / "trajectories.ucr.jsonl",
                         trajectory_records(source["stored"], source["sha256"], trajectory_binding, selection=selection), trajectory_binding)
    chapters, text_manifest = web_chapters(args.canon, args.text_manifest)
    for item in audio:
        item["chapter"] = int(Path(item["stored"]).stem.removeprefix("Psalm"))
    audio_binding = {**binding, "audio_runtime": runtime(), "text_manifest": text_manifest,
                     "alignment": "chapter correspondence only; no word/verse timings or forced transcript"}
    sound = write_records(args.output / "audio.ucr.jsonl",
                          audio_records(audio, chapters, text_manifest, audio_binding), audio_binding)
    report = {"acceptance": "author-checked", "trajectories": traj, "audio": sound,
              "adapter_scope": "native UCR adapters; not a completed Stage 1/2 production run",
              "scientific_admission": False, "continual_learning_demonstrated": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trajectory_records": traj["count"], "audio_and_text_records": sound["count"],
                      "status": "provisional", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(3)
