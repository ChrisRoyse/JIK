"""Complete AMI native candidate measurement and fresh selected CUDA replay."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.perf_counter()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses import timed_readings as tr, visual_bank, audio_bank
from senses.contract import canonical, digest
from senses.perception_corpus import pinned, sha
from senses.timed_speech import FLAGS, save
from lenscommon import take_lease, release_lease
from native_assets import state_sha


def load(medium, name, args):
    os.environ["HF_HUB_CACHE"] = str(args.text_cache if medium == "text" else args.cache)
    if medium == "text":
        from text.window_readings import LearnedLens, by_name
        return LearnedLens(by_name(name), "cuda")
    if name == "dinov2":
        from vision_encoder import VisionEncoder
        return VisionEncoder()
    if name == "whisper":
        from audio_encoder import AudioEncoder
        return AudioEncoder()
    from native_vision import NativeVisionEncoder
    from native_audio import NativeAudioEncoder
    cls = NativeVisionEncoder if medium == "image" else NativeAudioEncoder
    return cls(name, asset_root=args.assets, plan_path=args.asset_plan)


def execute(lens, medium, name, item, captures):
    row, value, blob = item
    if medium == "text":
        from text.window_readings import forward
        return forward(lens, [value])
    from audit.fsv_visual_bank import frontend_probe as image_probe
    from audit.audio_bank_readback import frontend_probe as audio_probe
    count = len(captures)
    hook = (image_probe(lens, name, blob, captures) if medium == "image"
            else audio_probe(lens, name, {**row, "clip": row["segment"]}, blob, captures)[0])
    rng = np.random.get_state()
    try:
        result = lens.measure(value, blob) if medium == "image" else lens.embed([value], blob)
    finally:
        hook.remove()
    after = np.random.get_state()
    tr.need(rng[0] == after[0] and np.array_equal(rng[1], after[1])
            and rng[2:] == after[2:], "RANDOM_SOURCE_SELECTION")
    tr.need(len(captures) > count, "FRONTEND_NOT_EXECUTED")
    return result


def values_windows(medium, output):
    if medium == "image":
        return output["tokens"], output["regions"]
    if medium == "audio":
        return output
    return output[0], output[2][0]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_CACHE"] = str(args.cache)
    inputs = tr.Inputs(args.inputs, args.input_pin, args.source)
    phases = dict(import_source_wrappers=time.perf_counter()-STARTED)
    tick = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    (args.output/"residents-before.txt").write_text(resident)
    (args.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    tr.need(not resident.strip(), "GPU_BUSY")
    take_lease("28", "timed-speech-native-readings")
    phases["resident_and_lease"] = time.perf_counter()-tick
    report = dict(input_pin=inputs.pin, source_report=args.input_pin, flags=FLAGS,
                  acceptance="author-checked only", lanes={}, errors=[], mode=args.mode)
    try:
        tick = time.perf_counter()
        import torch
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        bank = dict(schema=tr.SCHEMA, flags=FLAGS, input_pin=inputs.pin,
                    binding_code=sha(Path(tr.__file__)), instruments={}, readings={})
        reuse = pinned(args.reuse, args.reuse_pin) if args.reuse else None
        if reuse:
            tr.need(args.mode == "measure", "REUSE_MODE")
            reuse_root = Path(reuse["root"])
            reuse_report = pinned(reuse_root/"report.json", reuse["report_pin"])
            tr.need(reuse_report["input_pin"] == inputs.pin and reuse_report["flags"] == FLAGS,
                    "REUSE_SOURCE")
        if args.mode == "replay":
            original = pinned(args.bank/"bank.json", args.bank_pin)
            ids = {n: digest(c) for n, c in original["instruments"].items()}
            reader = tr.Bank(args.bank, args.bank_pin, inputs, ids)
            bytes_read = 0
            for lane in tr.ROSTER:
                for row in inputs.rows[lane.split("_", 1)[0]]:
                    _, array = reader.read(lane, row)
                    bytes_read += array.nbytes
            report["all_value_bytes_rehashed"] = bytes_read
        save(args.output/"inputs.json", dict(input_pin=inputs.pin, rows=inputs.rows))
        phases["runtime_and_cold_values"] = time.perf_counter()-tick
        for lane in tr.ROSTER:
            medium, name = lane.split("_", 1)
            root = args.output/lane
            if reuse and lane in reuse["entries"]:
                tick = time.perf_counter()
                prior = pinned(reuse_root/lane/"entries.json", reuse["entries"][lane])
                proof = reuse_report["lanes"][lane]
                tr.need(proof["identity"] == digest(prior["identity"])
                        and proof["state_before"] == proof["state_after"]
                        and [e["input"] for e in prior["entries"]] == inputs.rows[medium],
                        "REUSE_IDENTITY_OR_INPUT")
                for entry in prior["entries"]:
                    tr.need(entry["file"] == digest(dict(instrument=proof["identity"],
                                                         input=entry["input"]))+".bin"
                            and sha(reuse_root/lane/entry["file"]) == entry["sha256"],
                            "REUSE_BYTES")
                root.symlink_to(reuse_root/lane, target_is_directory=True)
                bank["instruments"][lane], bank["readings"][lane] = prior["identity"], prior["entries"]
                report["lanes"][lane] = {**proof, "reused": True,
                    "source_receipt": args.reuse_pin, "entries_pin": reuse["entries"][lane],
                    "phases": dict(reuse_readback=time.perf_counter()-tick)}
                phases[lane+"_reuse_readback"] = time.perf_counter()-tick
                print(json.dumps(dict(lane=lane, reused=len(prior["entries"]))), flush=True)
                continue
            root.mkdir()
            lane_phases, captures, replays, entries = {}, [], [], []
            lens = None
            try:
                tick = time.perf_counter()
                lens = load(medium, name, args)
                identity = tr.text_contract(lens) if medium == "text" else lens.identity
                before = state_sha(lens.model)
                instrument_id = digest(identity)
                lane_phases["load_and_state"] = time.perf_counter()-tick
                torch.cuda.reset_peak_memory_stats()
                if args.mode == "measure":
                    bank["instruments"][lane] = identity
                    tick = time.perf_counter()
                    for item in inputs.items[medium]:
                        output = execute(lens, medium, name, item, captures)
                        if medium == "text":
                            entry = tr.text_store(root, instrument_id, item[0], output)
                        else:
                            module = visual_bank if medium == "image" else audio_bank
                            entry = module.store(root, name, instrument_id, item[0], output)
                        entries.append(entry)
                    bank["readings"][lane] = entries
                    save(root/"entries.json", dict(identity=identity, entries=entries))
                    lane_phases["all_measure_audit_store"] = time.perf_counter()-tick
                else:
                    tr.need(identity == original["instruments"][lane], "REPLAY_IDENTITY")
                    entries = original["readings"][lane]
                tick = time.perf_counter()
                index = {digest(e["input"]): e for e in entries}
                for item in inputs.selected(medium):
                    entry = index[digest(item[0])]
                    output = execute(lens, medium, name, item, captures)
                    values, windows = values_windows(medium, output)
                    expected = (args.bank if args.mode == "replay" else args.output)/lane/entry["file"]
                    tr.need(values.astype("<f4").tobytes() == expected.read_bytes()
                            and windows == entry["native"], "GPU_REPLAY")
                    replays.append(dict(input=digest(item[0]), sha256=entry["sha256"], exact=True))
                after = state_sha(lens.model)
                tr.need(before == after and not lens.model.training
                        and all(not p.requires_grad for p in lens.model.parameters()), "FROZEN")
                lane_phases["selected_replay_and_state"] = time.perf_counter()-tick
                result = dict(identity=instrument_id, readings=len(entries), replays=replays,
                    frontends=captures, state_before=before, state_after=after,
                    value_bytes=sum(int(np.prod(e["shape"]))*4 for e in entries),
                    native_rows=sum(e["shape"][0] for e in entries),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), phases=lane_phases)
                if args.mode == "measure":
                    result["entries_pin"] = sha(root/"entries.json")
                report["lanes"][lane] = result
                save(root/"report.json", result)
                print(json.dumps(dict(lane=lane, readings=len(entries), replayed=len(replays),
                                      value_bytes=result["value_bytes"])), flush=True)
            except Exception as exc:
                error = dict(lane=lane, error=type(exc).__name__+": "+str(exc))
                report["errors"].append(error)
                save(root/"error.json", error)
                print(json.dumps(error), flush=True)
            finally:
                phases.update({lane+"_"+k: v for k, v in lane_phases.items()})
                tick = time.perf_counter()
                del lens
                gc.collect()
                torch.cuda.empty_cache()
                phases[lane+"_release"] = time.perf_counter()-tick
        tick = time.perf_counter()
        if not report["errors"] and args.mode == "measure":
            save(args.output/"bank.json", bank)
            report["bank_pin"] = sha(args.output/"bank.json")
            ids = {n: digest(c) for n, c in bank["instruments"].items()}
            reader = tr.Bank(args.output, report["bank_pin"], inputs, ids)
            report["all_value_bytes_rehashed"] = sum(reader.read(lane, row)[1].nbytes
                for lane in tr.ROSTER for row in inputs.rows[lane.split("_", 1)[0]])
        report["complete"] = not report["errors"]
        phases["publish_and_consume"] = time.perf_counter()-tick
    finally:
        release_lease()
    report["seconds"] = time.perf_counter()-STARTED
    report["phases"] = phases
    report["phase_coverage"] = sum(phases.values())/report["seconds"]
    report["code"] = sha(Path(__file__))
    save(args.output/"report.json", report)
    tr.need(report["complete"], "INCOMPLETE_BANK")
    tr.need(report["phase_coverage"] >= .9, "PHASE_COVERAGE")
    print(json.dumps(dict(report_pin=sha(args.output/"report.json"),
        seconds=report["seconds"], phase_coverage=report["phase_coverage"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "source", "assets", "asset-plan", "cache", "text-cache", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--input-pin", required=True)
    p.add_argument("--mode", choices=("measure", "replay"), default="measure")
    p.add_argument("--bank", type=Path)
    p.add_argument("--bank-pin")
    p.add_argument("--reuse", type=Path)
    p.add_argument("--reuse-pin")
    main(p.parse_args())
