"""Execute a fixed source observation through complete native candidate banks."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses import visual_bank, audio_bank
from senses.contract import digest
from senses.perception_corpus import pinned, save, sha
from senses.native_observation import public_views
from audit.fsv_visual_bank import frontend_probe as image_probe
from audit.audio_bank_readback import frontend_probe as audio_probe
from lenscommon import take_lease, release_lease
from native_assets import state_sha


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_WITNESS_"+code)


def controls(root, module, cls, inputs, input_pin, original, ids):
    """Exercise admission, identity, shape, roster and physical-byte boundaries."""
    results = []
    prefix = "MNLM_VISUAL_BANK_" if module is visual_bank else "MNLM_AUDIO_BANK_"
    names = module.NAMES
    cases = []
    v = copy.deepcopy(original); v["readings"].pop(names[-1])
    cases.append(("missing-instrument", v, "ROSTER", False))
    v = copy.deepcopy(original); v["readings"][names[0]][0]["instrument_id"] = ids[names[1]]
    cases.append(("swapped-instrument", v, "SWAPPED_INSTRUMENT", False))
    v = copy.deepcopy(original); v["readings"][names[0]][0]["shape"][0] = 1
    cases.append(("pooled-native-rows", v, "NATIVE_SHAPE", False))
    v = copy.deepcopy(original); v["flags"]["scientific_admission"] = True
    cases.append(("false-admission", v, "FLAGS", False))
    v = copy.deepcopy(original); v["instruments"][names[0]]["frozen"] = False
    cases.append(("changed-instrument", v, "IDENTITIES", False))
    cases.append(("premature-training", original, "NATIVE_ADMISSION_REQUIRED", True))
    for name, value, expected, consume in cases:
        target = root/name
        target.mkdir()
        pin = save(target/"bank.json", value)
        try:
            reader = cls(target, pin, inputs, input_pin, ids)
            if consume:
                list(reader.consume("training"))
        except ValueError as exc:
            need(str(exc) == prefix+expected, "CONTROL_REASON:"+name+":"+str(exc))
            results.append(dict(control=name, error=str(exc)))
        else:
            raise ValueError("MNLM_NATIVE_WITNESS_ACCEPTED:"+name)
    target = root/"corrupt-values"
    target.mkdir()
    entry = original["readings"][names[0]][0]
    raw = (root/entry["file"]).read_bytes()
    with (target/entry["file"]).open("xb") as f:
        f.write(bytes([raw[0]^1])+raw[1:])
    pin = save(target/"bank.json", original)
    try:
        list(cls(target, pin, inputs, input_pin, ids).consume())
    except ValueError as exc:
        need(str(exc) == prefix+"READING_BYTES", "CORRUPTION_REASON")
        results.append(dict(control="corrupt-values", error=str(exc)))
    else:
        raise ValueError("MNLM_NATIVE_WITNESS_ACCEPTED:corrupt-values")
    return results


def run(a):
    started = time.perf_counter()
    inputs = pinned(a.inputs/"report.json", a.input_pin)
    source_proof = pinned(a.source_readback, a.source_pin)
    need(source_proof["input_pin"] == a.input_pin and source_proof["author_checked"]
         and not source_proof["scientific_admission"], "SOURCE_READBACK")
    # A busy host refuses before any model CUDA allocation.
    resident = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], check=True, capture_output=True, text=True)
    need(not resident.stdout.strip(), "GPU_RESIDENT")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"resident-before.txt").write_text(resident.stdout, encoding="utf-8")
    gpu_info = subprocess.check_output(["nvidia-smi"], text=True)
    (a.output/"gpu-before.txt").write_text(gpu_info, encoding="utf-8")
    resident_at = datetime.now(timezone.utc).isoformat()
    take_lease("28", "native-observation-witness")
    phases, banks = dict(setup=time.perf_counter()-started), {}
    try:
        t = time.perf_counter()
        import torch
        from vision_encoder import VisionEncoder
        from audio_encoder import AudioEncoder
        from native_vision import NativeVisionEncoder
        from native_audio import NativeAudioEncoder
        os.environ["HF_HUB_CACHE"] = str(a.cache)
        torch.set_num_threads(32)
        phases["imports_runtime"] = time.perf_counter()-t
        for medium in a.media:
            module = visual_bank if medium == "image" else audio_bank
            cls = visual_bank.VisualBank if medium == "image" else audio_bank.AudioBank
            t = time.perf_counter()
            root = a.output/medium
            root.mkdir()
            input_root, input_pin = a.inputs/medium, inputs["input_pins"][medium]
            views = public_views(medium, module.input_views(input_root, input_pin))
            bank = dict(schema="native-"+("visual" if medium == "image" else "audio")+"-bank-v1",
                        input_pin=input_pin, flags=module.FLAGS, instruments={}, readings={})
            frontends, replays, states = [], [], {}
            phases[medium+"_inputs"] = time.perf_counter()-t
            for name in module.NAMES:
                t = time.perf_counter()
                if name == "dinov2":
                    lens = VisionEncoder()
                elif name == "whisper":
                    lens = AudioEncoder()
                else:
                    constructor = NativeVisionEncoder if medium == "image" else NativeAudioEncoder
                    lens = constructor(name, asset_root=a.assets, plan_path=a.asset_plan)
                before = state_sha(lens.model)
                identity = digest(lens.identity)
                bank["instruments"][name] = lens.identity
                phases[name+"_load_state"] = time.perf_counter()-t
                entries, timings = [], []
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                for row, value, blob in views:
                    previous = len(frontends)
                    if medium == "image":
                        hook, expected = image_probe(lens, name, blob, frontends), 1
                    else:
                        hook, expected = audio_probe(lens, name, row, blob, frontends)
                    rng = np.random.get_state()
                    try:
                        result = lens.measure(value, blob) if medium == "image" else lens.embed([value], blob)
                    finally:
                        hook.remove()
                    after_rng = np.random.get_state()
                    need(rng[0] == after_rng[0] and np.array_equal(rng[1],after_rng[1])
                         and rng[2:] == after_rng[2:], "RANDOM_SOURCE_SELECTION")
                    need(len(frontends) == previous+expected, "FRONTEND_ROSTER")
                    entries.append(module.store(root, name, identity, row, result))
                    if hasattr(lens, "last_phases"):
                        timings.append(lens.last_phases)
                phases[name+"_measure_audit_store"] = time.perf_counter()-t
                bank["readings"][name] = entries
                t = time.perf_counter()
                for (row,value,blob), entry in zip(views,entries):
                    result = lens.measure(value,blob) if medium == "image" else lens.embed([value],blob)
                    values, windows = (result["tokens"],result["regions"]) if medium == "image" else result
                    need(values.astype("<f4").tobytes() == (root/entry["file"]).read_bytes()
                         and windows == entry["native"], "REPLAY:"+name)
                    replays.append(dict(instrument=name, file=entry["file"], exact=True))
                after = state_sha(lens.model)
                need(before == after and not lens.model.training
                     and all(not p.requires_grad for p in lens.model.parameters()), "FROZEN:"+name)
                phases[name+"_replay_state"] = time.perf_counter()-t
                states[name] = dict(before=before, after=after, timings=timings,
                    frozen_parameters=all(not p.requires_grad for p in lens.model.parameters()),
                    model_training=lens.model.training,
                    value_bytes=sum(int(np.prod(x["shape"]))*4 for x in entries),
                    rows=sum(x["shape"][0] for x in entries),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
                need(states[name]["value_bytes"] ==
                     inputs["projected_slot_bytes"]["dino" if name == "dinov2" else name],
                     "NATIVE_BYTE_PROJECTION:"+name)
                del lens, result, values
                torch.cuda.empty_cache()
                print(json.dumps(dict(instrument=name, windows=len(entries),
                    measured_bytes=states[name]["value_bytes"], state_equal=True)), flush=True)
            t = time.perf_counter()
            pin = save(root/"bank.json", bank)
            ids = {n:digest(v) for n,v in bank["instruments"].items()}
            observed = sum(1 for _ in cls(root,pin,input_root,input_pin,ids).consume())
            need(observed == len(views)*len(module.NAMES), "CONSUMER_ROSTER")
            faults = controls(root,module,cls,input_root,input_pin,bank,ids)
            banks[medium] = dict(bank_pin=pin, input_pin=input_pin, instruments=ids,
                states=states, frontends=frontends, replays=replays, controls=faults)
            save(root/"report.json", banks[medium])
            phases[medium+"_consumer_controls"] = time.perf_counter()-t
    finally:
        release_lease()
    value = dict(author_checked=True, input_pin=a.input_pin, source_readback_pin=a.source_pin,
        banks=banks, phases=phases, code_pin=sha(Path(__file__)),
        resident_before=dict(timestamp=resident_at, compute_apps=resident.stdout,
            gpu_info_sha256=sha(a.output/"gpu-before.txt")),
        torch_cpu_threads=torch.get_num_threads(),
        scientific_admission=False, training_ready=False, seconds=time.perf_counter()-started)
    value["phase_coverage"] = sum(phases.values())/value["seconds"]
    need(value["phase_coverage"] >= 0.9, "PHASE_COVERAGE")
    pin = save(a.output/"report.json",value)
    print(json.dumps(dict(report_pin=pin, seconds=value["seconds"],
        phase_coverage=value["phase_coverage"], media=list(banks))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","source-readback","assets","asset-plan","cache","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("input-pin","source-pin"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--media",nargs="+",choices=("image","audio"),default=["image","audio"])
    run(p.parse_args())
