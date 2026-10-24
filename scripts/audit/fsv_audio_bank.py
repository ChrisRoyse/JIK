"""Two real stereo prefixes -> five native audio instruments -> strict cold readback."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses.audio_bank import AudioBank, FLAGS, NAMES, input_views, store
from senses.contract import canonical, digest
from senses.perception_instruments import sha
from native_assets import state_sha
from native_audio import NativeAudioEncoder
from audit.audio_bank_readback import cold, frontend_probe, need
from lenscommon import take_lease, release_lease
from measure import nvidia_smi


def faults(root, inputs, input_pin, original):
    controls = []
    identities = {n: digest(v) for n, v in original["instruments"].items()}
    def attempt(label, value, code, purpose=None):
        target = root/label
        target.mkdir()
        raw = canonical(value)
        (target/"bank.json").write_bytes(raw)
        try:
            bank = AudioBank(target, sha(raw), inputs, input_pin, identities)
            if purpose:
                list(bank.consume(purpose))
        except ValueError as exc:
            need(str(exc) == "MNLM_AUDIO_BANK_"+code, label+str(exc))
            controls.append(dict(name=label, error=str(exc)))
        else:
            raise AssertionError(label+" accepted")
    v = copy.deepcopy(original); v["readings"].pop("clap")
    attempt("missing-instrument", v, "ROSTER")
    v = copy.deepcopy(original); v["readings"]["ast"][0]["instrument_id"] = identities["hubert"]
    attempt("swapped-instrument", v, "SWAPPED_INSTRUMENT")
    v = copy.deepcopy(original); v["readings"]["wavlm"][0]["shape"] = [1, 768]
    attempt("pooled-temporal-rows", v, "NATIVE_SHAPE")
    v = copy.deepcopy(original); v["readings"]["whisper"].pop()
    attempt("missing-channel", v, "CHANNEL_ROSTER")
    v = copy.deepcopy(original); v["readings"]["hubert"][0]["native"].pop()
    attempt("missing-tail", v, "NATIVE_WINDOWS")
    v = copy.deepcopy(original); v["readings"]["clap"][0]["native"][0]["grid"] = [1, 64]
    attempt("changed-time-frequency-grid", v, "NATIVE_WINDOWS")
    v = copy.deepcopy(original); v["readings"]["whisper"][0]["native"][1]["resampled_start"] += 1
    attempt("shifted-time", v, "NATIVE_WINDOWS")
    v = copy.deepcopy(original); v["readings"]["wavlm"][0]["input"]["span"]["channel"] = 1
    attempt("mixed-channel", v, "CHANNEL_ROSTER")
    v = copy.deepcopy(original); v["instruments"]["ast"]["frozen"] = False
    attempt("changed-frozen-identity", v, "IDENTITIES")
    v = copy.deepcopy(original); v["input_pin"] = "0"*64
    attempt("wrong-source", v, "SOURCE")
    v = copy.deepcopy(original); v["flags"]["scientific_admission"] = True
    attempt("false-admission", v, "FLAGS")
    attempt("premature-training", original, "NATIVE_ADMISSION_REQUIRED", purpose="training")
    for label, nonfinite in (("corrupt-bytes", False), ("nonfinite", True)):
        target = root/label
        target.mkdir()
        v = copy.deepcopy(original)
        entry = v["readings"]["whisper"][0]
        raw = (root/entry["file"]).read_bytes()
        changed = np.float32(np.nan).tobytes()+raw[4:] if nonfinite else bytes([raw[0]^1])+raw[1:]
        (target/entry["file"]).write_bytes(changed)
        if nonfinite:
            entry["sha256"] = sha(changed)
        body = canonical(v)
        (target/"bank.json").write_bytes(body)
        try:
            list(AudioBank(target, sha(body), inputs, input_pin, identities).consume())
        except ValueError as exc:
            code = "MNLM_AUDIO_BANK_"+("NONFINITE" if nonfinite else "READING_BYTES")
            need(str(exc) == code, label+str(exc))
            controls.append(dict(name=label, error=code))
        else:
            raise AssertionError(label+" accepted")
    return controls


def run(args):
    import torch
    from audio_encoder import AudioEncoder
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    phases, frontends, replays, states = {}, [], [], {}
    t = time.perf_counter()
    (root/"resident-before.txt").write_text(nvidia_smi())
    views = input_views(args.inputs, args.input_pin)
    need(len(views) == 4 and len({v[0]["clip"] for v in views}) == 2, "declared real stereo corpus")
    phases["source_and_resident"] = time.perf_counter()-t
    bank = dict(schema="native-audio-bank-v1", input_pin=args.input_pin, flags=FLAGS,
                instruments={}, readings={})
    take_lease("28", "native-audio-bank")
    try:
        for name in NAMES:
            t = time.perf_counter()
            if name == "whisper":
                os.environ["HF_HUB_CACHE"] = str(args.whisper_cache)
                lens = AudioEncoder()
            else:
                lens = NativeAudioEncoder(name, asset_root=args.assets, plan_path=args.plan)
            before = state_sha(lens.model)
            phases[name+"_load_state"] = time.perf_counter()-t
            bank["instruments"][name] = lens.identity
            identity = digest(lens.identity)
            entries, timings = [], []
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            for row, span, blob in views:
                n = len(frontends)
                hook, expected = frontend_probe(lens, name, row, blob, frontends)
                rng_before = np.random.get_state()
                try:
                    result = lens.embed([span], blob)
                finally:
                    hook.remove()
                rng_after = np.random.get_state()
                need(rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1])
                     and rng_before[2:] == rng_after[2:], "no randomized source selection")
                need(len(frontends)-n == expected, "complete actual model frontend roster")
                entries.append(store(root, name, identity, row, result))
                if hasattr(lens, "last_phases"):
                    timings.append(lens.last_phases)
            phases[name+"_measure_audit_store"] = time.perf_counter()-t
            bank["readings"][name] = entries
            t = time.perf_counter()
            for (row, span, blob), entry in zip(views, entries):
                result = lens.embed([span], blob)
                need(result[0].astype("<f4").tobytes() == (root/entry["file"]).read_bytes()
                     and result[1] == entry["native"], name+" exact complete-stream replay")
                replays.append(dict(instrument=name, clip=row["clip"], channel=span["channel"],
                                    windows=len(entry["native"]), exact=True))
            after = state_sha(lens.model)
            need(before == after and not lens.model.training
                 and all(not p.requires_grad for p in lens.model.parameters()), name+" frozen")
            phases[name+"_replay_state"] = time.perf_counter()-t
            states[name] = dict(before=before, after=after, timings=timings,
                output_rows=sum(x["shape"][0] for x in entries),
                value_bytes=sum(int(np.prod(x["shape"]))*4 for x in entries),
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            del lens, result
            torch.cuda.empty_cache()
            print(json.dumps(dict(instrument=name, streams=len(entries), frozen=True)), flush=True)
    finally:
        release_lease()
    raw = canonical(bank)
    (root/"bank.json").write_bytes(raw)
    ids = {n: digest(v) for n, v in bank["instruments"].items()}
    t = time.perf_counter()
    need(sum(1 for _ in AudioBank(root, sha(raw), args.inputs, args.input_pin, ids).consume()) == 20,
         "actual complete typed consumer")
    controls = faults(root, args.inputs, args.input_pin, bank)
    phases["consumer_controls"] = time.perf_counter()-t
    value = dict(author_checked=True, input_pin=args.input_pin, bank_pin=sha(raw), instruments=ids,
                 states=states, frontends=frontends, replays=replays, controls=controls,
                 phases=phases, seconds=time.perf_counter()-started, flags=FLAGS)
    (root/"report.json").write_bytes(canonical(value))
    print(json.dumps(dict(report_pin=sha(canonical(value)), bank_pin=sha(raw),
                          frontends=len(frontends), replays=len(replays), controls=len(controls))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "assets", "plan", "whisper-cache", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--input-pin", required=True)
    ap.add_argument("--report-pin")
    ap.add_argument("--prefixes", type=Path)
    ap.add_argument("--episodes", type=Path)
    ap.add_argument("--episode-pin")
    args = ap.parse_args()
    cold(args) if args.report_pin else run(args)
