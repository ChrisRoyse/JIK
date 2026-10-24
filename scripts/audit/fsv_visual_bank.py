"""Real complete source frames -> five separate native visual candidate instruments."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses.contract import canonical, digest
from senses.visual_bank import FLAGS, NAMES, VisualBank, input_views, store, sha
from native_assets import state_sha
from native_vision import NativeVisionEncoder
from lenscommon import take_lease, release_lease
from measure import nvidia_smi


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def frontend_probe(lens, name, blob, capture):
    from PIL import Image
    with Image.open(io.BytesIO(blob.raw)) as image:
        rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    stride = {"dinov2": 14, "siglip2": 16, "mae": 16, "sam": 16, "convnext": 32}[name]
    ph = 1024 if name == "sam" else (height+stride-1)//stride*stride
    pw = 1024 if name == "sam" else (width+stride-1)//stride*stride
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:height, :width] = rgb
    preprocessing = lens.identity["preprocessing"]
    expected = (padded.astype(np.float32)/np.float32(255)
                - np.asarray(preprocessing["mean"], np.float32))/np.asarray(preprocessing["std"], np.float32)
    expected = np.ascontiguousarray(expected.transpose(2, 0, 1)[None])
    def hook(module, positional, kwargs):
        actual = kwargs["pixel_values"].detach().cpu().numpy()
        need(actual.shape == expected.shape and actual.dtype == np.float32
             and actual.tobytes() == expected.tobytes(), name+" actual frontend pixels")
        capture.append(dict(instrument=name, blob_sha256=blob.pin,
                            input_shape=list(actual.shape), tensor_sha256=sha(actual.tobytes()),
                            exact=True))
    return lens.model.register_forward_pre_hook(hook, with_kwargs=True)


def cold(args):
    from PIL import Image
    from senses.perception_native import NativePrefix
    root = args.output
    report_raw = (root/"report.json").read_bytes()
    need(sha(report_raw) == args.report_pin, "cold report pin")
    report = json.loads(report_raw)
    read = VisualBank(root, report["bank_pin"], args.inputs, args.input_pin, report["instruments"])
    counts = {}
    for name, _, values in read.consume():
        counts.setdefault(name, dict(windows=0, rows=0, bytes=0, dim=values.shape[1]))
        counts[name]["windows"] += 1
        counts[name]["rows"] += len(values)
        counts[name]["bytes"] += values.nbytes
    need(set(counts) == set(NAMES) and all(x["windows"] == 18 for x in counts.values()), "cold typed counts")
    # Reconstruct each complete frame from actual PNG files and compare native source bytes.
    views = input_views(args.inputs, args.input_pin)
    reconstructed = []
    for clip in ("video_10993", "video_874"):
        chosen = [x for x in views if x[0]["clip"] == clip]
        prefix = NativePrefix(args.prefixes/(args.host+"-prefix-"+clip),
                              expected=chosen[0][0]["prefix_pin"], episodes=args.episodes,
                              episode_pin=args.episode_pin)
        request = dict(clock_id=prefix.value["clock"]["id"], split_group=prefix.value["split_group"],
                       through=[1, 10], required_media=["image"])
        original = next(prefix.consume(**request))
        height, width, _ = original["values"].shape
        expected_rectangles = [[x, y, min(512, width-x), min(512, height-y)]
                               for y in range(0, height, 512) for x in range(0, width, 512)]
        need([x[0]["descriptor"]["rectangle"] for x in chosen] == expected_rectangles
             and all(x[0]["descriptor"]["node"] == original["node"] for x in chosen), "whole frame roster")
        rgb = np.empty_like(original["values"])
        for row, _, blob in chosen:
            x, y, w, h = row["descriptor"]["rectangle"]
            with Image.open(io.BytesIO(blob.raw)) as image:
                tile = np.asarray(image)
                need(tile.shape == (h, w, 3) and tile.dtype == np.uint8, "lossless tile geometry")
                rgb[y:y+h, x:x+w] = tile
        need(rgb.tobytes() == original["values"].tobytes(), "whole original RGB conservation")
        reconstructed.append(dict(clip=clip, bytes=rgb.nbytes, source_rgb_sha256=sha(rgb.tobytes())))
    need(len(report["frontends"]) == 90 and len(report["probes"]) == 20, "frontend and replay coverage")
    value = dict(author_checked=True, report_pin=args.report_pin, bank_pin=report["bank_pin"],
                 typed_counts=counts, original_frames=reconstructed, frontends=90, exact_replays=20,
                 controls=len(report["controls"]), flags=FLAGS)
    with (root/"cold.json").open("xb") as f:
        f.write(canonical(value))
    print(json.dumps(dict(cold_pin=sha(canonical(value)), counts=counts)), flush=True)


def faults(root, inputs, input_pin, original):
    controls = []
    identities = {n: digest(v) for n, v in original["instruments"].items()}
    def attempt(label, value, code, training=False):
        target = root/label
        target.mkdir()
        raw = canonical(value)
        (target/"bank.json").write_bytes(raw)
        try:
            bank = VisualBank(target, sha(raw), inputs, input_pin, identities)
            if training:
                list(bank.consume("training"))
        except ValueError as exc:
            need(str(exc) == "MNLM_VISUAL_BANK_"+code, label+": "+str(exc))
            controls.append(dict(name=label, error=str(exc)))
        else:
            raise AssertionError(label+" accepted")
    v = copy.deepcopy(original); v["readings"].pop("mae")
    attempt("missing-instrument", v, "ROSTER")
    v = copy.deepcopy(original); v["readings"]["siglip2"][0]["instrument_id"] = identities["mae"]
    attempt("swapped-instrument", v, "SWAPPED_INSTRUMENT")
    v = copy.deepcopy(original); v["readings"]["sam"][0]["shape"] = [1, 256]
    attempt("pooled-grid", v, "NATIVE_SHAPE")
    v = copy.deepcopy(original); v["readings"]["convnext"].pop()
    attempt("missing-edge-window", v, "WINDOW_ROSTER")
    v = copy.deepcopy(original); v["instruments"]["mae"]["frozen"] = False
    attempt("drifted-instrument", v, "IDENTITIES")
    v = copy.deepcopy(original); v["readings"]["dinov2"][0]["native"][0]["token_end"] = 1
    attempt("missing-patches", v, "NATIVE_WINDOW")
    v = copy.deepcopy(original); v["readings"]["sam"][0]["native"][0]["grid"] = [32, 128]
    attempt("changed-spatial-grid", v, "NATIVE_WINDOW")
    v = copy.deepcopy(original); v["input_pin"] = "0"*64
    attempt("wrong-source", v, "SOURCE")
    v = copy.deepcopy(original); v["flags"]["scientific_admission"] = True
    attempt("false-admission", v, "FLAGS")
    attempt("premature-training", original, "NATIVE_ADMISSION_REQUIRED", training=True)
    target = root/"corrupt-values"
    target.mkdir()
    raw = canonical(original)
    (target/"bank.json").write_bytes(raw)
    entry = original["readings"]["dinov2"][0]
    content = (root/entry["file"]).read_bytes()
    (target/entry["file"]).write_bytes(bytes([content[0]^1])+content[1:])
    try:
        list(VisualBank(target, sha(raw), inputs, input_pin, identities).consume())
    except ValueError as exc:
        need(str(exc) == "MNLM_VISUAL_BANK_READING_BYTES", str(exc))
        controls.append(dict(name="corrupt-values", error=str(exc)))
    else:
        raise AssertionError("corrupt values accepted")
    return controls


def run(args):
    import torch
    from vision_encoder import VisionEncoder
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (root/"resident-before.txt").write_text(nvidia_smi())
    views = input_views(args.inputs, args.input_pin)
    need(len(views) == 18 and len({v[0]["clip"] for v in views}) == 2, "complete declared input population")
    take_lease("28", "native-visual-bank")
    bank = dict(schema="native-visual-bank-v1", input_pin=args.input_pin,
                flags=FLAGS, instruments={}, readings={})
    phases, probes, states, frontends = {}, [], {}, []
    try:
        for name in NAMES:
            t = time.perf_counter()
            if name == "dinov2":
                os.environ["HF_HUB_CACHE"] = str(args.dino_cache)
                lens = VisionEncoder()
            else:
                lens = NativeVisionEncoder(name, asset_root=args.assets, plan_path=args.plan)
            phases[name+"_load"] = time.perf_counter()-t
            before = state_sha(lens.model)
            bank["instruments"][name] = lens.identity
            identity = digest(lens.identity)
            entries = []
            t = time.perf_counter()
            timings = []
            for row, record, blob in views:
                count = len(frontends)
                hook = frontend_probe(lens, name, blob, frontends)
                try:
                    result = lens.measure(record, blob)
                finally:
                    hook.remove()
                need(len(frontends) == count+1, "one actual frontend per view")
                entries.append(store(root, name, identity, row, result))
                if hasattr(lens, "last_phases"):
                    timings.append(lens.last_phases)
            phases[name+"_measure_store"] = time.perf_counter()-t
            bank["readings"][name] = entries
            t = time.perf_counter()
            for i in (0, 11, 12, 17):
                row, record, blob = views[i]
                result = lens.measure(record, blob)
                need(result["tokens"].astype("<f4").tobytes() == (root/entries[i]["file"]).read_bytes()
                     and result["regions"] == entries[i]["native"], name+" exact repeat")
                probes.append(dict(instrument=name, clip=row["clip"], tile=row["ordinal"], exact=True))
            after = state_sha(lens.model)
            need(before == after and not lens.model.training
                 and all(not p.requires_grad for p in lens.model.parameters()), name+" frozen")
            phases[name+"_replay_state"] = time.perf_counter()-t
            states[name] = dict(before=before, after=after, timings=timings,
                                output_rows=sum(x["shape"][0] for x in entries),
                                value_bytes=sum(np.prod(x["shape"]).item()*4 for x in entries),
                                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            del lens, result
            torch.cuda.empty_cache()
            print(json.dumps(dict(instrument=name, windows=len(entries), state_equal=True)), flush=True)
    finally:
        release_lease()
    raw = canonical(bank)
    (root/"bank.json").write_bytes(raw)
    ids = {n: digest(v) for n, v in bank["instruments"].items()}
    t = time.perf_counter()
    read = VisualBank(root, sha(raw), args.inputs, args.input_pin, ids)
    need(sum(1 for _ in read.consume()) == 90, "actual typed consumer")
    controls = faults(root, args.inputs, args.input_pin, bank)
    phases["consumer_controls"] = time.perf_counter()-t
    report = dict(bank_pin=sha(raw), input_pin=args.input_pin, instruments=ids, states=states,
                  probes=probes, frontends=frontends, controls=controls, phases=phases,
                  seconds=time.perf_counter()-started,
                  flags=FLAGS, author_checked=True)
    (root/"report.json").write_bytes(canonical(report))
    print(json.dumps(dict(report_pin=sha(canonical(report)), bank_pin=sha(raw),
                          controls=len(controls), probes=len(probes))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("inputs", "assets", "plan", "dino-cache", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--input-pin", required=True)
    ap.add_argument("--report-pin")
    ap.add_argument("--prefixes", type=Path)
    ap.add_argument("--episodes", type=Path)
    ap.add_argument("--episode-pin")
    ap.add_argument("--host", choices=("windows", "linux"), default="windows")
    args = ap.parse_args()
    cold(args) if args.report_pin else run(args)
