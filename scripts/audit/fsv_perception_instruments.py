"""Real native media -> unchanged frozen CUDA instruments -> typed cold readback."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
from senses.contract import canonical
from senses.perception_native import NativePrefix, fraction
from senses.perception_instruments import (Readings, descriptors, inputs, measure, sha, read_slot)
from lenscommon import take_lease, release_lease
from measure import nvidia_smi

CLIPS = ("video_10993", "video_874")


def need(ok, name):
    if not ok:
        raise AssertionError(name)


def state(lens):
    h = hashlib.sha256()
    for name, value in lens.model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def independent(root, publication, prefix, request):
    """Derive coverage from original record geometry, not the bridge roster helper."""
    v = json.loads((root/publication["path"]).read_bytes())
    need(sha((root/publication["path"]).read_bytes()) == publication["pin"], "publication pin")
    chosen = prefix.nodes(**request)
    h, w, _ = prefix.value["formats"]["image"]["shape"]
    visual = v["readings"]["image"]
    actual = [(r["descriptor"]["node"], r["descriptor"]["rectangle"]) for r in visual]
    expected = [(r["node"], [x, y, min(512, w-x), min(512, h-y)])
                for r in chosen["image"] for y in range(0, h, 512) for x in range(0, w, 512)]
    need(actual == expected, "complete frame/tile order")
    # Every native pixel occurs once: reconstruct from the actual lossless wrappers.
    from PIL import Image
    import soundfile as sf
    reconstructed, audio = {}, []
    roster = descriptors(prefix, request)
    for d, value, blob in inputs(prefix, request, roster):
        if d["modality"] == "image":
            node = d["node"]["id"]
            if node not in reconstructed:
                reconstructed[node] = np.empty((h, w, 3), np.uint8)
            x, y, width, height = d["rectangle"]
            with Image.open(io.BytesIO(blob.raw)) as im:
                reconstructed[node][y:y+height, x:x+width] = np.asarray(im)
        else:
            pcm, rate = sf.read(io.BytesIO(blob.raw), dtype="float32", always_2d=True)
            originals = [r for r in chosen["audio"] if r["node"] in d["nodes"]]
            raw = b"".join((prefix.root/r["file"]).read_bytes() for r in originals)
            need(raw == pcm.astype("<f4", copy=False).tobytes() and rate == d["rate"], "PCM exact")
            audio.append(dict(channel=d["channel"], span=d["span"], samples=len(pcm),
                              source_sha256=sha(raw), wrapper_sha256=blob.pin))
    for r in chosen["image"]:
        need(sha(reconstructed[r["node"]["id"]].tobytes()) == r["sha256"], "RGB pixel conservation")
    samples = sum(r["node"]["sample_count"] for r in chosen["audio"])
    channels = prefix.value["formats"]["audio"]["channels"]
    need([x["channel"] for x in audio] == list(range(channels))
         and all(x["samples"] == samples for x in audio), "complete distinct channels")
    totals = {}
    for m, entries in v["readings"].items():
        tokens, byte_count, hashes = 0, 0, []
        for e in entries:
            raw = (root/"slots"/(e["key"]+".bin")).read_bytes()
            dims = e["descriptor"]["shape"]
            a = np.frombuffer(raw, dtype="<f4")
            need(sha(raw) == e["sha256"] and len(raw) == np.prod(dims)*4
                 and dims[1] == 384 and bool(np.isfinite(a).all()), "native persisted values")
            tokens += dims[0]
            byte_count += len(raw)
            hashes.append(sha(raw))
        totals[m] = dict(windows=len(entries), tokens=tokens, bytes=byte_count, hashes=hashes)
    return dict(counts=totals, source_frames=len(chosen["image"]), audio=audio)


def faults(root, name, pin, prefix, request, ids):
    original = json.loads((root/(name+".json")).read_bytes())
    controls = []

    def attempt(label, fn, code):
        error = None
        try:
            fn()
        except ValueError as exc:
            error = str(exc)
        need(error == "MNLM_NATIVE_PREFIX_"+code, label+str(error))
        controls.append(dict(name=label, error=error))

    def mutate(label, fn, code):
        v = copy.deepcopy(original)
        fn(v)
        raw = canonical(v)
        path = root/("fault-"+label+".json")
        path.write_bytes(raw)
        attempt(label, lambda: Readings(root, path.stem, sha(raw), prefix, request, ids), code)

    mutate("pooled-image", lambda v: v["readings"]["image"].pop(), "READING_ROSTER")
    mutate("missing-channel", lambda v: v["readings"]["audio"].pop(), "READING_ROSTER")
    mutate("reordered-tiles", lambda v: v["readings"]["image"].reverse(), "READING_ROSTER")
    mutate("duplicate-tile", lambda v: v["readings"]["image"].append(v["readings"]["image"][0]), "READING_ROSTER")
    mutate("rectangle-shift", lambda v: v["readings"]["image"][0]["descriptor"]["rectangle"].__setitem__(0, 1),
           "READING_ROSTER")
    mutate("channel-mix", lambda v: v["readings"]["audio"][1]["descriptor"].__setitem__("channel", 0),
           "READING_ROSTER")
    mutate("future-cutoff", lambda v: v["request"].__setitem__("through", [1, 1]), "READING_SOURCE")
    mutate("wrong-clock", lambda v: v["clock"].__setitem__("id", "0"*64), "READING_SOURCE")
    mutate("wrong-source", lambda v: v["source"].__setitem__("media", "0"*64), "READING_SOURCE")
    mutate("claim-admission", lambda v: v["flags"].__setitem__("scientific_admission", True), "READING_FLAGS")
    mutate("instrument-swap", lambda v: v["instrument_ids"].__setitem__("image", ids["audio"]), "READING_INSTRUMENTS")
    mutate("flatten-window", lambda v: v["readings"]["image"][0]["descriptor"].__setitem__("shape", [1, 384]),
           "READING_ROSTER")
    mutate("discard-patches", lambda v: v["readings"]["image"][0]["native"][0].__setitem__("token_end", 1),
           "READING_NATIVE_WINDOWS")
    reader = Readings(root, name, pin, prefix, request, ids)
    attempt("training-without-admission", lambda: list(reader.consume(purpose="training")),
            "READING_NATIVE_ADMISSION_REQUIRED")
    attempt("missing-instrument", lambda: Readings(root, name, pin, prefix, request, {"image": ids["image"]}),
            "READING_INSTRUMENTS")
    # Own dedicated copy for byte corruption, never mutate published slots.
    bad = root/"corrupt-copy"
    (bad/"slots").mkdir(parents=True)
    entry = original["readings"]["image"][0]
    raw = (root/"slots"/(entry["key"]+".bin")).read_bytes()
    (bad/"slots"/(entry["key"]+".bin")).write_bytes(bytes([raw[0]^1])+raw[1:])
    attempt("corrupt-values", lambda: read_slot(bad, entry), "READING_BYTES")
    return controls


def cold(args):
    root = args.output
    report_raw = (root/"report.json").read_bytes()
    need(sha(report_raw) == args.report_pin, "cold report pin")
    report = json.loads(report_raw)
    receipts = []
    for clip in CLIPS:
        path = args.prefixes/(args.host+"-prefix-"+clip)
        pin = report["source_pins"][clip]
        prefix = NativePrefix(path, expected=pin, episodes=args.episodes, episode_pin=args.episode_pin)
        for entry in [e for e in report["publications"] if e["clip"] == clip]:
            reader = Readings(root/"readings", Path(entry["path"]).stem, entry["pin"],
                              prefix, entry["request"], report["instrument_ids"])
            got = {m: [] for m in ("image", "audio")}
            for value in reader.consume():
                need(not value["values"].flags.writeable, "read-only native arrays")
                got[value["modality"]].append(sha(value["values"].tobytes()))
            receipt = independent(root/"readings", entry, prefix, entry["request"])
            need(got == {m: r["hashes"] for m, r in receipt["counts"].items()}, "consumer independently matches")
            need(receipt == entry["independent"], "cold independent exact")
            receipts.append(dict(path=entry["path"], pin=entry["pin"], hashes=got))
    value = dict(acceptance="author-checked", report_pin=args.report_pin, receipts=receipts,
                 meaning_asserted=False, scientific_admission=False, model_learning=False)
    with (root/"cold.json").open("xb") as f:
        f.write(canonical(value))
    print(json.dumps(dict(cold_pin=sha(canonical(value)), publications=len(receipts))), flush=True)


def run(args):
    import torch
    from vision_encoder import VisionEncoder
    from audio_encoder import AudioEncoder
    start = time.perf_counter()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    phases, publications, source_pins, controls, probes = {}, [], {}, [], []
    t = time.perf_counter()
    (root/"resident-before.txt").write_text(nvidia_smi())
    take_lease("28", "native-instruments")
    lenses = {}
    phases["resident_lease"] = time.perf_counter()-t
    try:
        t = time.perf_counter()
        os.environ["HF_HUB_CACHE"] = str(args.vision_cache)
        lenses["image"] = VisionEncoder()
        os.environ["HF_HUB_CACHE"] = str(args.audio_cache)
        lenses["audio"] = AudioEncoder()
        before = {m: state(lens) for m, lens in lenses.items()}
        phases["load_attest"] = time.perf_counter()-t
        for clip in CLIPS:
            t = time.perf_counter()
            path = args.prefixes/(args.host+"-prefix-"+clip)
            pin = (path/"COMMITTED").read_text()
            source_pins[clip] = pin
            prefix = NativePrefix(path, expected=pin, episodes=args.episodes, episode_pin=args.episode_pin)
            phases[clip+"_bind"] = time.perf_counter()-t
            for through in ([1, 10], [1, 5]):
                request = dict(clock_id=prefix.value["clock"]["id"], split_group=prefix.value["split_group"],
                               through=through, required_media=["image", "audio"])
                name = clip+"-"+str(through[1])
                t = time.perf_counter()
                result = measure(prefix, request, lenses, root/"readings", name)
                phases[name+"_measure"] = time.perf_counter()-t
                t = time.perf_counter()
                pub = json.loads((root/"readings"/result["path"]).read_bytes())
                ids = pub["instrument_ids"]
                reader = Readings(root/"readings", name, result["pin"], prefix, request, ids)
                observed = list(reader.consume())
                receipt = independent(root/"readings", result, prefix, request)
                need(len(observed) == sum(r["windows"] for r in receipt["counts"].values()), "consumer count")
                del observed
                result.update(clip=clip, request=request, independent=receipt)
                publications.append(result)
                phases[name+"_readback"] = time.perf_counter()-t
                # Independent new forward on one interior + one boundary tile and both audio channels.
                t = time.perf_counter()
                roster = descriptors(prefix, request)
                witnesses = [roster["image"][0], roster["image"][-1], *roster["audio"]]
                for d, value, blob in inputs(prefix, request, roster):
                    if d not in witnesses:
                        continue
                    m = d["modality"]
                    if m == "image":
                        raw = lenses[m].measure(value, blob)
                        tokens, windows = raw["tokens"], raw["regions"]
                    else:
                        tokens, windows = lenses[m].embed([value], blob)
                    entry = next(e for e in pub["readings"][m] if e["descriptor"] == d)
                    need(tokens.astype("<f4", copy=False).tobytes() == read_slot(root/"readings", entry).tobytes()
                         and windows == entry["native"], "exact repeated native forward")
                    probes.append(dict(publication=name, key=entry["key"], modality=m, sha256=entry["sha256"]))
                phases[name+"_repeat"] = time.perf_counter()-t
                if not controls:
                    t = time.perf_counter()
                    controls = faults(root/"readings", name, result["pin"], prefix, request, ids)
                    phases["negative_controls"] = time.perf_counter()-t
        t = time.perf_counter()
        after = {m: state(lens) for m, lens in lenses.items()}
        need(before == after and not any(p.requires_grad for lens in lenses.values()
                                        for p in lens.model.parameters()), "frozen instruments")
        phases["final_frozen"] = time.perf_counter()-t
    finally:
        lenses.clear()
        torch.cuda.empty_cache()
        release_lease()
    elapsed = time.perf_counter()-start
    coverage = sum(phases.values())/elapsed
    need(.9 <= coverage <= 1.001, "cost coverage")
    report = dict(acceptance="author-checked", complete=True, source_pins=source_pins,
                  episode_pin=args.episode_pin, publications=publications, instrument_ids=ids,
                  frozen_before=before, frozen_after=after, controls=controls, repeated_forwards=probes,
                  phases=phases, seconds=elapsed, phase_coverage=coverage, meaning_asserted=False,
                  scientific_admission=False, model_learning=False, goal_complete=False)
    (root/"report.json").write_bytes(canonical(report))
    print(json.dumps(dict(report_pin=sha(canonical(report)), seconds=elapsed,
                          phase_coverage=coverage, controls=len(controls),
                          forwards=sum(p["forwarded"] for p in publications),
                          reused=sum(p["reused"] for p in publications))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for n in ("prefixes", "episodes", "output", "vision-cache", "audio-cache"):
        ap.add_argument("--"+n, type=Path, required=True)
    ap.add_argument("--episode-pin", required=True)
    ap.add_argument("--host", choices=("windows", "linux"), required=True)
    ap.add_argument("--report-pin")
    a = ap.parse_args()
    cold(a) if a.report_pin else run(a)
