"""Independent byte/shape/source readback; does not invoke model/checkpoint loaders."""
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from senses.contract import canonical, digest
from senses.perception_corpus import pinned, sha
from senses.timed_speech import save


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_TIMED_WORLD_READBACK_"+code)


def reference_hash(weights, specs, rows, source, prefix):
    h = hashlib.sha256(canonical(dict(source_id=source, instruments=[s["identity"] for s in specs])))
    for i, spec in enumerate(specs):
        h.update(canonical(dict(instrument=spec["name"], rows=rows[spec["name"]])))
        for kind in ("values", "padding", "absent"):
            a = weights[f"{prefix}{kind}_{i}"].contiguous().numpy()
            h.update(canonical([list(a.shape), a.dtype.str]))
            h.update(a.tobytes())
    return h.hexdigest()


def temporal_readback(a, r, moments, bank):
    """Reconstruct support from original bank metadata, not the model adapter."""
    records = pinned(a.run/"temporal-records.json", r["temporal_records"])
    check(sha(a.run/"temporal-positions.pt") == r["temporal_positions"], "TIME_BYTES")
    positions = torch.load(a.run/"temporal-positions.pt", weights_only=True)
    check(set(records) == set(positions) == set(moments), "TIME_ROWS")
    catalogue, checked, native_rows = {}, 0, 0
    for key, lanes in records.items():
        event = moments[key]
        check(set(lanes) == set(bank["readings"]) == set(positions[key]), "TIME_LANES")
        catalogue[key] = {}
        for lane, record in lanes.items():
            medium = lane.split("_", 1)[0]
            entries = [e for e in bank["readings"][lane]
                if e["input"]["segment"] == event["source"]["segment"]
                and (medium != "image" or e["input"]["frame"] == event["content"]["image"]["frame"])]
            check(len(entries) == 1, "TIME_JOIN")
            entry = entries[0]
            inp = entry["input"]
            check(record["source"] == event["source"] and record["at"] == event["content"]["at"]
                  and record["row"] == key and record["instrument"] == lane
                  and record["measurement"] == entry["instrument_id"]
                  and record["reading_sha256"] == entry["sha256"] and record["bank"] == r["bank_pin"]
                  and record["context"] == inp["support"]
                  and record["available_after"] == record["through"] == inp["available_after"]
                  and record["timing"] == event["content"]["timing"]
                  and record["physical_sync_calibrated"] is False
                  and record["active_language"] == event["content"]["language"]
                  and record["dependency"] == ("audio" if medium == "text" else "recording"),
                  "TIME_PROVENANCE")
            intervals = []
            if medium == "image":
                frame, tb = inp["frame"], Fraction(*inp["time_base"])
                intervals = [(0, entry["shape"][0], frame["pts"]*tb,
                              (frame["pts"]+frame["duration"])*tb)]
            elif medium == "audio":
                audio = inp["source_audio"]
                start = Fraction(audio["start_sample"], audio["rate"])
                stop = Fraction(audio["stop_sample"], audio["rate"])
                for w in entry["native"]:
                    rate = w.get("target_rate", 16000)
                    intervals.append((w["token_start"], w["token_end"],
                        start+Fraction(w["resampled_start"], rate),
                        min(stop, start+Fraction(w["resampled_end"], rate))))
            else:
                words, cursor = [], 0
                for word in inp["text"]["words"]:
                    if word["tag"] == "w":
                        end = cursor+len(word["text"] or "")
                        words.append((cursor, end, word))
                        cursor = end+1
                for i, w in enumerate(entry["native"]["windows"]):
                    x, y = w["source_char_span"]
                    selected = [word for lo, hi, word in words if x < hi and y > lo]
                    spans = [word["span"] for word in selected]
                    check(record["windows"][i]["components"] == spans
                          and record["windows"][i]["word_ids"] ==
                          [w["attributes"]["{http://nite.sourceforge.net/}id"] for w in selected],
                          "TIME_WORDS")
                    intervals.append((i, i+1, min(Fraction(*s[0]) for s in spans),
                                      max(Fraction(*s[1]) for s in spans)))
            check(len(intervals) == len(record["windows"]), "TIME_WINDOW_COUNT")
            expected = np.zeros((1, entry["shape"][0], 8), dtype=np.float32)
            at = Fraction(*record["at"])
            available = Fraction(*record["available_after"])
            context = list(map(lambda x: Fraction(*x), inp["support"]))
            for window, (start, end, lo, hi) in zip(record["windows"], intervals, strict=True):
                check(window["rows"] == [start, end] and
                      list(map(lambda x: Fraction(*x), window["interval"])) == [lo, hi],
                      "TIME_INTERVAL")
                offsets = (lo-at, hi-at, hi-lo, available-at, context[0]-at, context[1]-at, at)
                expected[0, start:end] = [math.copysign(math.log1p(abs(float(x))), float(x))
                                         for x in offsets]+[1.]
            check(np.array_equal(positions[key][lane].numpy(), expected), "TIME_MODEL_COORDINATES")
            catalogue[key][lane] = digest(record)
            checked += 1
            native_rows += entry["shape"][0]
    check(digest(catalogue) == r["temporal_id"], "TIME_CATALOGUE")
    check(sha(a.run/"temporal-sensitivity.pt") == r["temporal_sensitivity"], "TIME_SENSITIVITY_BYTES")
    probes = torch.load(a.run/"temporal-sensitivity.pt", weights_only=True)
    for probe in probes["reading_shift"].values():
        delta = float((probe["observed"]-probe["counterfactual"]).abs().max())
        check(delta > 0 and delta == probe["max_abs"] and probe["shift_seconds"] == [1, 1000],
              "TIME_SENSITIVITY")
    for probe in probes["model_ablation"].values():
        check(not torch.equal(probe["baseline"], probe["zero_time_projection"]), "TIME_MODEL_EFFECT")
    return dict(descriptors=checked, native_rows=native_rows, pin=r["temporal_id"],
                exact_intervals_and_coordinate_readback=True)


def main(a):
    r = pinned(a.run/"report.json", a.pin)
    source = pinned(a.inputs/"report.json", r["input_pin"])
    bank = pinned(a.bank/"bank.json", r["bank_pin"])
    check(sha(a.identity/"profile.md") == r["identity_pin"]
          == "5f839acceb76bd4bd2db269af9651b0df9e072b4a001588068434391a42ea4f6", "PROFILE")
    moments = {m["id"]: m for m in source["moments"]}
    temporal = temporal_readback(a, r, moments, bank)
    snapshots = {}
    for name in ("initial", "pending"):
        root = a.run/name
        manifest = pinned(root/"manifest.json", r[name])
        check(digest(manifest["temporal"]) == manifest["temporal_id"] == temporal["pin"],
              "TIME_SNAPSHOT")
        check((root/"COMMITTED").read_text() == r[name] and digest(manifest) == r[name], "COMMIT")
        check((root/"state.pt").stat().st_size == manifest["bytes"]
              <= manifest["config"]["checkpoint_bytes"] and sha(root/"state.pt") == manifest["sha256"], "BYTES")
        payload = torch.load(root/"state.pt", weights_only=True, map_location="cpu")
        weights = payload["model"]
        check(all(bool(torch.isfinite(v).all()) for v in weights.values()), "FINITE")
        check(reference_hash(weights, manifest["instruments"], manifest["world_rows"],
              r["binding"], "world_") == r["world"], "WORLD_PIN")
        check(reference_hash(weights, manifest["identity_instruments"], manifest["identity_rows"],
              manifest["identity_source"], "core.identity.ref_") == r["standard"], "STANDARD_PIN")
        checked = 0
        for i, spec in enumerate(manifest["instruments"]):
            definition = json.loads(spec["definition_json"])
            check(digest(definition) == spec["identity"]
                  and digest(bank["instruments"][spec["name"]]) == definition["measurement_identity"],
                  "INSTRUMENT")
            medium = spec["channel"]
            values, padding = weights[f"world_values_{i}"], weights[f"world_padding_{i}"]
            for j, row in enumerate(manifest["world_rows"][spec["name"]]):
                moment = moments[row]
                entries = [e for e in bank["readings"][spec["name"]]
                    if e["input"]["segment"] == moment["source"]["segment"] and
                    (medium != "image" or e["input"]["frame"] == moment["content"]["image"]["frame"])]
                check(len(entries) == 1, "SOURCE_JOIN")
                e = entries[0]
                path = a.bank/spec["name"]/e["file"]
                check(sha(path) == e["sha256"], "SOURCE_BYTES")
                native = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(e["shape"])
                actual = values[j, ~padding[j]].numpy()
                check(np.array_equal(native, actual) and not bool(values[j, padding[j]].any()), "NATIVE_WINDOWS")
                checked += 1
        check(len(manifest["contexts"]) == 2 and payload["world"].shape == (2, 16, 32)
              and payload["identity"].shape == (2, 6, 32), "CONTEXT_SHAPES")
        for i, context in enumerate(manifest["contexts"]):
            check(context["sequence"] == (0 if name == "initial" else 3), "SEQUENCE")
            if name == "pending":
                out = torch.load(a.run/"outputs"/f"2-{i}.pt", weights_only=True)
                check(torch.equal(payload["world"][i], out["world:0"])
                      and torch.equal(payload["identity"][i], out["identity:0"]), "PENDING_CONTEXT")
        snapshots[name] = dict(pin=r[name], native_rows=checked, bytes=manifest["bytes"])
    for record in r["outputs"]:
        path = a.run/"outputs"/record["file"]
        check(sha(path) == record["file_sha256"], "OUTPUT_HASH")
        out = torch.load(path, weights_only=True)
        for lane, shape in record["native_shapes"].items():
            value = out["slots:"+lane]
            check(list(value.shape) == shape and bool(torch.isfinite(value).all())
                  and bool((torch.linalg.vector_norm(value, dim=-1)-1).abs().lt(1e-4).all()),
                  "OUTPUT_NATIVE_SLOTS")
        check(len(record["native_shapes"]) == 16, "OUTPUT_ROSTER")
    save(a.output, dict(acceptance="author-checked", report=a.pin, snapshots=snapshots,
        temporal=temporal,
        output_records=len(r["outputs"]), profile_sha256=r["identity_pin"],
        direct_tensor_and_original_bank_readback=True, scientific_admission=False,
        learned_understanding=False, code=sha(Path(__file__)), seconds=time.perf_counter()-STARTED))
    print(json.dumps(dict(report=sha(a.output), snapshots=len(snapshots), outputs=len(r["outputs"]))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("run", "inputs", "bank", "identity", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--pin", required=True)
    main(p.parse_args())
