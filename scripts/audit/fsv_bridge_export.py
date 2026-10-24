"""Cold source-to-UCR readback and corruption/leakage controls on retained data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses import bridge, pixels
from senses.blobs import Blobs
from senses.contract import SenseError, canonical, digest, seal_record, validate_record
from senses.pipeline import file_hash, read_bound, write_records
from tfrecord import decode, frames, RecordError


def readback(root, exported):
    manifest = json.loads((exported / "observations.ucr.manifest.json").read_bytes())
    binding = manifest["input"]
    require = bridge.require
    for name, key in (("episodes.jsonl", "episodes_sha256"), ("blobs.json", "blobs_manifest_sha256"),
                      ("field-disposition.json", "field_disposition_sha256")):
        require(file_hash(exported / name) == binding[key], "MNLM_BRIDGE_EXPORT_DIGEST", name)
    blobs = Blobs(exported / "blobs.json", binding["blobs_manifest_sha256"])
    data_stream = read_bound(exported / "observations.ucr.jsonl", manifest["sha256"])
    rows = 0
    units, images = set(), set()
    witness = None
    for source in bridge.input_plan(root):
        for offset, payload in frames(source["path"], source["sha256"]):
            n, data, collection, language = bridge.episode(decode(payload))
            for step in range(n):
                r = next(data_stream)
                validate_record(r)
                ch, loc = r["channels"], r["locator"]
                require(loc["shard"] == source["path"].name + f"#offset={offset}" and loc["row"] == step
                        and loc["span"] == [step, step + 1] and loc["record_digest"] == hashlib.sha256(payload).hexdigest(),
                        "MNLM_BRIDGE_READBACK", "source coordinates/digest differ")
                require(r["unit"] == digest(["bridge-provider-collection", collection]), "MNLM_BRIDGE_READBACK", "collection split differs")
                got_state = {f["name"]: f["value"] for f in ch["fields"]["items"]}
                got_command = {f["name"]: f["value"] for f in ch["behaviour"]["items"][0]["measurements"]}
                require(got_state == {f"state_component_{i}": data["steps/observation/state"][step*7+i] for i in range(7)}
                        and got_command == {f"command_component_{i}": data["steps/action"][step*7+i] for i in range(7)},
                        "MNLM_BRIDGE_READBACK", "raw state/commands changed")
                require(ch["surface"]["items"] == ([{"text": language[step]}] if language[step] else []),
                        "MNLM_BRIDGE_READBACK", "source language changed")
                raw = data["steps/observation/image_0"][step]
                sha = hashlib.sha256(raw).hexdigest()
                require(len(ch["perception"]["items"]) == 1 and ch["perception"]["items"][0]["sha256"] == sha
                        and blobs.read(sha, "image/jpeg") == raw, "MNLM_BRIDGE_READBACK", "retained primary bytes changed")
                provenance = {f["name"]: f["value"] for f in ch["provenance"]["items"]}
                require(provenance["image_0_provider_present"] == bool(data["episode_metadata/has_image_0"][0])
                        and provenance["image_0_availability_conflict"] == (not bool(data["episode_metadata/has_image_0"][0])),
                        "MNLM_BRIDGE_READBACK", "availability contradiction was hidden")
                require(all(not ch[c]["items"] for c in ("relations", "history", "co_reference")),
                        "MNLM_BRIDGE_READBACK", "unobserved context invented")
                if witness is None:
                    witness = (r, data, collection, source, offset, payload, raw)
                rows += 1
                units.add(r["unit"])
                images.add(sha)
    require(next(data_stream, None) is None and rows == manifest["count"] == binding["counts"]["steps"],
            "MNLM_BRIDGE_READBACK", "export count differs; extra/missing rows")
    require(images == set(blobs.entries), "MNLM_BRIDGE_READBACK", "orphan/missing blobs")
    return {"rows": rows, "collection_units": len(units), "images": len(images), "sha256": manifest["sha256"]}, witness, blobs


def controls(out, witness, blobs, root):
    checks = []
    r, data, collection, source, offset, payload, raw = witness

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})

    def refusal(name, call, code):
        try:
            call()
        except (SenseError, RecordError) as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False, code="unexpected_success")

    binding = {"adapter": bridge.ADAPTER, "raw_sha256": source["sha256"], "policy_sha256": bridge.policy()[1]}
    def adapted(d):
        return bridge.step_record(d, collection, 0, source["path"].name, offset, hashlib.sha256(payload).hexdigest(),
                                  binding, hashlib.sha256(raw).hexdigest())
    check("first_step_replay", canonical(adapted(data)) == canonical(r))
    changed = copy.deepcopy(data)
    changed["steps/observation/state"][7:14] = [1234.0] * 7
    check("future_state_not_read_or_relabelled", canonical(adapted(changed)) == canonical(r))
    for name in ("reward", "discount", "is_terminal", "language_embedding"):
        changed = copy.deepcopy(data)
        changed["steps/" + name][0] += 1
        check(name + "_not_input", canonical(adapted(changed)) == canonical(r))
    changed = copy.deepcopy(data)
    changed["steps/action"][0] += 1
    check("raw_command_is_observed", adapted(changed)["channels"]["behaviour"] != r["channels"]["behaviour"])
    placeholder = data["steps/observation/image_1"][0]
    refusal("real_placeholder_absence", lambda: bridge.primary_image(placeholder), "MNLM_BRIDGE_PLACEHOLDER")
    refusal("malformed_jpeg", lambda: bridge.primary_image(raw[:30]), "MNLM_BRIDGE_IMAGE")

    a, b = pixels.measure(r, blobs), pixels.measure(r, blobs)
    check("pixel_encoder_byte_replay", a["tokens"].tobytes() == b["tokens"].tobytes() and a["regions"] == b["regions"])
    # Independent inversion checks every decoded RGB value, not a summary score.
    tokens = a["tokens"]
    reconstructed = np.rint(tokens[:, :192] * 255).astype(np.uint8).reshape(32, 32, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(256, 256, 3)
    check("all_decoded_pixels_survive", np.array_equal(reconstructed, pixels.decode_rgb(raw, "image/jpeg")), shape=list(tokens.shape))
    check("patch_coordinates_survive", np.array_equal(tokens[:, 192], np.tile(np.arange(0, 256, 8), 32))
          and np.array_equal(tokens[:, 193], np.repeat(np.arange(0, 256, 8), 32)))
    changed = copy.deepcopy(r)
    changed["channels"]["perception"]["items"][0]["sha256"] = "0" * 64
    refusal("unretained_image", lambda: pixels.measure(seal_record(changed), blobs), "MNLM_UCR_BLOB_MISSING")
    changed = copy.deepcopy(r)
    changed["channels"]["perception"]["items"][0]["measurements"][0]["value"] = 128
    refusal("false_geometry", lambda: pixels.measure(seal_record(changed), blobs), "MNLM_PIXEL_GEOMETRY")
    copied = out / "altered.jpg"
    corrupted = bytearray(raw)
    corrupted[-10] ^= 1
    copied.write_bytes(corrupted)
    manifest = out / "fault-blobs.json"
    manifest.write_bytes(canonical({"schema": "ucr-blobs-v1", "blobs": [{"sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(copied.resolve()), "bytes": len(raw), "media_type": "image/jpeg"}]}))
    wrong = Blobs(manifest, file_hash(manifest))
    refusal("tampered_retained_pixels", lambda: pixels.measure(r, wrong), "MNLM_UCR_BLOB_DIGEST")
    fault_root = out / "acquisition-fault"
    (fault_root / "census/bridge-observations-v1").mkdir(parents=True)
    (fault_root / "raw").mkdir()
    (fault_root / "census/bridge-observations-v1/selection.json").write_bytes((root / "census/bridge-observations-v1/selection.json").read_bytes())
    report_name = f"bounded-{bridge.SELECTION}-{bridge.ACQUISITION}.json"
    (fault_root / "raw" / report_name).write_bytes((root / "raw" / report_name).read_bytes() + b" ")
    refusal("changed_acquisition_cannot_redefine_raw_identity", lambda: bridge.input_plan(fault_root), "MNLM_BRIDGE_ACQUISITION")
    # A late source-digest failure must not publish a valid UCR export.
    def late_failure():
        yield r
        raise SenseError("MNLM_TF_FILE_DIGEST", "negative-control source mismatch")
    target = out / "must-not-publish.jsonl"
    refusal("late_failure_refused", lambda: write_records(target, late_failure(), {}), "MNLM_TF_FILE_DIGEST")
    check("late_failure_no_published_export", not target.exists() and not target.with_suffix(".manifest.json").exists())
    np.save(out / "primary.tokens.npy", tokens, allow_pickle=False)
    (out / "pixel-contract.json").write_bytes(canonical(pixels.contract()))
    (out / "witness.ucr.json").write_bytes(canonical(r))
    return checks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import time
    start = time.monotonic()
    result, witness, blobs = readback(args.root, args.export)
    checks = controls(args.out, witness, blobs, args.root)
    report = {"acceptance": "author-checked only", "readback": result, "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "wall_seconds": time.monotonic() - start, "stage3_admitted": False, "stage4_accepted": False}
    (args.out / "report.json").write_bytes(canonical(report))
    (args.out / "SHA256.json").write_bytes(canonical({str(p.relative_to(args.out)).replace("\\", "/"): file_hash(p)
                                                   for p in args.out.rglob("*") if p.is_file()}))
    print(json.dumps(report), flush=True)
    raise SystemExit(bool(report["failed"]))
