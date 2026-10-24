"""Real recorded-audio integration, boundary controls and cold slot readback.

No statistical admission. The interface labels only describe the retained
observation's modality. They are not an outcome oracle and are never assayed.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import blake3
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import SenseError, canonical, digest, present, seal_record
from senses.pipeline import file_hash, ingest_file, read_bound, write_records
from senses.store import Store
from senses.blobs import Blobs
from audio_encoder import AudioEncoder, SPEC, spans, waveform
from audio_ucr import visible_audio, measure_audio_ucr
from universal import ChannelReader
from lenscommon import LensError, Paths, entropy_bits, run_cli, sealed_registry
from admission_state import validate_ledger
from slotio import slot_digest
from measure import main as measurement_main, nvidia_smi


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **data):
        checks.append({"boundary": name, "pass": bool(ok), **data})

    def refuses(name, fn, code):
        error = None
        try:
            fn()
        except (SenseError, LensError) as exc:
            error = exc.code
        check(name, error == code, error=error)

    src = args.recordings / "audio.ucr.jsonl"
    manifest = json.loads(src.with_suffix(".manifest.json").read_bytes())
    originals = list(read_bound(src, manifest["sha256"]))
    audio = [r for r in originals if r["channels"]["perception"]["state"] == "present"]
    records = list(originals)
    # Same real sound with separate, real retained text: text cannot change this
    # instrument. This is an explicitly labelled boundary perturbation.
    extra = copy.deepcopy(audio[0])
    extra["locator"]["shard"] += "#text-control"
    extra["channels"]["surface"] = copy.deepcopy(originals[1]["channels"]["surface"])
    records.append(seal_record(extra))
    narrow = copy.deepcopy(audio[0])
    narrow["locator"]["shard"] += "#span-control"
    narrow["channels"]["perception"] = present(narrow["channels"]["perception"]["items"][1:5])
    region = spans(narrow)[0]
    narrow["locator"]["span"] = [region["start"], region["end"]]
    records.append(seal_record(narrow))
    gap = copy.deepcopy(audio[0])
    gap["locator"]["shard"] += "#gap-control"
    frames = gap["channels"]["perception"]["items"]
    gap["channels"]["perception"] = present(frames[:2] + frames[-2:])
    records.append(seal_record(gap))
    export = out / "audio.ucr.jsonl"
    m = write_records(export, records, {"source_sha256": manifest["sha256"], "scope": "two original recordings plus labelled boundary controls"})
    storepath = out / "channels.sqlite"
    store = Store(storepath, create=True)
    before = store.head
    ingest_file(store, export, m, before)
    generation = store.head
    census = store.audit()
    check("accepted_source_readback", census["records"] == len(records) and before is None and generation is not None, census=census)
    store.close()
    blobs = {p["sha256"]: {"sha256": p["sha256"], "path": str((args.raw / r["locator"]["shard"]).resolve()),
                          "bytes": (args.raw / r["locator"]["shard"]).stat().st_size, "media_type": p["media_type"]}
             for r in audio for p in r["channels"]["perception"]["items"][:1]}
    blobpath = out / "blobs.json"
    blobpath.write_bytes(canonical({"schema": "ucr-blobs-v1", "blobs": list(blobs.values())}))
    blobsha = file_hash(blobpath)
    resolver = Blobs(blobpath, blobsha)
    # Independent decoder geometry and authorized-range checks before model load.
    for r in audio:
        region = spans(r)[0]
        wave = waveform(resolver.read(region["sha256"], region["media_type"]), region)
        check(r["locator"]["shard"] + "_complete_resampling", len(wave) == (region["end"] * 16000 + region["rate"] - 1) // region["rate"], samples_16k=len(wave))
    family, version = "integration-audio", "audio"
    root = out / "lenses" / version
    cal = root / "cal"
    cal.mkdir(parents=True)
    rows = [{"i": i, **{k: r["locator"][k] for k in ("source", "shard", "row", "span")},
             "text": "\n".join(x["text"] for x in r["channels"]["surface"]["items"]),
             "y": int(r["channels"]["perception"]["state"] == "present")} for i, r in enumerate(records)]
    calpath = cal / (family + ".jsonl")
    calpath.write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
    labels = [r["y"] for r in rows]
    (cal / (family + ".summary.json")).write_bytes(canonical({"n": len(rows), "classes": {str(y): labels.count(y) for y in set(labels)},
        "H_Y_bits": entropy_bits(labels), "file_blake3": blake3.blake3(calpath.read_bytes()).hexdigest(),
        "status": "observation modality descriptor; not an admissible outcome; never assayed"}))
    policy = {"schema": "ucr-view-v1", "question": family, "allow_channels": ["perception", "surface"], "deny_fields": [],
              "deny_relations": [], "blocked_units": [], "label_aliases": [], "rationale": "source-bound audio integration; no statistical question"}
    policies = out / "policies.json"
    policies.write_bytes(canonical({family: policy}))
    (out / "resident-before.txt").write_text(nvidia_smi())
    sys.argv = ["measure.py", "--root", str(out), "--out", str(out), "--version", version, "--families", family,
                "--lenses", SPEC["name"], "--ucr-store", str(storepath), "--ucr-policies", str(policies),
                "--ucr-blobs", str(blobpath), "--ucr-blobs-sha256", blobsha, "--issue", "15"]
    prepared = time.monotonic()
    result = measurement_main()  # real dispatch, lease, registry publication and ledger path
    measured = time.monotonic()
    registry = sealed_registry(root)
    validate_ledger(Paths(out, out, version).ledger, registry)
    entry = registry["lenses"][SPEC["name"]]
    check("actual_measurement_entrypoint", result == 0 and not entry.get("error"), entry_error=entry.get("error"))
    if entry.get("error"):
        (out / "report.json").write_bytes(canonical({"acceptance": "author-checked", "checks": checks, "failed": 1}))
        return 1
    ent = entry["families"][family]
    path = Path(ent["slot"])
    with np.load(path, allow_pickle=False) as z:
        tokens, offsets, absent, windows = (z[k] for k in ("tokens", "offsets", "absent", "windows"))
    provenance = json.loads(path.with_suffix(".views.json").read_bytes())
    check("slot_and_provenance_cold_readback", slot_digest(path)["digest"] == ent["digest"] and digest(provenance) == ent["window_provenance_sha256"])
    check("native_temporal_layout", tokens.shape[1] == 384 and offsets[-1] == len(tokens) and windows.tolist() == [2, 0, 3, 0, 2, 1, 2],
          tokens=len(tokens), windows=windows.tolist(), offsets=offsets.tolist())
    check("text_observations_are_audio_absent", absent.tolist() == [False, True, False, True, False, False, False])
    check("text_cannot_change_audio", np.array_equal(tokens[offsets[0]:offsets[1]], tokens[offsets[4]:offsets[5]]))
    check("deterministic_probe", entry["probe"]["byte_identical"] and entry["probe"]["tokens"] > 0)
    check("frozen_cuda_encoder", bool(entry["frozen_encoder_state_sha256"]) and entry["contract"]["runtime"]["device"].startswith("NVIDIA")
          and entry["contract"]["preprocessing"]["decoder"] is False)
    for i, segments in enumerate(provenance["windows"]):
        expected_tokens = sum((w["resampled_end"] - w["resampled_start"] + 319) // 320 for w in segments)
        check(f"row_{i}_valid_tokens_only", offsets[i + 1] - offsets[i] == expected_tokens)
    reader = ChannelReader(storepath, policies)
    denied = {**policy, "allow_channels": ["surface"]}
    check("excluded_perception_reads_no_blob", visible_audio(reader, denied, rows[0])[0] == [])
    refuses("partial_mask_cannot_reopen_raw", lambda: visible_audio(reader, {**policy, "deny_fields": ["rms"]}, rows[0]), "MNLM_AUDIO_MASK_UNSUPPORTED")
    refuses("alias_mask_cannot_reopen_raw", lambda: visible_audio(reader, {**policy, "label_aliases": ["Lord"]}, rows[0]), "MNLM_AUDIO_MASK_UNSUPPORTED")
    reader.close()
    region = spans(audio[0])[0]
    raw = resolver.read(region["sha256"], region["media_type"])
    refuses("outside_retained_samples", lambda: waveform(raw, {**region, "end": region["end"] + 1}), "MNLM_AUDIO_SPAN")
    refuses("wrong_rate_refused", lambda: waveform(raw, {**region, "rate": 16000}), "MNLM_AUDIO_SPAN")
    refuses("wrong_channel_refused", lambda: waveform(raw, {**region, "channel": 1}), "MNLM_AUDIO_SPAN")
    refuses("wrong_media_refused", lambda: waveform(raw, {**region, "media_type": "audio/wav"}), "MNLM_UCR_BLOB_TYPE")
    corrupted = out / "corrupt-control.ogg"
    corrupted.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    badmanifest = out / "corrupt-blobs.json"
    changed = copy.deepcopy(blobs)
    changed[region["sha256"]]["path"] = str(corrupted.resolve())
    badmanifest.write_bytes(canonical({"schema": "ucr-blobs-v1", "blobs": list(changed.values())}))
    badresolver = Blobs(badmanifest, file_hash(badmanifest))
    refuses("corrupt_raw_refused", lambda: badresolver.read(region["sha256"], region["media_type"]), "MNLM_UCR_BLOB_DIGEST")
    refuses("manifest_pin_refused", lambda: Blobs(blobpath, "0" * 64), "MNLM_UCR_BLOB_MANIFEST")
    refuses("missing_manifest_refused", lambda: Blobs(out / "not-present.json", "0" * 64), "MNLM_UCR_BLOB_MANIFEST")
    # Directly replay a real accepted atom through the same frozen instrument,
    # and verify that corruption cannot produce a second output generation.
    encoder = AudioEncoder()
    repeated, _ = encoder.embed(spans(audio[0]), resolver)
    check("cold_model_replay", repeated.tobytes() == tokens[offsets[0]:offsets[1]].tobytes())
    refuses("corrupt_blob_cannot_measure", lambda: encoder.embed(spans(audio[0]), badresolver), "MNLM_UCR_BLOB_DIGEST")
    del encoder
    maskedpath = out / "masked-policies.json"
    maskedpath.write_bytes(canonical({family: denied}))
    refuses("entirely_absent_not_registered", lambda: measure_audio_ucr(SPEC, {family: rows}, out / "absent-slots", {},
            storepath, maskedpath, None, None), "MNLM_UCR_CHANNEL_UNOBSERVED")
    cold = Store(storepath)
    check("negative_controls_no_source_mutation", cold.head == generation and cold.audit() == census)
    cold.close()
    check("negative_controls_no_slot_mutation", slot_digest(path)["digest"] == ent["digest"] and not (out / "absent-slots").exists())
    import torch
    checked = time.monotonic()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "generation": generation,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "admitted": [], "served": False,
              "seconds": {"prepare": prepared - started, "measure": measured - prepared,
                          "readback_controls": checked - measured, "instrumented_total": checked - started},
              "limitations": ["two recordings; no statistical information admission", "frozen audio perception is not continual model learning",
                              "masked waveforms require reviewed source-span adapters", "no verified word alignment"]}
    (out / "report.json").write_bytes(canonical(report))
    (out / "SHA256.json").write_bytes(canonical({str(p.relative_to(out)): file_hash(p) for p in out.rglob("*")
                                                if p.is_file() and not p.name.endswith(("-shm", "-wal"))}))
    print(json.dumps({k: report[k] for k in ("passed", "failed", "gpu_peak_allocated_bytes")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(run_cli(main))
