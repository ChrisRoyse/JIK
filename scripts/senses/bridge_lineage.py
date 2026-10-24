"""Audit retained Bridge release bytes against a pinned, compatible converter.

O(raw bytes + export bytes + steps), one episode resident. Compatibility is not
build attestation, physical timing, task success, or scientific admission. Never
execute the provider's Python or load its NumPy/pickle files.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.bridge_inspect import META, STEP, episode
from canon.tfrecord import decode, frames
from senses.bridge import input_plan, policy
from senses.contract import SenseError, canonical, digest, require, validate_record
from senses.pipeline import file_hash, read_bound

EXPORT_SHA = "ec14c9c757227ab1798d2c67a49252e8f0768bc01c12e84bb99058e83335fde6"
PREFIX = "/nfs/kun2/users/homer/datasets/bridge_data_all/"
SOURCES = {
    "provider-rlds-builder.py": {
        "url": "https://raw.githubusercontent.com/kpertsch/bridge_rlds_builder/"
               "f0d16c5a8384c1476aa1c274a9aef3a5f76cbada/bridge_dataset/bridge_dataset_dataset_builder.py",
        "bytes": 9921, "sha256": "c80f41302fcbc088058db5a3bd418f19fe4ec12c2f8dc015e991798eff047eba"},
    "bridge-raw-converter-2023.py": {
        "url": "https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/"
               "e8d1639c20dd509689fa19bf5e451b26ae0cf517/data_processing/bridgedata_raw_to_numpy.py",
        "bytes": 9946, "sha256": "66b0320a8de7e2b1ee6121f6dd4e92438d50134bd2e3c9d070190c67d2610079"},
}
LIMITS = {
    "release_build_attested": False,
    "original_raw_to_numpy_bytes_bound": False,
    "physical_units_commissioned": False,
    "action_observation_timing_commissioned": False,
    "independent_sampling_proven": False,
    "terminal_reward_is_task_success": False,
    "commands_enabled_for_forecast": False,
    "stage3_admitted": False,
    "stage4_accepted": False,
}


def put(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def bind_sources(directory):
    """Hash the same bounded bytes retained as evidence; no import/eval/pickle."""
    result = {}
    for name, spec in SOURCES.items():
        with (directory / name).open("rb") as f:
            raw = f.read(spec["bytes"] + 1)
        require(len(raw) == spec["bytes"] and hashlib.sha256(raw).hexdigest() == spec["sha256"],
                "MNLM_BRIDGE_LINEAGE_SOURCE", name)
        result[name] = raw
    return result


def assess(n, data, source_path):
    """Observable footprint only. None of these tests proves source causality."""
    last = [0] * (n - 1) + [1]
    language = data[STEP + "language_instruction"]
    state = np.asarray(data[STEP + "observation/state"], dtype=np.float32).reshape(n, 7)
    action = np.asarray(data[STEP + "action"], dtype=np.float32).reshape(n, 7)
    # The known loader relabels the first six command components this way.
    # Counting equality is a diagnostic, never an authenticity test.
    eq = action[:-1, :6] == np.diff(state[:, :6], axis=0)
    footprint = {
        "numpy_collection_path": any(source_path.startswith(PREFIX + p + "/")
                                     for p in ("numpy_256", "scripted_numpy_256"))
                                 and source_path.endswith("/train/out.npy"),
        "all_image_flags_false": all(data[META + f"has_image_{i}"] == [0] for i in range(4)),
        "terminal_is_final_index": data[STEP + "is_terminal"] == last,
        "reward_is_final_index": data[STEP + "reward"] == last,
        "discount_is_one": data[STEP + "discount"] == [1.0] * n,
        "instruction_constant": all(s == language[0] for s in language),
        "language_flag_matches_bytes": data[META + "has_language"] == [int(bool(language[0]))],
    }
    return {"footprint": footprint, "steps": n, "transitions": n - 1,
            "next_delta_equal_rows_6d": int(eq.all(axis=1).sum()),
            "next_delta_equal_components_6d": int(eq.sum()),
            "last_command_all_zero": bool((action[-1] == 0).all()),
            "state_f32_sha256": hashlib.sha256(state.astype("<f4").tobytes()).hexdigest(),
            "command_f32_sha256": hashlib.sha256(action.astype("<f4").tobytes()).hexdigest()}


def match_step(r, data, step, source, offset, payload_sha, source_path, policy_sha):
    """Independent raw->UCR comparison; deliberately does not call the adapter."""
    validate_record(r)
    loc, ch = r["locator"], r["channels"]
    require(loc["source"] == "bridge-v2-rlds" and loc["shard"] == source["path"].name + f"#offset={offset}"
            and loc["row"] == step and loc["span"] == [step, step + 1]
            and loc["record_digest"] == payload_sha
            and r["unit"] == digest(["bridge-provider-collection", source_path]),
            "MNLM_BRIDGE_LINEAGE_JOIN", "source episode/order binding changed")
    fields = ch["fields"]["items"]
    command = ch["behaviour"]["items"]
    require(len(command) == 1 and command[0]["event"] == "recorded_command" and command[0]["mode"] == "observed",
            "MNLM_BRIDGE_LINEAGE_COMMAND", "recorded command missing or substituted")
    for items, prefix, key in ((fields, "state", "observation/state"),
                               (command[0]["measurements"], "command", "action")):
        expected = data[STEP + key][7 * step:7 * (step + 1)]
        require(len(items) == 7 and all(f["name"] == f"{prefix}_component_{i}" and f["type"] == "number"
                and f["unit"] is None and f["clock_role"] is None for i, f in enumerate(items))
                and [f["value"] for f in items] == expected
                and struct.pack("<7f", *(f["value"] for f in items)) == struct.pack("<7f", *expected),
                "MNLM_BRIDGE_LINEAGE_VALUES", "raw float32 bits or undeclared units changed")
    ev = command[0]["evidence"]
    require(ev == {"raw_sha256": source["sha256"], "episode_payload_sha256": payload_sha,
                   "offset": offset, "step": step, "alignment": "uncommissioned; no next-state relabeling"},
            "MNLM_BRIDGE_LINEAGE_CLAIM", "command provenance/unknown timing was changed")
    image = ch["perception"]["items"]
    require(len(image) == 1 and image[0]["sha256"] == hashlib.sha256(data[STEP + "observation/image_0"][step]).hexdigest(),
            "MNLM_BRIDGE_LINEAGE_IMAGE", "primary JPEG bytes changed")
    text = "\n".join(x["text"] for x in ch["surface"]["items"])
    require(text.encode() == data[STEP + "language_instruction"][step],
            "MNLM_BRIDGE_LINEAGE_TEXT", "instruction bytes changed")
    provenance = {x["name"]: x["value"] for x in ch["provenance"]["items"]}
    require(provenance["policy_sha256"] == policy_sha and provenance["raw_sha256"] == source["sha256"]
            and provenance["physical_units"] == "uncommissioned"
            and provenance["image_0_provider_present"] is bool(data[META + "has_image_0"][0]),
            "MNLM_BRIDGE_LINEAGE_CLAIM", "source policy/availability changed")


def summarize(rows):
    counts, footprint, errors, collections = Counter(), Counter(), Counter(), set()
    for row in rows:
        counts.update({k: int(row[k]) for k in ("steps", "transitions", "next_delta_equal_rows_6d",
                                               "next_delta_equal_components_6d", "last_command_all_zero")})
        counts["episodes"] += 1
        counts["canonical_steps_matched"] += row["canonical_steps_matched"]
        collections.add(row["source_path"])
        footprint.update({k: int(v) for k, v in row["footprint"].items()})
        errors.update(row["errors"])
    return {"counts": dict(counts), "footprint_episodes": dict(footprint),
            "collection_files": len(collections), "errors": dict(errors)}


def audit(root, source_dir, out):
    start = time.perf_counter()
    sources = bind_sources(source_dir)
    plan = input_plan(root)
    _, policy_sha = policy()
    export = root / "canon/bridge-observations-v2/observations.ucr.jsonl"
    require(sum(p["bytes"] for p in plan) == 1324579732 and export.stat().st_size == 113747310,
            "MNLM_BRIDGE_LINEAGE_SCOPE", "recommission changed audit cost/population")
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in sources.items():
        put(out / name, raw)
    put(out / "sources.json", canonical(SOURCES))
    records = read_bound(export, EXPORT_SHA)
    rows, seen = [], set()
    scanned = []
    prepare_done = time.perf_counter()
    with (out / "episodes.jsonl").open("xb") as f:
        for source in plan:
            for offset, payload in frames(source["path"], source["sha256"]):
                n, data, source_path, _ = episode(decode(payload))
                key = (source_path, data[META + "episode_id"][0])
                require(key not in seen, "MNLM_BRIDGE_LINEAGE_DUPLICATE", "duplicate episode identity")
                seen.add(key)
                payload_sha = hashlib.sha256(payload).hexdigest()
                row = {"shard": source["path"].name, "offset": offset, "payload_sha256": payload_sha,
                       "source_path": source_path, "episode_id": key[1], **assess(n, data, source_path),
                       "canonical_steps_matched": 0, "errors": {}}
                for step in range(n):
                    r = next(records, None)
                    require(r is not None, "MNLM_BRIDGE_LINEAGE_LENGTH", "canonical stream ended early")
                    try:
                        match_step(r, data, step, source, offset, payload_sha, source_path, policy_sha)
                        row["canonical_steps_matched"] += 1
                    except SenseError as exc:
                        row["errors"][exc.code] = row["errors"].get(exc.code, 0) + 1
                f.write(canonical(row) + b"\n")
                rows.append(row)
            scanned.append({"name": source["path"].name, "sha256": source["sha256"], "bytes": source["bytes"]})
        require(next(records, None) is None, "MNLM_BRIDGE_LINEAGE_LENGTH", "extra canonical rows")
        f.flush()
        os.fsync(f.fileno())
    summary = summarize(rows)
    require(summary["counts"]["episodes"] == 622 and summary["counts"]["steps"] == 23702
            and summary["collection_files"] == 416, "MNLM_BRIDGE_LINEAGE_SCOPE", "changed population")
    scan_done = time.perf_counter()
    compatible = all(v == 622 for v in summary["footprint_episodes"].values())
    report = {"schema": "bridge-lineage-audit-v1", "acceptance": "author-checked only",
              "export_sha256": EXPORT_SHA, "raw": scanned, "sources": SOURCES, "policy_sha256": policy_sha,
              **summary, "compatible_converter_footprint": compatible,
              "raw_commands_preserved": not summary["errors"], **LIMITS,
              "audit_code_sha256": file_hash(Path(__file__)),
              "interpretation": "Compatible code and observed footprint are not release-build attestation; "
                                "command/delta inequality does not prove pre-outcome timing. "
                                "Index-generated reward is not independent task-success evidence.",
              "cost_seconds": {"prepare": prepare_done - start, "scan_join": scan_done - prepare_done}}
    report["wall_seconds"] = time.perf_counter() - start
    put(out / "report.json", canonical(report))
    require(not summary["errors"], "MNLM_BRIDGE_LINEAGE_CANONICAL", "raw/canonical mismatch; see refused episode rows")
    put(out / "SHA256.json", canonical({p.name: file_hash(p) for p in sorted(out.iterdir())}))
    print(json.dumps({**summary, "compatible_converter_footprint": compatible,
                      "wall_seconds": report["wall_seconds"], **LIMITS}), flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "sources", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    try:
        audit(a.root, a.sources, a.out)
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        raise SystemExit(3)
