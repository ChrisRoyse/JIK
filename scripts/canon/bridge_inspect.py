"""Complete bounded Bridge RLDS physical census; no admission or source execution.

O(selected bytes + pixels), one <=64 MiB episode resident; schema fixed from the
first retained Example and checked against every record. Every source feature is
accounted for. Quarantined episodes retain their source offsets and error codes.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfrecord import decode, frames, require, RecordError

META = "episode_metadata/"
STEP = "steps/"
SCHEMA = {META + "episode_id": ("int64_list", 1), META + "file_path": ("bytes_list", 1)}
SCHEMA.update({META + k: ("int64_list", 1) for k in ("has_language", *(f"has_image_{i}" for i in range(4)))})
SCHEMA.update({STEP + k: ("int64_list", None) for k in ("is_first", "is_last", "is_terminal")})
SCHEMA.update({STEP + k: ("float_list", None) for k in ("action", "discount", "reward", "language_embedding", "observation/state")})
SCHEMA.update({STEP + k: ("bytes_list", None) for k in ("language_instruction", *(f"observation/image_{i}" for i in range(4)))})
WIDTHS = {"action": 7, "observation/state": 7, "language_embedding": 512}


def episode(features):
    require(set(features) == set(SCHEMA), "MNLM_BRIDGE_SCHEMA", f"unexpected keys: {sorted(set(features) ^ set(SCHEMA))}")
    n = len(features[STEP + "is_first"][1])
    require(n > 1, "MNLM_BRIDGE_LENGTH", "need a complete ordered episode")
    data = {}
    for key, (kind, count) in SCHEMA.items():
        typ, values = features[key]
        want = count if count is not None else n * WIDTHS.get(key.removeprefix(STEP), 1)
        require(typ == kind and len(values) == want, "MNLM_BRIDGE_SHAPE", key)
        data[key] = values
    for key, values in data.items():
        if key.startswith(META + "has_") or key in (STEP + "is_first", STEP + "is_last", STEP + "is_terminal"):
            require(all(x in (0, 1) for x in values), "MNLM_BRIDGE_BOOLEAN", key)
    require(data[STEP + "is_first"] == [1] + [0] * (n - 1)
            and data[STEP + "is_last"] == [0] * (n - 1) + [1], "MNLM_BRIDGE_BOUNDARY", "inconsistent sequence boundaries")
    try:
        path = data[META + "file_path"][0].decode("utf-8")
        language = [v.decode("utf-8") for v in data[STEP + "language_instruction"]]
    except UnicodeError as exc:
        raise RecordError("MNLM_BRIDGE_TEXT", "invalid UTF-8") from exc
    require(path and all("\x00" not in v for v in [path, *language]), "MNLM_BRIDGE_TEXT", "empty locator or NUL")
    return n, data, path, language


def inspect_images(data, n):
    result = []
    for i in range(4):
        images = data[STEP + f"observation/image_{i}"]
        constant = 0
        for raw in images:
            with Image.open(io.BytesIO(raw)) as im:
                require(im.format == "JPEG" and im.mode == "RGB" and im.size == (256, 256),
                        "MNLM_BRIDGE_IMAGE", "unexpected JPEG layout; do not resize silently")
                im.load()
                extrema = im.getextrema()
                constant += int(all(lo == hi for lo, hi in extrema))
        unique = len(set(images))  # whole byte equality, bounded to one camera episode
        flag = bool(data[META + f"has_image_{i}"][0])
        result.append({"camera": i, "provider_present": flag, "frames": n,
                       "distinct_jpeg_bytes": unique, "spatially_constant_frames": constant,
                       "availability_conflict": not flag and constant < n,
                       "usable_by_declared_flag": flag and constant < n})
    return result


def inspect(root, out):
    start = time.monotonic()
    selection_raw = (root / "census/bridge-observations-v1/selection.json").read_bytes()
    selection = json.loads(selection_raw)
    source = "bridge-v2-rlds"
    paths = selection["sources"][source]["paths"]
    receipts = {}
    for line in (root / "raw" / source / "receipts.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        receipts[r["path"]] = r
    require(set(paths) <= set(receipts), "MNLM_BRIDGE_INCOMPLETE", "wait for every selected acquisition receipt")
    out.mkdir(parents=True, exist_ok=False)
    seen, collections, language_keys = set(), Counter(), set()
    totals, by_camera, errors = Counter(), [Counter() for _ in range(4)], Counter()
    manifest = []
    with (out / "episodes.jsonl").open("wb") as output:
        for url in paths:
            receipt = receipts[url]
            require(receipt["size_match"] and receipt["verified"] is not False,
                    "MNLM_BRIDGE_RECEIPT", "unaccepted raw acquisition")
            path = root / "raw" / source / url.rsplit("/", 1)[-1]
            require(path.stat().st_size == receipt["bytes"], "MNLM_BRIDGE_SIZE", str(path))
            count = 0
            for offset, payload in frames(path, receipt["sha256"]):
                row = {"shard": path.name, "offset": offset, "payload_bytes": len(payload),
                       "payload_sha256": hashlib.sha256(payload).hexdigest(), "row": count}
                count += 1
                totals["episodes"] += 1
                totals["max_payload_bytes"] = max(totals["max_payload_bytes"], len(payload))
                try:
                    n, data, locator, language = episode(decode(payload))
                    key = (locator, data[META + "episode_id"][0])
                    row.update(steps=n, source_episode_id=key[1], source_path=locator,
                               episode_key=hashlib.sha256(json.dumps(key).encode()).hexdigest())
                    require(key not in seen, "MNLM_BRIDGE_DUPLICATE_EPISODE", "same collection/episode identity")
                    seen.add(key)
                    collections[locator] += 1
                    language_keys.update(language)
                    cameras = inspect_images(data, n)
                    row["cameras"] = cameras
                    row["language_present"] = bool(data[META + "has_language"][0])
                    row["language_consistent"] = all(bool(v) == row["language_present"] for v in language)
                    row["terminal_action_zero"] = all(x == 0 for x in data[STEP + "action"][-7:])
                    row["terminal_flags"] = data[STEP + "is_terminal"]
                    # Observed numerical relationships only, not physical-unit/causality assertions.
                    state = np.asarray(data[STEP + "observation/state"], dtype=np.float32).reshape(n, 7)
                    action = np.asarray(data[STEP + "action"], dtype=np.float32).reshape(n, 7)
                    row["state_distinct_rows"] = len({r.tobytes() for r in state})
                    row["action_vs_next_state_delta_mae_6d"] = float(np.mean(np.abs(action[:-1, :6] - np.diff(state[:, :6], axis=0))))
                    for i, c in enumerate(cameras):
                        by_camera[i].update({k: int(c[k]) for k in ("provider_present", "availability_conflict", "usable_by_declared_flag")})
                        by_camera[i]["frames"] += n
                        by_camera[i]["spatially_constant_frames"] += c["spatially_constant_frames"]
                    totals["steps"] += n
                    totals["language_present_episodes"] += row["language_present"]
                    totals["terminal_zero_action_episodes"] += row["terminal_action_zero"]
                    row["status"] = "decoded; semantic alignment uncommissioned"
                except (RecordError, OSError, ValueError) as exc:
                    row["status"] = "quarantine"
                    row["error"] = str(exc)
                    errors[getattr(exc, "code", type(exc).__name__)] += 1
                output.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            manifest.append({"name": path.name, "sha256": receipt["sha256"], "bytes": receipt["bytes"], "episodes": count})
    report = {"schema": "bridge-physical-census-v1", "acceptance": "author-checked only",
              "selection_sha256": hashlib.sha256(selection_raw).hexdigest(), "raw": manifest,
              "totals": dict(totals), "cameras": [dict(c) for c in by_camera], "quarantine": dict(errors),
              "unique_episode_keys": len(seen), "collection_files": len(collections),
              "distinct_instruction_strings": len(language_keys), "collection_counts": dict(collections),
              "unknowns": ["release-specific converter lineage", "physical units and action/state timing",
                           "independent task/collection sampling", "camera availability metadata conflicts"],
              "stage3_admitted": False, "wall_seconds": time.monotonic() - start}
    (out / "report.json").write_bytes((json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    hashes = {}
    for p in out.iterdir():
        with p.open("rb") as f:
            hashes[p.name] = hashlib.file_digest(f, "sha256").hexdigest()
    (out / "SHA256.json").write_bytes(json.dumps(hashes, sort_keys=True).encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ("raw", "collection_counts")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inspect(args.root, args.out)
