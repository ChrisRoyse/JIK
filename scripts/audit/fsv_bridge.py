"""Boundary controls on a real retained Bridge episode; fault copies are evidence only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from tfrecord import Example, decode, frames, masked_crc, RecordError
from bridge_inspect import episode, inspect_images, META, STEP


def packed(payload):
    size = struct.pack("<Q", len(payload))
    return size + struct.pack("<I", masked_crc(size)) + payload + struct.pack("<I", masked_crc(payload))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with args.raw.open("rb") as f:
        before = hashlib.file_digest(f, "sha256").hexdigest()
    stream = frames(args.raw)
    _, payload = next(stream)
    stream.close()
    source = decode(payload)
    n, data, locator, language = episode(source)
    checks = []

    def refusal(name, call, code):
        try:
            call()
        except RecordError as exc:
            checks.append({"name": name, "pass": exc.code == code, "code": exc.code})
        else:
            checks.append({"name": name, "pass": False, "code": "unexpected_success"})

    original = args.out / "real-episode.tfrecord"
    original.write_bytes(packed(payload))
    reread = list(frames(original, hashlib.sha256(original.read_bytes()).hexdigest()))
    checks.append({"name": "real_framing_to_features", "pass": len(reread) == 1 and decode(reread[0][1]) == source,
                   "steps": n, "fields": len(source)})
    for name, byte, code in (("length_crc", 8, "MNLM_TF_LENGTH_CRC"),
                             ("payload_crc", 16, "MNLM_TF_DATA_CRC")):
        changed = bytearray(original.read_bytes())
        changed[byte] ^= 1
        path = args.out / (name + ".tfrecord")
        path.write_bytes(changed)
        refusal(name, lambda p=path: list(frames(p)), code)
    cut = args.out / "truncated.tfrecord"
    cut.write_bytes(original.read_bytes()[:-1])
    refusal("truncation", lambda: list(frames(cut)), "MNLM_TF_TRUNCATED")
    refusal("record_budget", lambda: list(frames(original, max_record_bytes=16)), "MNLM_TF_BUDGET")
    refusal("file_digest", lambda: list(frames(original, "0" * 64)), "MNLM_TF_FILE_DIGEST")
    refusal("unknown_wire_field", lambda: decode(payload + b"\x10\x01"), "MNLM_TF_UNKNOWN_FIELD")
    obj = Example()
    obj.ParseFromString(payload)
    obj.features.feature[STEP + "action"].float_list.value[0] = float("nan")
    refusal("nonfinite", lambda: decode(obj.SerializeToString()), "MNLM_TF_NONFINITE")
    for name, mutation, code in (
        ("unknown_feature", lambda d: d.update(extra=("int64_list", [1])), "MNLM_BRIDGE_SCHEMA"),
        ("missing_feature", lambda d: d.pop(STEP + "reward"), "MNLM_BRIDGE_SCHEMA"),
        ("short_state", lambda d: d[STEP + "observation/state"][1].pop(), "MNLM_BRIDGE_SHAPE"),
        ("invalid_flag", lambda d: d[META + "has_image_0"][1].__setitem__(0, 2), "MNLM_BRIDGE_BOOLEAN"),
        ("wrong_boundary", lambda d: d[STEP + "is_last"][1].__setitem__(0, 1), "MNLM_BRIDGE_BOUNDARY"),
        ("wrong_utf8", lambda d: d[STEP + "language_instruction"][1].__setitem__(0, b"\xff"), "MNLM_BRIDGE_TEXT")):
        changed = copy.deepcopy(source)
        mutation(changed)
        refusal(name, lambda d=changed: episode(d), code)
    images = inspect_images(data, n)
    checks.append({"name": "flagged_absent_not_promoted", "pass": all(not c["usable_by_declared_flag"] for c in images),
                   "conflicts": sum(c["availability_conflict"] for c in images)})
    checks.append({"name": "real_placeholder_detected", "pass": images[1]["spatially_constant_frames"] == n})
    with args.raw.open("rb") as f:
        after = hashlib.file_digest(f, "sha256").hexdigest()
    checks.append({"name": "source_unchanged", "pass": before == after, "sha256": after})
    report = {"acceptance": "author-checked only", "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks)}
    (args.out / "report.json").write_bytes(json.dumps(report, indent=2, sort_keys=True).encode())
    inventory = {}
    for p in args.out.iterdir():
        with p.open("rb") as f:
            inventory[p.name] = hashlib.file_digest(f, "sha256").hexdigest()
    (args.out / "SHA256.json").write_bytes(json.dumps(inventory, sort_keys=True).encode())
    print(json.dumps(report))
    return bool(report["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
