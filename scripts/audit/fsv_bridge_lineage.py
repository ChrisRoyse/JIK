"""Cold raw/UCR readback and real-source negative controls for release lineage."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.bridge_inspect import episode, META, STEP
from canon.tfrecord import frames, decode
from senses.bridge import input_plan, policy
from senses.bridge_lineage import bind_sources, match_step, assess, SOURCES, LIMITS, EXPORT_SHA, put
from senses.contract import canonical, SenseError, require, seal_record
from senses.pipeline import file_hash, read_bound


def readback(root, audit_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **detail):
        checks.append({"name": name, "pass": bool(ok), **detail})
        require(ok, "MNLM_BRIDGE_LINEAGE_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False)

    hashes = json.loads((audit_dir / "SHA256.json").read_bytes())
    check("sealed_artifacts_cold_hash", all(file_hash(audit_dir / n) == h for n, h in hashes.items()))
    report = json.loads((audit_dir / "report.json").read_bytes())
    inventory = [json.loads(line) for line in (audit_dir / "episodes.jsonl").read_bytes().splitlines()]
    before = {p["path"]: file_hash(p["path"]) for p in input_plan(root)}
    code = bind_sources(audit_dir)
    check("provider_code_bound_without_execution", set(code) == set(SOURCES))
    bad_sources = out / "source-drift"
    bad_sources.mkdir()
    for name, raw in code.items():
        put(bad_sources / name, raw + (b"\n# drift" if name == "provider-rlds-builder.py" else b""))
    refused("changed_provider_code_refuses", lambda: bind_sources(bad_sources), "MNLM_BRIDGE_LINEAGE_SOURCE")

    rows = iter(inventory)
    export = root / "canon/bridge-observations-v2/observations.ucr.jsonl"
    inputs = read_bound(export, EXPORT_SHA)
    totals, footprints, collections = Counter(), Counter(), set()
    first = None
    raw_receipts = []
    for source in input_plan(root):
        for offset, payload in frames(source["path"], source["sha256"]):
            n, data, path, languages = episode(decode(payload))
            item = next(rows, None)
            require(item is not None, "MNLM_BRIDGE_LINEAGE_FSV", "missing inventory episode")
            psha = hashlib.sha256(payload).hexdigest()
            require((item["shard"], item["offset"], item["source_path"], item["episode_id"], item["payload_sha256"])
                    == (source["path"].name, offset, path, data[META + "episode_id"][0], psha),
                    "MNLM_BRIDGE_LINEAGE_FSV", "physical inventory binding mismatch")
            states = data[STEP + "observation/state"]
            commands = data[STEP + "action"]
            # Independent stdlib implementation of the persisted numerical counters.
            equal = [commands[7 * i + j] == struct.unpack("<f", struct.pack("<f", states[7 * (i + 1) + j] - states[7 * i + j]))[0]
                     for i in range(n - 1) for j in range(6)]
            observed = {"steps": n, "transitions": n - 1, "next_delta_equal_components_6d": sum(equal),
                        "next_delta_equal_rows_6d": sum(all(equal[i:i + 6]) for i in range(0, len(equal), 6)),
                        "last_command_all_zero": all(v == 0 for v in commands[-7:]), "canonical_steps_matched": n}
            for key, values in (("state", states), ("command", commands)):
                require(item[key + "_f32_sha256"] == hashlib.sha256(struct.pack("<" + "f" * len(values), *values)).hexdigest(),
                        "MNLM_BRIDGE_LINEAGE_FSV", "numerical source digest differs")
            require(all(item[k] == v for k, v in observed.items()) and not item["errors"],
                    "MNLM_BRIDGE_LINEAGE_FSV", "cold counter mismatch")
            fp = {"numpy_collection_path": path.startswith(tuple("/nfs/kun2/users/homer/datasets/bridge_data_all/" + p + "/"
                                                                for p in ("numpy_256", "scripted_numpy_256"))) and path.endswith("/train/out.npy"),
                  "all_image_flags_false": not any(data[META + f"has_image_{i}"][0] for i in range(4)),
                  "terminal_is_final_index": all(bool(v) == (i == n - 1) for i, v in enumerate(data[STEP + "is_terminal"])),
                  "reward_is_final_index": all(v == float(i == n - 1) for i, v in enumerate(data[STEP + "reward"])),
                  "discount_is_one": all(v == 1 for v in data[STEP + "discount"]),
                  "instruction_constant": len(set(languages)) == 1,
                  "language_flag_matches_bytes": bool(data[META + "has_language"][0]) == bool(languages[0])}
            require(item["footprint"] == fp, "MNLM_BRIDGE_LINEAGE_FSV", "cold footprint mismatch")
            for i in range(n):
                r = next(inputs, None)
                require(r is not None and r["locator"]["row"] == i and r["locator"]["record_digest"] == psha,
                        "MNLM_BRIDGE_LINEAGE_FSV", "cold canonical join differs")
                ch = r["channels"]
                for actual, expected in (([f["value"] for f in ch["fields"]["items"]], states[7 * i:7 * (i + 1)]),
                                         ([f["value"] for f in ch["behaviour"]["items"][0]["measurements"]], commands[7 * i:7 * (i + 1)])):
                    require(actual == expected and struct.pack("<7f", *actual) == struct.pack("<7f", *expected),
                            "MNLM_BRIDGE_LINEAGE_FSV", "cold recorded inputs differ")
                if first is None:
                    first = (n, data, path, source, offset, psha, r)
            totals.update({k: int(v) for k, v in observed.items()})
            totals["episodes"] += 1
            footprints.update({k: int(v) for k, v in fp.items()})
            collections.add(path)
        raw_receipts.append({"name": source["path"].name, "sha256": source["sha256"], "bytes": source["bytes"]})
    check("complete_raw_to_ucr_cold_replay", next(inputs, None) is None and next(rows, None) is None
          and dict(totals) == report["counts"] and report["collection_files"] == len(collections)
          and report["raw"] == raw_receipts, steps=totals["steps"], episodes=totals["episodes"])
    check("observable_converter_footprint_recomputed", dict(footprints) == report["footprint_episodes"]
          and report["compatible_converter_footprint"] == all(v == totals["episodes"] for v in footprints.values()))
    check("unknowns_not_promoted_to_claims", all(report.get(k) is v for k, v in LIMITS.items()))

    n, data, path, source, offset, psha, original = first
    _, policy_sha = policy()
    match = lambda r: match_step(r, data, 0, source, offset, psha, path, policy_sha)
    match(original)
    check("real_source_step_passes", True)
    # Persist the exact source and altered projections used by each negative control.
    put(out / "real-step.json", canonical(original))
    cases = []
    changed = copy.deepcopy(original)
    changed["channels"]["behaviour"]["items"][0]["measurements"][0]["value"] = 123.0
    cases.append(("invented_command", changed, "MNLM_BRIDGE_LINEAGE_VALUES"))
    changed = copy.deepcopy(original)
    for i in range(6):
        changed["channels"]["behaviour"]["items"][0]["measurements"][i]["value"] = float(
            struct.unpack("<f", struct.pack("<f", data[STEP + "observation/state"][7 + i] - data[STEP + "observation/state"][i]))[0])
    cases.append(("future_delta_substituted_for_recorded_command", changed, "MNLM_BRIDGE_LINEAGE_VALUES"))
    changed = copy.deepcopy(original)
    changed["locator"]["row"] = 1
    cases.append(("reordered_step", changed, "MNLM_BRIDGE_LINEAGE_JOIN"))
    changed = copy.deepcopy(original)
    changed["channels"]["fields"]["items"][0]["unit"] = "metre"
    cases.append(("invented_physical_units", changed, "MNLM_BRIDGE_LINEAGE_VALUES"))
    changed = copy.deepcopy(original)
    changed["channels"]["behaviour"]["items"][0]["evidence"]["alignment"] = "physically_verified"
    cases.append(("unsupported_timing_claim", changed, "MNLM_BRIDGE_LINEAGE_CLAIM"))
    for name, record, code in cases:
        seal_record(record)
        put(out / (name + ".json"), canonical(record))
        refused(name, lambda r=record: match(r), code)
    altered = copy.deepcopy(data)
    altered[STEP + "reward"][0] = 1.0
    put(out / "altered-reward.json", canonical(altered[STEP + "reward"]))
    check("nonfinal_reward_breaks_converter_footprint", not assess(n, altered, path)["footprint"]["reward_is_final_index"])
    check("index_reward_never_promoted_to_success", report["terminal_reward_is_task_success"] is False)
    check("raw_sources_unchanged", all(file_hash(p) == h for p, h in before.items()))
    check("audit_outputs_unchanged", all(file_hash(audit_dir / n) == h for n, h in hashes.items()))
    result = {"acceptance": "author-checked only", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "audit_manifest_sha256": file_hash(audit_dir / "SHA256.json"),
              "readback_code_sha256": file_hash(Path(__file__)), **LIMITS}
    put(out / "report.json", canonical(result))
    put(out / "SHA256.json", canonical({p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    print(json.dumps({"passed": result["passed"], "failed": result["failed"], "steps": totals["steps"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    readback(args.root, args.audit, args.out)
