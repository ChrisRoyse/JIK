"""Full-source cold prefix replay, temporal instrument intake and leakage controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import SenseError, canonical, require, seal_record
from senses.forecast import load_bundle, put
from senses.forecast_prefix import PrefixInputs, training_targets, validate_prefix
from senses.instruments import measure, slot_digest
from senses.pipeline import file_hash, read_bound
from lens.prefix_history import publish, load as load_native
from slotio import npz_member


def run(export, original, prefix, out):
    out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **detail):
        checks.append({"name": name, "pass": bool(ok), **detail})
        require(ok, "MNLM_PREFIX_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == code, code=exc.code)
        else:
            check(name, False)

    sha = file_hash(prefix / "manifest.json")
    reader = PrefixInputs(prefix, sha)
    targets = training_targets(prefix, reader)
    bm, bd = load_bundle(original)
    original_hashes = {n: file_hash(original / n) for n in ("manifest.json", *bm["files"])}
    pm = reader.manifest
    before = {n: file_hash(prefix / n) for n in ("manifest.json", *pm["files"])}
    expected_targets = [r for r in bd["labels.jsonl"] if r["split"] == "train"]
    expected_current = [r for r, label in zip(bd["inputs.jsonl"], bd["labels.jsonl"]) if label["split"] == "train"]
    check("training_targets_unchanged", targets == expected_targets and len(targets) == 239)
    check("training_current_selection_unchanged", [reader.records[k] for k in reader.examples] == expected_current)
    heldout = {r["unit"] for r in bd["labels.jsonl"] if r["split"] != "train"}
    check("heldout_and_future_records_absent", not heldout & {r["unit"] for r in reader.records.values()}
          and not {r["atom_id"] for r in bd["future-evidence.jsonl"]} & reader.records.keys())
    # Rebuild the expected prefix from every persisted source row, without the
    # builder's selection or prefix validator.
    by_series = {(r["locator"]["source"], r["locator"]["shard"]): r for r in expected_current}
    cold, ordered = {}, {r["atom_id"]: [] for r in expected_current}
    n = 0
    for r in read_bound(export, bm["source_sha256"]):
        n += 1
        loc = r["locator"]
        current = by_series.get((loc["source"], loc["shard"]))
        if current is None:
            continue
        end = current["locator"]["row"]
        if max(0, end - 8) <= loc["row"] <= end:
            cold[r["atom_id"]] = r
            if loc["row"] < end:
                ordered[current["atom_id"]].append(r["atom_id"])
    check("complete_cold_source_readback", n == 23702 and cold == reader.records, rows=n, retained=len(cold))
    check("complete_prefix_order_rederived", all(e["past"] == ordered[k] for k, e in reader.examples.items()))
    native_path = out / "native"
    published = publish(reader, native_path)
    nm, nr = load_native(native_path, file_hash(native_path / "manifest.json"))
    native = {k: npz_member(native_path / "history.npz", k) for k in ("tokens", "offsets", "absent", "windows")}
    check("native_manifest_cold_readback", nm == published)
    digests, empty, total_tokens = {}, 0, 0
    for i, key in enumerate(reader.examples):
        current, past, binding = reader.read(key)
        require(all(not r["channels"][c]["items"] for r in past for c in r["channels"] if c != "fields"),
                "MNLM_PREFIX_FSV", "annotation/command/surface entered historical coordinates")
        require(not current["channels"]["behaviour"]["items"], "MNLM_PREFIX_FSV", "uncommissioned command entered current")
        slot = measure(current, "history", past)
        repeat = measure(current, "history", past)
        require(slot_digest(slot) == slot_digest(repeat) and slot["dependencies"] == binding["visible_past"],
                "MNLM_PREFIX_FSV", "temporal instrument lost dependency binding or determinism")
        empty += slot["kind"] == "Absent"
        total_tokens += len(slot["tokens"])
        digests[key] = {"slot_digest": slot_digest(slot), "shape": slot["shape"], "binding": binding}
        a, b = native["offsets"][i:i + 2]
        require(np.array_equal(native["tokens"][a:b], slot["tokens"])
                and bool(native["absent"][i]) == (slot["kind"] == "Absent")
                and native["windows"][i] == len(past) and nr[i]["slot_digest"] == slot_digest(slot)
                and nr[i]["binding"] == binding, "MNLM_PREFIX_FSV", "cold native rows differ from source replay")
    for a in native.values():
        a._mmap.close()
    check("all_native_values_exact_cold_replay", True, value_digest=nm["value_digest"], tokens=total_tokens)
    put(out / "temporal-slots.json", canonical(digests))
    check("real_temporal_instrument_intake", empty == 15 and total_tokens > 0,
          examples=len(digests), absent=empty, tokens=total_tokens)
    key = next(k for k, e in reader.examples.items() if len(e["past"]) == 8)
    current = reader.records[key]
    past = [reader.records[a] for a in reader.examples[key]["past"]]
    future = next(r for r, label in zip(bd["future-evidence.jsonl"], bd["labels.jsonl"]) if label["atom"] == key)
    refused("current_in_history", lambda: validate_prefix(current, past[1:] + [current], reader.plan), "MNLM_PREFIX_ORDER")
    refused("future_in_history", lambda: validate_prefix(current, past[1:] + [future], reader.plan), "MNLM_PREFIX_ORDER")
    refused("truncated_history", lambda: validate_prefix(current, past[1:], reader.plan), "MNLM_PREFIX_ORDER")
    refused("reversed_history", lambda: validate_prefix(current, list(reversed(past)), reader.plan), "MNLM_PREFIX_ORDER")
    for name, mutate in (("cross_group", lambda r: r.__setitem__("unit", "wrong-group")),
                         ("cross_series", lambda r: r["locator"].__setitem__("shard", "another-series")),
                         ("unbound_clock", lambda r: r["locator"].__setitem__("manifest_digest", "0" * 64))):
        changed = copy.deepcopy(past)
        mutate(changed[0])
        seal_record(changed[0])
        put(out / (name + ".jsonl"), b"".join(canonical(r) + b"\n" for r in changed))
        refused(name, lambda p=changed: validate_prefix(current, p, reader.plan), "MNLM_PREFIX_DOMAIN")
    changed = copy.deepcopy(past)
    changed[0]["channels"]["fields"]["items"][0]["unit"] = "different-unit"
    seal_record(changed[0])
    refused("historical_coordinate_unit_drift", lambda: validate_prefix(current, changed, reader.plan), "MNLM_PREFIX_FIELDS")
    changed = copy.deepcopy(past)
    changed[0]["channels"]["structure"]["items"][-1]["value"] += 1
    seal_record(changed[0])
    refused("historical_clock_disagrees_with_locator", lambda: validate_prefix(current, changed, reader.plan), "MNLM_PREFIX_CLOCK")
    other = next(r for r, label in zip(bd["inputs.jsonl"], bd["labels.jsonl"]) if label["split"] != "train")
    refused("heldout_target_cannot_be_opened", lambda: reader.read(other["atom_id"]), "MNLM_PREFIX_NOT_FOUND")
    before_slot = digests[key]["slot_digest"]
    c, h, _ = reader.read(key)
    h[0]["channels"]["fields"]["items"][0]["value"] = 123.0
    c2, h2, _ = reader.read(key)
    check("returned_view_cannot_mutate_input_state", slot_digest(measure(c2, "history", h2)) == before_slot)

    # A sensing reader must work without ever opening the supervised target file.
    missing = out / "no-target-access"
    missing.mkdir()
    for name in ("manifest.json", *(pm["files"].keys() - {"training-targets.jsonl"})):
        shutil.copyfile(prefix / name, missing / name)
    blind = PrefixInputs(missing, sha)
    c, h, _ = blind.read(key)
    check("input_reader_has_no_target_file_dependency", slot_digest(measure(c, "history", h)) == before_slot)
    # Exact manifest/file pins must detect tampering before the encoder runs.
    with (missing / "records.jsonl").open("ab") as f:
        f.write(b"\n")
    refused("input_mutated_after_sealing", lambda: PrefixInputs(missing, sha), "MNLM_PREFIX_DIGEST")
    refused("unbound_manifest", lambda: PrefixInputs(prefix, "0" * 64), "MNLM_PREFIX_DIGEST")
    policy_fault = out / "resealed-policy-drift"
    policy_fault.mkdir()
    for name in ("manifest.json", *pm["files"]):
        shutil.copyfile(prefix / name, policy_fault / name)
    changed = json.loads((policy_fault / "policy.json").read_bytes())
    changed["allow_channels"].append("behaviour")
    (policy_fault / "policy.json").write_bytes(canonical(changed))
    fm = copy.deepcopy(pm)
    fm["files"]["policy.json"] = {"sha256": file_hash(policy_fault / "policy.json"), "bytes": (policy_fault / "policy.json").stat().st_size}
    (policy_fault / "manifest.json").write_bytes(canonical(fm))
    refused("resealed_policy_cannot_enable_commands", lambda: PrefixInputs(policy_fault, file_hash(policy_fault / "manifest.json")),
            "MNLM_PREFIX_POLICY")
    corrupted = out / "native-value-substitution"
    shutil.copytree(native_path, corrupted)
    # A valid native file with a changed value still fails the original seal.
    with np.load(corrupted / "history.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["tokens"][0, 0] += 1
    with (corrupted / "history.npz").open("wb") as f:
        np.savez(f, **arrays)
    refused("saved_native_value_substitution", lambda: load_native(corrupted, file_hash(corrupted / "manifest.json")),
            "MNLM_PREFIX_SLOT_DIGEST")
    check("original_forecast_and_consumed_test_unchanged", all(file_hash(original / n) == h for n, h in original_hashes.items()))
    check("prefix_artifact_unchanged", all(file_hash(prefix / n) == h for n, h in before.items()))
    report = {"acceptance": "author-checked only", "manifest_sha256": sha, "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "temporal_slots_sha256": file_hash(out / "temporal-slots.json"), "new_outcome_evaluation": False,
              "native_value_digest": nm["value_digest"], "native_manifest_sha256": file_hash(native_path / "manifest.json"),
              "readback_code_sha256": file_hash(Path(__file__)), "stage3_admitted": False, "stage4_accepted": False}
    put(out / "report.json", canonical(report))
    put(out / "SHA256.json", canonical({p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    print(json.dumps({k: report[k] for k in ("passed", "failed", "temporal_slots_sha256")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "original", "prefix", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    run(a.export, a.original, a.prefix, a.out)
