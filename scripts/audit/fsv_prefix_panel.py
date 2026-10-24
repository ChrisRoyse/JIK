"""Cold native-panel replay, prefix isolation, and corrupt-publication controls.

PC-07/13/35: one source replay and streaming seals; O(native bytes + tokenization),
O(largest reading) working memory. Shadows hardlink untouched native files; only
the small fields channel is copied for destructive controls. No outcome reads.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import CHANNELS, SenseError, require
from senses.instruments import measure, slot_digest as reading_digest
from senses.pipeline import file_hash
from senses.trajectory_population import publish, read_pinned
from senses.trajectory_prefix import PrefixInputs
from prefix_panel import load, native_state, validate_plan
from slotio import npz_member, slot_digest


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})
        require(ok, "MNLM_PREFIX_PANEL_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as e:
            check(name, e.code == code, code=e.code)
        else:
            check(name, False)

    reader = PrefixInputs(a.prefix, a.prefix_sha256)
    manifest, rows = load(a.panel, a.panel_sha256)
    plan = read_pinned(a.plan, a.plan_sha256)
    validate_plan(reader, plan)
    check("source_and_plan_bound", manifest["input_manifest_sha256"] == reader.identity
          and manifest["plan_sha256"] == a.plan_sha256
          and [r["atom"] for r in rows] == list(reader.examples))
    before = {p.name: file_hash(p) for p in a.panel.iterdir() if p.is_file()}
    source_before = {p.name: file_hash(p) for p in a.prefix.iterdir() if p.is_file()}

    # B1: independently remeasure every current observation and its strict past,
    # compare every native row and binding, rather than trust writer-return counts.
    arrays = {c: {k: npz_member(a.panel / (c + ".npz"), k)
                  for k in ("tokens", "offsets", "absent", "windows")} for c in CHANNELS}
    equal = {c: True for c in CHANNELS}
    states = {c: set() for c in CHANNELS}
    try:
        for i, row in enumerate(rows):
            current, past, binding = reader.read(row["atom"])
            require(binding == row["binding"], "MNLM_PREFIX_PANEL_FSV", "prefix binding differs")
            for c in CHANNELS:
                slot = measure(current, c, past if c == "history" else None)
                ar = arrays[c]
                lo, hi = ar["offsets"][i:i + 2]
                values = ar["tokens"][lo:hi]
                expected = row["slots"][c]
                require(np.array_equal(values, slot["tokens"]) and list(values.shape) == expected["shape"]
                        and bool(ar["absent"][i]) == (slot["kind"] == "Absent")
                        and slot["kind"] == expected["kind"] and slot["reason"] == expected["reason"]
                        and reading_digest(slot) == expected["reading_digest"]
                        and native_state(values) == expected["state_sha256"]
                        and int(ar["windows"][i]) == expected["windows"],
                        "MNLM_PREFIX_PANEL_FSV", "native payload/metadata differs from source replay")
                first = ar["tokens"][ar["offsets"][0]:ar["offsets"][1]]
                equal[c] = equal[c] and np.array_equal(first, values)
                states[c].add(native_state(values))
        for c, ar in arrays.items():
            v = manifest["variation"][c]
            check("complete_native_replay_" + c, v["exact_constant"] == equal[c]
                  and v["states"] == len(states[c]) and v["token_rows"] == len(ar["tokens"])
                  and v["absent"] == int(ar["absent"].sum()) and v["shannon_information_measured"] is False,
                  token_rows=len(ar["tokens"]), states=len(states[c]), constant=equal[c])
    finally:
        for ar in arrays.values():
            for value in ar.values():
                if hasattr(value, "_mmap"):
                    value._mmap.close()

    # B2: history access is limited to the sealed current IDs. Caller-owned copies
    # cannot change later readings, and IDs/dependencies cannot turn constants into
    # information. Payload changes must still change the physical state digest.
    atom = rows[0]["atom"]
    current, past, binding = reader.read(atom)
    expected = copy.deepcopy((current, past, binding))
    current["unit"] = "caller-mutation"
    past.reverse()
    binding["visible_past"].clear()
    check("caller_mutations_isolated", reader.read(atom) == expected)
    refused("past_id_cannot_open_current", lambda: reader.read(expected[2]["visible_past"][0]), "MNLM_PREFIX_NOT_FOUND")
    refused("unknown_id", lambda: reader.read("0" * 64), "MNLM_PREFIX_NOT_FOUND")
    current, past, _ = expected
    history = measure(current, "history", past)
    changed = {**history, "dependencies": ["different-id"]}
    check("identity_is_not_measurement_variation", reading_digest(changed) != reading_digest(history)
          and native_state(changed["tokens"]) == native_state(history["tokens"]))
    changed_values = history["tokens"].copy()
    changed_values[0, 0] += 1
    check("actual_payload_changes_state", native_state(changed_values) != native_state(history["tokens"]))
    check("reader_has_no_outcome_surface", set(reader.__dict__) == {
        "manifest", "examples", "records", "identity", "reader_code_sha256"})
    bad = copy.deepcopy(plan)
    bad["input_manifest_sha256"] = "0" * 64
    refused("different_input_plan", lambda: validate_plan(reader, bad), "MNLM_PREFIX_PANEL_PLAN")
    bad = copy.deepcopy(plan)
    bad["contracts"][0]["weights_sha256"] = "0" * 64
    refused("different_instrument_plan", lambda: validate_plan(reader, bad), "MNLM_PREFIX_PANEL_INSTRUMENT")

    # B3: trusted manifest pins every native file, layout, and row publication.
    # No native source hardlink is ever opened for writing.
    def shadow(name, change=None, bad_file=False):
        dest = a.output / name
        dest.mkdir()
        for c in CHANNELS:
            src, dst = a.panel / (c + ".npz"), dest / (c + ".npz")
            if bad_file and c == "fields":
                dst.write_bytes(src.read_bytes() + b"CORRUPTION")
            else:
                os.link(src, dst)
        os.link(a.panel / "rows.json", dest / "rows.json")
        m = copy.deepcopy(manifest)
        if change:
            change(m)
        sha = publish(dest / "manifest.json", m)
        publish(dest / "COMMITTED", {"manifest_sha256": sha})
        return dest, sha

    for name, change, bad_file, code in [
        ("native_corruption", None, True, "MNLM_PREFIX_PANEL_DIGEST"),
        ("layout_seal_changed", lambda m: m["slots"]["surface"]["seal"].update(layout_digest="0" * 64), False, "MNLM_PREFIX_PANEL_DIGEST"),
        ("channel_identity_changed", lambda m: m["slots"]["surface"]["contract"].update(channel="history"), False, "MNLM_PREFIX_PANEL_DIGEST"),
        ("admission_claim", lambda m: m.update(stage3_admitted=True), False, "MNLM_PREFIX_PANEL_MANIFEST"),
        ("production_claim", lambda m: m.update(stage4_accepted=True), False, "MNLM_PREFIX_PANEL_MANIFEST"),
    ]:
        dest, sha = shadow(name, change, bad_file)
        refused(name, lambda: load(dest, sha), code)
    dest, sha = shadow("commit_mismatch")
    (dest / "COMMITTED").write_text("{}", encoding="utf-8")
    refused("commit_mismatch", lambda: load(dest, sha), "MNLM_PREFIX_PANEL_UNCOMMITTED")
    check("source_prefix_unchanged", source_before == {n: file_hash(a.prefix / n) for n in source_before})
    check("native_publication_unchanged", before == {n: file_hash(a.panel / n) for n in before})
    report = {"schema": "prefix-panel-fsv-v1", "checks": checks, "rows": len(rows),
              "source_manifest_sha256": reader.identity, "panel_manifest_sha256": a.panel_sha256,
              "plan_sha256": a.plan_sha256, "native_files": manifest["slots"], "variation": manifest["variation"],
              "author_checked": True, "independently_adjudicated": False,
              "stage3_admitted": False, "stage4_accepted": False,
              "audit_code_sha256": file_hash(Path(__file__)), "wall_seconds": time.perf_counter() - started}
    sha = publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha, "checks": len(checks), "rows": len(rows), "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("prefix", "panel", "plan", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("prefix", "panel", "plan"):
        p.add_argument("--" + n + "-sha256", required=True)
    run(p.parse_args())
