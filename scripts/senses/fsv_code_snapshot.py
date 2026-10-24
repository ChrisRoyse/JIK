"""Real pre-test source -> UCR -> native slots -> model; cold boundaries and controls.

No training, label access by an encoder, or scientific admission. The private oracle
tree is read only by this audit to prove that newly applied tests stayed outside inputs.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.perf_counter()

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from oracle.export_observation import input_plan, validate as validate_plan
from senses.code_snapshot import SnapshotInputs, validate, native_plan, CODE
from senses.contract import CHANNELS, SenseError, canonical, digest, require, validate_record
from senses.fsv_fences import audit_ast
from senses.instruments import contracts, measure
from senses.pipeline import file_hash, read_bound
from senses.trajectory_population import publish, read_pinned
from prefix_panel import publish as panel_publish, load as panel_load


def model_input(panel, sha, atom):
    from contextlib import ExitStack
    from model.contracts import Instrument, ModelConfig
    from model.native import from_arrays
    from production.native_rows import mapped_rows, one
    m, rows = panel_load(panel, sha)
    config = ModelConfig()
    specs = tuple(Instrument.from_slot(s["contract"]["name"], s) for s in m["slots"].values())
    i = next(i for i, r in enumerate(rows) if r["atom"] == atom)
    native = {}
    with ExitStack() as stack:
        for spec in specs:
            arr = stack.enter_context(mapped_rows(panel / (spec.channel + ".npz")))
            native[spec.name] = from_arrays(spec, one(arr, i), [atom], config)
    return config, specs, native


def configure():
    import torch
    torch.manual_seed(112)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def replay(root, output):
    import torch
    from model.checkpoint import load, state_hash
    configure()
    binding = json.loads((root / "model-input.json").read_bytes())
    _, specs, native = model_input(root / "panel", binding["panel_sha256"], binding["atom"])
    model, manifest = load(root / "checkpoint", "cpu")
    assert model.specs == specs and manifest["input_sha256"] == digest(binding)
    with torch.no_grad():
        result = model(torch.tensor([[256]], dtype=torch.int64), native)
    out = {"logits_sha256": hashlib.sha256(result["logits"].numpy().tobytes()).hexdigest(),
           "state_sha256": state_hash(model), "instrument_count": result["instrument_count"].tolist(),
           "channel_count": result["channel_count"].tolist(), "meaning_asserted": result["meaning_asserted"]}
    publish(output, out)
    return out


def run(a):
    import numpy as np
    import torch
    from model.network import SensoryLanguageModel
    from model.checkpoint import save, state_hash
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok):
        require(bool(ok), "MNLM_SNAPSHOT_FSV", name)
        checks.append(name)

    def refused(name, fn, code):
        try:
            fn()
        except SenseError as exc:
            check(name, exc.code == code)
        else:
            check(name, False)

    original = read_pinned(a.oracle_plan, a.oracle_plan_sha256)
    reader = SnapshotInputs(a.snapshot, a.snapshot_sha256)
    licence = reader.manifest["licence"]
    plan = input_plan(original, licence=licence)
    pinned_input = read_pinned(a.input_plan, reader.manifest["source_binding"]["input_plan_sha256"])
    check("source_projection_matches_pinned_plan", plan == pinned_input)
    refused("missing_licence_not_defaulted", lambda: input_plan(original), "MNLM_PRETEST_INPUT")
    changed_licence = input_plan(original, licence="BSD-3-Clause control attestation")
    check("licence_not_hardcoded", changed_licence["licence"] == "BSD-3-Clause control attestation"
          and digest(changed_licence) != digest(plan))
    # B1: the projection is independent of every private oracle binding.
    for name in ("benchmark", "allowed_paths", "test_modules", "wheels", "image", "decision",
                 "target", "eval_logs", "test_patch", "patch", "execution"):
        bad = deepcopy(original)
        bad[name] = "PRIVATE_ORACLE_POISON"
        check("private_" + name + "_not_input", input_plan(bad, licence=licence) == plan)
    for name, value in (("target", True), ("cutoff", "after_external_evaluation")):
        bad = deepcopy(plan)
        bad[name] = value
        refused("plan_" + name, lambda b=bad: validate_plan(b), "MNLM_PRETEST_INPUT")
    bad_source = deepcopy(reader.manifest)
    bad_source["source"] = "cuyahoga-eighth-district"
    refused("excluded_court_source", lambda: validate(bad_source), "MNLM_SOURCE_EXCLUDED")
    selected = next(iter(reader.examples))
    returned, past, binding = reader.read(selected)
    returned["channels"]["surface"]["items"][0]["text"] = "MUTATION"
    binding["outcomes_visible"] = True
    check("callers_cannot_mutate_loaded_inputs", reader.read(selected)[0] != returned
          and reader.read(selected)[2]["outcomes_visible"] is False)
    saved = deepcopy(reader.bindings[selected])
    reader.bindings[selected]["outcomes_visible"] = True
    refused("changed_loaded_cutoff", lambda: reader.read(selected), CODE)
    reader.bindings[selected] = saved
    saved_record = reader.records.pop(selected)
    refused("missing_loaded_context", lambda: reader.read(selected), CODE)
    reader.records[selected] = saved_record
    before = {p.relative_to(a.snapshot).as_posix(): file_hash(p) for p in a.snapshot.rglob("*") if p.is_file()}
    # Independent archive reader, not the producer's Git extraction function.
    with tarfile.open(a.data_root / plan["archive"]["path"]) as archive:
        base = {x.name.split("/", 1)[1]: archive.extractfile(x).read() for x in archive if x.isfile()}
    check("archive_all_base_files_exact", len(base) == len(reader.manifest["snapshots"]["base"])
          and all((a.snapshot / "base" / p).read_bytes() == b for p, b in base.items()))
    changes = reader.manifest["changed"]
    check("changed_new_unchanged_present", any(p not in base for p in changes)
          and any(p in base for p in changes) and any(p not in changes for p in base))
    # Private benchmark test bytes genuinely differ, while submitted code is identical.
    hidden = original["allowed_paths"]["test"]
    check("hidden_test_patch_is_excluded", all((a.snapshot / "submitted" / p).read_bytes() == base[p]
          and base[p] != (a.prepared / "agent" / p).read_bytes() for p in hidden))
    check("nonoracle_agent_files_match_execution", all((a.snapshot / "submitted" / p).read_bytes()
          == (a.prepared / "agent" / p).read_bytes() for p in reader.manifest["snapshots"]["submitted"] if p not in hidden))
    for name, alter in (
        ("future_cutoff", lambda m: m.update(cutoff="after_external_evaluation")),
        ("reversed_clock", lambda m: m.update(order=["submitted", "base"])),
        ("changed_inventory", lambda m: m.update(changed=[])),
        ("false_admission", lambda m: m.update(admitted=True)),
        ("unknown_oracle_field", lambda m: m.update(target=True)),
    ):
        bad = deepcopy(reader.manifest)
        alter(bad)
        refused(name, lambda b=bad: validate(b), CODE)
    controls = a.output / "controls"
    shutil.copytree(a.snapshot, controls)
    path = next(p for p in changes if p in base)
    target = controls / "submitted" / path
    raw = target.read_bytes()
    target.write_bytes(raw + b"\n")
    refused("mutated_source", lambda: SnapshotInputs(controls, a.snapshot_sha256), CODE)
    target.write_bytes(raw)
    target.unlink()
    refused("missing_source", lambda: SnapshotInputs(controls, a.snapshot_sha256), CODE)
    target.write_bytes(raw)
    extra = controls / "submitted" / "unknown-evaluation.json"
    extra.write_bytes(b'{"target":true}')
    refused("undeclared_future_file", lambda: SnapshotInputs(controls, a.snapshot_sha256), CODE)
    extra.unlink()
    patch = controls / "submitted.patch"
    patch_bytes = patch.read_bytes()
    patch.write_bytes(patch_bytes + b"\n")
    refused("changed_patch", lambda: SnapshotInputs(controls, a.snapshot_sha256), CODE)
    patch.write_bytes(patch_bytes)
    source_done = time.perf_counter()

    # B2: persisted UCR surfaces, syntax and same-file history independently reread.
    manifest = reader.export(a.output / "ucr")
    cold = list(read_bound(a.output / "ucr/inputs.ucr.jsonl", manifest["sha256"]))
    check("all_ucr_ids_and_bytes", {validate_record(r)["atom_id"]: r for r in cold} == reader.records)
    nodes, parsed = 0, 0
    for r in cold:
        loc, ch = r["locator"], r["channels"]
        raw = (a.snapshot / loc["shard"]).read_bytes()
        check("surface_exact_" + str(len(checks)), ch["surface"]["items"] == [{"text": raw.decode()}]
              and loc["span"] == [0, len(raw)] and loc["record_digest"] == hashlib.sha256(raw).hexdigest())
        fs = ch["structure"]["items"]
        status = next(f["value"] for f in fs if f["name"] == "syntax.status")
        if status == "parsed":
            nodes += audit_ast(raw.decode(), fs)
            parsed += 1
    for atom, name in reader.examples.items():
        current, past, b = reader.read(atom)
        check("history_exact_" + name, all(r["unit"] == current["unit"] and r["locator"]["shard"] == "base/" + name
              and r["locator"]["row"] < current["locator"]["row"] for r in past)
              and b["future_visible"] is False and b["outcomes_visible"] is False)
    check("empty_bytes_retained_explicitly", len(reader.omitted) == 2
          and all(x["reason"] == "empty_file_no_nonempty_atom" for x in reader.omitted))
    refused("base_not_current", lambda: reader.read(next(r["atom_id"] for r in cold if r["locator"]["row"] == 0)), CODE)
    ucr_done = time.perf_counter()

    # B3: count without allocating, then preserve and cold-recount every native row.
    class CountControl:
        identity = "resource-boundary-control-only"
        examples = ["control"]

        def read(self, _):
            r, _, b = reader.read(next(iter(reader.examples)))
            r["channels"]["surface"]["items"] = [{"text": "token " * 8193}]
            return r, [], b
    refused("native_budget_before_allocation", lambda: native_plan(CountControl()), "MNLM_CODE_SNAPSHOT_NATIVE_BUDGET")
    plan = native_plan(reader)
    publish(a.output / "native-plan.json", plan)
    result = panel_publish(reader, plan, a.output / "panel")
    m, rows = panel_load(a.output / "panel", result["manifest_sha256"])
    for channel in CHANNELS:
        with np.load(a.output / "panel" / (channel + ".npz"), allow_pickle=False) as z:
            for i, row in enumerate(rows):
                current, past, _ = reader.read(row["atom"])
                values = measure(current, channel, past if channel == "history" else None)["tokens"]
                check("native_" + channel + "_" + str(i), np.array_equal(values, z["tokens"][z["offsets"][i]:z["offsets"][i+1]]))
    check("same_frozen_instruments", {c: s["contract"] for c, s in m["slots"].items()}
          == {c["channel"]: c for c in contracts()})
    check("no_scientific_promotion", not m["stage3_admitted"] and not m["stage4_accepted"])
    panel_done = time.perf_counter()

    # B4: an actual existing model consumes all native rows, with effective branches.
    configure()
    atom = next(atom for atom, name in reader.examples.items() if name == path)
    config, specs, native = model_input(a.output / "panel", result["manifest_sha256"], atom)
    model = SensoryLanguageModel(config, specs).eval()
    prefix = torch.tensor([[256]], dtype=torch.int64)
    with torch.no_grad():
        original_out = model(prefix, native)
        ablations = {}
        for spec in specs:
            if bool(native[spec.name].absent.all()):
                continue
            branch = model.blocks[0].branches[spec.name]
            gain = branch.gain.clone()
            branch.gain.zero_()
            changed = model(prefix, native)["logits"]
            branch.gain.copy_(gain)
            ablations[spec.channel] = float((changed - original_out["logits"]).abs().max())
    check("at_least_five_effective_branches", len(ablations) >= 5 and all(v > 0 for v in ablations.values()))
    check("seven_channels_observed", original_out["channel_count"].tolist() == [7])
    refused("meaning_still_refused", lambda: model.assert_meaning(native), "MNLM_MODEL_UNCOMMISSIONED")
    first = specs[0].name
    bad = {**native, first: replace(native[first], identity="0" * 64)}
    refused("wrong_instrument_identity", lambda: model(prefix, bad), "MNLM_MODEL_IDENTITY")
    bad = {**native, first: replace(native[first], rows=("different-observation",))}
    refused("misaligned_observation", lambda: model(prefix, bad), "MNLM_MODEL_ROWS")
    binding = {"panel_sha256": result["manifest_sha256"], "atom": atom, "selection": "first_changed_existing_file"}
    publish(a.output / "model-input.json", binding)
    checkpoint = save(model, a.output / "checkpoint", step=0, input_sha256=digest(binding))
    expected = {"logits_sha256": hashlib.sha256(original_out["logits"].numpy().tobytes()).hexdigest(),
                "state_sha256": state_hash(model), "instrument_count": original_out["instrument_count"].tolist(),
                "channel_count": original_out["channel_count"].tolist(), "meaning_asserted": False}
    publish(a.output / "model-before.json", expected)
    run = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--replay", str(a.output),
                          "--output", str(a.output / "model-replay.json")], capture_output=True, timeout=60)
    check("fresh_process_checkpoint_replay", run.returncode == 0
          and json.loads((a.output / "model-replay.json").read_bytes()) == expected)
    after = {p.relative_to(a.snapshot).as_posix(): file_hash(p) for p in a.snapshot.rglob("*") if p.is_file()}
    check("inputs_unchanged_after_model", before == after)
    model_done = time.perf_counter()
    report = {"schema": "code-snapshot-fsv-v1", "checks": checks, "source_snapshot_sha256": a.snapshot_sha256,
              "records": len(cold), "current_files": len(reader.examples), "parsed_files": parsed, "syntax_nodes": nodes,
              "omitted": reader.omitted, "native_rows": {c: p["token_rows"] for c, p in plan["channels"].items()},
              "native_value_bytes": plan["projected_values_bytes"], "ablations_max_abs_logits": ablations,
              "model_parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": checkpoint,
              "admitted": False, "stage4_accepted": False, "trained": False}
    publish(a.output / "report.json", report)
    end = time.perf_counter()
    publish(a.output / "receipt.json", {"wall_seconds": end-ENTRY_STARTED, "scope": "cli_including_imports", "phases": {
        "imports": start-ENTRY_STARTED, "source": source_done-start, "ucr": ucr_done-source_done, "native": panel_done-ucr_done,
        "model": model_done-panel_done, "publication": end-model_done}})
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path)
    for name in ("data-root", "snapshot", "oracle-plan", "input-plan", "prepared", "output"):
        p.add_argument("--" + name, type=Path, required=name == "output")
    p.add_argument("--snapshot-sha256")
    p.add_argument("--oracle-plan-sha256")
    a = p.parse_args()
    try:
        if a.replay:
            replay(a.replay, a.output)
        else:
            r = run(a)
            print(json.dumps({k: len(r[k]) if k == "checks" else r[k] for k in
                              ("checks", "records", "current_files", "syntax_nodes", "native_value_bytes", "admitted")}))
    except (SenseError, OSError, ValueError, KeyError, TypeError) as exc:
        e = exc if isinstance(exc, SenseError) else SenseError("MNLM_SNAPSHOT_FSV", str(exc))
        print(json.dumps(e.envelope()), file=sys.stderr)
        sys.exit(2)
