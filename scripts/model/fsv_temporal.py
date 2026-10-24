"""Real trained temporal instrument -> extensible decoder -> cold checkpoint replay."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from temporal_encoder import configure
import numpy as np
import torch
from senses.contract import canonical, digest, require, SenseError
from senses.forecast import put
from senses.forecast_prefix import PrefixInputs
from senses.instruments import measure, contracts
from senses.pipeline import file_hash
from model.contracts import ModelConfig, Instrument
from model.native import from_arrays
from model.temporal_input import read_temporal
from model.prefix_bank import read_bank, read_windowed
from senses.trajectory_population import read_pinned
from model.network import SensoryLanguageModel, training_step
from model.checkpoint import save, load, state_hash
from model.train_sensory import byte_targets
from model.fsv_sensory import native_hash, tensor_hash
from production.native_rows import joined
from lenscommon import take_lease, release_lease

CHANNELS = ("surface", "fields", "structure", "provenance", "history")


def inputs(prefix, temporal, config):
    reader = PrefixInputs(prefix, file_hash(prefix / "manifest.json"))
    ids = [k for k, e in reader.examples.items() if e["past"]][:8]
    require(len(ids) == 8, "MNLM_MODEL_SCOPE", "fixed eight-row training proof required")
    observed = [reader.read(k) for k in ids]
    catalog = {c["channel"]: c for c in contracts()}
    native, specs = {}, []
    for channel in CHANNELS:
        c = catalog[channel]
        spec = Instrument.from_slot(c["name"], {"contract": c, "seal": {"kind": "multi", "dims": 128}})
        rows = []
        for current, past, _ in observed:
            slot = measure(current, channel, past)
            rows.append({"tokens": slot["tokens"], "offsets": np.asarray([0, len(slot["tokens"])], np.int64),
                         "absent": np.asarray([slot["kind"] == "Absent"], bool), "windows": np.asarray([len(past)], np.int64)})
        native[spec.name] = from_arrays(spec, joined(rows, "multi", 128), ids, config).to("cuda")
        specs.append(spec)
    temporal_spec, temporal_input, receipt = read_temporal(temporal, file_hash(temporal / "manifest.json"), ids, config)
    text = [{"atom": k, "text": " ".join(i["text"] for i in cur["channels"]["surface"]["items"])}
            for k, (cur, _, _) in zip(ids, observed)]
    receipt = {"prefix_sha256": reader.identity, "temporal": receipt, "row_ids": ids,
               "base_native_sha256": native_hash(native), "text_sha256": digest(text), "config": asdict(config)}
    return tuple(specs), native, temporal_spec, temporal_input.to("cuda"), text, receipt


def selected_temporal(a, row_ids, config):
    if a.panel is None:
        return read_temporal(a.temporal, file_hash(a.temporal / "manifest.json"), row_ids, config)
    return read_windowed(a.temporal, a.temporal_sha256, a.panel_sha256, row_ids, config)


def prepare(a, config):
    if a.panel is None:
        return inputs(a.prefix, a.temporal, config)
    require(all((a.panel_sha256, a.prefix_sha256, a.temporal_sha256, a.plan, a.plan_sha256)),
            "MNLM_MODEL_SCOPE", "pin all source/publication/plan identities")
    from senses.trajectory_prefix import PrefixInputs as TrajectoryInputs
    reader = TrajectoryInputs(a.prefix, a.prefix_sha256)
    plan = read_pinned(a.plan, a.plan_sha256)
    require(plan["schema"] == "windowed-model-preflight-v1" and plan["admitted"] is False
            and plan["rows"] == 8 and plan["fixed_steps"] == 5 and plan["prefix_tokens"] == 8,
            "MNLM_MODEL_SCOPE", "fixed eight-row mechanical proof required")
    ids = plan["row_ids"]
    observed = [reader.read(k) for k in ids]
    specs, native, base_receipt = read_bank(a.panel, a.panel_sha256, ids, CHANNELS, config)
    temporal_spec, temporal_input, receipt = selected_temporal(a, ids, config)
    from model.contracts import parameter_count
    counts = {s.channel: int((~native[s.name].padding).sum()) for s in specs}
    maxima = {s.channel: native[s.name].values.shape[1] for s in specs}
    require(counts == plan["base_native_tokens"] and maxima == plan["maxima"]
            and int((~temporal_input.padding).sum()) == plan["learned_native_tokens"]
            and sum(x.values.numel() for x in native.values()) + temporal_input.values.numel() == plan["padded_native_elements"]
            and parameter_count(config, (*specs, temporal_spec)) == plan["expanded_parameters"]
            and 8 * config.heads * 8 * max(maxima.values()) == plan["largest_attention_elements"],
            "MNLM_MODEL_COST_DRIFT", "cold selected native input differs from preflight")
    # Bind current/past identities to the sealed source, not just matching ID strings.
    from prefix_panel import load as load_panel
    panel_manifest, panel_rows = load_panel(a.panel, a.panel_sha256)
    index = {r["atom"]: r for r in panel_rows}
    require(panel_manifest["input_manifest_sha256"] == reader.identity
            and all(index[k]["binding"] == observation[2] for k, observation in zip(ids, observed)),
            "MNLM_MODEL_INPUT_BINDING", "model/native prefix scopes differ")
    text = [{"atom": k, "text": " ".join(i["text"] for i in cur["channels"]["surface"]["items"])}
            for k, (cur, _, _) in zip(ids, observed)]
    native = {name: x.to("cuda") for name, x in native.items()}
    receipt = {"prefix_sha256": reader.identity, "temporal": receipt, "base_receipt": base_receipt,
               "row_ids": ids, "base_native_sha256": native_hash(native), "text_sha256": digest(text),
               "config": asdict(config), "plan_sha256": a.plan_sha256, "temporal_token_rows": plan["learned_native_tokens"]}
    return specs, native, temporal_spec, temporal_input.to("cuda"), text, receipt


def run(a):
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    take_lease(25, "temporal-decoder-integration-replay" if a.replay else "temporal-decoder-integration")
    try:
        configure()
        config = ModelConfig()
        specs, native, new_spec, new_input, rows, receipt = prepare(a, config)
        prefix, targets = byte_targets(rows, 8, "cuda")
        expanded = {**native, new_spec.name: new_input}
        if a.replay:
            report = json.loads((a.out / "report.json").read_bytes())
            require(digest(receipt) == report["input_sha256"] and native_hash(expanded) == report["native_sha256"],
                    "MNLM_MODEL_REPLAY", "cold source/slot/model input changed")
            cold, m = load(a.out / "checkpoint", "cuda")
            old = json.loads((a.out / "original-parameters.json").read_bytes())
            require(all(tensor_hash(cold.state_dict()[n]) == h for n, h in old.items()),
                    "MNLM_MODEL_REPLAY", "old bank parameters changed during expansion/training")
            with torch.inference_mode():
                actual = cold(prefix, expanded)["logits"].cpu().numpy()
            require(np.array_equal(actual, np.load(a.out / "logits.npy", allow_pickle=False))
                    and digest(m) == report["checkpoint_sha256"], "MNLM_MODEL_REPLAY", "cold decoder changed outputs")
            put(a.out / "cold-replay.json", canonical({"acceptance": "author-checked only", "exact": True,
                "input_sha256": digest(receipt), "checkpoint_sha256": digest(m), "logits_sha256": file_hash(a.out / "logits.npy"),
                "meaning_asserted": False, "stage4_accepted": False}))
            print(json.dumps({"cold_replay_exact": True, "checkpoint_sha256": digest(m)}))
            return
        a.out.mkdir(parents=True, exist_ok=False)
        put(a.out / "gpu-before.txt", resident.encode())
        put(a.out / "input.json", canonical(receipt))
        checks = []

        def check(name, ok):
            checks.append({"name": name, "pass": bool(ok)})
            require(ok, "MNLM_MODEL_FSV", name)

        def refused(name, action, code):
            try:
                action()
            except SenseError as e:
                check(name, e.code == code)
            else:
                check(name, False)

        check("local_training_lineage_retained", json.loads(new_spec.definition_json)["training_lineage"]["record"]["split"] == "train")
        original_hash = native_hash(expanded)
        torch.manual_seed(25)
        model = SensoryLanguageModel(config, specs).to("cuda")
        with torch.inference_mode():
            before = model(prefix, native)
        original_state = {n: tensor_hash(p) for n, p in model.state_dict().items()}
        put(a.out / "original-parameters.json", canonical(original_state))
        model.extend(new_spec)
        with torch.inference_mode():
            after = model(prefix, expanded)
            memories, _ = model.memories(expanded)
        check("new_instrument_preserves_initial_decoder_output", torch.equal(before["logits"], after["logits"]))
        check("six_instruments_still_five_possible_channels", torch.equal(before["channel_count"], after["channel_count"])
              and torch.equal(before["instrument_count"] + 1, after["instrument_count"]) and int(after["channel_count"].max()) == 5)
        check("native_temporal_rows_survive_into_attention", memories[new_spec.name].shape[:2] == new_input.values.shape[:2]
              and int((~new_input.padding).sum()) == receipt.get("temporal_token_rows", 294))
        refused("missing_temporal_slot", lambda: model(prefix, native), "MNLM_MODEL_ROSTER")
        bad = {**expanded, new_spec.name: replace(new_input, rows=tuple(reversed(new_input.rows)))}
        refused("misaligned_temporal_rows", lambda: model(prefix, bad), "MNLM_MODEL_ROWS")
        refused("duplicate_temporal_selection", lambda: selected_temporal(a, [rows[0]["atom"]] * 2, config), "MNLM_MODEL_ROWS")
        refused("missing_temporal_selection", lambda: selected_temporal(a, ["0" * 64], config), "MNLM_MODEL_ROWS")
        if bool((after["channel_count"] < 5).any()):
            refused("meaning_without_five_present_channels", lambda: model.assert_meaning(expanded), "MNLM_MODEL_MEANING_COVERAGE")
        if a.panel is not None:
            refused("native_budget_no_truncation", lambda: selected_temporal(a, receipt["row_ids"], replace(config, max_native_tokens=256)),
                    "MNLM_MODEL_INPUT_BUDGET")
            refused("different_source_binding", lambda: read_windowed(a.temporal, a.temporal_sha256, "0" * 64,
                    receipt["row_ids"], config), "MNLM_MODEL_INPUT_BINDING")
        # Rows that do have five channels still cannot bypass scientific admission.
        selected = torch.nonzero(after["channel_count"] == 5).flatten()
        complete = {n: replace(x, rows=tuple(x.rows[int(i)] for i in selected), values=x.values[selected],
                              padding=x.padding[selected], absent=x.absent[selected]) for n, x in expanded.items()}
        refused("five_channels_without_admission", lambda: model.assert_meaning(complete), "MNLM_MODEL_UNCOMMISSIONED")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, foreach=False, fused=False)
        state_before = state_hash(model)
        losses = [training_step(model, optimizer, prefix, targets, expanded) for _ in range(5)]
        check("new_branch_trains", state_hash(model) != state_before)
        check("old_model_parameters_retained", all(tensor_hash(model.state_dict()[n]) == h for n, h in original_state.items()))
        check("frozen_native_values_unchanged", native_hash(expanded) == original_hash)
        checkpoint = save(model, a.out / "checkpoint", step=5, input_sha256=digest(receipt))
        with torch.inference_mode():
            final = model(prefix, expanded)["logits"].cpu().numpy()
        gains = [b.branches[new_spec.name].gain.detach().clone() for b in model.blocks]
        try:
            with torch.no_grad():
                for b in model.blocks:
                    b.branches[new_spec.name].gain.zero_()
            with torch.inference_mode():
                disabled = model(prefix, expanded)["logits"].cpu().numpy()
            check("trained_new_instrument_affects_logits", not np.array_equal(final, disabled))
            check("disabled_new_branch_restores_old_function", np.array_equal(disabled, before["logits"].cpu().numpy()))
        finally:
            with torch.no_grad():
                for b, gain in zip(model.blocks, gains):
                    b.branches[new_spec.name].gain.copy_(gain)
        with (a.out / "logits.npy").open("xb") as f:
            np.save(f, final, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        cold, m = load(a.out / "checkpoint", "cuda")
        with torch.inference_mode():
            check("cold_model_output_exact", np.array_equal(final, cold(prefix, expanded)["logits"].cpu().numpy()))
        result = {"acceptance": "author-checked only", "input_sha256": digest(receipt), "checkpoint_sha256": checkpoint,
                  "native_sha256": original_hash, "checks": checks, "losses": losses, "rows": len(rows),
                  "channels_per_row": after["channel_count"].tolist(), "instruments_per_row": after["instrument_count"].tolist(),
                  "meaning_asserted": False, "stage4_accepted": False, "state_before": state_before, "state_after": m["state_sha256"]}
        put(a.out / "report.json", canonical(result))
        print(json.dumps({"checks": len(checks), "checkpoint_sha256": checkpoint, "report_sha256": file_hash(a.out / "report.json")}))
    finally:
        release_lease()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("prefix", "temporal", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    for n in ("panel", "plan"):
        ap.add_argument("--" + n, type=Path)
    for n in ("panel-sha256", "prefix-sha256", "temporal-sha256", "plan-sha256"):
        ap.add_argument("--" + n)
    ap.add_argument("--replay", action="store_true")
    run(ap.parse_args())
