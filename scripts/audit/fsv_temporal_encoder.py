"""Cold replay of real trained temporal instruments, native outputs and refusal boundaries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from temporal_encoder import FrozenTemporal, TemporalNet, configure, bounded_read
import numpy as np
import torch
from safetensors.torch import load as deserialize, save as serialize
from model.attention import Linear
from model.checkpoint import state_hash
from senses.contract import canonical, digest, require, SenseError
from senses.forecast import put
from senses.forecast_prefix import PrefixInputs, training_targets
from senses.instruments import measure, slot_digest as observation_digest
from senses.pipeline import file_hash
from prefix_history import load as load_history
from temporal_slots import load as load_temporal
from train_temporal import prepare
from slotio import npz_member
from lenscommon import take_lease, release_lease


def read_json(path):
    return json.loads(path.read_bytes())


def cold_metrics(x, y):
    # Independent stable softmax calculation over persisted predictions.
    x = np.asarray(x, np.float64)
    z = x - x.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    picked = logp[np.arange(len(y))[:, None], np.arange(7)[None, :], y]
    return {"n": len(y), "training_accuracy": float(np.mean(x.argmax(-1) == y)),
            "training_log_loss_bits": float(-np.mean(picked) / np.log(2)), "generalisation_or_CMI_claim": False}


def run(a):
    started = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})
        require(ok, "MNLM_TEMPORAL_FSV", name)

    def refused(name, action, code):
        try:
            action()
        except SenseError as exc:
            check(name, exc.code == code, actual_code=exc.code)
        else:
            check(name, False)

    require(file_hash(a.run / "SHA256.json") == a.run_sha256, "MNLM_TEMPORAL_FSV", "unbound training inventory")
    inventory = read_json(a.run / "SHA256.json")
    check("complete_training_file_readback", all(file_hash(a.run / n) == h for n, h in inventory.items()), files=len(inventory))
    report = read_json(a.run / "report.json")
    binding = read_json(a.run / "input.json")
    recipe = read_json(a.run / "recipe.json")
    reader = PrefixInputs(a.prefix, binding["prefix_manifest_sha256"])
    targets = training_targets(a.prefix, reader)
    hm, rows = load_history(a.native, binding["native_manifest_sha256"])
    native = {k: npz_member(a.native / "history.npz", k) for k in ("tokens", "offsets", "absent", "windows")}
    values = []
    try:
        for i, r in enumerate(rows):
            current, past, dependency = reader.read(r["atom"])
            source = measure(current, "history", past)
            lo, hi = native["offsets"][i:i + 2]
            require(np.array_equal(native["tokens"][lo:hi], source["tokens"])
                    and observation_digest(source) == r["slot_digest"] and dependency == r["binding"]
                    and native["windows"][i] == len(past) and bool(native["absent"][i]) == (len(source["tokens"]) == 0),
                    "MNLM_TEMPORAL_FSV", "source-prefix/native boundary differs")
            values.append(source["tokens"])
    finally:
        for v in native.values():
            v._mmap.close()
    check("all_training_inputs_rederived", len(values) == 239 and sum(map(len, values)) == 10465,
          rows=len(values), tokens=sum(map(len, values)), input_digest=hm["value_digest"])
    truth = np.asarray([r["sign"] for r in targets], np.int64) + 1
    present = np.asarray([i for i, v in enumerate(values) if len(v)], np.int64)
    check("label_role_and_order", [r["atom"] for r in rows] == [r["atom"] for r in targets] == binding["predictor_ids"]
          and binding["split"] == "train" and recipe["selection"] == "fixed_final_epoch; no validation or test")
    _, prepared_rows, prepared_values, prepared_truth, prepared_binding = prepare(a.prefix, binding["prefix_manifest_sha256"],
                                                                               a.native, binding["native_manifest_sha256"])
    check("training_entrypoint_binds_real_input_machinery", prepared_rows == rows and prepared_binding == binding
          and np.array_equal(prepared_truth.numpy(), truth) and all(np.array_equal(v, p) for v, p in zip(values, prepared_values)))
    wrong_input = a.out / "wrong-preprocessor"
    shutil.copytree(a.native, wrong_input)
    wrong = copy.deepcopy(hm)
    wrong["contract"]["weights_sha256"] = "0" * 64
    (wrong_input / "manifest.json").write_bytes(canonical(wrong))
    refused("different_history_machinery_even_with_valid_native_values", lambda: prepare(a.prefix, binding["prefix_manifest_sha256"],
            wrong_input, digest(wrong)), "MNLM_TEMPORAL_TRAIN_BINDING")
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    put(a.out / "gpu-before.txt", resident.encode())
    take_lease(12, "cold-temporal-instrument-readback")
    try:
        execution = configure()
        torch.manual_seed(recipe["seed"])
        initial_model, initial_head = TemporalNet().to("cuda"), Linear(32, 21).to("cuda")
        initial_state = {"encoder": state_hash(initial_model), "head": state_hash(initial_head)}
        check("real_and_control_initialization_rederived", report["real"]["before"] == report["shuffled"]["before"] == initial_state)
        observed = {}
        for variant in ("real", "shuffled"):
            root = a.run / variant
            vr = read_json(root / "report.json")
            frozen = FrozenTemporal(root / "instrument", vr["checkpoint_id"])
            head_bytes = bounded_read(root / "instrument" / "head.safetensors")
            check(variant + "_head_bytes_bound", hashlib.sha256(head_bytes).hexdigest() == frozen.manifest["head_sha256"])
            head = Linear(32, 21)
            state = deserialize(head_bytes)
            require(set(state) == set(head.state_dict()) and all(state[n].shape == t.shape and state[n].dtype == t.dtype
                    and bool(torch.isfinite(state[n]).all()) for n, t in head.state_dict().items()),
                    "MNLM_TEMPORAL_FSV", "auxiliary head layout mismatch")
            head.load_state_dict(state, strict=True)
            head.eval().requires_grad_(False).to("cuda")
            after = {"encoder": state_hash(frozen.model), "head": state_hash(head)}
            check(variant + "_frozen_weights_cold_readback", after == vr["after"] and after != initial_state
                  and after["encoder"] != initial_state["encoder"] and vr["updates"] == 280)
            label_map = read_json(root / "training-label-map.json")
            perm = np.random.default_rng(recipe["permutation_seed"]).permutation(present) if variant == "shuffled" else present
            check(variant + "_label_mapping_rederived", label_map == {"predictor_ids": [rows[i]["atom"] for i in present],
                  "label_source_ids": [rows[i]["atom"] for i in perm]})
            with np.load(root / "predictions.npz", allow_pickle=False) as z:
                arrays = {n: z[n] for n in z.files}
            check(variant + "_targets_unchanged", np.array_equal(arrays["present"], present)
                  and np.array_equal(arrays["truth"], truth[present]) and np.array_equal(arrays["fit_truth"], truth[perm]))
            nm, output_rows = load_temporal(root / "native", file_hash(root / "native" / "manifest.json"))
            native = {k: npz_member(root / "native" / "history.npz", k) for k in ("tokens", "offsets", "absent", "windows")}
            final, initial = [], []
            try:
                for i, r in enumerate(rows):
                    current, past, dep = reader.read(r["atom"])
                    slot = frozen.encode_record(current, past)
                    lo, hi = native["offsets"][i:i + 2]
                    saved = output_rows[i]
                    require(np.array_equal(native["tokens"][lo:hi], slot["tokens"])
                            and saved["atom"] == r["atom"] and saved["binding"] == dep
                            and saved["input"] == {k: r[k] for k in ("kind", "reason", "shape", "slot_digest")}
                            and saved["output"] == {k: slot[k] for k in ("kind", "reason", "shape")} | {"slot_digest": observation_digest(slot)}
                            and native["windows"][i] == len(past) and bool(native["absent"][i]) == (len(values[i]) == 0),
                            "MNLM_TEMPORAL_FSV", "learned native values or typed provenance mismatch")
                    if len(values[i]):
                        with torch.inference_mode():
                            final.append(head(torch.from_numpy(slot["tokens"][-1:]).to("cuda"))[0].reshape(7, 3).cpu().numpy())
                            v = torch.from_numpy(values[i]).to("cuda")[None]
                            h = initial_model(v, torch.tensor([len(values[i])], device="cuda"))[0, -1:]
                            initial.append(initial_head(h)[0].reshape(7, 3).cpu().numpy())
                check(variant + "_all_239_native_slots_cold_replay", nm["value_digest"] == vr["native_value_digest"],
                      tokens=10465, absent=15, value_digest=nm["value_digest"])
            finally:
                for v in native.values():
                    v._mmap.close()
            check(variant + "_initial_and_trained_predictions_exact", np.array_equal(initial, arrays["initial"])
                  and np.array_equal(final, arrays["final"]))
            for key, x, y in (("initial", arrays["initial"], truth[present]),
                              ("final_against_real_training_labels", arrays["final"], truth[present]),
                              ("final_against_fit_labels", arrays["final"], truth[perm])):
                m = cold_metrics(x, y)
                check(variant + "_" + key + "_recomputed", m["training_accuracy"] == vr[key]["training_accuracy"]
                      and abs(m["training_log_loss_bits"] - vr[key]["training_log_loss_bits"]) < 1e-12)
            counts = np.asarray([np.bincount(truth[present, j], minlength=3) for j in range(7)]) + 1.
            baseline = cold_metrics(np.broadcast_to(np.log(counts), (len(present), 7, 3)), truth[present])
            check(variant + "_training_marginal_recomputed", baseline["training_accuracy"] == vr["training_marginal"]["training_accuracy"]
                  and abs(baseline["training_log_loss_bits"] - vr["training_marginal"]["training_log_loss_bits"]) < 1e-12)
            check(variant + "_no_admission", not nm["production_admission"] and not nm["stage4_accepted"]
                  and not frozen.manifest["production_admission"] and frozen.contract["channel"] == "history")
            observed[variant] = {"checkpoint_id": vr["checkpoint_id"], "lens_id": frozen.contract["lens_id"],
                                 "native_digest": nm["value_digest"], "state": after,
                                 "training": vr["final_against_real_training_labels"], "marginal": baseline}
        # Inference fixture contains only encoder, manifest and commit marker.
        isolated = a.out / "encoder-only"
        isolated.mkdir()
        source = a.run / "real" / "instrument"
        for n in ("manifest.json", "COMMITTED", "encoder.safetensors"):
            shutil.copyfile(source / n, isolated / n)
        frozen = FrozenTemporal(isolated, report["real"]["checkpoint_id"])
        example = next(v for v in values if len(v) == 56)
        expected = frozen.encode(example)
        check("encoder_runs_without_targets_or_training_head", set(p.name for p in isolated.iterdir()) ==
              {"manifest.json", "COMMITTED", "encoder.safetensors"} and expected.shape == (56, 32))
        check("frozen_repeat_exact", np.array_equal(expected, frozen.encode(example)))
        check("absence_stays_absent", frozen.encode(np.empty((0, 128), np.float32)).shape == (0, 32))
        altered = example.copy()
        altered[28:] = altered[28:][::-1] + np.float32(.125)
        changed = frozen.encode(altered)
        check("causal_prefix_invariant_to_future_tokens", np.array_equal(expected[:28], changed[:28])
              and not np.array_equal(expected[28:], changed[28:]))
        check("temporal_order_matters", not np.array_equal(expected, frozen.encode(example[::-1].copy())[::-1]))
        for name, bad in (("dtype", example.astype(np.float64)), ("shape", example[:, :127]),
                          ("nonfinite", np.full_like(example, np.nan)), ("overflow", np.zeros((257, 128), np.float32))):
            refused("invalid_" + name, lambda b=bad: frozen.encode(b), "MNLM_TEMPORAL_INPUT")
        frozen.model.attention.train()
        refused("child_training_mode_drift", lambda: frozen.encode(example), "MNLM_TEMPORAL_RUNTIME_DRIFT")
        frozen.model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        refused("precision_runtime_drift", lambda: frozen.encode(example), "MNLM_TEMPORAL_RUNTIME_DRIFT")
        configure()
        param = next(frozen.model.parameters())
        param.requires_grad_(True)
        refused("unfrozen_parameter", lambda: frozen.encode(example), "MNLM_TEMPORAL_RUNTIME_DRIFT")
        param.requires_grad_(False)
        with torch.no_grad():
            param.add_(1.)
        refused("parameter_update_drift", lambda: frozen.encode(example), "MNLM_TEMPORAL_RUNTIME_DRIFT")
        bad = a.out / "corrupt-instrument"
        shutil.copytree(isolated, bad)
        weights = bytearray((bad / "encoder.safetensors").read_bytes())
        weights[-1] ^= 1
        (bad / "encoder.safetensors").write_bytes(weights)
        refused("changed_checkpoint_bytes", lambda: FrozenTemporal(bad, report["real"]["checkpoint_id"]), "MNLM_TEMPORAL_CHECKPOINT")
        shutil.copyfile(source / "encoder.safetensors", bad / "encoder.safetensors")
        manifest = read_json(bad / "manifest.json")
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["execution_code_sha256"]["lens/temporal_encoder.py"] = "0" * 64
        (bad / "manifest.json").write_bytes(canonical(changed_manifest))
        (bad / "COMMITTED").write_text(digest(changed_manifest))
        refused("code_drift_even_with_resealed_manifest", lambda: FrozenTemporal(bad, digest(changed_manifest)), "MNLM_TEMPORAL_CODE_DRIFT")
        (bad / "manifest.json").write_bytes(canonical(manifest))
        (bad / "COMMITTED").unlink()
        refused("missing_commit_marker", lambda: FrozenTemporal(bad, digest(manifest)), "MNLM_TEMPORAL_CHECKPOINT")
        corrupt_native = a.out / "corrupt-native"
        shutil.copytree(a.run / "real" / "native", corrupt_native)
        with np.load(corrupt_native / "history.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["tokens"][0, 0] += np.float32(1)
        np.savez(corrupt_native / "history.npz", **arrays)
        refused("altered_native_values", lambda: load_temporal(corrupt_native, file_hash(corrupt_native / "manifest.json")),
                "MNLM_TEMPORAL_SLOT_DIGEST")
        check("original_run_unchanged", all(file_hash(a.run / n) == h for n, h in inventory.items()))
        result = {"schema": "temporal-candidate-fsv-v1", "acceptance": "author-checked only", "checks": checks,
                  "run_sha256": a.run_sha256, "input": binding, "runtime": execution, "observed": observed,
                  "meaning_asserted": False, "stage3_admitted": False, "stage4_accepted": False,
                  "new_outcome_evaluation": False, "wall_seconds": time.perf_counter() - started}
        put(a.out / "report.json", canonical(result))
        put(a.out / "SHA256.json", canonical({p.relative_to(a.out).as_posix(): file_hash(p) for p in sorted(a.out.rglob("*")) if p.is_file()}))
        print(json.dumps({"checks": len(checks), "all_pass": all(c["pass"] for c in checks),
                          "report_sha256": file_hash(a.out / "report.json"), "wall_seconds": result["wall_seconds"]}), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("prefix", "native", "run", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    ap.add_argument("--run-sha256", required=True)
    run(ap.parse_args())
