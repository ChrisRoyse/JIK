"""Held-out recorded-observation probe of separate frozen native inputs.

Uses the engineering cross-attention consumer, not a commissioned meaning model.
Five fixed epochs, validation selection, then one test/ablation report. Never MI.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
import torch
from senses.contract import canonical, digest, require
from senses.forecast import load_bundle, put, SPLITS
from senses.pipeline import file_hash
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.network import SensoryLanguageModel, training_step
from model.publication_input import read_publication
from model.checkpoint import save, load, state_hash

CONFIG = ModelConfig(vocabulary=3, max_prefix=7)
RECIPE = {"schema": "recorded-forecast-probe-v1", "purpose": "diagnostic_predictive_performance_not_CMI",
          "seed": 42, "epochs": 5, "learning_rate": 0.001, "model": asdict(CONFIG), "device": "cuda",
          "selection": "minimum_validation_mean_cross_entropy; earliest_epoch_tie",
          "query": "seven_zero_tokens; ordinal_positions_only; no_target_teacher_forcing",
          "label": "recorded_difference_sign_plus_one", "cpu_native_budget_bytes": 384 << 20,
          "permutation": "seed43_permutation_of_whole_training_label_vectors"}


def metrics(prob, truth):
    require(prob.shape == (*truth.shape, 3) and bool(np.isfinite(prob).all()) and bool((prob > 0).all())
            and bool(np.allclose(prob.sum(-1), 1, atol=1e-6)), "MNLM_FORECAST_PROBE_VALUES", "invalid probabilities")
    correct = prob.argmax(-1) == truth
    losses = -np.log2(np.take_along_axis(prob, truth[..., None], -1)[..., 0])
    return {"rows": len(truth), "mean_log_loss_bits": float(losses.mean()), "accuracy": float(correct.mean()),
            "coordinate_accuracy": correct.mean(0).tolist(), "coordinate_log_loss_bits": losses.mean(0).tolist(),
            "sampling_intervals": "not asserted; collection exchangeability/IID unestablished"}


def native_hash(batches):
    h = hashlib.sha256()
    for split in SPLITS:
        for native, _, rows in batches[split]:
            h.update(canonical(rows))
            for name, x in sorted(native.items()):
                h.update(canonical([name, x.identity, list(x.rows)]))
                for v in (x.values, x.padding, x.absent, x.indices):
                    if v is not None:
                        h.update(memoryview(v.numpy()).cast("B"))
    return h.hexdigest()


def prepare(bundle, native_root):
    m, data = load_bundle(bundle)
    consumer = json.loads((native_root / "consumer.json").read_bytes())
    require(consumer["bundle_manifest_sha256"] == file_hash(bundle / "manifest.json")
            and consumer["stage3_admitted"] is False and consumer["stage4_accepted"] is False
            and consumer["plan_sha256"] == file_hash(native_root / "plan.json"),
            "MNLM_FORECAST_PROBE_BINDING", "native scope differs from frozen forecast")
    label_map = {r["atom"]: r for r in data["labels.jsonl"]}
    specs, batches, seen, receipts, total = None, {}, set(), [], 0
    for split in SPLITS:
        batches[split] = []
        for group in consumer["groups"][split]:
            current_specs, native, rows, receipt = read_publication(native_root / "native/publication", [group], CONFIG)
            require(specs is None or specs == current_specs, "MNLM_FORECAST_PROBE_ROSTER", "instrument drift")
            specs = current_specs
            ids = [r["atom"] for r in rows]
            require(len(ids) <= 8 and not set(ids) & seen and all(a in label_map and label_map[a]["split"] == split for a in ids),
                    "MNLM_FORECAST_PROBE_SPLIT", "row duplicates, wrong split or unrecognized predictors")
            seen.update(ids)
            target = torch.tensor([[v + 1 for v in label_map[a]["sign"]] for a in ids], dtype=torch.long)
            for x in native.values():
                total += sum(v.numel() * v.element_size() for v in (x.values, x.padding, x.absent, x.indices) if v is not None)
            require(total <= RECIPE["cpu_native_budget_bytes"], "MNLM_FORECAST_PROBE_BUDGET", "bounded native cache exceeded")
            batches[split].append((native, target, ids))
            receipts.append(receipt)
    require(seen == set(label_map) and len(specs) == 5 and m["rows"] == 416,
            "MNLM_FORECAST_PROBE_SCOPE", "incomplete exact forecast scope")
    return specs, batches, receipts, total


def predict(model, batches, ablate=None):
    probabilities, truths, row_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for native, target, ids in batches:
            native = {n: x.to("cuda") for n, x in native.items()}
            if ablate is not None:
                x = native[ablate]
                native[ablate] = replace(x, values=torch.zeros_like(x.values), padding=torch.ones_like(x.padding),
                                        absent=torch.ones_like(x.absent),
                                        indices=None if x.indices is None else torch.zeros_like(x.indices))
            prefix = torch.zeros((len(ids), 7), dtype=torch.long, device="cuda")
            probabilities.append(model(prefix, native)["logits"].double().softmax(-1).cpu().numpy())
            truths.append(target.numpy())
            row_ids.extend(ids)
    return np.concatenate(probabilities), np.concatenate(truths), row_ids


def train_variant(specs, batches, out, input_sha, permuted):
    out.mkdir()
    torch.manual_seed(RECIPE["seed"])
    model = SensoryLanguageModel(CONFIG, specs).to("cuda")
    initial = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["learning_rate"])
    originals = torch.cat([y for _, y, _ in batches["train"]])
    ids = [i for _, _, rows in batches["train"] for i in rows]
    order = np.random.default_rng(43).permutation(len(ids)) if permuted else np.arange(len(ids))
    shuffled = {a: originals[int(j)] for a, j in zip(ids, order)}
    put(out / "training-label-map.json", canonical({"predictor_ids": ids, "label_source_ids": [ids[int(j)] for j in order]}))
    history, best = [], None
    for epoch in range(1, RECIPE["epochs"] + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for native, _, rows in batches["train"]:
            x = {key: v.to("cuda") for key, v in native.items()}
            target = torch.stack([shuffled[a] for a in rows]).to("cuda")
            prefix = torch.zeros_like(target)  # no outcome tokens used as inputs
            loss_sum += training_step(model, optimizer, prefix, target, x) * len(rows)
            n += len(rows)
        probability, truth, _ = predict(model, batches["validation"])
        score = metrics(probability, truth)
        checkpoint = out / f"epoch-{epoch}"
        checkpoint_id = save(model, checkpoint, step=epoch * len(batches["train"]), input_sha256=input_sha)
        entry = {"epoch": epoch, "training_nll_nats": loss_sum / n, "validation": score, "checkpoint_sha256": checkpoint_id}
        history.append(entry)
        with (out / f"validation-{epoch}.npz").open("xb") as f:
            np.savez(f, probabilities=probability, truth=truth)
        if best is None or score["mean_log_loss_bits"] < best["validation"]["mean_log_loss_bits"]:
            best = entry
        print(json.dumps({"variant": out.name, **entry}), flush=True)
    put(out / "selection.json", canonical({"initial_state": initial, "history": history, "best_epoch": best["epoch"],
                                            "criterion": RECIPE["selection"], "test_used_for_selection": False}))
    frozen, manifest = load(out / f"epoch-{best['epoch']}", "cuda")
    p, y, test_ids = predict(frozen, batches["test"])
    with (out / "test.npz").open("xb") as f:
        np.savez(f, probabilities=p, truth=y)
    put(out / "test-ids.json", canonical(test_ids))
    again, _ = load(out / f"epoch-{best['epoch']}", "cuda")
    replay, replay_y, replay_ids = predict(again, batches["test"])
    require(np.array_equal(p, replay) and np.array_equal(y, replay_y) and test_ids == replay_ids,
            "MNLM_FORECAST_PROBE_REPLAY", "cold checkpoint replay changed predictions")
    result = {"best_epoch": best["epoch"], "test": metrics(p, y), "cold_replay_exact": True,
              "state_changed": initial != manifest["state_sha256"], "parameters": sum(p.numel() for p in frozen.parameters())}
    if not permuted:
        result["ablations"] = {}
        for spec in specs:
            p, y, _ = predict(frozen, batches["test"], ablate=spec.name)
            with (out / ("ablate-" + spec.name + ".npz")).open("xb") as f:
                np.savez(f, probabilities=p, truth=y)
            result["ablations"][spec.name] = metrics(p, y)
        require(state_hash(frozen) == manifest["state_sha256"], "MNLM_FORECAST_PROBE_MUTATION", "evaluation mutated model")
    return result


def run(args):
    start = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    put(out / "recipe.json", canonical(RECIPE))  # fixed before training or test metrics
    specs, batches, receipts, retained = prepare(args.bundle, args.native)
    put(out / "input-receipts.json", canonical(receipts))
    before = native_hash(batches)
    input_sha = digest([receipts, file_hash(args.bundle / "manifest.json"), RECIPE])
    prepare_done = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi"], text=True)
    put(out / "gpu-before.txt", resident.encode())
    take_lease(12, "recorded-observation-diagnostic-probe")
    try:
        require(torch.cuda.is_available(), "MNLM_FORECAST_PROBE_CUDA", "declared CUDA runtime required; no fallback")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        real = train_variant(specs, batches, out / "real", input_sha, False)
        real_done = time.perf_counter()
        null = train_variant(specs, batches, out / "permuted", input_sha, True)
        null_done = time.perf_counter()
        train_y = torch.cat([y for _, y, _ in batches["train"]]).numpy()
        test_y = torch.cat([y for _, y, _ in batches["test"]]).numpy()
        marginal = np.stack([(np.bincount(train_y[:, j], minlength=3) + 1) / (len(train_y) + 3) for j in range(7)])
        baseline = np.broadcast_to(marginal, (len(test_y), 7, 3)).copy()
        with (out / "baseline.npz").open("xb") as f:
            np.savez(f, probabilities=baseline, truth=test_y, training_truth=train_y)
        require(native_hash(batches) == before, "MNLM_FORECAST_PROBE_MUTATION", "frozen native inputs changed")
        report = {"schema": "recorded-forecast-probe-result-v1", "acceptance": "author-checked only", "recipe": RECIPE,
                  "input_sha256": input_sha, "native_before_after_sha256": before, "cpu_native_bytes": retained,
                  "real": real, "permuted": null, "baseline": metrics(baseline, test_y),
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "native_inputs_unchanged": True,
                  "stage3_admitted": False, "stage4_accepted": False, "meaning_asserted": False,
                  "conditional_information_measured": False, "independent_acceptance": False,
                  "cost_seconds": {"prepare": prepare_done - start, "real": real_done - prepare_done,
                                   "permuted": null_done - real_done, "final_readback": time.perf_counter() - null_done}}
        report["wall_seconds"] = time.perf_counter() - start
        put(out / "report.json", canonical(report))
        put(out / "SHA256.json", canonical({str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
            for p in out.rglob("*") if p.is_file() and p.name != "SHA256.json"}))
        print(json.dumps({"real": real["test"], "permuted": null["test"], "baseline": report["baseline"],
                          "wall_seconds": report["wall_seconds"]}), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "native", "out"):
        ap.add_argument("--" + key, required=True, type=Path)
    run(ap.parse_args())
