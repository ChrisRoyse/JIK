"""Create frozen temporal candidates from sealed TRAIN-ONLY native history.

Two fixed recipes (real/shuffled targets). No validation/test access, early
stopping, scientific information estimate or admission. O(E*B*(T*H^2+T^2*H)).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from senses.contract import canonical, digest, require
from senses.forecast import put
from senses.forecast_prefix import PrefixInputs, training_targets
from senses.instruments import contracts
from senses.pipeline import file_hash
from temporal_encoder import CONFIG, TemporalNet, FrozenTemporal, configure, freeze
from model.attention import Linear
from model.checkpoint import state_hash
from prefix_history import load as load_history
from slotio import npz_member
from temporal_slots import publish as native_outputs
from lenscommon import take_lease, release_lease

RECIPE = {"schema": "train-temporal-candidate-v1", "purpose": "train_only_instrument_construction",
          "seed": 42, "permutation_seed": 43, "epochs": 20, "batch": 16, "learning_rate": 0.001,
          "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "config": asdict(CONFIG),
          "outputs": 7, "classes": 3, "selection": "fixed_final_epoch; no validation or test",
          "scope": "one recorded temporal-realisation family", "production_admission": False}


def prepare(prefix, prefix_id, native, native_id):
    reader = PrefixInputs(prefix, prefix_id)
    labels = training_targets(prefix, reader)
    m, rows = load_history(native, native_id)
    upstream = next(c for c in contracts() if c["channel"] == "history")
    # CPU runtime identity is retained by the source manifest; the frozen
    # preprocessing algorithm, channel and native type must match this encoder.
    require(all(m["contract"].get(k) == upstream[k] for k in
                ("name", "channel", "kind", "shape", "dtype", "norm", "contract", "weights_sha256")),
            "MNLM_TEMPORAL_TRAIN_BINDING", "history preprocessing machinery differs; recommission inputs")
    require(m["input_manifest_sha256"] == prefix_id and m["reader_code_sha256"] == reader.reader_code_sha256
            and [r["atom"] for r in rows] == [r["atom"] for r in labels],
            "MNLM_TEMPORAL_TRAIN_BINDING", "native/label/reader identities differ")
    mapped = {k: npz_member(native / "history.npz", k) for k in ("tokens", "offsets", "absent")}
    try:
        values = [np.array(mapped["tokens"][a:b], copy=True) for a, b in zip(mapped["offsets"][:-1], mapped["offsets"][1:])]
        require(len(values) == 239 and sum(bool(len(v)) for v in values) == 224
                and sum(len(v) for v in values) == 10465 and max(len(v) for v in values) == 56,
                "MNLM_TEMPORAL_TRAIN_SCOPE", "changed observed scope; recommission cost before fitting")
        require(all(v.dtype == np.float32 and v.shape[1] == CONFIG.dims for v in values)
                and all(bool(a) == (len(v) == 0) for a, v in zip(mapped["absent"], values)),
                "MNLM_TEMPORAL_TRAIN_INPUT", "native dimension/absence mismatch")
    finally:
        for a in mapped.values():
            a._mmap.close()
    target = np.asarray([r["sign"] for r in labels])
    require(target.shape == (239, 7) and target.dtype.kind in "iu" and np.isin(target, [-1, 0, 1]).all(),
            "MNLM_TEMPORAL_TARGET", "seven ternary recorded outcomes required")
    return reader, rows, values, torch.tensor(target + 1, dtype=torch.long), {
        "prefix_manifest_sha256": prefix_id, "native_manifest_sha256": native_id,
        "native_value_digest": m["value_digest"], "targets_sha256": reader.manifest["files"]["training-targets.jsonl"]["sha256"],
        "predictor_ids": [r["atom"] for r in rows], "split": "train", "data_scope": reader.manifest["source_sha256"]}


def batch(values, indices):
    lengths = [len(values[i]) for i in indices]
    require(lengths and min(lengths) > 0, "MNLM_TEMPORAL_ABSENT", "absent rows never enter a learned forward")
    padded = np.zeros((len(indices), max(lengths), CONFIG.dims), np.float32)
    for row, i in enumerate(indices):
        padded[row, :lengths[row]] = values[i]
    return torch.from_numpy(padded).to("cuda"), torch.tensor(lengths, dtype=torch.long, device="cuda")


def logits(model, head, values, indices):
    x, lengths = batch(values, indices)
    h = model(x, lengths)
    return head(h[torch.arange(len(indices), device="cuda"), lengths - 1]).reshape(-1, 7, 3)


def predictions(model, head, values, present):
    model.eval()
    head.eval()
    outputs = []
    with torch.inference_mode():
        # Same one-observation arithmetic as the frozen instrument.
        for i in present:
            outputs.append(logits(model, head, values, [int(i)])[0].cpu().numpy())
    return np.asarray(outputs, np.float32)


def metrics(logit, labels):
    x = np.asarray(logit, np.float64)
    logprob = x - np.logaddexp.reduce(x, axis=-1, keepdims=True)
    losses = -np.take_along_axis(logprob, labels[..., None], axis=-1)[..., 0] / np.log(2.)
    return {"n": len(labels), "training_log_loss_bits": float(losses.mean()),
            "training_accuracy": float((x.argmax(-1) == labels).mean()), "generalisation_or_CMI_claim": False}


def train(values, target, rows, binding, out, permuted):
    out.mkdir()
    present = np.asarray([i for i, v in enumerate(values) if len(v)], np.int64)
    torch.manual_seed(RECIPE["seed"])
    model, head = TemporalNet().to("cuda"), Linear(CONFIG.width, 21).to("cuda")
    require(sum(p.numel() for p in [*model.parameters(), *head.parameters()]) == 17589,
            "MNLM_TEMPORAL_COST", "training parameter count differs from preflight")
    before = {"encoder": state_hash(model), "head": state_hash(head)}
    initial = predictions(model, head, values, present)
    permutation = np.random.default_rng(RECIPE["permutation_seed"]).permutation(present) if permuted else present.copy()
    targets = target.clone()
    targets[present] = target[permutation]
    label_map = {"predictor_ids": [rows[i]["atom"] for i in present], "label_source_ids": [rows[i]["atom"] for i in permutation]}
    put(out / "training-label-map.json", canonical(label_map))
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=RECIPE["learning_rate"],
        betas=tuple(RECIPE["betas"]), eps=RECIPE["epsilon"], weight_decay=RECIPE["weight_decay"], foreach=False, fused=False)
    rng = np.random.default_rng(RECIPE["seed"])
    losses, updates = [], 0
    started = time.perf_counter()
    for epoch in range(RECIPE["epochs"]):
        order = rng.permutation(present)
        model.train()
        head.train()
        total = 0.
        for at in range(0, len(order), RECIPE["batch"]):
            ids = order[at:at + RECIPE["batch"]]
            optimizer.zero_grad(set_to_none=True)
            y = targets[ids].to("cuda")
            loss = torch.nn.functional.cross_entropy(logits(model, head, values, ids).reshape(-1, 3), y.reshape(-1))
            require(bool(torch.isfinite(loss)), "MNLM_TEMPORAL_NONFINITE", "nonfinite training loss")
            loss.backward()
            params = [*model.parameters(), *head.parameters()]
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
                    "MNLM_TEMPORAL_NONFINITE", "missing/nonfinite gradients; update refused")
            optimizer.step()
            total += loss.item() * len(ids)
            updates += 1
        losses.append({"epoch": epoch + 1, "loss_nats": total / len(present), "updates": updates})
    require(all(bool(torch.isfinite(p).all()) for p in [*model.parameters(), *head.parameters()]),
            "MNLM_TEMPORAL_NONFINITE", "nonfinite final parameters")
    fit_seconds = time.perf_counter() - started
    final = predictions(model, head, values, present)
    lineage = {**binding, "recipe_sha256": digest(RECIPE), "training_code_sha256": file_hash(Path(__file__)),
               "label_map_sha256": file_hash(out / "training-label-map.json"), "variant": "shuffled" if permuted else "real",
               "updates": updates, "rows_total": 239, "rows_with_history": 224, "absent_rows": 15}
    checkpoint = out / "instrument"
    checkpoint_id = freeze(model, head, checkpoint, lineage)
    cold = FrozenTemporal(checkpoint, checkpoint_id)
    cold_logits = []
    with torch.inference_mode():
        for i in present:
            encoded = cold.encode(values[i])
            cold_logits.append(head(torch.from_numpy(encoded[-1:]).to("cuda"))[0].reshape(7, 3).cpu().numpy())
    require(np.array_equal(final, np.asarray(cold_logits)), "MNLM_TEMPORAL_REPLAY", "cold trained predictions differ")
    # All statistics below are recomputable from these persisted arrays.
    with (out / "predictions.npz").open("xb") as f:
        np.savez(f, initial=initial, final=final, truth=target[present].numpy(), fit_truth=targets[present].numpy(), present=present)
        f.flush()
        os.fsync(f.fileno())
    native = native_outputs(cold, values, rows, out / "native")
    truth = target[present].numpy()
    counts = np.asarray([np.bincount(truth[:, j], minlength=3) for j in range(7)], np.float64) + 1.
    marginal = np.broadcast_to(np.log(counts / counts.sum(-1, keepdims=True)), (len(present), 7, 3))
    report = {"acceptance": "author-checked only", "variant": lineage["variant"], "before": before,
              "after": {"encoder": state_hash(model), "head": state_hash(head)}, "updates": updates,
              "loss_trace": losses, "initial": metrics(initial, truth), "final_against_real_training_labels": metrics(final, truth),
              "final_against_fit_labels": metrics(final, targets[present].numpy()), "training_marginal": metrics(marginal, truth),
              "checkpoint_id": checkpoint_id, "native_value_digest": native["value_digest"], "fit_seconds": fit_seconds,
              "frozen_replay_exact": True, "production_admission": False, "stage4_accepted": False}
    put(out / "report.json", canonical(report))
    return report


def run(a):
    started = time.perf_counter()
    reader, rows, values, target, binding = prepare(a.prefix, a.prefix_sha256, a.native, a.native_sha256)
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    take_lease(12, "train-frozen-temporal-candidates")
    try:
        execution = configure()
        a.out.mkdir(parents=True, exist_ok=False)
        put(a.out / "recipe.json", canonical(RECIPE))
        put(a.out / "input.json", canonical(binding))
        put(a.out / "gpu-before.txt", resident.encode())
        prepared = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        real = train(values, target, rows, binding, a.out / "real", False)
        real_done = time.perf_counter()
        shuffled = train(values, target, rows, binding, a.out / "shuffled", True)
        finished = time.perf_counter()
        require(real["before"] == shuffled["before"], "MNLM_TEMPORAL_CONTROL", "control initialization differs")
        report = {"acceptance": "author-checked only", "runtime": execution, "real": real, "shuffled": shuffled,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "new_outcome_evaluation": False,
                  "production_admission": False, "stage4_accepted": False,
                  "cost_seconds": {"prepare": prepared - started, "real": real_done - prepared, "shuffled": finished - real_done},
                  "wall_seconds": finished - started}
        put(a.out / "report.json", canonical(report))
        put(a.out / "SHA256.json", canonical({p.relative_to(a.out).as_posix(): file_hash(p) for p in sorted(a.out.rglob("*")) if p.is_file()}))
        print(json.dumps({"real": real["final_against_real_training_labels"], "shuffled": shuffled["final_against_real_training_labels"],
                          "wall_seconds": report["wall_seconds"], "gpu_peak_allocated_bytes": report["gpu_peak_allocated_bytes"]}), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("prefix", "native", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    for n in ("prefix", "native"):
        ap.add_argument("--" + n + "-sha256", required=True)
    run(ap.parse_args())
