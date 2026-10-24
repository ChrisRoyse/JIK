"""Bounded masked constellation associations with persisted train/control arms."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check, sha
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.loop_training_checkpoint import fingerprint, finite
from model.text_masked_associations import MaskedData, RECIPE, build_probe
from senses.contract import digest
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def optimizer_for(probe):
    return torch.optim.AdamW(probe.parameters(), lr=RECIPE["learning_rate"],
        weight_decay=RECIPE["weight_decay"], foreach=False, fused=False)


def checkpoint(root, probe, optimizer, data, step):
    probe.check_reference()
    check(finite(probe.state_dict()) and finite(optimizer.state_dict()), "MASKED_NONFINITE_SNAPSHOT")
    root.mkdir()
    stream = io.BytesIO()
    torch.save(dict(model=probe.state_dict(), optimizer=optimizer.state_dict()), stream)
    raw = stream.getvalue()
    check(len(raw) <= probe.config.checkpoint_bytes, "MASKED_CHECKPOINT_BUDGET")
    with (root/"state.pt").open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    manifest = dict(schema="masked-association-checkpoint-v1", binding=data.pin,
        step=step, state_sha256=sha(raw), bytes=len(raw), fingerprint=fingerprint(probe, optimizer))
    publish(root/"manifest.json", manifest)
    return file_sha(root/"manifest.json")


def restore(root, pin, probe, data):
    check(file_sha(root/"manifest.json") == pin, "MASKED_CHECKPOINT_PIN")
    manifest = json.loads((root/"manifest.json").read_bytes())
    check(manifest["binding"] == data.pin and manifest["bytes"] <= probe.config.checkpoint_bytes
          and (root/"state.pt").stat().st_size == manifest["bytes"], "MASKED_CHECKPOINT_BINDING")
    raw = (root/"state.pt").read_bytes()
    check(sha(raw) == manifest["state_sha256"], "MASKED_CHECKPOINT_BYTES")
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    check(set(state) == {"model", "optimizer"} and finite(state), "MASKED_CHECKPOINT_STATE")
    probe.load_state_dict(state["model"], strict=True)
    optimizer = optimizer_for(probe)
    optimizer.load_state_dict(state["optimizer"])
    check(fingerprint(probe, optimizer) == manifest["fingerprint"], "MASKED_CHECKPOINT_READBACK")
    probe.check_reference()
    return optimizer


def update(probe, optimizer, data, indices, shuffled):
    optimizer.zero_grad(set_to_none=True)
    terms = data.loss(probe, indices, shuffled=shuffled)
    loss = terms.mean()
    check(bool(torch.isfinite(loss)), "MASKED_LOSS_NONFINITE")
    loss.backward()
    check(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
              for p in probe.parameters()), "MASKED_GRADIENT_COVERAGE")
    branch = {n: float(p.grad.abs().max()) for n, p in probe.named_parameters()
              if n.endswith("weight") and (n.startswith("projectors.") or n.startswith("heads."))}
    check(branch and min(branch.values()) > 0, "MASKED_BRANCH_DISCONNECTED")
    norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.)
    check(bool(torch.isfinite(norm)), "MASKED_GRADIENT_NORM")
    optimizer.step()
    probe.check_reference()
    return dict(loss_nats=float(loss.detach()), per_slot_nats=terms.detach().cpu().tolist(),
        atoms=len(indices), missing_slot_examples=len(indices)*len(data.lanes), gradient_max=branch)


def predict(probe, data):
    with torch.no_grad():
        return torch.stack([probe(data.observed(data.nodes, n), n) for n in data.lanes]).cpu().numpy()


def scores(probe, data, values):
    result = {}
    # Compute scoring with torch on CPU; raw inputs and outputs remain in native files.
    teacher = torch.from_numpy(data.teacher.astype(np.float64))/RECIPE["temperature"]
    predictions = torch.from_numpy(values.astype(np.float64))/RECIPE["temperature"]
    train = teacher[:, data.train_nodes].clone()
    valid = torch.tensor(data.train_nodes)[:, None] != torch.tensor(data.train_nodes)[None]
    prior = train.masked_fill(~valid[None], -torch.inf).softmax(-1).mean(1)
    for g in data.groups:
        if g["split"] != "development":
            continue
        indices = g["indices"]
        lp, lq = teacher[:, indices].log_softmax(-1), predictions[:, indices].log_softmax(-1)
        divergence = (lp.exp()*(lp-lq)).sum(-1)/np.log(2)
        prior_kl = (lp.exp()*(lp-prior[:, None].log())).sum(-1)/np.log(2)
        expected = teacher[:, indices].argsort(dim=-1, descending=True, stable=True)[..., :5]
        ranked = predictions[:, indices].argsort(dim=-1, descending=True, stable=True)[..., :5]
        recall = (expected[..., :, None] == ranked[..., None, :]).any(-1).double().mean(-1)
        # Five-slot reference selection is a routing decision, never a merged native slot.
        neighbors = []
        for lane in range(len(data.lanes)):
            evidence = teacher[[j for j in range(len(data.lanes)) if j != lane]][:, indices].sum(0)
            selected = evidence.argmax(-1)
            target = teacher[lane, data.train_nodes][selected].log_softmax(-1)
            neighbors.append((lp[lane].exp()*(lp[lane]-target)).sum(-1)/np.log(2))
        result[g["domain"]] = {n: dict(kl_bits=float(divergence[j].mean()),
            prior_kl_bits=float(prior_kl[j].mean()), neighbor_baseline_kl_bits=float(neighbors[j].mean()),
            recall_at_5=float(recall[j].mean()),
            correct_vs_shuffled_bits=float((lp[j].exp()*(lq[j]-lq[j].roll(1, 0))).sum(-1).mean()/np.log(2)),
            prediction_std=float(torch.from_numpy(values[j, indices]).double().std(0, unbiased=False).mean()))
            for j, n in enumerate(data.lanes)}
    return result


def controls(probe, data):
    indices = data.train_nodes[:6]
    evidence = {}
    with torch.no_grad():
        for missing in data.lanes:
            observed = data.observed(indices, missing)
            base = probe(observed, missing)
            single = torch.cat([probe({n: x[i:i+1] for n, x in observed.items()}, missing)
                                for i in range(len(indices))])
            difference = float((base-single).abs().max())
            check(difference <= 1e-6, "MASKED_BATCH_EQUIVALENCE")
            saved = [data.cached[i][missing] for i in indices]
            try:
                for i in indices:
                    data.cached[i][missing] = object()  # cannot be embedded or inspected
                check(torch.equal(base, probe(data.observed(indices, missing), missing)),
                      "MASKED_TARGET_INPUT_LEAK")
            finally:
                for i, value in zip(indices, saved):
                    data.cached[i][missing] = value
            observed_change = {}
            for name, x in observed.items():
                altered = {**observed, name: x.roll(1, 0)}
                delta = float((base-probe(altered, missing)).abs().max())
                check(delta > 0, "MASKED_OBSERVATION_IGNORED")
                observed_change[name] = delta
            refused = []
            bad = [({**observed, missing: saved[0][None]}, "MASKED_OBSERVED_ROSTER"),
                   ({n: x for n, x in observed.items() if n != next(iter(observed))}, "MASKED_OBSERVED_ROSTER")]
            name = next(iter(observed))
            poisoned = observed[name].clone()
            poisoned[0, 0, 0] = torch.nan
            bad.append(({**observed, name: poisoned}, "MASKED_NATIVE_INPUT"))
            for candidate, code in bad:
                try:
                    probe(candidate, missing)
                except ValueError as exc:
                    check(str(exc) == "MNLM_CODATA_"+code, "MASKED_NEGATIVE_CODE")
                    refused.append(code)
                else:
                    raise ValueError("MNLM_CODATA_MASKED_NEGATIVE_ACCEPTED")
            evidence[missing] = dict(batch_max_error=difference, target_poison_invariant=True,
                observed_change=observed_change, refused=refused)
    return evidence


def train(a, probe, data):
    np.save(a.output/"teacher.npy", data.teacher, allow_pickle=False)
    controls_before = controls(probe, data)
    optimizer = optimizer_for(probe)
    pins = {"initial": checkpoint(a.output/"initial", probe, optimizer, data, 0)}
    values = predict(probe, data)
    np.save(a.output/"initial.npy", values, allow_pickle=False)
    metrics, updates = {"initial": scores(probe, data, values)}, {}
    step = 0
    for cycle, domains in (("cycle-1", ["general", "legal"]), ("cycle-2", ["general", "legal", "science"])):
        indices = data.indices(domains)
        check(len(indices) == (88 if cycle == "cycle-1" else 107), "MASKED_TRAIN_COUNT")
        updates[cycle] = []
        for epoch in range(RECIPE["epochs"]):
            result = update(probe, optimizer, data, indices, a.shuffled)
            updates[cycle].append(result)
            step += 1
            print(json.dumps(dict(cycle=cycle, step=step, loss=result["loss_nats"])), flush=True)
        pins[cycle] = checkpoint(a.output/cycle, probe, optimizer, data, step)
        values = predict(probe, data)
        np.save(a.output/(cycle+".npy"), values, allow_pickle=False)
        metrics[cycle] = scores(probe, data, values)
    controls_after = controls(probe, data)
    continuation = []
    for _ in range(2):
        optimizer = restore(a.output/"cycle-1", pins["cycle-1"], probe, data)
        result = update(probe, optimizer, data, data.train_nodes, a.shuffled)
        continuation.append(dict(update=result, fingerprint=fingerprint(probe, optimizer)))
    check(continuation[0] == continuation[1], "MASKED_OPTIMIZER_RESTART")
    restore(a.output/"initial", pins["initial"], probe, data)
    check(np.array_equal(predict(probe, data), np.load(a.output/"initial.npy", allow_pickle=False)),
          "MASKED_ROLLBACK_PREDICTION")
    return dict(schema="masked-relationship-learning-v1", data=data.bind, recipe=RECIPE,
        shuffled=a.shuffled, identity_unchanged=data.identity_before,
        checkpoints=pins, metrics=metrics, updates=updates, controls_before=controls_before,
        controls_after=controls_after, continuation=continuation[0], rollback=True,
        auxiliary_restart_updates=2, auxiliary_restart_atom_exposures=214,
        successful_acquisition_demonstrated=False, flags=RECIPE["flags"])


def cold(a, probe, data):
    check(file_sha(a.cold/"report.json") == a.cold_pin, "MASKED_COLD_PIN")
    old = json.loads((a.cold/"report.json").read_bytes())
    check(old["data"] == data.bind and old["shuffled"] == a.shuffled, "MASKED_COLD_BINDING")
    check(np.array_equal(data.teacher, np.load(a.cold/"teacher.npy", allow_pickle=False)), "MASKED_COLD_TARGET")
    metrics = {}
    for name, pin in old["checkpoints"].items():
        restore(a.cold/name, pin, probe, data)
        values = predict(probe, data)
        check(np.array_equal(values, np.load(a.cold/(name+".npy"), allow_pickle=False)), "MASKED_COLD_PREDICTION")
        metrics[name] = scores(probe, data, values)
        check(metrics[name] == old["metrics"][name], "MASKED_COLD_METRICS")
    optimizer = restore(a.cold/"cycle-1", old["checkpoints"]["cycle-1"], probe, data)
    value = update(probe, optimizer, data, data.train_nodes, a.shuffled)
    check(dict(update=value, fingerprint=fingerprint(probe, optimizer)) == old["continuation"], "MASKED_COLD_OPTIMIZER")
    restore(a.cold/"initial", old["checkpoints"]["initial"], probe, data)
    check(np.array_equal(predict(probe, data), np.load(a.cold/"initial.npy", allow_pickle=False)), "MASKED_COLD_ROLLBACK")
    return dict(schema="masked-relationship-cold-v1", source=a.cold_pin, exact=True, metrics=metrics,
        optimizer_exact=True, rollback=True, flags=RECIPE["flags"])


def run(a):
    start = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = MaskedData(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    phases = dict(source_and_fixed_targets=time.perf_counter()-start)
    device_tick = time.perf_counter()
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "MASKED_GPU_BUSY")
    take_lease("28", "masked-native-association-diagnostic")
    try:
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        phases["device_preflight_and_setup"] = time.perf_counter()-device_tick
        tick = time.perf_counter()
        probe = build_probe(data, a)
        phases["model_and_native_cache"] = time.perf_counter()-tick
        tick = time.perf_counter()
        report = cold(a, probe, data) if a.cold else train(a, probe, data)
        phases["training_and_checks" if not a.cold else "cold_replay"] = time.perf_counter()-tick
        report.update(phases_seconds=phases, main_function_wall_seconds=time.perf_counter()-start,
            process_startup_included=False, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            acceptance="author-checked only")
        report["phase_coverage"] = sum(phases.values())/report["main_function_wall_seconds"]
        check(.9 <= report["phase_coverage"] <= 1.001, "MASKED_PHASE_COVERAGE")
        publish(a.output/"report.json", report)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), seconds=report["main_function_wall_seconds"])), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("corpus", "bank", "identity-population", "identity-native", "identity-source", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("corpus-pin", "bank-pin"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--shuffled", action="store_true")
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    run(p.parse_args())
