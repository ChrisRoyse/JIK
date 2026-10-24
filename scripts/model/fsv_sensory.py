"""Real retained encoder outputs -> training -> cold checkpoint -> bank expansion.

Engineering only. No labels/encoder admission, meaning predictor or language
quality claim. Run with -I. Negative mutations never enter the real publication.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
import torch

from senses.contract import canonical, digest, SenseError
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig, parameter_count
from model.network import SensoryLanguageModel, training_step
from model.publication_input import read_publication
from model.checkpoint import save, load, state_hash

GROUPS = [[f"{tier}-{i}" for tier in ("cheap-semantic", "deterministic", "multi", "panel")] for i in (0, 1)]


def tensor_hash(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(canonical([a.dtype.str, list(a.shape)]) + a.tobytes()).hexdigest()


def native_hash(native):
    return digest({n: {"identity": x.identity, "rows": x.rows, "values": tensor_hash(x.values),
                       "padding": tensor_hash(x.padding), "absent": tensor_hash(x.absent),
                       "indices": tensor_hash(x.indices) if x.indices is not None else None} for n, x in native.items()})


def prefixes(rows, device, length):
    # Byte surface reconstruction is a bounded wiring target, never an anchor.
    prefix = torch.full((len(rows), length), 257, dtype=torch.long, device=device)
    target = torch.full_like(prefix, -100)
    for i, row in enumerate(rows):
        raw = list(row["text"].encode("utf-8")[:length])
        prefix[i, 0] = 256
        if raw:
            target[i, :len(raw)] = torch.tensor(raw, device=device)
            if len(raw) > 1:
                prefix[i, 1:len(raw)] = torch.tensor(raw[:-1], device=device)
    return prefix, target


def configure():
    torch.manual_seed(25)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def replay(args):
    configure()
    take_lease(25, "sensory-cold-replay")
    try:
        model, manifest = load(args.checkpoint, args.device)
        specs, native, rows, receipt = read_publication(args.publication, GROUPS, model.config)
        assert digest(receipt) == manifest["input_sha256"]
        assert all(s in specs for s in model.specs)
        native = {s.name: native[s.name].to(args.device) for s in model.specs}
        prefix, _ = prefixes(rows, args.device, model.config.max_prefix)
        with torch.no_grad():
            output = model(prefix, native)
        result = {"state_sha256": state_hash(model), "logits_sha256": tensor_hash(output["logits"]),
                  "input_sha256": digest(receipt), "manifest_sha256": digest(manifest),
                  "trainable": [n for n, p in model.named_parameters() if p.requires_grad], "meaning_asserted": False}
        Path(args.out).write_bytes(canonical(result))
    finally:
        release_lease()


def run(args):
    start = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **evidence):
        checks.append({"name": name, "pass": bool(ok), **evidence})
        (out / "checks.json").write_bytes(canonical(checks))
        assert ok, name

    def refuses(name, action, code):
        try:
            action()
        except SenseError as e:
            check(name, e.code == code, error=e.code)
        else:
            check(name, False, expected=code)

    config = ModelConfig()
    configure()
    specs, original, rows, receipt = read_publication(args.publication, GROUPS, config)
    (out / "input.json").write_bytes(canonical(receipt))
    check("real_scope", len(rows) == 8 and len(specs) == 12, rows=len(rows), instruments=len(specs))
    original_hash = native_hash(original)
    (out / "gpu-before.txt").write_text(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True), encoding="utf-8")
    take_lease(25, "sensory-model-fsv")
    try:
        native = {n: x.to(args.device) for n, x in original.items()}
        held = next(s for s in specs if s.name == "ucr-all-minilm-l6-v2")
        initial_specs = tuple(s for s in specs if s != held)
        model = SensoryLanguageModel(config, initial_specs).to(args.device)
        initial = {s.name: native[s.name] for s in initial_specs}
        prefix, targets = prefixes(rows, args.device, config.max_prefix)
        memories, counts = model.memories(initial)
        check("separate_native_memories", set(memories) == set(initial)
              and all(memories[n].shape[:2] == x.values.shape[:2] for n, x in initial.items()),
              shapes={n: list(x.values.shape) for n, x in native.items()},
              instrument_counts=counts[0].tolist(), channel_counts=counts[1].tolist())
        check("dense_sparse_multi_real", {s.layout for s in specs} == {"dense", "sparse", "multi"})
        check("no_native_gradient", all(not x.values.requires_grad for x in native.values()))
        before = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        losses = [training_step(model, optimizer, prefix, targets, initial) for _ in range(5)]
        after = state_hash(model)
        check("real_consumer_optimizer_update", before != after, before=before, after=after, losses=losses)
        with torch.no_grad():
            logits = model(prefix, initial)["logits"]
            again = model(prefix, initial)["logits"]
        check("repeat_logits_exact", torch.equal(logits, again))
        model.eval()
        with torch.no_grad():
            check("eval_mode_does_not_switch_arithmetic", torch.equal(logits, model(prefix, initial)["logits"]))
        model.train()
        base_id = save(model, out / "base", step=5, input_sha256=digest(receipt))
        cold, manifest = load(out / "base", args.device)
        with torch.no_grad():
            check("cold_reload_exact", torch.equal(logits, cold(prefix, initial)["logits"])
                  and state_hash(cold) == after, manifest_sha256=base_id)
        del cold
        old_parameters = {n: tensor_hash(p) for n, p in model.named_parameters()}
        model.extend(held)
        with torch.no_grad():
            expanded = model(prefix, native)["logits"]
        check("expansion_preserves_existing_outputs", torch.equal(logits, expanded))
        check("expansion_preserves_old_parameters", all(tensor_hash(dict(model.named_parameters())[n]) == h
                                                       for n, h in old_parameters.items()))
        refuses("stale_optimizer", lambda: training_step(model, optimizer, prefix, targets, native), "MNLM_MODEL_STALE_OPTIMIZER")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
        extension_losses = [training_step(model, optimizer, prefix, targets, native) for _ in range(3)]
        check("new_branch_learns_old_parameters_frozen", all(tensor_hash(dict(model.named_parameters())[n]) == h
                                                            for n, h in old_parameters.items())
              and any(float(b.branches[held.name].gain.detach()) != 0 for b in model.blocks), losses=extension_losses)
        held_input = native[held.name]
        explicit_absence = replace(held_input, values=torch.zeros_like(held_input.values),
                                   padding=torch.ones_like(held_input.padding), absent=torch.ones_like(held_input.absent))
        with torch.no_grad():
            check("old_view_retained_after_extension_training", torch.equal(logits, model(prefix, {**native, held.name: explicit_absence})["logits"]))
            trained_logits = model(prefix, native)["logits"]
            check("new_branch_changes_observed_outputs", not torch.equal(logits, trained_logits))
        expanded_id = save(model, out / "expanded", step=8, input_sha256=digest(receipt), parent=base_id)
        (out / "expected-replay.json").write_bytes(canonical({"state_sha256": state_hash(model),
            "logits_sha256": tensor_hash(trained_logits), "input_sha256": digest(receipt), "manifest_sha256": expanded_id,
            "trainable": [n for n, p in model.named_parameters() if p.requires_grad], "meaning_asserted": False}))

        # Five interventions target distinct intake/learning failure mechanisms.
        altered = prefix.clone()
        altered[:, -1] = (altered[:, -1] + 1) % config.vocabulary
        with torch.no_grad():
            changed = model(altered, native)["logits"]
        check("future_surface_cannot_change_past_logits", torch.equal(changed[:, :-1], trained_logits[:, :-1]))
        mm = next(s for s in specs if s.name == "ucr-answerai-colbert-small")
        x = native[mm.name]
        swapped = x.values.clone()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        with torch.no_grad():
            check("multi_order_is_observable", not torch.equal(trained_logits, model(prefix, {**native, mm.name: replace(x, values=swapped)})["logits"]))
        refuses("four_instruments_not_model", lambda: SensoryLanguageModel(config, specs[:4]), "MNLM_MODEL_MIN_INSTRUMENTS")
        refuses("implicit_missing_instrument", lambda: model(prefix, initial), "MNLM_MODEL_ROSTER")
        refuses("wrong_version", lambda: model(prefix, {**native, held.name: replace(held_input, identity="0" * 64)}), "MNLM_MODEL_IDENTITY")
        refuses("wrong_rows", lambda: model(prefix, {**native, held.name: replace(held_input, rows=tuple(reversed(held_input.rows)))}), "MNLM_MODEL_ROWS")
        refuses("wrong_dimension", lambda: model(prefix, {**native, held.name: replace(held_input, values=held_input.values[:, :, :-1])}), "MNLM_MODEL_SHAPE")
        refuses("absent_disagrees_with_mask", lambda: model(prefix, {**native, held.name: replace(held_input, absent=~held_input.absent)}), "MNLM_MODEL_MASK")
        bad = held_input.values.clone()
        bad[0, 0, 0] = float("nan")
        refuses("nonfinite_observation", lambda: model(prefix, {**native, held.name: replace(held_input, values=bad)}), "MNLM_MODEL_VALUES")
        refuses("frozen_native_gradients", lambda: model(prefix, {**native, held.name: replace(held_input, values=held_input.values.clone().requires_grad_())}), "MNLM_MODEL_VALUES")
        blank = {n: replace(x, values=torch.zeros_like(x.values), padding=torch.ones_like(x.padding),
                            absent=torch.ones_like(x.absent), indices=None if x.indices is None else torch.zeros_like(x.indices))
                 for n, x in native.items()}
        refuses("all_absent", lambda: model(prefix, blank), "MNLM_MODEL_NO_OBSERVATION")
        refuses("partial_rows_cannot_assert_meaning", lambda: model.assert_meaning(native), "MNLM_MODEL_MEANING_COVERAGE")
        first = {n: replace(x, rows=x.rows[:1], values=x.values[:1], padding=x.padding[:1], absent=x.absent[:1],
                            indices=None if x.indices is None else x.indices[:1]) for n, x in native.items()}
        refuses("five_channels_do_not_bypass_admission", lambda: model.assert_meaning(first), "MNLM_MODEL_UNCOMMISSIONED")
        before_failure = state_hash(model)
        refuses("duplicate_extension", lambda: model.extend(held), "MNLM_MODEL_DUPLICATE")
        check("failed_extension_does_not_mutate", state_hash(model) == before_failure)
        refuses("parameter_preflight", lambda: SensoryLanguageModel(replace(config, max_parameters=100), specs), "MNLM_MODEL_PARAMETER_BUDGET")
        from model.attention import Attention
        tight_attention = Attention(replace(config, max_attention_elements=1)).to(args.device)
        refuses("attention_preflight", lambda: tight_attention(torch.zeros((1, 2, config.width), device=args.device),
                torch.zeros((1, 2, config.width), device=args.device)), "MNLM_MODEL_ATTENTION_BUDGET")
        sparse = next(s for s in specs if s.layout == "sparse")
        sx = native[sparse.name]
        ix = sx.indices.clone()
        ix[0, 0] = sparse.dims
        refuses("sparse_coordinate_bound", lambda: model(prefix, {**native, sparse.name: replace(sx, indices=ix)}), "MNLM_MODEL_SPARSE")
        refuses("whole_bank_preflight", lambda: read_publication(args.publication, GROUPS,
                replace(config, max_native_elements=1)), "MNLM_MODEL_INPUT_BUDGET")
        refuses("checkpoint_never_overwrites", lambda: save(model, out / "expanded", step=8, input_sha256=digest(receipt)), "MNLM_MODEL_CHECKPOINT_EXISTS")
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            refuses("changed_runtime_precision", lambda: load(out / "expanded", args.device), "MNLM_MODEL_RUNTIME_DRIFT")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        shutil.copytree(out / "expanded", out / "corrupt")
        path = out / "corrupt" / "weights.pt"
        with path.open("r+b") as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 1]))
        refuses("corrupt_checkpoint", lambda: load(out / "corrupt", args.device), "MNLM_MODEL_CHECKPOINT_CORRUPT")
        (out / "corrupt" / "COMMITTED").unlink()
        refuses("uncommitted_checkpoint", lambda: load(out / "corrupt", args.device), "MNLM_MODEL_CHECKPOINT_INCOMPLETE")
        check("native_inputs_unchanged", native_hash(native) == original_hash)
        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    finally:
        release_lease()
    # A new process loads the actual files and real source again, after lease release.
    child = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--replay", "--publication", args.publication,
                            "--checkpoint", str(out / "expanded"), "--device", args.device,
                            "--out", str(out / "cold-replay.json")], capture_output=True, text=True)
    (out / "replay.stdout").write_text(child.stdout, encoding="utf-8")
    (out / "replay.stderr").write_text(child.stderr, encoding="utf-8")
    check("separate_process_replay", child.returncode == 0 and (out / "cold-replay.json").is_file()
          and (out / "cold-replay.json").read_bytes() == (out / "expected-replay.json").read_bytes(), exit=child.returncode)
    _, reread, _, after_receipt = read_publication(args.publication, GROUPS, config)
    check("source_of_truth_unchanged", receipt == after_receipt and native_hash(reread) == original_hash)
    report = {"schema": "sensory-model-fsv-v1", "acceptance": "author-checked only", "passed": len(checks), "failed": 0,
              "checks": checks, "parameters": parameter_count(config, specs), "input_sha256": digest(receipt),
              "gpu_peak_allocated_bytes": peak, "wall_seconds": time.perf_counter() - start,
              "stage3_accepted": False, "stage4_accepted": False, "trained_language_quality_claimed": False,
              "meaning_predictor_implemented": False, "optimizer_resume_supported": False}
    (out / "report.json").write_bytes(canonical(report))
    inventory = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.rglob("*") if p.is_file()}
    (out / "SHA256.json").write_bytes(canonical(inventory))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    try:
        replay(args) if args.replay else run(args)
    except SenseError as e:
        print(json.dumps(e.envelope()), file=sys.stderr)
        raise SystemExit(2)
