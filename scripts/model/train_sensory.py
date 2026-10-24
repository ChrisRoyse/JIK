"""Train the sensory surface consumer from sealed native slots (engineering only).

Example: python -I scripts/model/train_sensory.py --publication PATH
  --groups groups.json --config config/sensory-engineering.json --out NEW_PATH
This is neither Stage 3 admission nor the meaning predictor. Native measurements
are never modified; checkpoints cannot be promoted to production by this CLI.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import torch
from senses.contract import canonical, digest, require, SenseError
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.network import SensoryLanguageModel, training_step
from model.checkpoint import save, load, state_hash
from model.publication_input import read_publication


def byte_targets(rows, length, device):
    """Only the rendering target is a bounded prefix; all native inputs survive."""
    prefix = torch.full((len(rows), length), 257, dtype=torch.long, device=device)
    targets = torch.full_like(prefix, -100)
    prefix[:, 0] = 256
    for i, row in enumerate(rows):
        data = list(row["text"].encode("utf-8")[:length])
        if data:
            targets[i, :len(data)] = torch.tensor(data, device=device)
            if len(data) > 1:
                prefix[i, 1:len(data)] = torch.tensor(data[:-1], device=device)
    require(bool((targets != -100).any()), "MNLM_MODEL_TARGET", "selected observations have no retained surface target")
    return prefix, targets


def run(args):
    start = time.perf_counter()
    recipe = json.loads(Path(args.config).read_text(encoding="utf-8"))
    require(set(recipe) == {"schema", "purpose", "model", "steps", "learning_rate", "seed"}
            and recipe["schema"] == "sensory-training-v1" and recipe["purpose"] == "engineering_surface_decoder",
            "MNLM_MODEL_TRAINING_PLAN", "explicit engineering training plan required")
    config = ModelConfig(**recipe["model"])
    config.validate()
    require(config.vocabulary == 258 and type(recipe["steps"]) is int and 1 <= recipe["steps"] <= 1000
            and type(recipe["seed"]) is int and 0 <= recipe["seed"] < 2**32
            and type(recipe["learning_rate"]) in (int, float) and 0 < recipe["learning_rate"] <= 0.01,
            "MNLM_MODEL_TRAINING_PLAN", "byte-target vocabulary, bounded steps, finite rate and explicit seed required")
    out = Path(args.out)
    require(not out.exists(), "MNLM_MODEL_CHECKPOINT_EXISTS", "choose a new run directory")
    groups = json.loads(Path(args.groups).read_text(encoding="utf-8"))
    specs, native, rows, receipt = read_publication(args.publication, groups, config)
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    take_lease(25, "sensory-surface-training")
    try:
        require(torch.cuda.is_available(), "MNLM_MODEL_CUDA_REQUIRED", "declared CUDA runtime unavailable; no CPU fallback")
        torch.manual_seed(recipe["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = SensoryLanguageModel(config, specs).to("cuda")
        native = {n: x.to("cuda") for n, x in native.items()}
        prefix, targets = byte_targets(rows, config.max_prefix, "cuda")
        out.mkdir(parents=True, exist_ok=False)
        (out / "plan.json").write_bytes(canonical(recipe))
        (out / "input.json").write_bytes(canonical(receipt))
        (out / "gpu-before.txt").write_text(resident, encoding="utf-8")
        before = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
        losses = []
        for step in range(recipe["steps"]):
            losses.append(training_step(model, optimizer, prefix, targets, native))
            (out / "progress.json").write_bytes(canonical({"steps": step + 1, "losses": losses, "accepted": False}))
        checkpoint_id = save(model, out / "checkpoint", step=recipe["steps"], input_sha256=digest(receipt))
        with torch.no_grad():
            expected = model(prefix, native)["logits"]
        cold, manifest = load(out / "checkpoint", "cuda")
        with torch.no_grad():
            require(torch.equal(expected, cold(prefix, native)["logits"]), "MNLM_MODEL_REPLAY", "cold checkpoint changed outputs")
        report = {"acceptance": "author-checked only", "purpose": recipe["purpose"], "input_sha256": digest(receipt),
                  "checkpoint_sha256": checkpoint_id, "state_before": before, "state_after": manifest["state_sha256"],
                  "rows": len(rows), "instruments": len(specs), "losses": losses, "cold_replay_exact": True,
                  "wall_seconds": time.perf_counter() - start, "meaning_asserted": False, "stage4_accepted": False}
        (out / "report.json").write_bytes(canonical(report))
        print(json.dumps(report))
    finally:
        release_lease()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("publication", "groups", "config", "out"):
        parser.add_argument("--" + key, required=True)
    try:
        run(parser.parse_args())
    except SenseError as e:
        print(json.dumps(e.envelope()), file=sys.stderr)
        raise SystemExit(2)
