"""Train an immutable extension; assess active-branch retention before selection.

No serving state is modified. A failed assessment selects the unchanged parent.
The held-out benchmark is consumed once; replay may only reopen the same run.
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
from senses.forecast import put
from senses.pipeline import file_hash
from senses.trajectory_population import read_pinned
from lenscommon import take_lease, release_lease
from model.continual_input import prepare, on_device
from model.network import SensoryLanguageModel, training_step
from model.checkpoint import save, load, state_hash
from model.retention import assess, batch_hash


def code_hashes():
    root = Path(__file__).resolve().parent
    return {n: file_hash(root / n) for n in ("retention.py", "continual_input.py", "train_continual.py",
                                            "train_sensory.py", "prefix_bank.py")}


def configure(seed):
    require(torch.cuda.is_available(), "MNLM_MODEL_CUDA_REQUIRED", "CUDA is required; no CPU fallback")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def fit(model, batches, role, steps, rate):
    selected = [b for b in batches if b["role"] == role]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate)
    history = []
    for step in range(steps):
        b = selected[step % len(selected)]
        loss = training_step(model, optimizer, b["prefix"], b["targets"],
                             {s.name: b["native"][s.name] for s in model.specs})
        history.append({"step": step + 1, "atom": b["atom"], "unit": b["unit"], "loss_nats": loss})
    return history


def run(args):
    started = time.perf_counter()
    plan = read_pinned(args.plan, args.plan_sha256)
    config, specs, cpu, receipt = prepare(args.panel, args.records, plan)
    input_hash = digest({"plan": plan, "batch_sha256": batch_hash(cpu)})
    prepared = time.perf_counter()
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    take_lease(25, "continual-retention-replay" if args.replay else "continual-retention")
    try:
        configure(plan["seed"])
        batches = on_device(cpu, "cuda")
        require(batch_hash(batches) == batch_hash(cpu), "MNLM_RETENTION_BINDING", "device move changed inputs")
        if args.replay:
            report = json.loads((args.out / "report.json").read_bytes())
            require(report["code_sha256"] == code_hashes() and report["input_sha256"] == input_hash,
                    "MNLM_RETENTION_BINDING", "restore the exact run source/code for replay")
            base, bm = load(args.out / "base", "cuda")
            candidate, cm = load(args.out / "candidate", "cuda")
            result = assess(base, candidate, bm, cm, plan, batches)
            require(digest(result) == report["assessment_sha256"]
                    and (args.out / "assessment.json").read_bytes() == canonical(result),
                    "MNLM_RETENTION_REPLAY", "cold metrics or selection changed")
            put(args.replay, canonical({"exact": True, "assessment_sha256": digest(result),
                "input_sha256": input_hash, "wall_seconds": time.perf_counter() - started,
                "acceptance": "author-checked only", "stage4_accepted": False}))
            print(json.dumps({"cold_replay_exact": True, "selected": result["selected_checkpoint_sha256"]}))
            return
        require(not args.out.exists(), "MNLM_RETENTION_RUN_EXISTS", "immutable run already exists; never overwrite")
        args.out.mkdir(parents=True)
        put(args.out / "plan.json", canonical(plan))
        put(args.out / "input.json", canonical(receipt))
        put(args.out / "gpu-before.txt", resident.encode())
        setup = time.perf_counter()
        model = SensoryLanguageModel(config, specs[:-1]).to("cuda")
        old_history = fit(model, batches, "old_train", plan["base_steps"], plan["learning_rate"])
        base_id = save(model, args.out / "base", step=plan["base_steps"], input_sha256=input_hash)
        trained_base = time.perf_counter()
        # Reload the baseline we actually persisted before extending it.
        model, bm = load(args.out / "base", "cuda")
        model.extend(specs[-1])
        new_history = fit(model, batches, "new_train", plan["extension_steps"], plan["learning_rate"])
        candidate_id = save(model, args.out / "candidate", step=plan["base_steps"] + plan["extension_steps"],
                            input_sha256=input_hash, parent=base_id)
        put(args.out / "training.json", canonical({"baseline": old_history, "extension": new_history}))
        trained_candidate = time.perf_counter()
        base, bm = load(args.out / "base", "cuda")
        candidate, cm = load(args.out / "candidate", "cuda")
        require(digest(cm) == candidate_id, "MNLM_RETENTION_LINEAGE", "cold candidate identity changed")
        put(args.out / "EVALUATION_STARTED", canonical({"plan_sha256": digest(plan), "baseline": base_id,
                                                       "candidate": candidate_id}))
        result = assess(base, candidate, bm, cm, plan, batches)
        require(input_hash == digest({"plan": plan, "batch_sha256": batch_hash(batches)}),
                "MNLM_RETENTION_BINDING", "training/evaluation mutated native inputs or targets")
        put(args.out / "assessment.json", canonical(result))
        finished = time.perf_counter()
        phases = {"input": prepared - started, "setup": setup - prepared,
                  "baseline": trained_base - setup, "candidate": trained_candidate - trained_base,
                  "cold_assessment": finished - trained_candidate}
        report = {"schema": "sensory-continual-run-v1", "acceptance": "author-checked only",
                  "code_sha256": code_hashes(), "input_sha256": input_hash,
                  "assessment_sha256": digest(result), "decision": result["decision"],
                  "instrument_count": len(specs), "native_value_bytes": receipt["native_value_bytes"],
                  "parameters": sum(p.numel() for p in candidate.parameters()),
                  "phases_seconds": phases, "wall_seconds": finished - started,
                  "stage4_accepted": False, "meaning_asserted": False}
        put(args.out / "report.json", canonical(report))
        print(json.dumps({"decision": result["decision"], "wall_seconds": finished - started}))
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "records", "plan", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--replay", type=Path)
    try:
        run(p.parse_args())
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        raise SystemExit(2)
