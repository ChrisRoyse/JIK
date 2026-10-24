"""Read real checkpoints/observations and probe the active-branch retention boundary."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
import torch
from senses.contract import canonical, digest, require, SenseError
from senses.forecast import put
from senses.pipeline import file_hash
from senses.trajectory_population import read_pinned
from lenscommon import take_lease, release_lease
from model.continual_input import prepare, on_device, validate_plan
from model.checkpoint import load, state_hash
from model.retention import assess, batch_hash, decide, validate_split
from model.train_continual import configure


def inventory(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def run(a):
    start = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **details):
        checks.append({"name": name, "pass": bool(ok), **details})
        put(a.out / (f"check-{len(checks):02d}.json"), canonical(checks[-1]))
        require(ok, "MNLM_RETENTION_FSV", name)

    def refuses(name, action, code):
        try:
            action()
        except SenseError as e:
            check(name, e.code == code, actual=e.code, expected=code)
        else:
            check(name, False, expected=code)

    plan = read_pinned(a.plan, a.plan_sha256)
    config, specs, cpu, receipt = prepare(a.panel, a.records, plan)
    original_files = inventory(a.run)
    original_source = {"panel": inventory(a.panel), "records": file_hash(a.records)}
    report = json.loads((a.run / "report.json").read_bytes())
    recorded = json.loads((a.run / "assessment.json").read_bytes())
    put(a.out / "gpu-before.txt", subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]))
    take_lease(25, "retention-boundary-readback")
    try:
        configure(plan["seed"])
        batches = on_device(cpu, "cuda")
        base, bm = load(a.run / "base", "cuda")
        candidate, cm = load(a.run / "candidate", "cuda")
        before = {"base": state_hash(base), "candidate": state_hash(candidate), "native": batch_hash(batches)}
        actual = assess(base, candidate, bm, cm, plan, batches)
        check("cold_assessment_exact", canonical(actual) == canonical(recorded)
              and digest(actual) == report["assessment_sha256"])
        training = json.loads((a.run / "training.json").read_bytes())
        for name, role, steps in (("baseline", "old_train", plan["base_steps"]),
                                  ("extension", "new_train", plan["extension_steps"])):
            allowed = {r["atom"]: r["unit"] for r in plan["rows"] if r["role"] == role}
            check(name + "_training_units_only", len(training[name]) == steps
                  and all(r["atom"] in allowed and allowed[r["atom"]] == r["unit"] for r in training[name]))
        check("trained_extension_active", all(float(b.branches[specs[-1].name].gain.detach()) != 0 for b in candidate.blocks))
        check("frozen_weights_do_not_imply_identical_behavior", all(torch.equal(t, candidate.state_dict()[n])
              for n, t in base.state_dict().items())
              and any(x["logits_sha256"] != y["logits_sha256"] for x, y in zip(actual["baseline"], actual["candidate"])))
        independent = []
        with torch.inference_mode():
            for model, key in ((base, "baseline"), (candidate, "candidate")):
                for b in batches:
                    if b["role"] not in ("retention", "acquisition"):
                        continue
                    logits = model(b["prefix"], {s.name: b["native"][s.name] for s in model.specs})["logits"].cpu().numpy().astype(np.float64)[0]
                    targets = b["targets"].cpu().numpy()[0]
                    z = logits - logits.max(-1, keepdims=True)
                    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
                    losses = -lp[np.arange(len(targets))[targets != -100], targets[targets != -100]] / np.log(2)
                    row = next(r for r in actual[key] if r["atom"] == b["atom"])
                    check(key + "_numpy_loss_" + b["atom"][:8], np.allclose(losses, row["loss_bits"], rtol=0, atol=2e-14))
                    independent.append({"version": key, "atom": b["atom"], "loss_bits": losses.tolist()})
        put(a.out / "independent-losses.json", canonical(independent))
        decision = actual["decision"]
        check("selection_uses_gate", actual["selected_checkpoint_sha256"] == digest(cm if decision["eligible_on_finite_benchmark"] else bm))
        check("never_scientific_promotion", all(decision[k] is False for k in
              ("production_admission", "meaning_asserted", "stage4_accepted", "population_retention_proven")))

        rows = copy.deepcopy(plan["rows"])
        rows[1]["unit"] = rows[0]["unit"]
        refuses("sibling_training_leakage", lambda: validate_split(rows), "MNLM_RETENTION_LEAKAGE")
        refuses("missing_old_holdouts", lambda: validate_split([r for r in plan["rows"] if r["role"] != "retention"]), "MNLM_RETENTION_SPLIT")
        refuses("duplicate_observations", lambda: validate_split(plan["rows"] + [plan["rows"][0]]), "MNLM_RETENTION_SPLIT")
        bad_plan = {**plan, "retention_tolerance": .1}
        refuses("cannot_add_slack_after_result", lambda: validate_plan(bad_plan), "MNLM_RETENTION_PLAN")
        bad_plan = copy.deepcopy(plan)
        bad_plan["rows"][1]["unit"] = "invented-independent-unit"
        refuses("unit_identity_bound_to_source", lambda: prepare(a.panel, a.records, bad_plan), "MNLM_RETENTION_BINDING")
        bad = copy.deepcopy(actual["candidate"])
        bad[0]["loss_bits"][0] = float("nan")
        refuses("nonfinite_metrics", lambda: decide(plan["rows"], actual["baseline"], bad), "MNLM_RETENTION_NONFINITE")
        refuses("missing_retained_metrics", lambda: decide(plan["rows"], actual["baseline"], actual["candidate"][:-1]), "MNLM_RETENTION_COVERAGE")
        bad = copy.deepcopy(actual["candidate"])
        bad[0]["loss_bits"].pop()
        refuses("changed_target_extent", lambda: decide(plan["rows"], actual["baseline"], bad), "MNLM_RETENTION_BINDING")
        equal = decide(plan["rows"], actual["baseline"], actual["baseline"])
        check("identity_is_not_learning", equal["reasons"] == ["MNLM_RETENTION_NO_ACQUISITION_GAIN"])
        # Arithmetic-only positive/negative controls, never training observations.
        controlled = copy.deepcopy(actual["baseline"])
        for r in controlled:
            if r["role"] == "acquisition":
                r["loss_bits"] = [x - .1 for x in r["loss_bits"]]
        check("finite_gate_positive_arithmetic_control", decide(plan["rows"], actual["baseline"], controlled)["eligible_on_finite_benchmark"])
        old = [r for r in controlled if r["role"] == "retention"]
        old[0]["loss_bits"] = [x + .1 for x in old[0]["loss_bits"]]
        old[1]["loss_bits"] = [x - .2 for x in old[1]["loss_bits"]]
        controlled_decision = decide(plan["rows"], actual["baseline"], controlled)
        check("average_cannot_hide_forgetting", controlled_decision["reasons"] == ["MNLM_RETENTION_REGRESSION"])
        wrong_cm = {**cm, "parent": "0" * 64}
        refuses("wrong_parent", lambda: assess(base, candidate, bm, wrong_cm, plan, batches), "MNLM_RETENTION_LINEAGE")
        altered = [{**b} for b in batches]
        altered[1]["targets"] = altered[1]["targets"].clone()
        altered[1]["targets"][0, 0] = (altered[1]["targets"][0, 0] + 1) % 258
        refuses("changed_targets", lambda: assess(base, candidate, bm, cm, plan, altered), "MNLM_RETENTION_BINDING")
        altered = [{**b, "native": dict(b["native"])} for b in batches]
        x = altered[1]["native"][specs[-1].name]
        altered[1]["native"][specs[-1].name] = replace(x, values=torch.zeros_like(x.values),
            padding=torch.ones_like(x.padding), absent=torch.ones_like(x.absent))
        refuses("cannot_disable_new_branch_for_retention", lambda: assess(base, candidate, bm, cm, plan, altered), "MNLM_RETENTION_BINDING")
        with torch.no_grad():
            candidate.blocks[0].branches[specs[-1].name].gain.add_(1)
        refuses("changed_candidate", lambda: assess(base, candidate, bm, cm, plan, batches), "MNLM_RETENTION_LINEAGE")
        candidate, _ = load(a.run / "candidate", "cuda")
        corrupt = a.out / "corrupt-checkpoint"
        shutil.copytree(a.run / "candidate", corrupt)
        raw = (corrupt / "weights.pt").read_bytes()
        (corrupt / "weights.pt").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        refuses("corrupt_weight_bytes", lambda: load(corrupt, "cuda"), "MNLM_MODEL_CHECKPOINT_CORRUPT")
        check("before_action_after_state_unchanged", before == {"base": state_hash(base), "candidate": state_hash(candidate), "native": batch_hash(batches)})
        check("persisted_run_unchanged", original_files == inventory(a.run))
        check("persisted_source_unchanged", original_source == {"panel": inventory(a.panel), "records": file_hash(a.records)})
        put(a.out / "report.json", canonical({"checks": checks, "source": original_source, "run_files": original_files,
            "before_after": before, "decision": decision, "wall_seconds": time.perf_counter() - start,
            "acceptance": "author-checked only", "stage4_accepted": False}))
        print(json.dumps({"checks": len(checks), "passed": all(c["pass"] for c in checks)}))
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "records", "plan", "run", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    run(p.parse_args())
