"""Matched true / Gaussian / actual-predictor conditioning, diagnostic only.

Every arm starts from the same from-scratch weights and consumes the same complete
targets, batch order and optimizer budget. Actual predictions are computed once.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import read, sha, check
from model.renderer_conditioning import gpu_lease
from model.surface_renderer import Renderer
from model.strict_surface import encode_complete, losses_encoded, render_complete
from senses.trajectory_population import publish
from lens.lenscommon import run_cli

ARMS = ("true", "gaussian", "predicted")


def read_conditioning(root, pin, population, population_pin):
    check(sha(root/"report.json") == pin, "CONDITIONING_PIN")
    r = json.loads((root/"report.json").read_bytes())
    p = json.loads((population/"report.json").read_bytes())
    check(r["schema"] == "renderer-conditioning-v1" and r["population_sha256"] == population_pin, "CONDITIONING_SOURCE")
    check(r["lenses"] == p["lenses"] and r["dims"] == p["dims"] and r["rows"] == p["selected"], "CONDITIONING_ROWS")
    check(r.get("scientific_admission") is False and r.get("composition_success") is False, "CONDITIONING_CLAIM")
    check(all(r.get(k) is True for k in ("target_not_input", "target_mutation_invariant", "history_mutation_changes_output", "predictor_frozen", "unsnapped")), "CONDITIONING_CONTROLS")
    check(set(r["files"]) == {"gpu-before.txt", "predicted-codes.npy"} | {n+".npy" for n in p["lenses"]}, "CONDITIONING_FILES")
    for name, meta in r["files"].items():
        check(sha(root/name) == meta["sha256"] and (root/name).stat().st_size == meta["bytes"], "CONDITIONING_FILE")
    for name, d in zip(p["lenses"], p["dims"]):
        x = np.load(root/(name+".npy"), allow_pickle=False)
        check(x.shape == (p["selected"], d) and x.dtype == np.float32 and np.isfinite(x).all(), "CONDITIONING_VALUES")
        check(np.all(abs(np.linalg.norm(x, axis=1)-1) < 2e-5), "CONDITIONING_NORM")
    return r


@torch.inference_mode()
def evaluate(model, true, predicted, texts, rows, indices, output, packed, lengths):
    model.eval(); records = []
    # A fixed cyclic derangement has no self-pairs, unlike an unconstrained shuffle.
    other = np.roll(indices, 1)
    for route, arrays, chosen in (("true", true, indices), ("predicted", predicted, indices),
                                  ("other_true", true, other), ("other_predicted", predicted, other)):
        for start in range(0, len(indices), 32):
            targets = indices[start:start+32]; source = chosen[start:start+32]
            slots = [x[torch.as_tensor(source, device="cuda")] for x in arrays]
            length = int(lengths[targets].max()); ix = torch.as_tensor(targets, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, byte_bits, eos_bits, counts = losses_encoded(model, slots, packed[0][ix, :length], packed[1][ix, :length])
            for i, source_i, bb, eb, count in zip(targets, source, byte_bits.tolist(), eos_bits.tolist(), counts.tolist()):
                records.append(dict(row=int(i), conditioning_row=int(source_i), split=rows[i]["split"], route=route,
                                    byte_nll_bits=bb, eos_nll_bits=eb, bytes=count))
    (output/"scores.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    # Predetermined first 16 evaluation rows; no selection by generated quality.
    render_indices = np.asarray([i for i in indices if rows[i]["split"] == "evaluation"][:16])
    generated = []
    for route, arrays in (("true", true), ("predicted", predicted)):
        slots = [x[torch.as_tensor(render_indices, device="cuda")] for x in arrays]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rendered = render_complete(model, slots)
        for i, value in zip(render_indices, rendered):
            generated.append(dict(row=int(i), route=route, target=texts[i], **value,
                                  exact_target=value["text"] == texts[i], asserted=False))
    (output/"renders.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in generated), encoding="utf-8")
    metrics = {}
    for split in ("calibration", "evaluation"):
        metrics[split] = {}
        for route in ("true", "predicted", "other_true", "other_predicted"):
            part = [r for r in records if r["split"] == split and r["route"] == route]
            metrics[split][route] = dict(rows=len(part), bytes=sum(r["bytes"] for r in part),
                    bits_per_byte=sum(r["byte_nll_bits"] for r in part)/sum(r["bytes"] for r in part),
                    mean_eos_bits=sum(r["eos_nll_bits"] for r in part)/len(part))
    return metrics, dict(Counter(r["status"] for r in generated))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "conditioning", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--population-sha256", required=True)
    p.add_argument("--conditioning-sha256", required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    p.add_argument("--steps", type=int, default=800)
    a = p.parse_args(); check(not a.output.exists() and a.steps > 0, "OUTPUT_OR_STEPS")
    started = time.perf_counter(); phases = {}
    read(a.population, a.population_sha256)
    read_conditioning(a.conditioning, a.conditioning_sha256, a.population, a.population_sha256)
    rep = json.loads((a.population/"report.json").read_bytes())
    rows = [json.loads(line) for line in (a.population/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [r["atoms"][-1]["text"] for r in rows]
    train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"])
    evaluation = np.asarray([i for i, r in enumerate(rows) if r["split"] != "train"])
    batches = np.random.default_rng(25).choice(train, size=(a.steps, 32), replace=True)
    phases["input_readback_and_order"] = time.perf_counter()-started
    tick = time.perf_counter()
    residents = gpu_lease(a.reviewed_desktop_residents, "matched-renderer-conditioning")
    # The full target-byte plan is invariant across steps and arms (PC-07).
    packed = tuple(x.cuda() for x in encode_complete(texts, "cpu"))
    lengths = np.asarray([len(t.encode("utf-8"))+2 for t in texts])
    true = [torch.tensor(np.load(a.population/(n+".npy"))[:, -1], device="cuda", dtype=torch.float32) for n in rep["lenses"]]
    predicted = [torch.tensor(np.load(a.conditioning/(n+".npy")), device="cuda") for n in rep["lenses"]]
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    np.save(a.output/"batches.npy", batches)
    cfg = dict(width=256, layers=4, heads=8, batch=32, steps=a.steps, seed=25, lr=3e-4, capacity=512, gaussian_sigma=.3)
    phases["gpu_data_and_plan"] = time.perf_counter()-tick
    arms = {}
    for arm in ARMS:
        tick = time.perf_counter(); out = a.output/arm; out.mkdir()
        torch.manual_seed(25)
        model = Renderer(rep["dims"], cfg["width"], cfg["layers"], cfg["heads"]).cuda()
        # Save common initialization once, then prove exact state identity per arm.
        if arm == ARMS[0]:
            torch.save(model.state_dict(), a.output/"initial.pt")
        initial = torch.load(a.output/"initial.pt", map_location="cuda", weights_only=True)
        check(all(torch.equal(v, initial[k]) for k, v in model.state_dict().items()), "INITIALIZATION_MISMATCH")
        del initial
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.05, betas=(.9, .95))
        gen = torch.Generator(device="cuda").manual_seed(25)
        log = (out/"training.jsonl").open("w", encoding="utf-8")
        model.train()
        for step, ix in enumerate(batches):
            source = predicted if arm == "predicted" else true
            s = [x[torch.as_tensor(ix, device="cuda")] for x in source]
            if arm == "gaussian":
                s = [torch.nn.functional.normalize(x + cfg["gaussian_sigma"]*torch.randn(x.shape, device="cuda", generator=gen)/math.sqrt(x.shape[-1]), dim=-1) for x in s]
            lr = cfg["lr"]*min(1, (step+1)/50)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
            for group in opt.param_groups:
                group["lr"] = lr
            length = int(lengths[ix].max()); target_ix = torch.as_tensor(ix, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _, _ = losses_encoded(model, s, packed[0][target_ix, :length], packed[1][target_ix, :length])
            check(bool(torch.isfinite(loss)), "TRAIN_NONFINITE")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            if step % 100 == 0 or step == a.steps-1:
                record = dict(arm=arm, step=step+1, loss=float(loss.detach()), seconds=time.perf_counter()-tick)
                log.write(json.dumps(record)+"\n"); log.flush(); print(json.dumps(record), flush=True)
        log.close(); torch.cuda.synchronize()
        phases[arm+"_train"] = time.perf_counter()-tick
        tick = time.perf_counter(); torch.save(model.state_dict(), out/"renderer.pt")
        metrics, statuses = evaluate(model, true, predicted, texts, rows, evaluation, out, packed, lengths)
        arms[arm] = dict(metrics=metrics, render_statuses=statuses, parameters=sum(x.numel() for x in model.parameters()),
                         checkpoint_sha256=sha(out/"renderer.pt"), initialization_sha256=sha(a.output/"initial.pt"))
        phases[arm+"_evaluation"] = time.perf_counter()-tick
        del model, opt
    tick = time.perf_counter()
    check(sha(a.population/"report.json") == a.population_sha256 and sha(a.conditioning/"report.json") == a.conditioning_sha256, "SOURCE_DRIFT")
    read(a.population, a.population_sha256)
    read_conditioning(a.conditioning, a.conditioning_sha256, a.population, a.population_sha256)
    files = {p.relative_to(a.output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in a.output.rglob("*") if p.is_file()}
    phases["final_hashes"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    publish(a.output/"report.json", dict(schema="renderer-comparison-v1", config=cfg, arms=arms, files=files,
            population_sha256=a.population_sha256, conditioning_sha256=a.conditioning_sha256,
            phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), platform=sys.platform),
            source_code_sha256=sha(Path(__file__)), dependency_sha256={name: sha(Path(sys.modules[name].__file__))
                for name in ("model.surface_renderer", "model.strict_surface")},
            guard_evaluated=False, independent_anchors_evaluated=False,
            scientific_admission=False, composition_success=False, serving_ready=False, final_test=False))
    print(json.dumps(dict(report_sha256=sha(a.output/"report.json"), wall_seconds=wall, arms=arms)))


if __name__ == "__main__":
    sys.exit(run_cli(main))
