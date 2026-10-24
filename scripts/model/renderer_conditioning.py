"""Actual frozen region-predictor outputs from a pinned diagnostic population."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import read, sha, check
from model.recognising_lm import Recogniser
from model.recognition import Codebook
from senses.commissioned_windows import check_residents
from senses.trajectory_population import publish
from lens.lenscommon import LensError, run_cli, take_lease


def gpu_lease(reviewed, job):
    check(torch.cuda.is_available(), "CUDA_REQUIRED")
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    try:
        check_residents(residents, json.loads(reviewed.read_bytes()) if reviewed else None)
    except LensError as exc:
        raise LensError("MNLM_RENDERER_GPU_RESIDENT", "wait for unapproved GPU residents; inspect the saved GPU probe") from exc
    take_lease("25", job)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return residents


def load_predictor(root):
    rep = json.loads((root/"predictor-report.json").read_bytes())
    cfg = rep["config"]; book = Codebook(root/"codebook.npz", "cuda")
    model = Recogniser(rep["dims"], book.sizes, book.schema, cfg["width"], cfg["layers"], cfg["heads"], cfg["window"], 0.0)
    model.load_state_dict(torch.load(root/"predictor.pt", map_location="cpu", weights_only=True))
    return model.requires_grad_(False).eval().cuda()


@torch.inference_mode()
def predict(model, slots, codes):
    # Target index 8 is deliberately never passed into any model branch.
    history = [s[:, :8] for s in slots]
    h = model.encode(history, codes[:, :8])
    guess = model.recognise(h)
    return [x[:, -1].float() for x in model.place(h, history, guess)], guess[:, -1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS")
    started = time.perf_counter(); phases = {}
    readback = read(a.population, a.report_sha256)
    r = json.loads((a.population/"report.json").read_bytes())
    phases["input_cold_read"] = time.perf_counter()-started
    tick = time.perf_counter()
    residents = gpu_lease(a.reviewed_desktop_residents, "renderer-actual-predictor-conditioning")
    model = load_predictor(a.population)
    slots = [np.load(a.population/(n+".npy"), mmap_mode="r") for n in r["lenses"]]
    codes = np.load(a.population/"codes.npy")
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    phases["model_load"] = time.perf_counter()-tick
    tick = time.perf_counter(); values = [[] for _ in slots]; guessed = []
    for j in range(0, r["selected"], 32):
        s = [torch.tensor(x[j:j+32], device="cuda", dtype=torch.float32) for x in slots]
        c = torch.tensor(codes[j:j+32], device="cuda", dtype=torch.long)
        out, code = predict(model, s, c)
        for k, x in enumerate(out):
            check(bool(torch.isfinite(x).all()), "NONFINITE_PREDICTION")
            values[k].append(x.cpu().numpy())
        guessed.append(code.cpu().numpy())
        if j == 0:
            # Future/target changes must be irrelevant; changed history must matter.
            mutated = [x.clone() for x in s]
            for x in mutated:
                x[:, -1] *= -1
            changed_codes = c.clone(); changed_codes[:, -1] = (changed_codes[:, -1]+1) % 256
            future, fc = predict(model, mutated, changed_codes)
            check(all(torch.equal(x, y) for x, y in zip(out, future)) and torch.equal(code, fc), "TARGET_LEAKAGE")
            history_changed = [x.clone() for x in s]
            for x in history_changed:
                x[:, :8] *= -1
            altered, _ = predict(model, history_changed, c)
            check(any(not torch.equal(x, y) for x, y in zip(out, altered)), "IGNORED_HISTORY")
    torch.cuda.synchronize(); phases["inference_and_causal_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    metrics = {}
    for k, name in enumerate(r["lenses"]):
        v = np.vstack(values[k]); np.save(a.output/(name+".npy"), v)
        truth = slots[k][:, -1].astype(np.float32)
        metrics[name] = dict(mean_cos_to_target=float((v*truth).sum(1).mean()),
                             mean_l2_error=float(np.linalg.norm(v-truth, axis=1).mean()))
    np.save(a.output/"predicted-codes.npy", np.vstack(guessed))
    check(sha(a.population/"predictor.pt") == r["files"]["predictor.pt"]["sha256"], "WEIGHT_DRIFT")
    check(sha(a.population/"codebook.npz") == r["files"]["codebook.npz"]["sha256"], "BOOK_DRIFT")
    read(a.population, a.report_sha256)
    files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in a.output.iterdir() if p.is_file()}
    phases["publication_and_frozen_hashes"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    publish(a.output/"report.json", dict(schema="renderer-conditioning-v1", population_sha256=a.report_sha256,
            rows=r["selected"], lenses=r["lenses"], dims=r["dims"], files=files, metrics=metrics,
            target_not_input=True, target_mutation_invariant=True, history_mutation_changes_output=True,
            predictor_frozen=True, unsnapped=True, runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            platform=sys.platform, device=torch.cuda.get_device_name(0), precision="fp32", batch=32),
            phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
            source_code_sha256=sha(Path(__file__)), dependency_sha256={name: sha(Path(sys.modules[name].__file__))
                for name in ("model.recognising_lm", "model.recognition")},
            scientific_admission=False, composition_success=False))
    print(json.dumps(dict(rows=r["selected"], report_sha256=sha(a.output/"report.json"), phases_seconds=phases)))


if __name__ == "__main__":
    sys.exit(run_cli(main))
