"""Fresh leased GPU replay of persisted predictor-conditioning batches."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import read, sha, check
from model.renderer_conditioning import gpu_lease, load_predictor
from model.renderer_comparison import read_conditioning
from senses.trajectory_population import publish
from lens.lenscommon import run_cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "conditioning", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--population-sha256", required=True)
    p.add_argument("--conditioning-sha256", required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args(); check(not a.output.exists(), "REPLAY_OUTPUT_EXISTS")
    read(a.population, a.population_sha256)
    c = read_conditioning(a.conditioning, a.conditioning_sha256, a.population, a.population_sha256)
    p = json.loads((a.population/"report.json").read_bytes())
    check(c["runtime"]["platform"] == sys.platform and c["runtime"]["torch"] == str(torch.__version__), "REPLAY_RUNTIME")
    residents = gpu_lease(a.reviewed_desktop_residents, "renderer-conditioning-replay")
    check(c["runtime"]["cuda"] == torch.version.cuda and c["runtime"]["device"] == torch.cuda.get_device_name(0), "REPLAY_DEVICE")
    model = load_predictor(a.population)
    for name, pin in c["dependency_sha256"].items():
        check(sha(Path(sys.modules[name].__file__)) == pin, "REPLAY_CODE")
    source = [np.load(a.population/(n+".npy"), mmap_mode="r") for n in p["lenses"]]
    target = [np.load(a.conditioning/(n+".npy"), mmap_mode="r") for n in p["lenses"]]
    codes = np.load(a.population/"codes.npy")
    expected_codes = np.load(a.conditioning/"predicted-codes.npy")
    starts = sorted({0, p["selected"]//2//32*32, (p["selected"]-1)//32*32})
    replayed = 0; saved = {}
    with torch.inference_mode():
        for start in starts:
            stop = min(start+32, p["selected"])
            history = [torch.tensor(x[start:stop, :8], device="cuda", dtype=torch.float32) for x in source]
            cc = torch.tensor(codes[start:stop, :8], device="cuda", dtype=torch.long)
            h = model.encode(history, cc)
            guess = model.recognise(h)
            actual = model.place(h, history, guess)
            check(np.array_equal(guess[:, -1].cpu().numpy(), expected_codes[start:stop]), "REPLAY_CODES")
            for k, name in enumerate(p["lenses"]):
                value = actual[k][:, -1].float().cpu().numpy()
                check(np.array_equal(value, target[k][start:stop]), "REPLAY_VALUES")
                saved[str(start)+"__"+name] = value
            replayed += stop-start
    # Recount every reported cosine and residual from independent NumPy arrays.
    for k, name in enumerate(p["lenses"]):
        truth = source[k][:, -1].astype(np.float32)
        metrics = dict(mean_cos_to_target=float((target[k]*truth).sum(1).mean()),
                       mean_l2_error=float(np.linalg.norm(target[k]-truth, axis=1).mean()))
        check(metrics == c["metrics"][name], "REPLAY_METRICS")
    read(a.population, a.population_sha256)
    read_conditioning(a.conditioning, a.conditioning_sha256, a.population, a.population_sha256)
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    np.savez(a.output/"replayed.npz", **saved)
    publish(a.output/"report.json", dict(acceptance="author-checked", population_sha256=a.population_sha256,
            conditioning_sha256=a.conditioning_sha256, fixed_batch_starts=starts, rows_replayed=replayed,
            all_rows_replayed=False, all_replayed_predictions_and_codes_exact=True, all_metrics_recounted=True,
            artifact_sha256=sha(a.output/"replayed.npz"), scientific_admission=False))
    print(json.dumps(dict(rows_replayed=replayed, exact=True)))


if __name__ == "__main__":
    sys.exit(run_cli(main))
