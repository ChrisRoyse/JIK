"""Fresh leased CUDA recomputation of every persisted renderer evaluation score."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_comparison_readback import inspect
from audit.renderer_population_readback import sha, check
from model.renderer_conditioning import gpu_lease
from model.surface_renderer import Renderer
from model.strict_surface import encode_complete
from senses.trajectory_population import publish
from lens.lenscommon import run_cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "population", "conditioning", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args(); check(not a.output.exists(), "SCORE_REPLAY_EXISTS")
    inspect(a.comparison, a.population, a.conditioning, a.report_sha256)
    report = json.loads((a.comparison/"report.json").read_bytes())
    pop = json.loads((a.population/"report.json").read_bytes()); cfg = report["config"]
    check(report["runtime"]["torch"] == str(torch.__version__) and report["runtime"]["platform"] == sys.platform, "SCORE_REPLAY_RUNTIME")
    residents = gpu_lease(a.reviewed_desktop_residents, "renderer-score-cold-replay")
    check(report["runtime"]["cuda"] == torch.version.cuda and report["runtime"]["device"] == torch.cuda.get_device_name(0), "SCORE_REPLAY_DEVICE")
    for module, pin in report["dependency_sha256"].items():
        check(sha(Path(sys.modules[module].__file__)) == pin, "SCORE_REPLAY_CODE")
    rows = [json.loads(s) for s in (a.population/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [r["atoms"][-1]["text"] for r in rows]
    packed = [v.cuda() for v in encode_complete(texts, "cpu")]
    lengths = np.asarray([len(t.encode())+2 for t in texts])
    true = [torch.tensor(np.load(a.population/(n+".npy"))[:, -1], device="cuda", dtype=torch.float32) for n in pop["lenses"]]
    predicted = [torch.tensor(np.load(a.conditioning/(n+".npy")), device="cuda") for n in pop["lenses"]]
    replayed = {}; maximum_delta = 0.0
    with torch.inference_mode():
        for arm in ("true", "gaussian", "predicted"):
            model = Renderer(pop["dims"], cfg["width"], cfg["layers"], cfg["heads"]).cuda().eval()
            model.load_state_dict(torch.load(a.comparison/arm/"renderer.pt", map_location="cuda", weights_only=True))
            records = [json.loads(s) for s in (a.comparison/arm/"scores.jsonl").read_text().splitlines()]
            values = []
            for route in ("true", "predicted", "other_true", "other_predicted"):
                part = [r for r in records if r["route"] == route]
                for start in range(0, len(part), 32):
                    group = part[start:start+32]
                    target = np.asarray([r["row"] for r in group]); source = [r["conditioning_row"] for r in group]
                    length = int(lengths[target].max())
                    tokens, padding = [v[torch.as_tensor(target, device="cuda"), :length] for v in packed]
                    arrays = predicted if "predicted" in route else true
                    slots = [v[torch.as_tensor(source, device="cuda")] for v in arrays]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(slots, tokens[:, :-1], padding[:, :-1]).float()
                    # Independent likelihood calculation via log-softmax + gather.
                    labels = tokens[:, 1:]
                    nll = -torch.log_softmax(logits.transpose(1, 2), dim=1).gather(1, labels[:, None]).squeeze(1)
                    bb = (nll*((labels < 256) & ~padding[:, 1:])).sum(1)/np.log(2)
                    eb = (nll*((labels == 257) & ~padding[:, 1:])).sum(1)/np.log(2)
                    counts = ((labels < 256) & ~padding[:, 1:]).sum(1)
                    actual = np.column_stack((bb.cpu().numpy(), eb.cpu().numpy(), counts.cpu().numpy()))
                    expected = np.asarray([[r["byte_nll_bits"], r["eos_nll_bits"], r["bytes"]] for r in group])
                    delta = abs(actual-expected); maximum_delta = max(maximum_delta, float(delta.max()))
                    # Independent CE vs log-softmax kernels may round at fp32 ULPs.
                    tolerance = 8*np.finfo(np.float32).eps*np.maximum(1, abs(expected))
                    if not np.all(delta <= tolerance):
                        publish(a.output.with_name(a.output.name+"-refusal.json"), dict(arm=arm, route=route, start=start,
                                actual=actual.tolist(), expected=expected.tolist(), max_difference=float(delta.max()),
                                report_sha256=a.report_sha256, scientific_admission=False))
                    check(np.all(delta <= tolerance) and np.array_equal(actual[:, 2], expected[:, 2]), "SCORE_REPLAY_NLL")
                    values.extend(actual.tolist())
            replayed[arm] = np.asarray(values)
            del model
    inspect(a.comparison, a.population, a.conditioning, a.report_sha256)
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    np.savez(a.output/"scores.npz", **replayed)
    publish(a.output/"report.json", dict(acceptance="author-checked", comparison_sha256=a.report_sha256,
            scored_rows=sum(len(v) for v in replayed.values()), every_saved_score_replayed=True,
            calculation="independent log_softmax/gather", tolerance="8 fp32 eps times max(1, absolute expected)",
            max_absolute_difference=maximum_delta, artifacts_sha256=sha(a.output/"scores.npz"),
            generation_replayed=False, scientific_admission=False, code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(rows=sum(len(v) for v in replayed.values()), max_absolute_difference=maximum_delta)))


if __name__ == "__main__":
    sys.exit(run_cli(main))
