"""Fixed calibration/evaluation rendering for a strict per-slot guard diagnostic.

No training, retrieval, threshold fitting, anchor inference or lens forward here.
Complete native bytes are retained; malformed outputs stay typed refusals.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_comparison_readback import inspect
from audit.renderer_population_readback import sha, check
from model.renderer_conditioning import gpu_lease
from model.renderer_comparison import ARMS
from model.surface_renderer import Renderer
from model.strict_surface import render_complete
from senses.contract import canonical, digest
from senses.trajectory_population import publish
from lens.lenscommon import run_cli


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(canonical([name, str(value.dtype), list(value.shape)]))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def jobs(rows):
    calibration = [i for i, r in enumerate(rows) if r["split"] == "calibration"]
    evaluation = [i for i, r in enumerate(rows) if r["split"] == "evaluation"]
    check(len(calibration) == len(evaluation) == 256, "GUARD_POPULATION")
    return [("true", "calibration", "true", calibration)] + [
        (arm, "evaluation", route, evaluation) for arm in ARMS for route in ("true", "predicted")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "population", "conditioning", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--comparison-sha256", required=True)
    ap.add_argument("--reviewed-desktop-residents", type=Path)
    a = ap.parse_args()
    check(not a.output.exists(), "GUARD_OUTPUT_EXISTS")
    start = time.perf_counter(); phases = {}
    inspect(a.comparison, a.population, a.conditioning, a.comparison_sha256)
    comparison = json.loads((a.comparison / "report.json").read_bytes())
    population = json.loads((a.population / "report.json").read_bytes())
    rows = [json.loads(s) for s in (a.population / "rows.jsonl").read_bytes().splitlines()]
    plan = jobs(rows)
    check(str(torch.__version__) == comparison["runtime"]["torch"]
          and torch.version.cuda == comparison["runtime"]["cuda"]
          and sys.platform == comparison["runtime"]["platform"], "GUARD_RENDERER_RUNTIME")
    phases["input_readback"] = time.perf_counter() - start
    t = time.perf_counter()
    residents = gpu_lease(a.reviewed_desktop_residents, "strict-guard-fixed-rendering")
    a.output.mkdir(parents=True)
    (a.output / "gpu-before.txt").write_text(residents, encoding="utf-8")
    values = {"true": [], "predicted": []}
    for name in population["lenses"]:
        values["true"].append(torch.tensor(np.load(a.population / (name + ".npy"))[:, -1], dtype=torch.float32, device="cuda"))
        values["predicted"].append(torch.tensor(np.load(a.conditioning / (name + ".npy")), dtype=torch.float32, device="cuda"))
    phases["gpu_and_native_slots"] = time.perf_counter() - t
    checkpoint_pins, state_pins, replays, candidates = {}, {}, [], []
    model, active = None, None
    cfg = comparison["config"]
    for arm, split, route, indices in plan:
        t = time.perf_counter()
        if arm != active:
            if model is not None:
                check(state_sha(model) == state_pins[active], "GUARD_RENDERER_DRIFT")
                del model
            checkpoint = a.comparison / arm / "renderer.pt"
            checkpoint_pins[arm] = sha(checkpoint)
            check(checkpoint_pins[arm] == comparison["arms"][arm]["checkpoint_sha256"], "GUARD_RENDERER_PIN")
            model = Renderer(population["dims"], cfg["width"], cfg["layers"], cfg["heads"])
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.requires_grad_(False).eval().cuda()
            state_pins[arm] = state_sha(model)
            active = arm
        load_seconds = time.perf_counter() - t
        t = time.perf_counter()
        for start_row in range(0, len(indices), 32):
            chosen = indices[start_row:start_row + 32]
            ix = torch.tensor(chosen, device="cuda")
            slots = [x[ix] for x in values[route]]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                rendered = render_complete(model, slots)
                if start_row in (0, 128, 224):
                    repeated = render_complete(model, slots)
                    check(rendered == repeated, "GUARD_RENDER_REPLAY")
                    replays.append({"arm": arm, "split": split, "route": route, "rows": chosen,
                                    "sha256": digest(rendered), "byte_identical": True})
            for row, value in zip(chosen, rendered):
                candidates.append({"case": digest(["strict-guard-case-v1", arm, split, route, row]),
                    "arm": arm, "split": split, "route": route, "row": row,
                    "doc": rows[row]["doc"], **value, "asserted": False,
                    "independent_anchor_status": "MNLM_ANCHOR_UNAVAILABLE"})
            print(json.dumps({"arm": arm, "split": split, "route": route, "completed_rows": start_row + len(chosen)}), flush=True)
        torch.cuda.synchronize()
        phases[f"{arm}_{split}_{route}_render_and_replay"] = time.perf_counter() - t
        phases[f"{arm}_{split}_{route}_load"] = load_seconds
    check(state_sha(model) == state_pins[active], "GUARD_RENDERER_DRIFT")
    del model
    t = time.perf_counter()
    check(len(candidates) == 1792 and len({c["case"] for c in candidates}) == 1792, "GUARD_CASE_COUNT")
    texts = {}

    def retain(text):
        raw = text.encode("utf-8", "strict")
        check(text.strip() and len(raw) <= 512, "GUARD_COMPLETE_TEXT")
        key = hashlib.sha256(raw).hexdigest()
        check(key not in texts or texts[key]["text"] == text, "GUARD_TEXT_COLLISION")
        texts[key] = {"id": key, "text": text, "bytes": len(raw)}
        return key

    target_rows = [i for i, r in enumerate(rows) if r["split"] in ("calibration", "evaluation")]
    targets = [{"row": i, "doc": rows[i]["doc"], "split": rows[i]["split"], "text_id": retain(rows[i]["atoms"][-1]["text"])} for i in target_rows]
    for case in candidates:
        case["text_id"] = retain(case["text"]) if case["status"] == "complete_unasserted" else None
    for filename, items in (("candidates.jsonl", candidates), ("targets.jsonl", targets), ("texts.jsonl", sorted(texts.values(), key=lambda x: x["id"]))):
        with (a.output / filename).open("wb") as out:
            for item in items:
                out.write(canonical(item) + b"\n")
    (a.output / "replays.json").write_bytes(canonical(replays))
    check(sha(a.comparison / "report.json") == a.comparison_sha256
          and sha(a.population / "report.json") == comparison["population_sha256"]
          and sha(a.conditioning / "report.json") == comparison["conditioning_sha256"], "GUARD_SOURCE_DRIFT")
    for arm, pin in checkpoint_pins.items():
        check(sha(a.comparison / arm / "renderer.pt") == pin, "GUARD_RENDERER_DRIFT")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in a.output.iterdir() if p.is_file()}
    phases["retention_and_readback_pins"] = time.perf_counter() - t
    wall = time.perf_counter() - start
    report = {"schema": "strict-guard-renders-v1", "comparison_sha256": a.comparison_sha256,
              "population_sha256": comparison["population_sha256"], "conditioning_sha256": comparison["conditioning_sha256"],
              "source_code_sha256": sha(Path(__file__)), "strict_surface_code_sha256": sha(Path(sys.modules["model.strict_surface"].__file__)),
              "runtime": comparison["runtime"], "checkpoint_sha256": checkpoint_pins, "frozen_state_sha256": state_pins,
              "cases": len(candidates), "targets": len(targets), "unique_complete_texts": len(texts),
              "unique_utf8_bytes": sum(x["bytes"] for x in texts.values()), "statuses": dict(Counter(c["status"] for c in candidates)),
              "calibration_policy": {"arm": "true", "route": "true", "split": "calibration", "quantile": .1,
                                     "method": "linear", "minimum_complete": 50, "semantic_far_certified": False},
              "replayed_rows": sum(len(x["rows"]) for x in replays), "files": files,
              "phases_seconds": phases, "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall,
              "guard_evaluated": False, "independent_anchors_evaluated": False, "scientific_admission": False,
              "composition_success": False, "final_test": False}
    publish(a.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("cases", "unique_complete_texts", "statuses", "wall_seconds", "phase_coverage")}))


if __name__ == "__main__":
    sys.exit(run_cli(main))
