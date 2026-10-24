"""Leased, frozen, per-instrument readback of the complete strict-guard pool.

Uses historical pooling/prefix/window/budget with a pinned runtime, fresh
identity and exact repeated-forward probes. Historical incomplete
runtime provenance prevents scientific identity/admission claims.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from audit.renderer_population_readback import sha, check
from audit.guard_render_readback import lines
from lens.learned import snapshot_dir, weights_sha256, preprocessing_sha256
from lens.lenscommon import run_cli
from model.renderer_conditioning import gpu_lease
from model.guard_render import state_sha
from senses.contract import canonical, digest
from senses.trajectory_population import publish


def runtime_matches(rep):
    matched = str(torch.__version__) == rep.get("historical_torch", rep["runtime"]["torch"])
    check(rep["runtime"] == {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
          "platform": sys.platform, "cuda": torch.version.cuda, "historical_torch_matched": matched}
          and (matched or rep.get("runtime_fork_declared") is True), "GUARD_SENSOR_RUNTIME")


def batches(texts, rows, side, budget):
    # Match the historical fixed-block, character-length ordering and budget splits.
    for begin in range(0, len(rows), 8192):
        ordered = [i for i in range(begin, min(begin + 8192, len(rows))) if rows[i]["status"] == "ready"]
        ordered.sort(key=lambda i: (len(texts[i]["text"]), i))
        queue = [ordered[i:i + 256] for i in range(0, len(ordered), 256)]
        while queue:
            chosen = queue.pop(0)
            if max(rows[i]["tokens"] for i in chosen) * len(chosen) > budget:
                check(len(chosen) > 1, "GUARD_SENSOR_SINGLE_WINDOW_BUDGET")
                half = len(chosen) // 2
                queue[:0] = [chosen[:half], chosen[half:]]
            else:
                yield chosen


@torch.inference_mode()
def forward(model, encoded, pool):
    h = model(**encoded).last_hidden_state.float()
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    if pool == "mean":
        value = (h * mask).sum(1) / mask.sum(1)
    elif pool == "cls":
        value = h[:, 0]
    else:
        check(pool == "last", "GUARD_SENSOR_POOL")
        value = h[:, -1]
    finite = torch.isfinite(value).all(1) & (value.norm(dim=-1) > 0)
    value = torch.nn.functional.normalize(value, dim=-1)
    return value, finite


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "renders", "output"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--reviewed-desktop-residents", type=Path)
    a = ap.parse_args()
    start = time.perf_counter(); phases = {}
    check(not a.output.exists() and sha(a.plan / "report.json") == a.plan_sha256, "GUARD_MEASURE_PLAN_PIN")
    rep = json.loads((a.plan / "report.json").read_bytes())
    check(rep["schema"] == "strict-guard-sensor-plan-v1" and rep["scientific_admission"] is False, "GUARD_MEASURE_PLAN")
    runtime_matches(rep)
    check(sha(a.renders / "report.json") == rep["renders_sha256"]
          and sha(a.renders / "texts.jsonl") == rep["texts_sha256"], "GUARD_MEASURE_TEXT_PIN")
    texts = lines(a.renders / "texts.jsonl")
    check(len(texts) <= 2304, "GUARD_MEASURE_SCOPE")
    plans = {}
    for name in rep["lenses"]:
        check(sha(a.plan / (name + ".json")) == rep["instruments"][name]["token_plan_sha256"], "GUARD_MEASURE_TOKEN_PIN")
        plans[name] = json.loads((a.plan / (name + ".json")).read_bytes())
        check([r["text_id"] for r in plans[name]] == [t["id"] for t in texts], "GUARD_MEASURE_TOKEN_ORDER")
    phases["input_pins"] = time.perf_counter() - start
    t = time.perf_counter()
    residents = gpu_lease(a.reviewed_desktop_residents, "strict-renderer-guard-sensors")
    a.output.mkdir(parents=True)
    (a.output / "gpu-before.txt").write_text(residents, encoding="utf-8")
    phases["gpu_lease"] = time.perf_counter() - t
    identities, readings = {}, {}
    for name in rep["lenses"]:
        t = time.perf_counter()
        item, rows = rep["instruments"][name], plans[name]
        spec = item["recipe"]
        snap = snapshot_dir(spec["repo"])
        before, files = weights_sha256(snap)
        prep = preprocessing_sha256(snap)
        check(before == item["weights_sha256"] and files == item["weight_files"]
              and prep == item["preprocessing_sha256"] and snap.name == item["revision"], "GUARD_MEASURE_ASSET_PIN")
        tokenizer = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False, local_files_only=True)
        tokenizer.padding_side = item["padding_side"]
        model = AutoModel.from_pretrained(str(snap), dtype=torch.float32, trust_remote_code=False, local_files_only=True)
        model.eval().requires_grad_(False).cuda()
        check(model.config.hidden_size == spec["dims"], "GUARD_MEASURE_DIMS")
        state_before = state_sha(model)
        phases[name + "_assets_and_load"] = time.perf_counter() - t
        t = time.perf_counter()
        output = np.zeros((len(texts), spec["dims"]), dtype=np.float16)
        valid = np.zeros(len(texts), dtype=np.bool_)
        statuses = [r["status"] for r in rows]
        geometry = list(batches(texts, rows, tokenizer.padding_side, spec["batch_token_budget"]))
        check(geometry, "GUARD_MEASURE_NO_COMPLETE_INPUT")
        probes = sorted({0, len(geometry) // 2, len(geometry) - 1})
        replay = []
        for batch, chosen in enumerate(geometry):
            encoded = tokenizer.pad([rows[i]["inputs"] for i in chosen], padding=True, return_tensors="pt")
            check(encoded["input_ids"].numel() <= spec["batch_token_budget"], "GUARD_MEASURE_PAD_BUDGET")
            encoded = {k: v.cuda() for k, v in encoded.items()}
            value, finite = forward(model, encoded, spec["pool"])
            if batch in probes:
                second, ok = forward(model, encoded, spec["pool"])
                check(torch.equal(finite, ok) and torch.equal(value[finite], second[finite]), "GUARD_MEASURE_DETERMINISM")
                replay.append({"rows": chosen, "fp32_sha256": hashlib.sha256(value[finite].cpu().numpy().tobytes()).hexdigest(), "byte_identical": True})
                replay[-1]["stored_sha256"] = hashlib.sha256(value[finite].cpu().numpy().astype(np.float16).tobytes()).hexdigest()
            good = finite.cpu().numpy()
            selected = np.asarray(chosen)[good]
            output[selected] = value[finite].cpu().numpy().astype(np.float16)
            valid[selected] = True
            for i, ok in zip(chosen, good):
                statuses[i] = "measured" if ok else "MNLM_GUARD_NONFINITE_OR_ZERO"
        torch.cuda.synchronize()
        phases[name + "_forward_and_replay"] = time.perf_counter() - t
        t = time.perf_counter()
        check(state_sha(model) == state_before and weights_sha256(snap) == (before, files)
              and preprocessing_sha256(snap) == prep, "GUARD_MEASURE_FROZEN_DRIFT")
        np.save(a.output / (name + ".npy"), output, allow_pickle=False)
        np.save(a.output / (name + ".valid.npy"), valid, allow_pickle=False)
        identity = {"name": name, "recipe": spec, "revision": snap.name, "weights_sha256": before,
                    "preprocessing_sha256": prep, "runtime": rep["runtime"], "device": torch.cuda.get_device_name(0),
                    "state_sha256": state_before, "execution_code_sha256": sha(Path(__file__)),
                    "batch_geometry_sha256": digest(geometry), "float32_repeated_forward": True,
                    "deterministic_algorithms": True, "tf32": False, "forward_dtype": "float32",
                    "stored_dtype": "float16", "historical_identity_equivalence_certified": False}
        identities[name] = identity
        readings[name] = {"identity_sha256": digest(identity), "rows": len(texts), "measured": int(valid.sum()),
                          "statuses": statuses, "replays": replay, "batches": geometry}
        del model
        torch.cuda.empty_cache()
        phases[name + "_frozen_readback_and_retention"] = time.perf_counter() - t
        print(json.dumps({"name": name, "measured": int(valid.sum()), "rows": len(texts)}), flush=True)
    t = time.perf_counter()
    check(sha(a.plan / "report.json") == a.plan_sha256 and sha(a.renders / "texts.jsonl") == rep["texts_sha256"], "GUARD_MEASURE_SOURCE_DRIFT")
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in a.output.iterdir() if p.is_file()}
    phases["final_hashes"] = time.perf_counter() - t
    wall = time.perf_counter() - start
    report = {"schema": "strict-guard-measurements-v1", "plan_sha256": a.plan_sha256, "renders_sha256": rep["renders_sha256"],
              "texts_sha256": rep["texts_sha256"], "lenses": rep["lenses"], "identities": identities, "readings": readings,
              "files": files, "source_code_sha256": sha(Path(__file__)), "phases_seconds": phases, "wall_seconds": wall,
              "phase_coverage": sum(phases.values()) / wall, "scientific_admission": False, "guard_evaluated": False,
              "historical_identity_equivalence_certified": False, "independent_anchors_evaluated": False,
              "runtime_fork_declared": rep.get("runtime_fork_declared", False)}
    publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha(a.output / "report.json"), "wall_seconds": wall}))


if __name__ == "__main__":
    sys.exit(run_cli(main))
